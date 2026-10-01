# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from orbit.rubrics_rl.execution import run_batch_scoring as execute_reward_batch
from orbit.rubrics_rl.reward import compute_score as rubrics_compute_score

from .options import OverlongBufferConfig

logger = logging.getLogger(__name__)


def run_batch_scoring(
    compute_score_fn: Callable,
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[dict | None],
    num_processes: int = 4,
    timeout: float = 240.0,
    max_inflight_samples: int | None = None,
):
    """Keep the manager's request limits; execution is shared with other managers."""
    return execute_reward_batch(
        compute_score_fn,
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
        num_processes=num_processes,
        timeout=timeout,
        max_inflight_samples=max_inflight_samples,
    )


@register("dapo_rubrics_async")
class DAPORewardManagerAsync(AbstractRewardManager):
    """Score responses in a bounded worker pool and emit token-level rewards."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable | None = None,
        reward_fn_key: str = "data_source",
        max_resp_len: int | None = None,
        overlong_buffer_cfg: OverlongBufferConfig | None = None,
        num_processes: int = 4,
        scoring_timeout: float = 240.0,
        max_inflight_samples: int | None = None,
    ) -> None:
        """
        Args:
            tokenizer: 分词器
            num_examine: 每个数据源打印的样本数量
            compute_score: 自定义的打分函数
            reward_fn_key: 从non_tensor_batch中获取数据源的key
            max_resp_len: 最大回复长度
            overlong_buffer_cfg: 超长惩罚配置
            num_processes: 异步进程池大小
            scoring_timeout: 单个样本打分超时时间（秒）
            max_inflight_samples: 同时提交到进程池的样本上限
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = rubrics_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_processes = num_processes
        self.scoring_timeout = scoring_timeout
        self.max_inflight_samples = max(1, max_inflight_samples or num_processes)

        # 验证配置
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                "max_resp_len must be provided if overlong_buffer_cfg is set, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        主要的调用接口

        Args:
            data: DataProto对象，包含batch数据
            return_dict: 是否返回字典格式

        Returns:
            reward_tensor 或包含额外信息的字典
        """
        # 如果已经有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # 初始化返回值
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # ====================
        # 第一步：批量提取数据（批量处理）
        # ====================
        # 获取prompt和response的基本信息
        prompt_ids = data.batch["prompts"]  # shape: [batch_size, prompt_length]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]  # shape: [batch_size, response_length]

        # 计算每个样本的有效response长度
        # attention_mask[:, prompt_length:] 表示response部分的mask
        # sum(dim=-1) 得到每个样本response的有效token数量
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )  # shape: [batch_size]

        # 批量解码所有response
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # 移除所有response结尾的eos_token
        eos_token = self.tokenizer.eos_token
        if eos_token:
            sequences_str = [
                s[: -len(eos_token)] if s.endswith(eos_token) else s for s in sequences_str
            ]

        # 批量获取数据源
        data_sources = data.non_tensor_batch[self.reward_fn_key]  # list of data sources

        # 批量获取ground_truth
        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data
        ]

        # 批量获取extra_info
        extra_infos = [data_item.non_tensor_batch.get("extra_info", None) for data_item in data]

        # ====================
        # 第二步：异步并行计算所有分数
        # ====================
        try:
            results = run_batch_scoring(
                compute_score_fn=self.compute_score,
                data_sources=data_sources,
                solution_strs=sequences_str,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                num_processes=self.num_processes,
                timeout=self.scoring_timeout,
                max_inflight_samples=self.max_inflight_samples,
            )
        except Exception:
            logger.warning("Reward batch failed; awarding zero to %d samples", len(data))
            results = [None] * len(data)

        # ====================
        # 第三步：处理结果并填充reward_tensor
        # ====================
        for i in range(len(data)):
            result = results[i]
            valid_response_length = valid_response_lengths[i].item()
            data_source = data_sources[i]

            # 处理计算结果
            if result is None or isinstance(result, Exception):
                # 超时或失败的情况，设置默认分数为0
                score = 0.0
                reward_extra_info["acc"].append(0.0)
            elif isinstance(result, dict):
                # 结果是字典，提取score和其他信息
                score = result.get("score", 0.0)
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                # 结果是标量
                score = float(result)
                reward_extra_info["acc"].append(score)

            reward = score

            # 应用超长惩罚（如果配置了）
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = cast(int, self.max_resp_len) - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                # 只有超过expected_len才有惩罚
                overlong_reward = min(
                    -exceed_len / overlong_buffer_len * overlong_penalty_factor, 0
                )
                reward += overlong_reward

                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # 将reward填充到对应位置（最后一个有效token的位置）
            reward_tensor[i, valid_response_length - 1] = reward

            # 打印调试信息
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                logger.debug("Reward sample index=%d score=%f reward=%f", i, score, reward)

        # ====================
        # 第四步：返回结果
        # ====================
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
