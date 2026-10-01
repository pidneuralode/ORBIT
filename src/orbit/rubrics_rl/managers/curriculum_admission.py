# Copyright 2024 PRIME team and/or its affiliates
# ... (Apache License) ...

import copy
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from orbit.rubrics_rl.curriculum import CurriculumStateMixin, is_mastered, update_ema
from orbit.rubrics_rl.execution import run_batch_scoring
from orbit.rubrics_rl.reward import compute_dynamics_score as rubrics_compute_score

from .options import OverlongBufferConfig

logger = logging.getLogger(__name__)


@register("dapo_rubrics_curriculum_admission")
class DAPORewardManagerCurriculumAdmission(CurriculumStateMixin, AbstractRewardManager):
    """Gate criterion admission by active-window EMA mastery."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable | None = None,
        reward_fn_key: str = "data_source",
        max_resp_len: int | None = None,
        overlong_buffer_cfg: OverlongBufferConfig | None = None,
        num_processes: int = 32,
        scoring_timeout: float = 300.0,
        # --- 课程学习的核心参数 ---
        query_id_key: str = "query_id",  # extra_info 中用于标识 query 的 key
        active_window_size: int = 6,  # 保持 N 个 Rubrics 处于 "active" 状态
        mastery_threshold: float = 0.9,  # 90% EMA 通过率视为 "mastered"
        ema_alpha: float = 0.1,  # EMA 更新的平滑因子
        admission_threshold: float = 0.75,
        max_admit_per_step: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = rubrics_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_processes = num_processes
        self.scoring_timeout = scoring_timeout

        # --- 课程学习初始化 ---
        self.query_id_key = query_id_key
        self.active_window_size = active_window_size
        self.mastery_threshold = mastery_threshold
        self.ema_alpha = ema_alpha

        # 对于rubrics进入的速率进行限制
        self.admission_threshold = admission_threshold
        self.max_admit_per_step = max_admit_per_step

        # curriculum_state 现在是一个空字典
        # 它将在运行时, 第一次遇到 query_id 时被动态填充
        # 格式: {query_id: {orig_idx_str: {"ema": 0.0, "status": "candidate"|"active"|"mastered"}, ...}}
        self.curriculum_state = {}

        logger.info("Curriculum manager event")

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                "max_resp_len must be provided if overlong_buffer_cfg is set, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # ====================
        # 第一步：批量提取数据
        # ====================
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        eos_token = self.tokenizer.eos_token
        if eos_token:
            sequences_str = [
                s[: -len(eos_token)] if s.endswith(eos_token) else s for s in sequences_str
            ]

        data_sources = data.non_tensor_batch[self.reward_fn_key]
        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data
        ]

        original_extra_infos = [
            data_item.non_tensor_batch.get("extra_info", {}) for data_item in data
        ]
        # new_extra_infos 只是一个临时的、用于 *这一轮* 调用的副本
        new_extra_infos = [copy.deepcopy(ei) for ei in original_extra_infos]
        # 用于 verl 进行相关的监控措施 记录rubrics窗口的滑动情况
        curriculum_stats_for_logging = []

        # ====================
        # 第二步：Pacing (起搏) - 构建 "active_rubrics"
        # ====================
        for ei in new_extra_infos:
            query_id = ei.get(self.query_id_key)
            all_rubrics = ei.get("rubrics")
            sorted_indices = ei.get("sorted_rubric_indices")

            # 添加默认的监控指标
            default_stats = {
                "progress": 0,
                "active": len(all_rubrics or []),
                "mastered": 0,
                "total": len(all_rubrics or []),
                "avg_pos": 0.0,
                "avg_ema": 0.0,
            }

            if not query_id or not all_rubrics or not isinstance(all_rubrics, list):
                logger.warning("Curriculum manager event")
                curriculum_stats_for_logging.append(default_stats)
                continue

            # 动态状态管理
            # 1. 检查是否是新 query, 如果是, 则初始化状态
            if query_id not in self.curriculum_state:
                if not self._initialize_state_for_query(query_id, ei):
                    # 初始化失败 (e.g., 'sorted_rubric_indices' 缺失)
                    curriculum_stats_for_logging.append(default_stats)
                    continue  # 同样, 回退到使用所有 rubrics

            query_state = self.curriculum_state[query_id]
            if query_state is None:  # 之前尝试过初始化但失败了
                curriculum_stats_for_logging.append(default_stats)
                continue

            # 2. "起搏" (Pacing) 逻辑:
            #    根据 "query_state" (动态) 和 "sorted_indices" (静态, 来自ei)
            #    构建 "active_rubrics_list"

            active_rubrics_list = []
            num_active = 0
            num_mastered = 0
            active_positions = []
            active_ema_scores = []

            # 遍历 "课程表" (来自 ei)
            for position, orig_idx in enumerate(sorted_indices):
                orig_idx_str = str(orig_idx)

                if orig_idx_str not in query_state:
                    logger.warning("Curriculum manager event")
                    continue

                state = query_state[orig_idx_str]

                # 检查 "active" 状态的 rubrics
                if state["status"] == "active":
                    if is_mastered(state["ema"], self.mastery_threshold):
                        state["status"] = "mastered"
                        num_mastered += 1
                        logger.info("Curriculum manager event")
                    else:
                        rubric_obj = copy.deepcopy(all_rubrics[orig_idx])
                        rubric_obj["_internal_original_index"] = orig_idx
                        active_rubrics_list.append(rubric_obj)
                        num_active += 1
                        active_positions.append(position)
                        active_ema_scores.append(state["ema"])

                # "mastered" 状态的 rubrics 会被自动跳过
                elif state["status"] == "mastered":
                    num_mastered += 1

            # 开始加入新的rubrics 同时需要保证整体过渡的稳定性
            current_avg_ema = np.mean(active_ema_scores) if active_ema_scores else 1.0
            is_gate_open = current_avg_ema > self.admission_threshold
            num_active = len(active_rubrics_list)
            num_newly_admitted = 0

            # rubrics的冷启动操作
            is_cold_start = (num_active == 0) and (num_mastered == 0)

            if is_cold_start:
                admission_limit_this_step = self.active_window_size
                logger.info("Curriculum manager event")
            else:
                admission_limit_this_step = self.max_admit_per_step

            if num_active < self.active_window_size and is_gate_open:
                for position, orig_idx in enumerate(sorted_indices):
                    # 同时检查三个条件
                    if num_active >= self.active_window_size:
                        break  # 1. 窗口满了
                    if num_newly_admitted >= admission_limit_this_step:
                        break  # 2. 达到本轮“限速”

                    orig_idx_str = str(orig_idx)
                    if orig_idx_str not in query_state:
                        continue
                    state = query_state[orig_idx_str]

                    if state["status"] == "candidate":
                        state["status"] = "active"
                        logger.info("Curriculum manager event")
                        rubric_obj = copy.deepcopy(all_rubrics[orig_idx])
                        rubric_obj["_internal_original_index"] = orig_idx
                        active_rubrics_list.append(rubric_obj)
                        num_active += 1
                        active_positions.append(position)
                        num_newly_admitted += 1

            ei["active_rubrics"] = active_rubrics_list

            avg_pos = np.mean(active_positions) if active_positions else 0.0
            total_count = len(sorted_indices)
            progress = (num_mastered / total_count) if total_count > 0 else 0

            curriculum_stats_for_logging.append(
                {
                    "progress": progress,
                    "active": num_active,
                    "mastered": num_mastered,
                    "total": total_count,
                    "avg_pos": avg_pos,
                    "avg_ema": current_avg_ema,
                }
            )

        # ====================
        # 第三步：异步并行计算所有分数
        # ====================
        try:
            results = run_batch_scoring(
                compute_score_fn=self.compute_score,
                data_sources=data_sources,
                solution_strs=sequences_str,
                ground_truths=ground_truths,
                extra_infos=new_extra_infos,  # [修改] 使用我们动态构建的 new_extra_infos
                num_processes=self.num_processes,
                timeout=self.scoring_timeout,
            )
        except Exception:
            logger.warning("Batch scoring failed; awarding zero to %d samples", len(data))
            results = [None] * len(data)

        # ====================
        # 第四步：处理结果 + 更新状态 (Update)
        # ====================
        for i in range(len(data)):
            result_dict = results[i]
            valid_response_length = valid_response_lengths[i].item()
            data_source = data_sources[i]

            # 1. 解析分数
            if not result_dict or isinstance(result_dict, Exception):
                score = 0.0
                reward_extra_info["acc"].append(0.0)
            else:
                score = result_dict.get("normalized_score", 0.0)
                reward_extra_info["acc"].append(score)

            reward = score

            # 2. 更新课程状态 (Update State)
            if result_dict and "rubric_results_map" in result_dict:
                query_id = new_extra_infos[i].get(self.query_id_key)

                # 检查 state 是否存在 (如果初始化失败, 则 query_state 为 None)
                query_state = self.curriculum_state.get(query_id)

                if query_state:
                    results_map = result_dict["rubric_results_map"]  # {orig_idx_str: {"met": bool}}

                    for orig_idx_str, met_dict in results_map.items():
                        if orig_idx_str in query_state:
                            state = query_state[orig_idx_str]
                            # 只更新 "active" 状态的 EMA
                            if state["status"] == "active":
                                is_met = met_dict.get("met", False)
                                points = met_dict.get("points", 0.0)
                                success_signal = 0.0
                                if points < 0:
                                    success_signal = 1.0 if not is_met else 0.0
                                else:
                                    success_signal = 1.0 if is_met else 0.0
                                old_ema = state["ema"]
                                state["ema"] = update_ema(old_ema, success_signal, self.ema_alpha)

            # 3. 超长惩罚
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

            # 4. 填充 Tensor
            reward_tensor[i, valid_response_length - 1] = reward

            stats = curriculum_stats_for_logging[i]
            reward_extra_info["curriculum/progress"].append(stats["progress"])
            reward_extra_info["curriculum/active_count"].append(stats["active"])
            reward_extra_info["curriculum/mastered_count"].append(stats["mastered"])
            reward_extra_info["curriculum/total_count"].append(stats["total"])
            reward_extra_info["curriculum/avg_pos"].append(stats["avg_pos"])
            reward_extra_info["curriculum/avg_ema"].append(stats["avg_ema"])

            # 5. 打印日志
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                logger.debug("Reward sample index=%d score=%f", i, score)

        # ====================
        # 第五步：返回结果
        # ====================
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
