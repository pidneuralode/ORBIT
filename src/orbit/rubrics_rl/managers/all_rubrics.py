"""verl tensor adapter for shared ORBIT scoring; no import-time API client."""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from orbit.rubrics_rl.execution import run_batch_scoring
from orbit.rubrics_rl.reward import score_many

from .options import OverlongBufferConfig

logger = logging.getLogger(__name__)


def score_batch_in_worker(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> list[float]:
    """Score the complete batch in one terminable worker, preserving score_many."""
    if extra_info is None:
        return []
    return score_many(extra_info["solutions"], extra_info["extra_infos"])


@register("dapo_rubrics_async_all")
class DAPORewardManagerAsyncAll(AbstractRewardManager):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable | None = None,
        reward_fn_key: str = "data_source",
        max_resp_len: int | None = None,
        overlong_buffer_cfg: OverlongBufferConfig | None = None,
        num_processes: int = 64,
        scoring_timeout: float = 600.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.scoring_timeout = scoring_timeout

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None
            assert self.max_resp_len >= self.overlong_buffer_cfg.len

        logger.info("ORBIT shared scorer initialized")

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # 1. 准备数据
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
        extra_infos = [data_item.non_tensor_batch.get("extra_info", None) for data_item in data]

        # Run the original whole-batch scorer in one owned, terminable process.
        # num_processes remains a compatibility argument: batch semantics use one worker.
        try:
            batch_results = run_batch_scoring(
                score_batch_in_worker,
                ["batch"],
                [""],
                [None],
                [{"solutions": sequences_str, "extra_infos": extra_infos}],
                num_processes=1,
                timeout=self.scoring_timeout,
            )
            results = batch_results[0]
            if not isinstance(results, list) or len(results) != len(sequences_str):
                raise ValueError("Batch result shape mismatch")
        except Exception:
            logger.warning("Reward batch execution failed; awarding zero")
            results = [0.0] * len(sequences_str)

        # 3. 填充结果
        for i in range(len(data)):
            score = float(results[i]) if results[i] is not None else 0.0
            data_source = data_sources[i]
            valid_response_length = valid_response_lengths[i].item()

            reward_extra_info["acc"].append(score)
            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = cast(int, self.max_resp_len) - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(
                    -exceed_len / overlong_buffer_len * overlong_penalty_factor, 0
                )
                reward += overlong_reward

                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            idx = max(0, min(valid_response_length - 1, reward_tensor.shape[1] - 1))
            reward_tensor[i, idx] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                logger.debug("Reward sample index=%d score=%f reward=%f", i, score, reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
