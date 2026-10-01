# Copyright 2024 PRIME team and/or its affiliates
# ... (Apache License) ...

import copy
import logging
import random
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import cast

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


# =============================================================================
#  Part 1: 异步和进程池包装器 (通用工具函数)
# =============================================================================


@register("dapo_rubrics_stochastic_review")
class DAPORewardManagerStochasticReview(CurriculumStateMixin, AbstractRewardManager):
    """Review mastered criteria and weight judgments by updated EMA."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable | None = None,
        reward_fn_key: str = "data_source",
        max_resp_len: int | None = None,
        overlong_buffer_cfg: OverlongBufferConfig | None = None,
        num_processes: int = 64,
        scoring_timeout: float = 500.0,
        # --- 自适应课程参数 ---
        query_id_key: str = "query_id",
        active_window_size: int = 5,  # 窗口总大小
        mastery_threshold: float = 0.9,  # EMA > 0.9 视为毕业
        ema_alpha: float = 0.1,  # EMA 平滑因子
        # [新机制参数]
        review_ratio: float = 0.5,  # Fraction of the active window reserved for mastered criteria
        random_seed: int | None = None,
        weighting_power: float = 1.0,  # 加权力度：Weight = (1-EMA)^power. power=1(线性), power=2(平方,更激进)
        # 柔性加权方式
        min_weight: float = 0.2,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = rubrics_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_processes = num_processes
        self.scoring_timeout = scoring_timeout

        # 课程参数
        self.query_id_key = query_id_key
        self.active_window_size = active_window_size
        self.mastery_threshold = mastery_threshold
        self.ema_alpha = ema_alpha
        self.review_ratio = review_ratio
        self.random = random if random_seed is None else random.Random(random_seed)
        self.weighting_power = weighting_power
        self.min_weight = min_weight

        # 状态存储
        self.curriculum_state = {}

        logger.info("Stochastic curriculum initialized")

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None
            assert self.max_resp_len >= self.overlong_buffer_cfg.len

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
        # Step 1: 提取数据
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
        new_extra_infos = [copy.deepcopy(ei) for ei in original_extra_infos]

        curriculum_stats_for_logging: list[Mapping[str, float | int]] = []

        # ====================
        # Step 2: Pacing (Stochastic Review 逻辑)
        # ====================
        for ei in new_extra_infos:
            query_id = ei.get(self.query_id_key)
            all_rubrics = ei.get("rubrics")
            sorted_indices = ei.get("sorted_rubric_indices")

            default_stats = {
                "progress": 0,
                "active": len(all_rubrics or []),
                "pool_mastered": 0,
                "window_size": 0,
            }

            if not query_id or not all_rubrics:
                curriculum_stats_for_logging.append(default_stats)
                continue

            # 初始化状态
            if query_id not in self.curriculum_state:
                if not self._initialize_state_for_query(query_id, ei):
                    curriculum_stats_for_logging.append(default_stats)
                    continue
            query_state = self.curriculum_state[query_id]
            if query_state is None:
                curriculum_stats_for_logging.append(default_stats)
                continue

            # --- [A] 维护三个池子 ---
            pool_unfinished_active = []  # 必须保留：正在学，还没学会
            pool_mastered = []  # 复习池：已经学会
            pool_candidate = []  # 候选池：还没开始

            for orig_idx in sorted_indices:
                orig_idx_str = str(orig_idx)
                if orig_idx_str not in query_state:
                    continue
                state = query_state[orig_idx_str]

                # 检查毕业逻辑
                if state["status"] == "active":
                    if is_mastered(state["ema"], self.mastery_threshold):
                        state["status"] = "mastered"
                        # 毕业不打印大量日志，保持清爽，但在debug时很有用
                        # logger.debug(f"Rubric {orig_idx} graduated!")

                # 构建 rubric 对象
                rubric_obj = copy.deepcopy(all_rubrics[orig_idx])
                rubric_obj["_internal_original_index"] = orig_idx

                if state["status"] == "active":
                    pool_unfinished_active.append(rubric_obj)
                elif state["status"] == "mastered":
                    pool_mastered.append(rubric_obj)
                elif state["status"] == "candidate":
                    pool_candidate.append(rubric_obj)

            # --- [B] 组装窗口 (Stochastic Selection) ---
            final_active_list = []

            # 1. 优先保留：尚未毕业的 active 题目 (基础)
            final_active_list.extend(pool_unfinished_active)

            slots_left = self.active_window_size - len(final_active_list)

            # 2. 随机复习 (从已毕业的题目中随机抽)
            if slots_left > 0:
                max_review_count = int(self.active_window_size * self.review_ratio)
                num_to_review = min(slots_left, max_review_count, len(pool_mastered))

                if num_to_review > 0:
                    # 随机抽取！防止遗忘，且填补窗口
                    review_items = self.random.sample(pool_mastered, num_to_review)
                    final_active_list.extend(review_items)
                    slots_left -= num_to_review

            # 3. 填补新题 (从 Candidate 中按顺序取)
            if slots_left > 0:
                # 限制一下进入的速度 避免一下子加入太难的问题
                num_to_add = min(slots_left, len(pool_candidate))
                if num_to_add > 0:
                    new_items = pool_candidate[:num_to_add]
                    final_active_list.extend(new_items)
                    # 激活状态
                    for item in new_items:
                        idx_str = str(item["_internal_original_index"])
                        query_state[idx_str]["status"] = "active"
                        logger.info("Curriculum manager event")

            ei["active_rubrics"] = final_active_list

            # 记录日志
            total_len = len(sorted_indices)
            curriculum_stats_for_logging.append(
                {
                    "progress": len(pool_mastered) / total_len if total_len else 0,
                    "active": len(pool_unfinished_active),  # "真正"在学的困难题目数
                    "pool_mastered": len(pool_mastered),
                    "window_size": len(final_active_list),
                }
            )

        # ====================
        # Step 3: 异步计算分数 (无需修改)
        # ====================
        try:
            results = run_batch_scoring(
                compute_score_fn=self.compute_score,
                data_sources=data_sources,
                solution_strs=sequences_str,
                ground_truths=ground_truths,
                extra_infos=new_extra_infos,
                num_processes=self.num_processes,
                timeout=self.scoring_timeout,
            )
        except Exception:
            logger.error("Curriculum manager event")
            results = [None] * len(data)

        # ====================
        # Step 4: 结果处理 & 柔性加权 (Soft Weighting)
        # ====================
        for i in range(len(data)):
            result_dict = results[i]
            valid_response_length = valid_response_lengths[i].item()

            score = 0.0

            if result_dict and "rubric_results_map" in result_dict:
                query_id = new_extra_infos[i].get(self.query_id_key)
                query_state = self.curriculum_state.get(query_id)

                if query_state:
                    results_map = result_dict["rubric_results_map"]

                    total_weighted_score = 0.0
                    total_weight = 0.0

                    # 遍历当前窗口内的结果
                    for orig_idx_str, met_dict in results_map.items():
                        if orig_idx_str in query_state:
                            state = query_state[orig_idx_str]

                            # 计算 Met 信号
                            is_met = met_dict.get("met", False)
                            points = met_dict.get("points", 0.0)
                            if points < 0:
                                success_signal = 1.0 if not is_met else 0.0
                            else:
                                success_signal = 1.0 if is_met else 0.0

                            # 更新 EMA (即使是复习题，如果做错了，EMA下降，下次权重会自动变大)
                            current_ema = state["ema"]
                            state["ema"] = update_ema(current_ema, success_signal, self.ema_alpha)

                            # --- [核心逻辑] 柔性加权 ---
                            # Interpolate between min_weight and full weight using difficulty.
                            # 简单题 (EMA=0.9) -> weight 极小
                            # 难题 (EMA=0.1) -> weight 极大
                            flex_factor = pow((1.0 - state["ema"]), self.weighting_power)
                            weight = self.min_weight + (1.0 - self.min_weight) * flex_factor

                            total_weighted_score += weight * success_signal
                            total_weight += weight

                    if total_weight > 0:
                        score = total_weighted_score / total_weight
                    else:
                        score = result_dict.get("normalized_score", 0.0)
                else:
                    score = result_dict.get("normalized_score", 0.0)
            elif result_dict:
                score = result_dict.get("normalized_score", 0.0)

            reward_extra_info["acc"].append(score)
            reward = score

            # Overlong Penalty (保留超长惩罚逻辑)
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

            reward_tensor[i, valid_response_length - 1] = reward

            # 日志
            stats = curriculum_stats_for_logging[i]
            reward_extra_info["curriculum/progress"].append(stats["progress"])
            reward_extra_info["curriculum/pool_mastered"].append(stats["pool_mastered"])
            reward_extra_info["curriculum/active_unfinished"].append(stats["active"])

            if data_sources[i] not in already_print_data_sources:
                already_print_data_sources[data_sources[i]] = 0
            if already_print_data_sources[data_sources[i]] < self.num_examine:
                already_print_data_sources[data_sources[i]] += 1
                logger.debug("Reward sample index=%d score=%f", i, score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
