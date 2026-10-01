"""Bounded process execution for CPU reward callbacks without payload logging."""

import asyncio
import logging
import math
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

logger = logging.getLogger("orbit.adapters.execution")


def _stop_workers(executor: ProcessPoolExecutor) -> None:
    # Python 3.10 has no public terminate_workers API. Snapshot before shutdown,
    # then terminate only processes belonging to this explicitly owned executor.
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


async def single_compute_score_wrapper(
    compute_score_fn: Callable[..., Any],
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    executor: ProcessPoolExecutor,
    timeout: float = 300.0,
) -> Any:
    try:
        future = asyncio.get_running_loop().run_in_executor(
            executor,
            partial(
                compute_score_fn,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Reward task timed out; result omitted")
        return None
    except Exception:
        logger.warning("Reward task failed; result omitted")
        return None


async def parallel_compute_scores_async(
    compute_score_fn: Callable[..., Any],
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[dict[str, Any] | None],
    num_processes: int = 64,
    timeout: float = 500.0,
    max_inflight_samples: int | None = None,
) -> list[Any]:
    if (
        isinstance(num_processes, bool)
        or not isinstance(num_processes, int)
        or not 1 <= num_processes <= 1024
    ):
        raise ValueError("num_processes must be from 1 to 1024")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    inputs = list(zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True))
    if not inputs:
        return []
    executor = ProcessPoolExecutor(
        max_workers=min(num_processes, len(inputs)), mp_context=multiprocessing.get_context("spawn")
    )
    # A bounded window is used by the baseline manager; the other managers keep
    # their existing whole-batch scheduling when no limit is supplied.
    limit = len(inputs) if max_inflight_samples is None else max(1, max_inflight_samples)
    results: list[Any] = [None] * len(inputs)
    pending: dict[asyncio.Task[Any], int] = {}
    next_index = 0

    def submit(index: int) -> None:
        ds, solution, reference, info = inputs[index]
        task = asyncio.create_task(
            single_compute_score_wrapper(
                compute_score_fn, ds, solution, reference, info, executor, timeout
            )
        )
        pending[task] = index

    try:
        while next_index < len(inputs) and len(pending) < limit:
            submit(next_index)
            next_index += 1
        while pending:
            completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                results[pending.pop(task)] = task.result()
                if next_index < len(inputs):
                    submit(next_index)
                    next_index += 1
        return results
    finally:
        _stop_workers(executor)


def run_batch_scoring(
    compute_score_fn: Callable[..., Any],
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[dict[str, Any] | None],
    num_processes: int = 128,
    timeout: float = 300.0,
    max_inflight_samples: int | None = None,
) -> list[Any]:
    """Synchronous training adapter; call the async API inside an event loop."""
    return asyncio.run(
        parallel_compute_scores_async(
            compute_score_fn,
            data_sources,
            solution_strs,
            ground_truths,
            extra_infos,
            num_processes,
            timeout,
            max_inflight_samples,
        )
    )
