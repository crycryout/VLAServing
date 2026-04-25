#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from bench_gr00t_unified_multistage_microbatch_runtime import (
    UnifiedRuntime,
    _build_phase_locked_requests,
    _run_unified_replay_async_backpressure,
    _warmup_unified_async_backpressure,
)


OUT = Path("/root/autodl-tmp/VLAServing/results/_tmp_gr00t_unified_async_fullgraph_single.json")
SEED = 20260425


def run() -> dict:
    runtime = UnifiedRuntime(
        "nvidia/GR00T-N1.6-bridge",
        "cuda:0",
        "bfloat16",
        num_slots=8,
        inference_steps=4,
        local_files_only=False,
    )
    runtime.enable_step_cudagraph([1, 2, 4])
    runtime.enable_vlm_full_cudagraph(
        llm_attn_impl="sdpa",
        vision_attn_impl="flash_attention_2",
    )
    config = {
        "step_batch_trigger": 2,
        "vlm_slot_count": 1,
        "vlm_max_prefill_ready": 1,
        "vlm_max_llm_queue_depth": 1,
        "prefill_priority": 0,
        "llm_priority": 0,
        "step_priority": 0,
        "pause_prefill_when_steps_ready": False,
        "pause_llm_when_steps_ready": False,
    }
    _warmup_unified_async_backpressure(runtime, **config)
    runtime.generator.manual_seed(SEED)
    result = _run_unified_replay_async_backpressure(
        runtime,
        _build_phase_locked_requests(
            cohort_size=2,
            num_cohorts=2,
            waves=8,
            period_ms=80.0,
            phase_stride_ms=20.0,
        ),
        deadline_ms=100.0,
        **config,
    )
    payload = {
        "config": config,
        "result": result,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    try:
        result = run()
        print(json.dumps(result, indent=2))
        print(OUT)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
