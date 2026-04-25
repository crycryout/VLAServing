#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
from pathlib import Path
import time
import statistics

import torch

from bench_gr00t_unified_multistage_microbatch_runtime import UnifiedRuntime
from gr00t_vlm_full_cudagraph import VLMFullCudaGraphExecutor


OUT = Path("/root/autodl-tmp/VLAServing/results/_tmp_gr00t_vlm_full_cudagraph_executor.json")


def _cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    ts = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - ts) * 1000.0


def run() -> dict:
    runtime = UnifiedRuntime(
        "nvidia/GR00T-N1.6-bridge",
        "cuda:0",
        "bfloat16",
        num_slots=8,
        inference_steps=4,
        local_files_only=False,
    )
    slot_id = 0
    with torch.inference_mode():
        base_task, _ = runtime.prepare_task(0, slot_id, 0.0)
        for _ in range(3):
            runtime.prepare_task(0, slot_id, 0.0)
        base_samples = [
            _cuda_ms(lambda: runtime.prepare_task(0, slot_id, 0.0))
            for _ in range(8)
        ]

    executor = VLMFullCudaGraphExecutor(
        runtime,
        sample_slot_id=slot_id,
        llm_attn_impl="sdpa",
        vision_attn_impl="flash_attention_2",
    )
    with torch.inference_mode():
        executor.launch(slot_id)
        graph_task = executor.materialize_task(0, slot_id, 0.0)
        for _ in range(3):
            executor.launch(slot_id)
        graph_samples = [
            _cuda_ms(lambda: executor.launch(slot_id))
            for _ in range(8)
        ]

    payload = {
        "baseline_prepare_ms": {
            "mean_ms": float(sum(base_samples) / len(base_samples)),
            "p50_ms": float(statistics.median(base_samples)),
        },
        "full_graph_prepare_ms": {
            "mean_ms": float(sum(graph_samples) / len(graph_samples)),
            "p50_ms": float(statistics.median(graph_samples)),
        },
        "backbone_max_abs": float(
            (base_task.backbone_features - graph_task.backbone_features).abs().max().item()
        ),
        "backbone_mean_abs": float(
            (base_task.backbone_features - graph_task.backbone_features).abs().mean().item()
        ),
        "state_max_abs": float(
            (base_task.state_features - graph_task.state_features).abs().max().item()
        ),
        "state_mean_abs": float(
            (base_task.state_features - graph_task.state_features).abs().mean().item()
        ),
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
