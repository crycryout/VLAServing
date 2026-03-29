#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag

from benchmark_segmented_batch_n1d6 import Config as BenchConfig
from benchmark_segmented_batch_n1d6 import _apply_torch_compile
from benchmark_segmented_batch_n1d6 import _load_model
from benchmark_segmented_batch_n1d6 import _make_dummy_inputs
from benchmark_segmented_batch_n1d6 import _maybe_override_inference_steps
from benchmark_segmented_batch_n1d6 import _move_batch_to_device
from benchmark_segmented_batch_n1d6 import _timed_call
from benchmark_segmented_batch_n1d6 import _torch_dtype


@dataclass
class Config:
    model_path: str = "/root/autodl-tmp/local_gr00t_n1d6_models/gr00t_n1d6-libero"
    base_anchor_path: str = "nvidia/GR00T-N1.6-3B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    warmup: int = 3
    iterations: int = 8
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    compile_target: str = "dit"
    compile_backend: str = "inductor"
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    inference_steps: int = 1
    output_json: str = "/root/autodl-tmp/VLAServing/results/groot_n1d6_same_model_batch_curve_step1_compile.json"


def _stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_req_s_mean": float(1000.0 / arr.mean()),
        "throughput_samples_s_mean": float((1000.0 / arr.mean()) * 0.0),  # filled later
        "per_sample_mean_ms": 0.0,  # filled later
        "per_sample_p50_ms": 0.0,  # filled later
    }


def run(cfg: Config) -> dict[str, Any]:
    bench_cfg = BenchConfig(
        model_paths=[cfg.model_path],
        base_anchor_path=cfg.base_anchor_path,
        device=cfg.device,
        dtype=cfg.dtype,
        local_files_only=cfg.local_files_only,
        warmup=cfg.warmup,
        iterations=cfg.iterations,
        single_model_batch_size=1,
        segmented_batch_size=1,
        embodiment_tag=cfg.embodiment_tag,
        compile_target=cfg.compile_target,
        compile_backend=cfg.compile_backend,
        compile_mode=cfg.compile_mode,
        compile_fullgraph=cfg.compile_fullgraph,
        compile_dynamic=cfg.compile_dynamic,
        inference_steps=cfg.inference_steps,
    )

    dtype = _torch_dtype(cfg.dtype)
    loaded = _load_model(cfg.model_path, bench_cfg, move_to_device=True, device=cfg.device)
    _maybe_override_inference_steps(loaded.model, cfg.inference_steps)
    _apply_torch_compile([loaded], bench_cfg)

    results: list[dict[str, Any]] = []
    for batch_size in cfg.batch_sizes:
        dummy_inputs = _move_batch_to_device(
            _make_dummy_inputs(loaded.model, bench_cfg, batch_size=batch_size),
            cfg.device,
            dtype,
        )
        for _ in range(cfg.warmup):
            _ = loaded.model.get_action(dict(dummy_inputs))

        times_ms: list[float] = []
        for _ in range(cfg.iterations):
            elapsed_ms, _ = _timed_call(lambda: loaded.model.get_action(dict(dummy_inputs)))
            times_ms.append(float(elapsed_ms))

        stats = _stats(times_ms)
        stats["batch_size"] = int(batch_size)
        stats["throughput_samples_s_mean"] = float((1000.0 / stats["mean_ms"]) * batch_size)
        stats["per_sample_mean_ms"] = float(stats["mean_ms"] / batch_size)
        stats["per_sample_p50_ms"] = float(stats["p50_ms"] / batch_size)
        stats["service_ms_for_scheduler"] = float(stats["p50_ms"])
        results.append(stats)
        print(
            f"[batch={batch_size}] "
            f"mean={stats['mean_ms']:.2f} ms p50={stats['p50_ms']:.2f} ms "
            f"per_sample_p50={stats['per_sample_p50_ms']:.2f} ms "
            f"throughput={stats['throughput_samples_s_mean']:.2f} samples/s",
            flush=True,
        )

    payload = {
        "model_path": cfg.model_path,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "warmup": cfg.warmup,
        "iterations": cfg.iterations,
        "batch_sizes": cfg.batch_sizes,
        "compile_target": cfg.compile_target,
        "compile_mode": cfg.compile_mode,
        "compile_dynamic": cfg.compile_dynamic,
        "inference_steps": cfg.inference_steps,
        "results": results,
    }
    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
