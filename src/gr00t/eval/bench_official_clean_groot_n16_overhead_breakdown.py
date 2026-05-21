#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


OFFICIAL_ROOT = Path("/root/autodl-tmp/Isaac-GR00T-official-clean-20260520")
OUT = Path(
    "/root/autodl-tmp/VLAServing/results/"
    "official_clean_groot_n16_noncompute_overhead_breakdown_20260520.json"
)


def stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "num_samples": 0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "num_samples": int(arr.size),
    }


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git_rev(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return float(sum(end - start for start, end in merged))


def interval_gaps_us(intervals: list[tuple[float, float]]) -> list[float]:
    if not intervals:
        return []
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [float(merged[i + 1][0] - merged[i][1]) for i in range(len(merged) - 1)]


def event_device_us(evt: Any) -> float:
    if hasattr(evt, "self_device_time_total"):
        return float(evt.self_device_time_total)
    if hasattr(evt, "self_cuda_time_total"):
        return float(evt.self_cuda_time_total)
    return 0.0


def is_cuda_event(evt: Any) -> bool:
    return "CUDA" in str(getattr(evt, "device_type", "")).upper()


def classify_device_event(name: str) -> str:
    low = name.lower()
    if "memcpy" in low or "copy" in low and "kernel" not in low:
        return "device_memcpy"
    if "memset" in low:
        return "device_memset"
    return "device_kernel"


def sum_cpu_by_prefix(events: list[Any], prefixes: tuple[str, ...]) -> tuple[float, int]:
    total = 0.0
    count = 0
    for evt in events:
        name = str(getattr(evt, "name", getattr(evt, "key", "")))
        if any(name.startswith(prefix) for prefix in prefixes):
            total += float(getattr(evt, "self_cpu_time_total", 0.0)) / 1000.0
            count += 1
    return total, count


def parse_profile(prof: Any) -> dict[str, Any]:
    events = list(prof.events())
    cuda_events = [evt for evt in events if is_cuda_event(evt)]
    device_by_class: dict[str, list[tuple[float, float]]] = {
        "device_kernel": [],
        "device_memcpy": [],
        "device_memset": [],
    }
    device_rows = []
    for evt in cuda_events:
        tr = getattr(evt, "time_range", None)
        if tr is None:
            continue
        start = float(tr.start)
        end = float(tr.end)
        cls = classify_device_event(str(evt.name))
        device_by_class.setdefault(cls, []).append((start, end))
        device_rows.append(
            {
                "name": str(evt.name),
                "class": cls,
                "start_us": start,
                "end_us": end,
                "duration_us": float(end - start),
            }
        )

    all_device_intervals = [iv for rows in device_by_class.values() for iv in rows]
    if all_device_intervals:
        timeline_span_us = max(end for _, end in all_device_intervals) - min(
            start for start, _ in all_device_intervals
        )
    else:
        timeline_span_us = 0.0
    device_union_us = interval_union_us(all_device_intervals)
    gaps = interval_gaps_us(all_device_intervals)

    launch_total, launch_count = sum_cpu_by_prefix(
        events, ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaGraphLaunch")
    )
    graph_total, graph_count = sum_cpu_by_prefix(events, ("cudaGraphLaunch",))
    launch_kernel_total, launch_kernel_count = sum_cpu_by_prefix(
        events, ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel")
    )
    sync_total, sync_count = sum_cpu_by_prefix(
        events, ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")
    )
    memcpy_runtime_total, memcpy_runtime_count = sum_cpu_by_prefix(
        events, ("cudaMemcpy", "cudaMemcpyAsync")
    )
    memset_runtime_total, memset_runtime_count = sum_cpu_by_prefix(events, ("cudaMemsetAsync",))
    malloc_total, malloc_count = sum_cpu_by_prefix(events, ("cudaMalloc", "cudaFree"))

    kernel_sum_us = sum(end - start for start, end in device_by_class.get("device_kernel", []))
    memcpy_sum_us = sum(end - start for start, end in device_by_class.get("device_memcpy", []))
    memset_sum_us = sum(end - start for start, end in device_by_class.get("device_memset", []))
    return {
        "device_timeline_span_ms": float(timeline_span_us / 1000.0),
        "device_active_union_ms": float(device_union_us / 1000.0),
        "device_idle_gap_sum_ms": float(max(timeline_span_us - device_union_us, 0.0) / 1000.0),
        "device_idle_gap_stats_ms": stats([g / 1000.0 for g in gaps]),
        "device_kernel_sum_ms": float(kernel_sum_us / 1000.0),
        "device_kernel_union_ms": float(interval_union_us(device_by_class.get("device_kernel", [])) / 1000.0),
        "device_memcpy_sum_ms": float(memcpy_sum_us / 1000.0),
        "device_memcpy_union_ms": float(interval_union_us(device_by_class.get("device_memcpy", [])) / 1000.0),
        "device_memset_sum_ms": float(memset_sum_us / 1000.0),
        "device_memset_union_ms": float(interval_union_us(device_by_class.get("device_memset", [])) / 1000.0),
        "device_event_count": int(len(device_rows)),
        "device_kernel_count": int(len(device_by_class.get("device_kernel", []))),
        "device_memcpy_count": int(len(device_by_class.get("device_memcpy", []))),
        "device_memset_count": int(len(device_by_class.get("device_memset", []))),
        "cpu_cuda_launch_runtime_ms": float(launch_total),
        "cpu_cuda_launch_runtime_count": int(launch_count),
        "cpu_cuda_kernel_launch_runtime_ms": float(launch_kernel_total),
        "cpu_cuda_kernel_launch_count": int(launch_kernel_count),
        "cpu_cuda_graph_launch_runtime_ms": float(graph_total),
        "cpu_cuda_graph_launch_count": int(graph_count),
        "cpu_cuda_sync_runtime_ms": float(sync_total),
        "cpu_cuda_sync_runtime_count": int(sync_count),
        "cpu_cuda_memcpy_runtime_ms": float(memcpy_runtime_total),
        "cpu_cuda_memcpy_runtime_count": int(memcpy_runtime_count),
        "cpu_cuda_memset_runtime_ms": float(memset_runtime_total),
        "cpu_cuda_memset_runtime_count": int(memset_runtime_count),
        "cpu_cuda_alloc_free_runtime_ms": float(malloc_total),
        "cpu_cuda_alloc_free_count": int(malloc_count),
        "top_device_events": sorted(device_rows, key=lambda row: row["duration_us"], reverse=True)[:12],
    }


@torch.inference_mode()
def measure_baseline(fn: Callable[[], Any], iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    host_enqueue = []
    sync_wait = []
    wall = []
    cuda_event = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        _ = fn()
        t1 = time.perf_counter()
        end.record()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        host_enqueue.append((t1 - t0) * 1000.0)
        sync_wait.append((t2 - t1) * 1000.0)
        wall.append((t2 - t0) * 1000.0)
        cuda_event.append(float(start.elapsed_time(end)))
    return {
        "host_enqueue_ms": stats(host_enqueue),
        "sync_wait_after_enqueue_ms": stats(sync_wait),
        "wall_sync_ms": stats(wall),
        "cuda_event_timeline_ms": stats(cuda_event),
    }


def profile_repeats(fn: Callable[[], Any], repeats: int, profiler_warmup: bool) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    if profiler_warmup:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            _ = fn()
            torch.cuda.synchronize()

    rows = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _ = fn()
            torch.cuda.synchronize()
        rows.append(parse_profile(prof))

    keys = [
        "device_timeline_span_ms",
        "device_active_union_ms",
        "device_idle_gap_sum_ms",
        "device_kernel_sum_ms",
        "device_kernel_union_ms",
        "device_memcpy_sum_ms",
        "device_memcpy_union_ms",
        "device_memset_sum_ms",
        "device_memset_union_ms",
        "cpu_cuda_launch_runtime_ms",
        "cpu_cuda_kernel_launch_runtime_ms",
        "cpu_cuda_graph_launch_runtime_ms",
        "cpu_cuda_sync_runtime_ms",
        "cpu_cuda_memcpy_runtime_ms",
        "cpu_cuda_memset_runtime_ms",
        "cpu_cuda_alloc_free_runtime_ms",
    ]
    counts = [
        "device_event_count",
        "device_kernel_count",
        "device_memcpy_count",
        "device_memset_count",
        "cpu_cuda_launch_runtime_count",
        "cpu_cuda_kernel_launch_count",
        "cpu_cuda_graph_launch_count",
        "cpu_cuda_sync_runtime_count",
        "cpu_cuda_memcpy_runtime_count",
        "cpu_cuda_memset_runtime_count",
        "cpu_cuda_alloc_free_count",
    ]
    summary = {key: stats([float(row[key]) for row in rows]) for key in keys}
    for key in counts:
        summary[key] = stats([float(row[key]) for row in rows])
    summary["device_idle_gap_p50_each_gap_ms"] = stats(
        [float(row["device_idle_gap_stats_ms"]["p50"]) for row in rows]
    )
    summary["device_idle_gap_p95_each_gap_ms"] = stats(
        [float(row["device_idle_gap_stats_ms"]["p95"]) for row in rows]
    )
    return {"summary": summary, "repeats": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=2)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    off = load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_benchmark_inference_overhead",
    )
    import gr00t  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset_path = args.dataset_path or str(official_root / "demo_data" / "gr1.PickNPlace")
    policy = off.Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=off.EmbodimentTag(args.embodiment_tag),
        device="cuda",
        strict=True,
    )
    policy.model.action_head.model.forward = torch.compile(
        policy.model.action_head.model.forward,
        mode=args.compile_mode,
    )

    observation = _load_observation(off, policy, dataset_path, args.embodiment_tag)
    data_processing = _measure_data_processing(off, policy, observation, args.iterations, args.warmup)
    collated = off.prepare_model_inputs(policy, observation)
    backbone_inputs, action_inputs = policy.model.prepare_input(collated)

    # Trigger compile and cudagraph paths before profiling.
    for _ in range(args.warmup + 3):
        _ = policy.model.get_action(collated)
    torch.cuda.synchronize()
    backbone_outputs = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    targets: dict[str, Callable[[], Any]] = {
        "backbone_prepared": lambda: policy.model.backbone(backbone_inputs),
        "action_head_prepared": lambda: policy.model.action_head.get_action(
            backbone_outputs, action_inputs
        ),
        "full_model_prepared": lambda: policy.model.get_action(collated),
    }

    results: dict[str, Any] = {}
    for name, fn in targets.items():
        baseline = measure_baseline(fn, args.iterations, args.warmup)
        profiled = profile_repeats(fn, args.profile_repeats, profiler_warmup=True)
        b_event = float(baseline["cuda_event_timeline_ms"]["p50"])
        p = profiled["summary"]
        results[name] = {
            "baseline": baseline,
            "profiled": profiled,
            "normalized_to_baseline_cuda_event_p50": {
                "device_kernel_union_ms": p["device_kernel_union_ms"]["p50"],
                "device_memcpy_union_ms": p["device_memcpy_union_ms"]["p50"],
                "device_memset_union_ms": p["device_memset_union_ms"]["p50"],
                "device_idle_gap_sum_ms": p["device_idle_gap_sum_ms"]["p50"],
                "cpu_cuda_launch_runtime_ms": p["cpu_cuda_launch_runtime_ms"]["p50"],
                "cpu_cuda_sync_runtime_ms": p["cpu_cuda_sync_runtime_ms"]["p50"],
                "unattributed_vs_baseline_ms": float(
                    b_event
                    - p["device_kernel_union_ms"]["p50"]
                    - p["device_memcpy_union_ms"]["p50"]
                    - p["device_memset_union_ms"]["p50"]
                    - p["device_idle_gap_sum_ms"]["p50"]
                ),
            },
        }

    payload = {
        "meta": {
            "date": "2026-05-20",
            "scope": (
                "Fresh official Isaac-GR00T n1.6-release torch.compile non-compute "
                "overhead breakdown using CUDA events plus torch.profiler raw CUDA events."
            ),
            "official_root": str(official_root),
            "official_git_rev": git_rev(official_root),
            "gr00t_import_file": str(Path(gr00t.__file__).resolve()),
            "model_path": args.model_path,
            "dataset_path": dataset_path,
            "embodiment_tag": args.embodiment_tag,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "compile_mode": args.compile_mode,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "profile_repeats": args.profile_repeats,
            "limitations": [
                "torch.profiler perturbs execution and can change torch.compile/cudagraph behavior; baseline CUDA-event numbers are the timing anchor.",
                "device_idle_gap_sum_ms is measured from raw CUDA event gaps inside profiler and includes kernel boundary, stream idle, launch delay, and graph/runtime handoff gaps.",
                "intermediate tensor materialization through HBM inside kernels is not directly measurable with torch.profiler; only explicit memcpy/memset and allocator/runtime proxies are reported.",
            ],
        },
        "data_processing_wall_ms": data_processing,
        "targets": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    compact = {"data_processing_p50_ms": data_processing["p50"]}
    for name, row in results.items():
        b = row["baseline"]
        p = row["profiled"]["summary"]
        compact[name] = {
            "baseline_cuda_event_p50_ms": b["cuda_event_timeline_ms"]["p50"],
            "baseline_wall_p50_ms": b["wall_sync_ms"]["p50"],
            "baseline_host_enqueue_p50_ms": b["host_enqueue_ms"]["p50"],
            "baseline_sync_wait_p50_ms": b["sync_wait_after_enqueue_ms"]["p50"],
            "profile_kernel_union_p50_ms": p["device_kernel_union_ms"]["p50"],
            "profile_memcpy_union_p50_ms": p["device_memcpy_union_ms"]["p50"],
            "profile_memset_union_p50_ms": p["device_memset_union_ms"]["p50"],
            "profile_idle_gap_sum_p50_ms": p["device_idle_gap_sum_ms"]["p50"],
            "profile_cpu_launch_runtime_p50_ms": p["cpu_cuda_launch_runtime_ms"]["p50"],
            "profile_cpu_graph_launch_runtime_p50_ms": p[
                "cpu_cuda_graph_launch_runtime_ms"
            ]["p50"],
            "profile_cpu_sync_runtime_p50_ms": p["cpu_cuda_sync_runtime_ms"]["p50"],
        }
    print(out)
    print(json.dumps(compact, indent=2))


def _load_observation(off: Any, policy: Any, dataset_path: str, embodiment_tag: str):
    modality_config = policy.get_modality_config()
    dataset = off.LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )
    episode_data = dataset[0]
    step_data = off.extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=off.EmbodimentTag(embodiment_tag),
        allow_padding=False,
    )
    return {
        "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
        "state": {k: step_data.states[k][None] for k in step_data.states},
        "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
    }


def _measure_data_processing(
    off: Any, policy: Any, observation: dict[str, Any], iterations: int, warmup: int
) -> dict[str, Any]:
    for _ in range(warmup):
        _ = off.prepare_model_inputs(policy, observation)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = off.prepare_model_inputs(policy, observation)
        times.append((time.perf_counter() - t0) * 1000.0)
    return stats(times)


if __name__ == "__main__":
    main()
