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
    "official_clean_groot_n16_cudagraph_overheads_20260521.json"
)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git_rev(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "num_samples": 0,
        }
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "num_samples": int(arr.size),
    }


def observation_from_official(off: Any, policy: Any, dataset_path: str, embodiment_tag: str):
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


def summarize_components(components: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "data_processing": stats(components["data_processing"]),
        "backbone": stats(components["backbone"]),
        "action_head": stats(components["action_head"]),
        "e2e": stats(components["e2e"]),
        "frequency_hz_from_e2e_p50": float(1000.0 / np.percentile(components["e2e"], 50)),
    }


@torch.inference_mode()
def measure_call(fn: Callable[[], Any], iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    host_enqueue = []
    sync_wait = []
    wall = []
    event_ms = []
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
        event_ms.append(float(start.elapsed_time(end)))

    return {
        "host_enqueue_ms": stats(host_enqueue),
        "sync_wait_after_enqueue_ms": stats(sync_wait),
        "wall_sync_ms": stats(wall),
        "cuda_event_timeline_ms": stats(event_ms),
    }


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


def is_cuda_event(evt: Any) -> bool:
    return "CUDA" in str(getattr(evt, "device_type", "")).upper()


def classify_device_event(name: str) -> str:
    low = name.lower()
    if "memcpy" in low or ("copy" in low and "kernel" not in low):
        return "device_memcpy"
    if "memset" in low:
        return "device_memset"
    return "device_kernel"


def sum_cpu_by_prefix(key_avgs: list[Any], prefixes: tuple[str, ...]) -> tuple[float, int]:
    total_us = 0.0
    count = 0
    for evt in key_avgs:
        key = str(evt.key)
        if any(key.startswith(prefix) for prefix in prefixes):
            total_us += float(evt.self_cpu_time_total)
            count += int(evt.count)
    return float(total_us / 1000.0), int(count)


def sum_cpu_by_contains(key_avgs: list[Any], patterns: tuple[str, ...]) -> tuple[float, int]:
    total_us = 0.0
    count = 0
    for evt in key_avgs:
        key = str(evt.key)
        if any(pattern in key for pattern in patterns):
            total_us += float(evt.self_cpu_time_total)
            count += int(evt.count)
    return float(total_us / 1000.0), int(count)


def parse_profile(prof: Any) -> dict[str, Any]:
    key_avgs = list(prof.key_averages())
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
    active_union_us = interval_union_us(all_device_intervals)
    gaps = interval_gaps_us(all_device_intervals)

    launch_total, launch_count = sum_cpu_by_prefix(
        key_avgs, ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaGraphLaunch")
    )
    kernel_launch_total, kernel_launch_count = sum_cpu_by_prefix(
        key_avgs, ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel")
    )
    graph_launch_total, graph_launch_count = sum_cpu_by_prefix(key_avgs, ("cudaGraphLaunch",))
    sync_total, sync_count = sum_cpu_by_prefix(
        key_avgs, ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")
    )
    memcpy_total, memcpy_count = sum_cpu_by_prefix(key_avgs, ("cudaMemcpy", "cudaMemcpyAsync"))
    memset_total, memset_count = sum_cpu_by_prefix(key_avgs, ("cudaMemsetAsync",))
    alloc_total, alloc_count = sum_cpu_by_prefix(
        key_avgs, ("cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync")
    )
    aten_alloc_total, aten_alloc_count = sum_cpu_by_contains(
        key_avgs,
        (
            "aten::empty",
            "aten::empty_strided",
            "aten::zeros",
            "aten::full",
            "aten::arange",
            "aten::randn",
            "aten::cat",
            "aten::_to_copy",
            "aten::copy_",
            "aten::clone",
            "aten::contiguous",
        ),
    )

    def sum_intervals(cls: str) -> float:
        return float(sum(end - start for start, end in device_by_class.get(cls, [])) / 1000.0)

    return {
        "device_timeline_span_ms": float(timeline_span_us / 1000.0),
        "device_active_union_ms": float(active_union_us / 1000.0),
        "device_idle_gap_sum_ms": float(max(timeline_span_us - active_union_us, 0.0) / 1000.0),
        "device_idle_gap_each_ms": stats([g / 1000.0 for g in gaps]),
        "device_kernel_sum_ms": sum_intervals("device_kernel"),
        "device_kernel_union_ms": float(
            interval_union_us(device_by_class.get("device_kernel", [])) / 1000.0
        ),
        "device_memcpy_sum_ms": sum_intervals("device_memcpy"),
        "device_memcpy_union_ms": float(
            interval_union_us(device_by_class.get("device_memcpy", [])) / 1000.0
        ),
        "device_memset_sum_ms": sum_intervals("device_memset"),
        "device_memset_union_ms": float(
            interval_union_us(device_by_class.get("device_memset", [])) / 1000.0
        ),
        "device_event_count": int(len(device_rows)),
        "device_kernel_count": int(len(device_by_class.get("device_kernel", []))),
        "device_memcpy_count": int(len(device_by_class.get("device_memcpy", []))),
        "device_memset_count": int(len(device_by_class.get("device_memset", []))),
        "cpu_cuda_launch_runtime_ms": launch_total,
        "cpu_cuda_launch_runtime_count": launch_count,
        "cpu_cuda_kernel_launch_runtime_ms": kernel_launch_total,
        "cpu_cuda_kernel_launch_count": kernel_launch_count,
        "cpu_cuda_graph_launch_runtime_ms": graph_launch_total,
        "cpu_cuda_graph_launch_count": graph_launch_count,
        "cpu_cuda_sync_runtime_ms": sync_total,
        "cpu_cuda_sync_runtime_count": sync_count,
        "cpu_cuda_memcpy_runtime_ms": memcpy_total,
        "cpu_cuda_memcpy_runtime_count": memcpy_count,
        "cpu_cuda_memset_runtime_ms": memset_total,
        "cpu_cuda_memset_runtime_count": memset_count,
        "cpu_cuda_alloc_free_runtime_ms": alloc_total,
        "cpu_cuda_alloc_free_count": alloc_count,
        "cpu_aten_alloc_like_self_ms": aten_alloc_total,
        "cpu_aten_alloc_like_count": aten_alloc_count,
        "top_device_events": sorted(device_rows, key=lambda row: row["duration_us"], reverse=True)[:16],
    }


def profile_repeats(fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    rows = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
        ) as prof:
            with torch.inference_mode():
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
        "cpu_aten_alloc_like_self_ms",
    ]
    count_keys = [
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
        "cpu_aten_alloc_like_count",
    ]
    summary: dict[str, Any] = {key: stats([row[key] for row in rows]) for key in keys}
    for key in count_keys:
        summary[key] = stats([float(row[key]) for row in rows])
    return {"summary": summary, "repeats": rows}


def cuda_memory_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


@torch.inference_mode()
def try_capture_cuda_graph(
    name: str,
    fn: Callable[[], Any],
    warmup: int,
) -> tuple[dict[str, Any], Callable[[], Any] | None]:
    try:
        # The official action head samples CUDA random noise. Resetting the
        # generator makes this probe robust after torch.compile/profiler runs
        # that may have used internal CUDA Graph/RNG bookkeeping.
        torch.cuda.manual_seed_all(1234)
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - warmup failure is benchmark data.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return (
            {
                "name": name,
                "capture_ok": False,
                "failure_phase": "warmup_before_capture",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "memory_after": cuda_memory_snapshot(),
            },
            None,
        )

    torch.cuda.reset_peak_memory_stats()
    before = cuda_memory_snapshot()
    graph = torch.cuda.CUDAGraph()
    static_out: Any = None
    t0 = time.perf_counter()
    try:
        torch.cuda.manual_seed_all(1234)
        with torch.cuda.graph(graph):
            static_out = fn()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001 - capture failures are benchmark data.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        after_fail = cuda_memory_snapshot()
        return (
            {
                "name": name,
                "capture_ok": False,
                "failure_phase": "capture",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "capture_wall_ms": (time.perf_counter() - t0) * 1000.0,
                "memory_before": before,
                "memory_after": after_fail,
            },
            None,
        )

    after = cuda_memory_snapshot()

    def replay() -> Any:
        graph.replay()
        return static_out

    return (
        {
            "name": name,
            "capture_ok": True,
            "capture_wall_ms": capture_ms,
            "memory_before": before,
            "memory_after": after,
            "memory_delta": {
                key: int(after[key] - before[key])
                for key in ["allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes"]
            },
        },
        replay,
    )


def compact_target(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile", {}).get("summary", {})
    out = {
        "wall_p50_ms": row["timing"]["wall_sync_ms"]["p50_ms"],
        "cuda_event_p50_ms": row["timing"]["cuda_event_timeline_ms"]["p50_ms"],
        "host_enqueue_p50_ms": row["timing"]["host_enqueue_ms"]["p50_ms"],
        "sync_wait_p50_ms": row["timing"]["sync_wait_after_enqueue_ms"]["p50_ms"],
    }
    for key in [
        "device_kernel_union_ms",
        "device_memcpy_union_ms",
        "device_memset_union_ms",
        "device_idle_gap_sum_ms",
        "cpu_cuda_launch_runtime_ms",
        "cpu_cuda_kernel_launch_runtime_ms",
        "cpu_cuda_graph_launch_runtime_ms",
        "cpu_cuda_sync_runtime_ms",
        "cpu_cuda_alloc_free_runtime_ms",
        "cpu_aten_alloc_like_self_ms",
    ]:
        if key in profile:
            out[key + "_p50"] = profile[key]["p50_ms"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--overhead-iterations", type=int, default=8)
    parser.add_argument("--profile-repeats", type=int, default=2)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    off = load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_cudagraph_overhead_benchmark_inference",
    )
    import gr00t  # noqa: PLC0415

    off.set_seed(args.seed)
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
    observation = observation_from_official(off, policy, dataset_path, args.embodiment_tag)

    # Trigger the official torch.compile path before measurement.
    collated = off.prepare_model_inputs(policy, observation)
    for _ in range(args.warmup + 3):
        _ = policy.model.get_action(collated)
    torch.cuda.synchronize()

    shared_data_processing = off.benchmark_data_processing(
        policy,
        observation,
        args.num_iterations,
        warmup=10,
    )
    official_components_raw = off.benchmark_components(
        policy,
        observation,
        args.num_iterations,
        warmup=args.warmup,
    )
    official_components = {
        "data_processing": np.asarray(shared_data_processing, dtype=np.float64),
        "backbone": np.asarray(official_components_raw["backbone"], dtype=np.float64),
        "action_head": np.asarray(official_components_raw["action_head"], dtype=np.float64),
    }
    official_components["e2e"] = off.compute_e2e_from_components(official_components)

    # Fixed prepared tensors for overhead and CUDA Graph probes.
    collated = off.prepare_model_inputs(policy, observation)
    backbone_inputs, action_inputs = policy.model.prepare_input(collated)
    backbone_outputs = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    target_fns: dict[str, Callable[[], Any]] = {
        "backbone_component_prepared": lambda: policy.model.backbone(backbone_inputs),
        "action_head_component_prepared": lambda: policy.model.action_head.get_action(
            backbone_outputs,
            action_inputs,
        ),
        "full_model_gpu_only_prepared": lambda: policy.model.get_action(collated),
    }

    baseline_profiles: dict[str, Any] = {}
    for name, fn in target_fns.items():
        timing = measure_call(fn, args.overhead_iterations, warmup=2)
        profile = profile_repeats(fn, args.profile_repeats)
        baseline_profiles[name] = {"timing": timing, "profile": profile}

    graph_results: dict[str, Any] = {
        "full_model_gpu_only_prepared": {
            "name": "full_model_gpu_only_prepared",
            "capture_ok": False,
            "not_attempted": True,
            "failure_type": "SkippedByDesign",
            "failure_message": (
                "The official full model contains the uncapturable SigLIP2 dynamic "
                "indexed-windowing path already tested in the backbone capture, plus "
                "action-head CUDA RNG initialization. The script records the backbone "
                "capture failure and uses action-head graph replay as the valid "
                "official-scope CUDA Graph control."
            ),
        }
    }
    graph_timings: dict[str, dict[str, Any]] = {}
    # Capture the action head first. It samples CUDA random noise, and after a
    # successful RNG capture, later eager RNG calls in the same process can
    # invalidate the CUDA generator offset. Baseline/profiler measurements are
    # therefore completed before this point, and full-model eager capture is
    # skipped. Backbone capture is attempted afterward only to record whether
    # the official VLM path is graph-capturable.
    for name in [
        "action_head_component_prepared",
        "backbone_component_prepared",
    ]:
        capture, replay = try_capture_cuda_graph(name, target_fns[name], warmup=2)
        graph_results[name] = capture
        if replay is not None:
            timing = measure_call(replay, args.overhead_iterations, warmup=2)
            profile = profile_repeats(replay, args.profile_repeats)
            graph_timings[name] = {"timing": timing, "profile": profile}

    official_graph_e2e: dict[str, Any] = {
        "scope": (
            "Official README component-sum scope. Uses official torch.compile "
            "data/backbone p50 and replaces action_head with explicit CUDA Graph "
            "replay only if that capture succeeded."
        )
    }
    action_graph = graph_timings.get("action_head_component_prepared")
    if action_graph is not None:
        data_p50 = float(np.percentile(official_components["data_processing"], 50))
        bb_p50 = float(np.percentile(official_components["backbone"], 50))
        ah_graph_p50 = float(action_graph["timing"]["cuda_event_timeline_ms"]["p50_ms"])
        official_graph_e2e.update(
            {
                "data_processing_p50_ms": data_p50,
                "backbone_torch_compile_p50_ms": bb_p50,
                "action_head_cuda_graph_replay_p50_ms": ah_graph_p50,
                "hybrid_e2e_p50_ms": data_p50 + bb_p50 + ah_graph_p50,
                "frequency_hz_from_hybrid_e2e_p50": 1000.0 / (data_p50 + bb_p50 + ah_graph_p50),
                "baseline_torch_compile_e2e_p50_ms": float(
                    np.percentile(official_components["e2e"], 50)
                ),
            }
        )

    payload = {
        "meta": {
            "date": "2026-05-21",
            "scope": (
                "Official README component-sum benchmark scope plus explicit CUDA Graph "
                "capture/replay probes and torch.profiler overhead breakdown."
            ),
            "official_root": str(official_root),
            "official_git_rev": git_rev(official_root),
            "gr00t_import_file": str(Path(gr00t.__file__).resolve()),
            "benchmark_module_file": str(
                official_root / "scripts" / "deployment" / "benchmark_inference.py"
            ),
            "model_path": args.model_path,
            "dataset_path": dataset_path,
            "embodiment_tag": args.embodiment_tag,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "compile_mode": args.compile_mode,
            "num_iterations": args.num_iterations,
            "warmup": args.warmup,
            "overhead_iterations": args.overhead_iterations,
            "profile_repeats": args.profile_repeats,
            "seed": args.seed,
            "measurement_notes": {
                "official_e2e": "data_processing + backbone + action_head component arrays.",
                "host_enqueue_ms": "Python/PyTorch call wall time before explicit synchronize.",
                "sync_wait_after_enqueue_ms": "Explicit synchronize wait; mostly outstanding GPU work, not pure sync overhead.",
                "cpu_cuda_launch_runtime_ms": "CPU runtime spent in cudaLaunchKernel/cuLaunchKernel/cudaGraphLaunch during profiler collection.",
                "device_idle_gap_sum_ms": "Profiler CUDA timeline gaps between device events; includes launch/handoff/stream idle effects and is profiler-perturbed.",
                "cpu_aten_alloc_like_self_ms": "CPU self time of allocation/copy-like aten ops; not a direct allocator-only measurement.",
                "cuda_alloc_free": "cudaMalloc/cudaFree runtime events after warmup; zero means the caching allocator avoided request-path device allocations.",
            },
        },
        "official_torch_compile_readme_scope": summarize_components(official_components),
        "baseline_torch_compile_overheads": baseline_profiles,
        "cuda_graph_capture": graph_results,
        "cuda_graph_replay_overheads": graph_timings,
        "official_scope_cuda_graph_hybrid": official_graph_e2e,
        "compact": {
            "official_torch_compile_e2e_p50_ms": float(np.percentile(official_components["e2e"], 50)),
            "official_torch_compile_data_p50_ms": float(
                np.percentile(official_components["data_processing"], 50)
            ),
            "official_torch_compile_backbone_p50_ms": float(
                np.percentile(official_components["backbone"], 50)
            ),
            "official_torch_compile_action_head_p50_ms": float(
                np.percentile(official_components["action_head"], 50)
            ),
            "baseline": {name: compact_target(row) for name, row in baseline_profiles.items()},
            "cuda_graph_replay": {
                name: compact_target(row) for name, row in graph_timings.items()
            },
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(payload["compact"], indent=2))
    print(json.dumps(payload["cuda_graph_capture"], indent=2))
    print(json.dumps(payload["official_scope_cuda_graph_hybrid"], indent=2))


if __name__ == "__main__":
    main()
