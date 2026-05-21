#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


DEFAULT_OFFICIAL_ROOT = Path("/root/autodl-tmp/Isaac-GR00T-official-clean-20260520")
DEFAULT_OUT = Path(
    "/root/autodl-tmp/VLAServing/results/"
    "official_clean_groot_n16_torchcompile_launch_sync_20260520.json"
)


def stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "num_samples": 0,
        }
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "num_samples": int(arr.size),
    }


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git_rev(root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def load_observation(off: Any, policy: Any, dataset_path: str, embodiment_tag: str) -> dict[str, Any]:
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


@torch.inference_mode()
def data_processing_times(
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


@torch.inference_mode()
def make_backbone_outputs(off: Any, policy: Any, observation: dict[str, Any]):
    collated = off.prepare_model_inputs(policy, observation)
    backbone_inputs, action_inputs = policy.model.prepare_input(collated)
    backbone_outputs = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()
    return backbone_outputs, action_inputs


def target_runner(
    name: str,
    off: Any,
    policy: Any,
    observation: dict[str, Any],
    backbone_outputs: Any,
    action_inputs: Any,
) -> Callable[[], Any]:
    if name == "backbone":
        def run():
            collated = off.prepare_model_inputs(policy, observation)
            backbone_inputs, _ = policy.model.prepare_input(collated)
            return policy.model.backbone(backbone_inputs)

        return run
    if name == "action_head":
        def run():
            return policy.model.action_head.get_action(backbone_outputs, action_inputs)

        return run
    if name == "model_gpu_only":
        def run():
            collated = off.prepare_model_inputs(policy, observation)
            return policy.model.get_action(collated)

        return run
    raise ValueError(f"unknown target {name}")


@torch.inference_mode()
def measure_async_call(fn: Callable[[], Any], iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    host_enqueue = []
    sync_wait = []
    wall = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        host_enqueue.append((t1 - t0) * 1000.0)
        sync_wait.append((t2 - t1) * 1000.0)
        wall.append((t2 - t0) * 1000.0)
    return {
        "host_enqueue_ms": stats(host_enqueue),
        "sync_wait_after_enqueue_ms": stats(sync_wait),
        "wall_sync_ms": stats(wall),
    }


@torch.inference_mode()
def measure_cuda_event(fn: Callable[[], Any], iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(float(start.elapsed_time(end)))
    return {"cuda_event_timeline_ms": stats(elapsed)}


def profiler_breakdown(fn: Callable[[], Any], iterations: int) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            _ = fn()
        torch.cuda.synchronize()

    key_avgs = prof.key_averages()

    def cuda_self_us(evt: Any) -> float:
        if hasattr(evt, "self_cuda_time_total"):
            return float(evt.self_cuda_time_total)
        if hasattr(evt, "self_device_time_total"):
            return float(evt.self_device_time_total)
        return 0.0

    key_avgs = prof.key_averages()
    cuda_events = [
        evt for evt in prof.events() if "CUDA" in str(getattr(evt, "device_type", "")).upper()
    ]
    cuda_kernel_total_ms = float(sum(cuda_self_us(evt) for evt in cuda_events) / 1000.0)
    cpu_total_ms = float(sum(evt.self_cpu_time_total for evt in key_avgs) / 1000.0)

    launch_keys = (
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cuLaunchKernel",
        "cudaGraphLaunch",
    )
    sync_keys = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")
    copy_keys = ("cudaMemcpyAsync", "cudaMemcpy", "cudaMemsetAsync")
    event_keys = ("cudaEventRecord",)

    def sum_cpu_ms(prefixes: tuple[str, ...]) -> float:
        return float(
            sum(
                evt.self_cpu_time_total
                for evt in key_avgs
                if any(str(evt.key).startswith(prefix) for prefix in prefixes)
            )
            / 1000.0
        )

    runtime_rows = []
    for evt in key_avgs:
        key = str(evt.key)
        if key.startswith("cuda") or key.startswith("cuLaunch"):
            runtime_rows.append(
                {
                    "key": key,
                    "cpu_self_total_ms": float(evt.self_cpu_time_total / 1000.0),
                    "cuda_self_total_ms": float(cuda_self_us(evt) / 1000.0),
                    "count": int(evt.count),
                }
            )
    runtime_rows.sort(key=lambda row: row["cpu_self_total_ms"], reverse=True)

    return {
        "profile_iterations": int(iterations),
        "cuda_device_event_count": int(len(cuda_events)),
        "cuda_kernel_self_total_ms": cuda_kernel_total_ms,
        "cuda_kernel_self_per_iter_ms": cuda_kernel_total_ms / float(iterations),
        "cpu_profiler_self_total_ms": cpu_total_ms,
        "cpu_profiler_self_per_iter_ms": cpu_total_ms / float(iterations),
        "cpu_cuda_launch_total_ms": sum_cpu_ms(launch_keys),
        "cpu_cuda_launch_per_iter_ms": sum_cpu_ms(launch_keys) / float(iterations),
        "cpu_cuda_copy_total_ms": sum_cpu_ms(copy_keys),
        "cpu_cuda_copy_per_iter_ms": sum_cpu_ms(copy_keys) / float(iterations),
        "cpu_cuda_event_record_total_ms": sum_cpu_ms(event_keys),
        "cpu_cuda_event_record_per_iter_ms": sum_cpu_ms(event_keys) / float(iterations),
        "cpu_cuda_sync_total_ms": sum_cpu_ms(sync_keys),
        "cpu_cuda_sync_per_iter_ms": sum_cpu_ms(sync_keys) / float(iterations),
        "runtime_rows_top": runtime_rows[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--profile-iterations", type=int, default=2)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--output-json", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    off = load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_benchmark_inference",
    )
    import gr00t  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = str(official_root / "demo_data" / "gr1.PickNPlace")

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

    observation = load_observation(off, policy, dataset_path, args.embodiment_tag)
    data_processing = data_processing_times(off, policy, observation, args.iterations, args.warmup)

    # Compile and warm action head with full model calls first.
    for _ in range(args.warmup + 2):
        collated = off.prepare_model_inputs(policy, observation)
        _ = policy.model.get_action(collated)
    torch.cuda.synchronize()

    backbone_outputs, action_inputs = make_backbone_outputs(off, policy, observation)

    results: dict[str, Any] = {}
    for target in ["backbone", "action_head", "model_gpu_only"]:
        fn = target_runner(target, off, policy, observation, backbone_outputs, action_inputs)
        async_part = measure_async_call(fn, args.iterations, warmup=1)
        event_part = measure_cuda_event(fn, args.iterations, warmup=1)
        prof_part = profiler_breakdown(fn, args.profile_iterations)
        event_p50 = event_part["cuda_event_timeline_ms"]["p50"]
        kernel_per_iter = prof_part["cuda_kernel_self_per_iter_ms"]
        prof_part["timeline_gap_proxy_p50_ms"] = float(event_p50 - kernel_per_iter)
        prof_part["kernel_sum_fraction_of_cuda_event_p50"] = float(
            kernel_per_iter / event_p50 if event_p50 > 0 else 0.0
        )
        results[target] = {**async_part, **event_part, "profiler": prof_part}

    payload = {
        "meta": {
            "date": "2026-05-20",
            "scope": (
                "Fresh official Isaac-GR00T n1.6-release torch.compile launch/sync "
                "probe. The code import path is the clean clone, not the local "
                "VLAServing or edited Isaac-GR00T tree."
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
            "iterations": args.iterations,
            "warmup": args.warmup,
            "profile_iterations": args.profile_iterations,
            "measurement_notes": {
                "host_enqueue_ms": (
                    "wall time for the Python/PyTorch call to enqueue work, without "
                    "synchronizing at return"
                ),
                "sync_wait_after_enqueue_ms": (
                    "explicit torch.cuda.synchronize wait after the call returns; this "
                    "mostly waits for outstanding GPU work"
                ),
                "cuda_event_timeline_ms": (
                    "CUDA event elapsed time around the call; includes kernel execution "
                    "and stream idle gaps caused by host-side handoff"
                ),
                "cuda_kernel_self_per_iter_ms": (
                    "torch.profiler sum of CUDA kernel self time per iteration; proxy "
                    "for pure kernel compute time"
                ),
                "timeline_gap_proxy_p50_ms": (
                    "cuda_event_timeline p50 minus profiler kernel-sum per iteration; "
                    "proxy for launch, stream idle, device runtime, and non-kernel gaps"
                ),
            },
        },
        "data_processing_wall_ms": data_processing,
        "targets": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    compact = {
        target: {
            "host_enqueue_p50_ms": row["host_enqueue_ms"]["p50"],
            "sync_wait_p50_ms": row["sync_wait_after_enqueue_ms"]["p50"],
            "wall_sync_p50_ms": row["wall_sync_ms"]["p50"],
            "cuda_event_p50_ms": row["cuda_event_timeline_ms"]["p50"],
            "kernel_sum_per_iter_ms": row["profiler"]["cuda_kernel_self_per_iter_ms"],
            "timeline_gap_proxy_p50_ms": row["profiler"]["timeline_gap_proxy_p50_ms"],
            "cpu_cuda_launch_per_iter_ms": row["profiler"]["cpu_cuda_launch_per_iter_ms"],
            "kernel_sum_fraction_of_event": row["profiler"][
                "kernel_sum_fraction_of_cuda_event_p50"
            ],
        }
        for target, row in results.items()
    }
    compact["data_processing_p50_ms"] = data_processing["p50"]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
