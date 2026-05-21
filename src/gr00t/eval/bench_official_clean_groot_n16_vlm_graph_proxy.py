#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


OFFICIAL_ROOT = Path("/root/autodl-tmp/Isaac-GR00T-official-clean-20260520")
OUT = Path(
    "/root/autodl-tmp/VLAServing/results/"
    "official_clean_groot_n16_vlm_graph_proxy_20260521.json"
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


def load_observation(off: Any, policy: Any, dataset_path: str, embodiment_tag: str):
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


def cuda_event_measure(fn: Callable[[], Any], iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    wall = []
    event = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall.append((t1 - t0) * 1000.0)
        event.append(float(start.elapsed_time(end)))
    return {"wall_ms": stats(wall), "cuda_event_ms": stats(event)}


def profile_cuda(fn: Callable[[], Any], iterations: int) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            _ = fn()
        torch.cuda.synchronize()

    cuda_events = [
        evt for evt in prof.events() if "CUDA" in str(getattr(evt, "device_type", "")).upper()
    ]

    def dur_ms(evt: Any) -> float:
        tr = getattr(evt, "time_range", None)
        if tr is not None:
            return float(tr.end - tr.start) / 1000.0
        if hasattr(evt, "self_device_time_total"):
            return float(evt.self_device_time_total) / 1000.0
        return float(getattr(evt, "self_cuda_time_total", 0.0)) / 1000.0

    rows = [
        {"name": str(evt.name), "duration_ms": dur_ms(evt)}
        for evt in cuda_events
        if dur_ms(evt) > 0
    ]
    key_avgs = prof.key_averages()
    launch_cpu_ms = float(
        sum(
            evt.self_cpu_time_total
            for evt in key_avgs
            if str(evt.key).startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch"))
        )
        / 1000.0
    )
    return {
        "profile_iterations": int(iterations),
        "cuda_event_count": len(rows),
        "cuda_kernel_sum_per_iter_ms": float(sum(row["duration_ms"] for row in rows) / iterations),
        "cpu_cuda_launch_per_iter_ms": float(launch_cpu_ms / iterations),
        "top_cuda_events": sorted(rows, key=lambda row: row["duration_ms"], reverse=True)[:16],
    }


def tensor_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def flatten_tensors(value: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        out["tensor"] = value
        return out
    if hasattr(value, "items"):
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                out[str(key)] = item
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if isinstance(item, torch.Tensor):
            out[name] = item
    return out


def output_diff(lhs: Any, rhs: Any) -> dict[str, Any]:
    lhs_t = flatten_tensors(lhs)
    rhs_t = flatten_tensors(rhs)
    keys = sorted(set(lhs_t) & set(rhs_t))
    return {key: tensor_diff(lhs_t[key], rhs_t[key]) for key in keys}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--profile-iterations", type=int, default=2)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    off = load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_vlm_graph_proxy_benchmark_inference",
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
    observation = load_observation(off, policy, dataset_path, args.embodiment_tag)
    collated = off.prepare_model_inputs(policy, observation)
    backbone_inputs, _ = policy.model.prepare_input(collated)

    for _ in range(args.warmup + 4):
        _ = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    uncaptured_fn = lambda: policy.model.backbone(backbone_inputs)
    ref = uncaptured_fn()
    torch.cuda.synchronize()

    graph_status: dict[str, Any] = {"captured": False}
    graph_measure = None
    graph_profile = None
    correctness = None
    try:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                _ = uncaptured_fn()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = uncaptured_fn()
        torch.cuda.synchronize()

        def replay_fn():
            graph.replay()
            return graph_output

        for _ in range(args.warmup):
            _ = replay_fn()
        torch.cuda.synchronize()
        graph_out = replay_fn()
        torch.cuda.synchronize()
        correctness = output_diff(graph_out, ref)
        graph_measure = cuda_event_measure(replay_fn, args.iterations, warmup=1)
        graph_profile = profile_cuda(replay_fn, args.profile_iterations)
        graph_status = {"captured": True}
    except Exception:
        graph_status = {"captured": False, "traceback": traceback.format_exc()}

    uncaptured_measure = cuda_event_measure(uncaptured_fn, args.iterations, warmup=1)
    uncaptured_profile = profile_cuda(uncaptured_fn, args.profile_iterations)

    result = {
        "meta": {
            "date": "2026-05-21",
            "scope": (
                "Fresh official Isaac-GR00T n1.6-release prepared VLM/backbone "
                "CUDA Graph proxy. This uses the clean official checkout and does "
                "not import VLAServing UnifiedRuntime."
            ),
            "official_root": str(official_root),
            "official_git_rev": git_rev(official_root),
            "gr00t_import_file": str(Path(gr00t.__file__).resolve()),
            "model_path": args.model_path,
            "dataset_path": dataset_path,
            "embodiment_tag": args.embodiment_tag,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "profile_iterations": args.profile_iterations,
        },
        "uncaptured_prepared_backbone": {
            "timing": uncaptured_measure,
            "profile": uncaptured_profile,
        },
        "cuda_graph_prepared_backbone": {
            "status": graph_status,
            "timing": graph_measure,
            "profile": graph_profile,
            "correctness_vs_uncaptured": correctness,
        },
    }
    if graph_measure is not None:
        uncaptured_p50 = uncaptured_measure["cuda_event_ms"]["p50"]
        graph_p50 = graph_measure["cuda_event_ms"]["p50"]
        result["comparison"] = {
            "uncaptured_cuda_event_p50_ms": uncaptured_p50,
            "graph_replay_cuda_event_p50_ms": graph_p50,
            "p50_speedup": float(uncaptured_p50 / graph_p50) if graph_p50 > 0 else None,
            "p50_delta_ms": float(uncaptured_p50 - graph_p50),
            "uncaptured_kernel_sum_per_iter_ms": uncaptured_profile[
                "cuda_kernel_sum_per_iter_ms"
            ],
            "graph_kernel_sum_per_iter_ms": graph_profile["cuda_kernel_sum_per_iter_ms"],
            "math_kernel_sum_ratio_graph_over_uncaptured": float(
                graph_profile["cuda_kernel_sum_per_iter_ms"]
                / max(uncaptured_profile["cuda_kernel_sum_per_iter_ms"], 1e-9)
            ),
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(result.get("comparison", result["cuda_graph_prepared_backbone"]["status"]), indent=2))


if __name__ == "__main__":
    main()
