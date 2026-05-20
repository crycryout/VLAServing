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


ROOT = Path("/root/autodl-tmp/VLAServing")
ISAAC_ROOT = Path("/root/autodl-tmp/Isaac-GR00T")
OFFICIAL_PROXY = ROOT / "src" / "gr00t" / "eval" / "bench_gr00t_official_mpk_potential.py"
OUT = ROOT / "results" / "gr00t_cpu_gpu_coprocessing_profile_20260520.json"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_ROOT))

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


OFF = load_module(OFFICIAL_PROXY, "bench_gr00t_official_mpk_potential_for_coprocessing")


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


def time_wall_ms(fn: Callable[[], Any], iterations: int) -> dict[str, Any]:
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats(samples)


def measure_cuda_event_ms(fn: Callable[[], Any], iterations: int) -> dict[str, Any]:
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return stats(samples)


@torch.inference_mode()
def sequential_requests(policy: Any, observation: dict[str, Any], iterations: int) -> dict[str, Any]:
    per_request_ms = []
    t0_total = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        collated = OFF.prepare_model_inputs(policy, observation)
        OFF.run_model(policy, collated)
        torch.cuda.synchronize()
        per_request_ms.append((time.perf_counter() - t0) * 1000.0)
    total_ms = (time.perf_counter() - t0_total) * 1000.0
    return {
        "total_ms": total_ms,
        "per_request_wall_ms": stats(per_request_ms),
        "throughput_requests_per_s": float(iterations * 1000.0 / total_ms),
    }


@torch.inference_mode()
def overlapped_prepare_next(
    policy: Any,
    observation: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    """Single-thread pipeline: launch GPU model, prepare next request while GPU runs.

    This is intentionally conservative. If prepare_model_inputs touches CUDA on
    the default stream, the measured overlap will shrink or vanish.
    """

    current = OFF.prepare_model_inputs(policy, observation)
    torch.cuda.synchronize()
    per_iteration_ms = []
    prepare_next_ms = []
    t0_total = time.perf_counter()
    for idx in range(iterations):
        t0 = time.perf_counter()
        OFF.run_model(policy, current)
        next_inputs = None
        if idx + 1 < iterations:
            p0 = time.perf_counter()
            next_inputs = OFF.prepare_model_inputs(policy, observation)
            prepare_next_ms.append((time.perf_counter() - p0) * 1000.0)
        torch.cuda.synchronize()
        per_iteration_ms.append((time.perf_counter() - t0) * 1000.0)
        if next_inputs is not None:
            current = next_inputs
    total_ms = (time.perf_counter() - t0_total) * 1000.0
    return {
        "total_ms": total_ms,
        "per_iteration_wall_ms": stats(per_iteration_ms),
        "prepare_next_wall_ms": stats(prepare_next_ms),
        "throughput_requests_per_s": float(iterations * 1000.0 / total_ms),
    }


def transfer_case(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    iterations: int,
    pin_memory: bool,
) -> dict[str, Any]:
    cpu = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
    if dtype == torch.uint8:
        cpu.copy_(torch.randint(0, 256, shape, dtype=dtype))
    else:
        cpu.uniform_(-1, 1)
    gpu = torch.empty_like(cpu, device="cuda")
    cpu_out = torch.empty_like(cpu, pin_memory=pin_memory)
    bytes_total = int(cpu.numel() * cpu.element_size())

    h2d_event = []
    h2d_wall = []
    d2h_event = []
    d2h_wall = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        gpu.copy_(cpu, non_blocking=pin_memory)
        end.record()
        torch.cuda.synchronize()
        h2d_wall.append((time.perf_counter() - t0) * 1000.0)
        h2d_event.append(float(start.elapsed_time(end)))

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        cpu_out.copy_(gpu, non_blocking=pin_memory)
        end.record()
        torch.cuda.synchronize()
        d2h_wall.append((time.perf_counter() - t0) * 1000.0)
        d2h_event.append(float(start.elapsed_time(end)))

    def with_bw(samples_ms: list[float]) -> dict[str, Any]:
        st = stats(samples_ms)
        p50_s = max(st["p50"] / 1000.0, 1e-12)
        st["p50_GBps"] = float(bytes_total / p50_s / 1e9)
        return st

    return {
        "name": name,
        "shape": list(shape),
        "dtype": str(dtype),
        "pin_memory": bool(pin_memory),
        "bytes": bytes_total,
        "MiB": float(bytes_total / (1024.0 * 1024.0)),
        "h2d_event_ms": with_bw(h2d_event),
        "h2d_wall_ms": with_bw(h2d_wall),
        "d2h_event_ms": with_bw(d2h_event),
        "d2h_wall_ms": with_bw(d2h_wall),
    }


def run_transfer_profiles(iterations: int) -> dict[str, Any]:
    cases = [
        ("raw_uint8_image_1x256x256x3", (1, 256, 256, 3), torch.uint8),
        ("fp16_image_tensor_1x3x256x256", (1, 3, 256, 256), torch.float16),
        ("vlm_hidden_203x2048_bf16", (203, 2048), torch.bfloat16),
        ("larger_tokens_1024x2048_bf16", (1024, 2048), torch.bfloat16),
        ("action_chunk_16x32_fp32", (16, 32), torch.float32),
    ]
    rows = []
    for name, shape, dtype in cases:
        for pin in [False, True]:
            rows.append(transfer_case(name, shape, dtype, iterations, pin))
    return {"cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-bridge")
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--transfer-iterations", type=int, default=20)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    result: dict[str, Any] = {
        "meta": {
            "date": "2026-05-20",
            "device": torch.cuda.get_device_name(0),
            "scope": (
                "KTransformers-inspired CPU/GPU co-processing probe for VLA Serving. "
                "This tests CPU preprocessing overlap and transfer costs; it does not "
                "offload dense VLM/DiT matmul to CPU."
            ),
            "caveat": (
                "Synthetic GR1 observation is used by default. Overlap is measured in a "
                "single Python process and is conservative if preprocessing touches CUDA."
            ),
        }
    }

    if not args.skip_model:
        policy = OFF.Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=OFF.EmbodimentTag(args.embodiment_tag),
            device=args.device,
            strict=True,
        )
        OFF.maybe_override_inference_steps(policy, args.inference_steps)
        observation = OFF.synthetic_observation(policy)
        collated = OFF.prepare_model_inputs(policy, observation)
        policy.model.action_head.model.forward = torch.compile(
            policy.model.action_head.model.forward,
            mode=args.compile_mode,
        )
        for _ in range(args.warmup):
            OFF.run_model(policy, collated)
        torch.cuda.synchronize()

        standalone_preprocess = time_wall_ms(
            lambda: OFF.prepare_model_inputs(policy, observation),
            args.iterations,
        )
        standalone_model_event = measure_cuda_event_ms(
            lambda: OFF.run_model(policy, collated),
            args.iterations,
        )
        seq = sequential_requests(policy, observation, args.iterations)
        overlap = overlapped_prepare_next(policy, observation, args.iterations)
        cpu_p50 = float(standalone_preprocess["p50"])
        gpu_p50 = float(standalone_model_event["p50"])
        result["real_gr00t_synthetic_observation"] = {
            "iterations": args.iterations,
            "standalone_preprocess_wall_ms": standalone_preprocess,
            "standalone_model_cuda_event_ms": standalone_model_event,
            "sequential_preprocess_then_gpu": seq,
            "overlap_gpu_current_prepare_next": overlap,
            "ideal_pipeline_service_interval_p50_ms": max(cpu_p50, gpu_p50),
            "ideal_pipeline_e2e_first_request_p50_ms": cpu_p50 + gpu_p50,
            "interpretation": (
                "If preprocessing is CPU-only, steady-state throughput can approach "
                "max(preprocess, GPU inference). If measured overlap is small, the "
                "current preprocessing path likely has CUDA/default-stream work or "
                "Python scheduling overhead."
            ),
        }

    result["transfer_profiles"] = run_transfer_profiles(args.transfer_iterations)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)
    compact = {"transfer_cases": len(result["transfer_profiles"]["cases"])}
    if "real_gr00t_synthetic_observation" in result:
        gr = result["real_gr00t_synthetic_observation"]
        compact.update(
            {
                "preprocess_p50_ms": gr["standalone_preprocess_wall_ms"]["p50"],
                "gpu_model_p50_ms": gr["standalone_model_cuda_event_ms"]["p50"],
                "sequential_throughput_rps": gr["sequential_preprocess_then_gpu"][
                    "throughput_requests_per_s"
                ],
                "overlap_throughput_rps": gr["overlap_gpu_current_prepare_next"][
                    "throughput_requests_per_s"
                ],
            }
        )
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
