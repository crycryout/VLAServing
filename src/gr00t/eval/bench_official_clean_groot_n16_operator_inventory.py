#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


OFFICIAL_ROOT = Path("/root/autodl-tmp/Isaac-GR00T-official-clean-20260520")
OUT = Path(
    "/root/autodl-tmp/VLAServing/results/"
    "official_clean_groot_n16_operator_inventory_20260520.json"
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


def module_inventory(module: torch.nn.Module) -> dict[str, Any]:
    counts = Counter(type(m).__name__ for m in module.modules())
    examples: dict[str, list[str]] = defaultdict(list)
    for name, m in module.named_modules():
        cls = type(m).__name__
        if len(examples[cls]) < 8:
            examples[cls].append(name or "<root>")
    return {
        "class_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "examples": dict(sorted(examples.items())),
    }


def classify_cuda_kernel(name: str) -> str:
    low = name.lower()
    if "memcpy" in low:
        return "memcpy"
    if "memset" in low:
        return "memset"
    if "gemm" in low or "cutlass" in low or "mm" in low or "matmul" in low:
        return "gemm/mm"
    if "flash" in low or "attention" in low or "fmha" in low or "sdpa" in low:
        return "attention"
    if "layernorm" in low or "norm" in low or "rms" in low:
        return "norm"
    if "silu" in low or "gelu" in low or "activation" in low or "sigmoid" in low:
        return "activation"
    if "cat" in low or "slice" in low or "index" in low or "gather" in low or "scatter" in low:
        return "shape/index"
    if "copy" in low or "cast" in low or "convert" in low:
        return "copy/cast"
    if "elementwise" in low or "vectorized" in low or "unrolled" in low:
        return "elementwise"
    return "other"


def profile_once(fn) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        _ = fn()
        torch.cuda.synchronize()
    key_rows = []
    for evt in prof.key_averages():
        key = str(evt.key)
        key_rows.append(
            {
                "key": key,
                "count": int(evt.count),
                "cpu_self_ms": float(evt.self_cpu_time_total / 1000.0),
                "device_self_ms": float(getattr(evt, "self_device_time_total", 0.0) / 1000.0),
            }
        )
    aten_rows = [r for r in key_rows if r["key"].startswith("aten::")]
    cuda_runtime_rows = [
        r
        for r in key_rows
        if r["key"].startswith("cuda") or r["key"].startswith("cuLaunch")
    ]

    cuda_events = []
    for evt in prof.events():
        if "CUDA" not in str(getattr(evt, "device_type", "")).upper():
            continue
        name = str(evt.name)
        dur_ms = float(getattr(evt, "self_device_time_total", 0.0) / 1000.0)
        cuda_events.append({"name": name, "class": classify_cuda_kernel(name), "duration_ms": dur_ms})
    class_counts = Counter(row["class"] for row in cuda_events)
    class_time = Counter()
    kernel_counts = Counter()
    kernel_time = Counter()
    for row in cuda_events:
        class_time[row["class"]] += row["duration_ms"]
        kernel_counts[row["name"]] += 1
        kernel_time[row["name"]] += row["duration_ms"]

    return {
        "aten_ops_by_count": sorted(aten_rows, key=lambda r: (-r["count"], r["key"]))[:80],
        "aten_ops_by_cpu_self_ms": sorted(aten_rows, key=lambda r: -r["cpu_self_ms"])[:80],
        "cuda_runtime_ops": sorted(cuda_runtime_rows, key=lambda r: -r["cpu_self_ms"])[:40],
        "cuda_kernel_class_counts": dict(sorted(class_counts.items())),
        "cuda_kernel_class_time_ms": dict(sorted(class_time.items())),
        "cuda_kernel_names_by_count": [
            {"name": name, "count": int(count), "time_ms": float(kernel_time[name])}
            for name, count in kernel_counts.most_common(80)
        ],
        "cuda_kernel_names_by_time_ms": [
            {"name": name, "count": int(kernel_counts[name]), "time_ms": float(time_ms)}
            for name, time_ms in kernel_time.most_common(80)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    off = load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_benchmark_inference_operator_inventory",
    )
    import gr00t  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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

    dataset_path = str(official_root / "demo_data" / "gr1.PickNPlace")
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
        embodiment_tag=off.EmbodimentTag(args.embodiment_tag),
        allow_padding=False,
    )
    import numpy as np

    observation = {
        "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
        "state": {k: step_data.states[k][None] for k in step_data.states},
        "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
    }
    collated = off.prepare_model_inputs(policy, observation)
    backbone_inputs, action_inputs = policy.model.prepare_input(collated)

    # Warm compile and cudagraph paths.
    for _ in range(5):
        _ = policy.model.get_action(collated)
    torch.cuda.synchronize()
    backbone_outputs = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    payload = {
        "meta": {
            "date": "2026-05-20",
            "scope": "Fresh official Isaac-GR00T n1.6-release operator inventory for torch.compile inference.",
            "official_root": str(official_root),
            "official_git_rev": git_rev(official_root),
            "gr00t_import_file": str(Path(gr00t.__file__).resolve()),
            "model_path": args.model_path,
            "dataset_path": dataset_path,
            "embodiment_tag": args.embodiment_tag,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "compile_mode": args.compile_mode,
        },
        "module_inventory": {
            "full_model": module_inventory(policy.model),
            "backbone": module_inventory(policy.model.backbone),
            "action_head": module_inventory(policy.model.action_head),
            "dit_model": module_inventory(policy.model.action_head.model),
        },
        "profiler_inventory": {
            "backbone_prepared": profile_once(lambda: policy.model.backbone(backbone_inputs)),
            "action_head_prepared": profile_once(
                lambda: policy.model.action_head.get_action(backbone_outputs, action_inputs)
            ),
            "full_model_prepared": profile_once(lambda: policy.model.get_action(collated)),
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    compact = {
        "module_top_counts": {
            "backbone": list(payload["module_inventory"]["backbone"]["class_counts"].items())[:20],
            "action_head": list(payload["module_inventory"]["action_head"]["class_counts"].items())[:20],
        },
        "cuda_kernel_classes": {
            target: payload["profiler_inventory"][target]["cuda_kernel_class_counts"]
            for target in payload["profiler_inventory"]
        },
        "top_aten_by_count": {
            target: payload["profiler_inventory"][target]["aten_ops_by_count"][:20]
            for target in payload["profiler_inventory"]
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
