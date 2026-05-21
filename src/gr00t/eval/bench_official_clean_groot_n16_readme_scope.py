#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


OFFICIAL_ROOT = Path("/root/autodl-tmp/Isaac-GR00T-official-clean-20260520")
OUT = Path(
    "/root/autodl-tmp/VLAServing/results/"
    "official_clean_groot_n16_readme_scope_benchmark_20260521.json"
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


def stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
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


def summarize_component_arrays(components: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "data_processing": stats(components["data_processing"]),
        "backbone": stats(components["backbone"]),
        "action_head": stats(components["action_head"]),
        "e2e": stats(components["e2e"]),
        "frequency_hz_from_e2e_p50": float(1000.0 / np.percentile(components["e2e"], 50)),
    }


def run_mode(
    off: Any,
    *,
    model_path: str,
    embodiment_tag: str,
    device: str,
    observation: Any,
    shared_data_processing_times: np.ndarray,
    num_iterations: int,
    warmup: int,
    compile_action_head: bool,
) -> tuple[Any, dict[str, np.ndarray]]:
    policy = off.Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=off.EmbodimentTag(embodiment_tag),
        device=device,
        strict=True,
    )
    if compile_action_head:
        policy.model.action_head.model.forward = torch.compile(
            policy.model.action_head.model.forward,
            mode="max-autotune",
        )
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    # This is exactly the official README scope:
    # component timings are measured by benchmark_components, but E2E uses the
    # shared data-processing array plus backbone/action-head arrays.
    times_components = off.benchmark_components(
        policy,
        observation,
        num_iterations=num_iterations,
        warmup=warmup,
    )
    components = {
        "data_processing": np.asarray(shared_data_processing_times, dtype=np.float64),
        "backbone": np.asarray(times_components["backbone"], dtype=np.float64),
        "action_head": np.asarray(times_components["action_head"], dtype=np.float64),
    }
    components["e2e"] = off.compute_e2e_from_components(components)
    return policy, components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--embodiment-tag", default="gr1")
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    off = load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_readme_scope_benchmark_inference",
    )
    import gr00t  # noqa: PLC0415

    off.set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    device_name = off.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset_path = args.dataset_path or str(official_root / "demo_data" / "gr1.PickNPlace")
    policy_for_obs = off.Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=off.EmbodimentTag(args.embodiment_tag),
        device=device,
        strict=True,
    )
    observation = observation_from_official(off, policy_for_obs, dataset_path, args.embodiment_tag)
    denoising_steps = int(policy_for_obs.model.action_head.num_inference_timesteps)
    action_horizon = int(policy_for_obs.model.action_head.action_horizon)

    shared_data_processing_times = off.benchmark_data_processing(
        policy_for_obs,
        observation,
        args.num_iterations,
        warmup=10,
    )

    _, eager_components = run_mode(
        off,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device=device,
        observation=observation,
        shared_data_processing_times=shared_data_processing_times,
        num_iterations=args.num_iterations,
        warmup=args.warmup,
        compile_action_head=False,
    )
    _, compile_components = run_mode(
        off,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device=device,
        observation=observation,
        shared_data_processing_times=shared_data_processing_times,
        num_iterations=args.num_iterations,
        warmup=args.warmup + 2,
        compile_action_head=True,
    )

    payload = {
        "meta": {
            "date": "2026-05-21",
            "scope": (
                "Official README benchmark scope: data processing measured "
                "separately; backbone and action head measured component-wise "
                "with synchronize; E2E is component-array sum, not full request "
                "wall time."
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
            "device_short_name": device_name,
            "torch_version": torch.__version__,
            "num_iterations": args.num_iterations,
            "warmup": args.warmup,
            "seed": args.seed,
            "denoising_steps": denoising_steps,
            "action_horizon": action_horizon,
        },
        "results": {
            "PyTorch Eager": summarize_component_arrays(eager_components),
            "torch.compile": summarize_component_arrays(compile_components),
        },
        "readme_markdown_rows": {
            mode: {
                "Device": device_name,
                "Mode": mode,
                "Data Processing": f"{row['data_processing']['p50_ms']:.0f} ms",
                "Backbone": f"{row['backbone']['p50_ms']:.0f} ms",
                "Action Head": f"{row['action_head']['p50_ms']:.0f} ms",
                "E2E": f"{row['e2e']['p50_ms']:.0f} ms",
                "Frequency": f"{row['frequency_hz_from_e2e_p50']:.1f} Hz",
            }
            for mode, row in {
                "PyTorch Eager": summarize_component_arrays(eager_components),
                "torch.compile": summarize_component_arrays(compile_components),
            }.items()
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(payload["readme_markdown_rows"], indent=2))


if __name__ == "__main__":
    main()
