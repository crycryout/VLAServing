#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import bench_official_clean_groot_n16_cudagraph_overheads as h


OFFICIAL_ROOT = Path("/root/autodl-tmp/Isaac-GR00T-official-clean-20260520")
OUT = Path(
    "/root/autodl-tmp/VLAServing/results/"
    "official_clean_groot_n16_eager_cudagraph_control_20260521.json"
)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=str(OUT))
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    sys.path.insert(0, str(official_root))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    off = h.load_module(
        official_root / "scripts" / "deployment" / "benchmark_inference.py",
        "official_clean_n16_eager_graph_control_benchmark_inference",
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
    observation = h.observation_from_official(off, policy, dataset_path, args.embodiment_tag)

    # Official README component-sum scope for the eager baseline.
    data_times = off.benchmark_data_processing(policy, observation, args.num_iterations, warmup=10)
    component_raw = off.benchmark_components(
        policy,
        observation,
        args.num_iterations,
        warmup=args.warmup,
    )
    components = {
        "data_processing": np.asarray(data_times, dtype=np.float64),
        "backbone": np.asarray(component_raw["backbone"], dtype=np.float64),
        "action_head": np.asarray(component_raw["action_head"], dtype=np.float64),
    }
    components["e2e"] = off.compute_e2e_from_components(components)

    collated = off.prepare_model_inputs(policy, observation)
    backbone_inputs, action_inputs = policy.model.prepare_input(collated)
    backbone_outputs = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    def action_fn():
        return policy.model.action_head.get_action(backbone_outputs, action_inputs)

    eager_action = {
        "timing": h.measure_call(action_fn, args.overhead_iterations, warmup=2),
        "profile": h.profile_repeats(action_fn, args.profile_repeats),
    }
    capture, replay = h.try_capture_cuda_graph(
        "eager_action_head_component_prepared",
        action_fn,
        warmup=2,
    )

    graph_action = None
    graph_e2e = {
        "scope": (
            "Official README component-sum scope for an eager+manual-CUDA-Graph "
            "control: data/backbone from official eager component timing, action "
            "head from explicit CUDA Graph replay if capture succeeded."
        )
    }
    if replay is not None:
        graph_action = {
            "timing": h.measure_call(replay, args.overhead_iterations, warmup=2),
            "profile": h.profile_repeats(replay, args.profile_repeats),
        }
        data_p50 = float(np.percentile(components["data_processing"], 50))
        bb_p50 = float(np.percentile(components["backbone"], 50))
        ah_graph_p50 = float(graph_action["timing"]["cuda_event_timeline_ms"]["p50_ms"])
        graph_e2e.update(
            {
                "data_processing_p50_ms": data_p50,
                "backbone_eager_p50_ms": bb_p50,
                "action_head_manual_cudagraph_p50_ms": ah_graph_p50,
                "hybrid_e2e_p50_ms": data_p50 + bb_p50 + ah_graph_p50,
                "baseline_eager_e2e_p50_ms": float(np.percentile(components["e2e"], 50)),
                "frequency_hz_from_hybrid_e2e_p50": 1000.0 / (data_p50 + bb_p50 + ah_graph_p50),
            }
        )

    payload = {
        "meta": {
            "date": "2026-05-21",
            "scope": (
                "Fresh official clean GR00T N1.6 eager policy plus explicit manual "
                "CUDA Graph action-head control under official README component-sum scope."
            ),
            "official_root": str(official_root),
            "official_git_rev": h.git_rev(official_root),
            "gr00t_import_file": str(Path(gr00t.__file__).resolve()),
            "benchmark_module_file": str(
                official_root / "scripts" / "deployment" / "benchmark_inference.py"
            ),
            "model_path": args.model_path,
            "dataset_path": dataset_path,
            "embodiment_tag": args.embodiment_tag,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "num_iterations": args.num_iterations,
            "warmup": args.warmup,
            "overhead_iterations": args.overhead_iterations,
            "profile_repeats": args.profile_repeats,
            "seed": args.seed,
            "limitations": [
                "This is not torch.compile+outer-CUDA-Graph. The compiled action head cannot be wrapped by a manual outer CUDA Graph because Inductor already performs graph replay internally.",
                "The full VLM/backbone path remains uncapturable in the unmodified official stack due dynamic SigLIP2 indexing.",
            ],
        },
        "official_eager_readme_scope": h.summarize_components(components),
        "eager_action_head_overheads": eager_action,
        "manual_cuda_graph_capture": capture,
        "manual_cuda_graph_action_head_overheads": graph_action,
        "official_scope_eager_manual_cudagraph_hybrid": graph_e2e,
        "compact": {
            "official_eager_e2e_p50_ms": float(np.percentile(components["e2e"], 50)),
            "official_eager_data_p50_ms": float(np.percentile(components["data_processing"], 50)),
            "official_eager_backbone_p50_ms": float(np.percentile(components["backbone"], 50)),
            "official_eager_action_head_p50_ms": float(np.percentile(components["action_head"], 50)),
            "eager_action": h.compact_target(eager_action),
            "manual_cuda_graph_action": h.compact_target(graph_action) if graph_action else None,
            "manual_cuda_graph_capture_ok": bool(capture["capture_ok"]),
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(payload["compact"], indent=2))
    print(json.dumps(payload["manual_cuda_graph_capture"], indent=2))
    print(json.dumps(payload["official_scope_eager_manual_cudagraph_hybrid"], indent=2))


if __name__ == "__main__":
    main()
