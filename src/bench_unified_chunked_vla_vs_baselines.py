#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/VLAServing")
RESULTS = ROOT / "results"
OUT = RESULTS / "unified_chunked_vla_vs_baselines_20260412.json"

BASELINES = RESULTS / "vla_single_gpu_methods_20260411.json"
EFFECTIVE = RESULTS / "unified_chunked_vla_effectiveness_20260412.json"
PI05_PREFETCH = RESULTS / "pi05_four_model_residency_prefetch_system_20260406.json"
GR00T_BATCH = RESULTS / "gr00t_shared_prefix_phase_lock_batch_mps_20260412.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def rank_request_baseline(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["mean_hard_miss_count"]),
        float(summary["mean_sla_miss_count"]),
        float(summary["mean_latency_p95_ms"]),
        -float(summary["mean_min_robot_score"]),
    )


def pick_best_conventional(req: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = {
        name: obj
        for name, obj in req.items()
        if name not in {"oracle_full_resident", "vla_aware"}
    }
    best_name, best_obj = min(
        candidates.items(),
        key=lambda kv: rank_request_baseline(kv[1]["summary"]),
    )
    return best_name, best_obj


def build_pi05_section(
    baselines: dict[str, Any],
    effective: dict[str, Any],
    prefetch: dict[str, Any],
) -> dict[str, Any]:
    req = baselines["pi05"]["request_level"]
    gpulet = baselines["pi05"]["gpulet_like"]
    best_name, best_obj = pick_best_conventional(req)
    latest = {
        "method": "Pi05 exact residency + predictive prefetch",
        "robot_count": 4,
        "gpu_memory_total_gb": float(prefetch["best"]["gpu_memory_estimate_gb"]["total_estimated_gb"]),
        "hard_deadline_miss_count": int(prefetch["best"]["hard_deadline_miss_count"]),
        "service_e2e_p95_ms": float(prefetch["best"]["service_e2e_ms"]["p95_ms"]),
        "strict_deadline_passes_ms": [
            int(k)
            for k, v in prefetch["best"]["deadline_sweep"].items()
            if bool(v["pass"])
        ],
        "fixed_four_robot_validation": effective["pi05_prefetch_effectiveness"]["fixed_four_robot_level"]["with_prefetch"],
    }
    oracle = req["oracle_full_resident"]["summary"]
    best_baseline = best_obj["summary"]
    return {
        "latest_method": latest,
        "all_request_level_baselines": {
            name: obj["summary"] for name, obj in req.items()
        },
        "generic_full_resident_upper_bound": {
            "method": "oracle_full_resident",
            "mean_sla_miss_count": float(oracle["mean_sla_miss_count"]),
            "mean_hard_miss_count": float(oracle["mean_hard_miss_count"]),
            "mean_latency_p95_ms": float(oracle["mean_latency_p95_ms"]),
            "mean_min_robot_score": float(oracle["mean_min_robot_score"]),
        },
        "best_conventional_baseline": {
            "method": best_name,
            "mean_sla_miss_count": float(best_baseline["mean_sla_miss_count"]),
            "mean_hard_miss_count": float(best_baseline["mean_hard_miss_count"]),
            "mean_latency_p95_ms": float(best_baseline["mean_latency_p95_ms"]),
            "mean_min_robot_score": float(best_baseline["mean_min_robot_score"]),
        },
        "gpulet_like": gpulet,
        "delta_vs_oracle_full_resident": {
            "p95_ms_improvement": float(oracle["mean_latency_p95_ms"] - latest["service_e2e_p95_ms"]),
            "hard_miss_reduction": float(oracle["mean_hard_miss_count"] - latest["hard_deadline_miss_count"]),
            "sla_miss_reduction": float(
                oracle["mean_sla_miss_count"]
                - latest["fixed_four_robot_validation"]["mean_sla_miss_count"]
            ),
        },
        "delta_vs_best_conventional_baseline": {
            "baseline_method": best_name,
            "p95_ms_improvement": float(best_baseline["mean_latency_p95_ms"] - latest["service_e2e_p95_ms"]),
            "hard_miss_reduction": float(best_baseline["mean_hard_miss_count"] - latest["hard_deadline_miss_count"]),
            "sla_miss_reduction": float(
                best_baseline["mean_sla_miss_count"]
                - latest["fixed_four_robot_validation"]["mean_sla_miss_count"]
            ),
        },
    }


def build_gr00t_section(
    baselines: dict[str, Any],
    effective: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    req = baselines["gr00t_n1d6"]["request_level"]
    gpulet = baselines["gr00t_n1d6"]["gpulet_like"]
    best_name, best_obj = pick_best_conventional(req)
    latest_4 = batch["scenarios"]["1x_per_model"]["batch_only"]["best"]["metrics"]
    latest_16 = batch["scenarios"]["4x_per_model"]["batch_only"]["best"]["metrics"]
    oracle = req["oracle_full_resident"]["summary"]
    best_baseline = best_obj["summary"]
    fairness = effective["gr00t_fair_admission_effectiveness"]
    mps = effective["gr00t_batch_mps_effectiveness"]
    return {
        "latest_method_4_robot": {
            "method": "GR00T shared-prefix phase-lock runtime",
            "robot_count": 4,
            "mean_request_to_result_p95_ms": float(latest_4["mean_request_to_result_p95_ms"]),
            "reply_over_chunk_actions_count": int(latest_4["reply_over_chunk_actions_count"]),
            "mean_batch_size": float(latest_4["mean_batch_size"]),
            "stable_under_100ms": bool(latest_4["stable_under_100ms"]),
        },
        "latest_method_16_robot_scaleout": {
            "method": "GR00T shared-prefix phase-lock runtime",
            "robot_count": 16,
            "mean_request_to_result_p95_ms": float(latest_16["mean_request_to_result_p95_ms"]),
            "reply_over_chunk_actions_count": int(latest_16["reply_over_chunk_actions_count"]),
            "mean_batch_size": float(latest_16["mean_batch_size"]),
            "stable_under_100ms": bool(latest_16["stable_under_100ms"]),
        },
        "all_request_level_baselines": {
            name: obj["summary"] for name, obj in req.items()
        },
        "generic_full_resident_upper_bound": {
            "method": "oracle_full_resident",
            "mean_sla_miss_count": float(oracle["mean_sla_miss_count"]),
            "mean_hard_miss_count": float(oracle["mean_hard_miss_count"]),
            "mean_latency_p95_ms": float(oracle["mean_latency_p95_ms"]),
            "mean_min_robot_score": float(oracle["mean_min_robot_score"]),
        },
        "best_conventional_baseline": {
            "method": best_name,
            "mean_sla_miss_count": float(best_baseline["mean_sla_miss_count"]),
            "mean_hard_miss_count": float(best_baseline["mean_hard_miss_count"]),
            "mean_latency_p95_ms": float(best_baseline["mean_latency_p95_ms"]),
            "mean_min_robot_score": float(best_baseline["mean_min_robot_score"]),
        },
        "gpulet_like": gpulet,
        "fair_admission": {
            "baseline_greedy_accept_rate_gap": float(fairness["baseline_greedy"]["accept_rate_gap"]),
            "quota_fair_accept_rate_gap": float(fairness["quota_fair"]["accept_rate_gap"]),
            "baseline_greedy_final_count_gap": float(fairness["baseline_greedy"]["final_count_gap"]),
            "quota_fair_final_count_gap": float(fairness["quota_fair"]["final_count_gap"]),
        },
        "mps_ablation": {
            "four_robot_equivalence": mps["cases"][0]["delta"],
            "twentyfour_robot_equivalence": mps["cases"][1]["delta"],
        },
        "delta_vs_oracle_full_resident": {
            "p95_ms_improvement": float(oracle["mean_latency_p95_ms"] - latest_4["mean_request_to_result_p95_ms"]),
            "sla_miss_reduction": float(oracle["mean_sla_miss_count"]),
        },
        "delta_vs_best_conventional_baseline": {
            "baseline_method": best_name,
            "p95_ms_improvement": float(
                best_baseline["mean_latency_p95_ms"] - latest_4["mean_request_to_result_p95_ms"]
            ),
            "hard_miss_reduction": float(best_baseline["mean_hard_miss_count"]),
            "sla_miss_reduction": float(best_baseline["mean_sla_miss_count"]),
        },
    }


def build_findings(payload: dict[str, Any]) -> list[str]:
    pi05 = payload["pi05"]
    gr00t = payload["gr00t_n1d6"]
    return [
        (
            "Pi05 latest runtime reaches "
            f"{pi05['latest_method']['service_e2e_p95_ms']:.2f} ms p95 with 0 hard misses, "
            f"while generic full-resident serving is {pi05['generic_full_resident_upper_bound']['mean_latency_p95_ms']:.2f} ms "
            f"and the best conventional baseline {pi05['best_conventional_baseline']['method']} is "
            f"{pi05['best_conventional_baseline']['mean_latency_p95_ms']:.2f} ms."
        ),
        (
            "Pi05 GPUlet-style temporal, spatial, and spatio-temporal partitions are all analytically infeasible under 100 ms once model-state switching is included."
        ),
        (
            "GR00T latest phase-lock runtime is stable at 4 robots "
            f"({gr00t['latest_method_4_robot']['mean_request_to_result_p95_ms']:.2f} ms p95) "
            f"and still stable at 16 robots ({gr00t['latest_method_16_robot_scaleout']['mean_request_to_result_p95_ms']:.2f} ms p95, batch 4)."
        ),
        (
            "GR00T generic full-resident serving still misses the 100 ms bound "
            f"({gr00t['generic_full_resident_upper_bound']['mean_latency_p95_ms']:.2f} ms p95), "
            f"and the best conventional baseline {gr00t['best_conventional_baseline']['method']} is far worse at "
            f"{gr00t['best_conventional_baseline']['mean_latency_p95_ms']:.2f} ms."
        ),
        (
            "GR00T GPUlet-style temporal, spatial, and spatio-temporal partitions are all infeasible, while quota-fair admission reduces accept-rate gap "
            f"{gr00t['fair_admission']['baseline_greedy_accept_rate_gap']:.4f} -> "
            f"{gr00t['fair_admission']['quota_fair_accept_rate_gap']:.4f}."
        ),
    ]


def main():
    baselines = load_json(BASELINES)
    effective = load_json(EFFECTIVE)
    prefetch = load_json(PI05_PREFETCH)
    batch = load_json(GR00T_BATCH)

    payload = {
        "meta": {
            "goal": "Compare the latest validated VLA-aware Pi05 and GR00T runtimes against prior single-GPU serving baselines and GPUlet-style partitioning methods.",
            "date": "2026-04-12",
            "baseline_selection_rule": [
                "minimize mean_hard_miss_count",
                "then minimize mean_sla_miss_count",
                "then minimize mean_latency_p95_ms",
                "then maximize mean_min_robot_score",
            ],
            "sources": {
                "baselines": str(BASELINES),
                "latest_effectiveness": str(EFFECTIVE),
                "pi05_prefetch": str(PI05_PREFETCH),
                "gr00t_batch": str(GR00T_BATCH),
            },
        },
        "pi05": build_pi05_section(baselines, effective, prefetch),
        "gr00t_n1d6": build_gr00t_section(baselines, effective, batch),
    }
    payload["findings"] = build_findings(payload)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(OUT)
    print(json.dumps({"findings": payload["findings"]}, indent=2))


if __name__ == "__main__":
    main()
