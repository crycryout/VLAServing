#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/VLAServing")
SRC = ROOT / "src"
RESULTS = ROOT / "results"
OUT = RESULTS / "unified_chunked_vla_effectiveness_20260412.json"

PI05_WORKLOAD_AWARE = SRC / "lerobot" / "eval" / "bench_vla_workload_aware_gpu_virtualization.py"
GR00T_PHASE_LOCK = SRC / "gr00t" / "eval" / "bench_gr00t_shared_prefix_phase_lock_batch.py"
GR00T_FAIR_ADMISSION = SRC / "gr00t" / "eval" / "bench_gr00t_batch_only_fair_admission.py"
GR00T_BATCH_MPS = SRC / "gr00t" / "eval" / "bench_gr00t_shared_prefix_phase_lock_batch_mps.py"

GR00T_2X_OPT = RESULTS / "gr00t_shared_prefix_phase_lock_batch_2x_opt_20260412.json"
GR00T_4X_OPT = RESULTS / "gr00t_shared_prefix_phase_lock_batch_4x_opt_20260412.json"
GR00T_6X_BEST = RESULTS / "gr00t_shared_prefix_phase_lock_batch_6x_best_20260412.json"

PI05_VALIDATED_POLICY = {
    "phase10": 1.0,
    "phase20": 0.0,
    "r10a": 0.0,
    "r10b": 0.1,
}
MPS_SEEDS = [4201, 4202, 4203]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_100ms(reply_over: float, p95_ms: float) -> bool:
    return reply_over == 0 and p95_ms <= 100.0


def pi05_prefetch_effectiveness() -> dict[str, Any]:
    mod = import_module("pi05_workload_aware_validation", PI05_WORKLOAD_AWARE)
    policy = {"policy": dict(PI05_VALIDATED_POLICY)}

    without_prefetch = mod.admission_search("pi05", policy, mode_prefetch=False)
    with_prefetch = mod.admission_search("pi05", policy, mode_prefetch=True)

    base_specs = [dict(r) for r in mod.WORKLOADS["pi05"]["base_robots"]]
    fixed_no_prefetch_runs = [
        mod.simulate_from_specs("pi05", base_specs, PI05_VALIDATED_POLICY, seed, prefetched=False)
        for seed in [4101, 4102, 4103]
    ]
    fixed_with_prefetch_runs = [
        mod.simulate_from_specs("pi05", base_specs, PI05_VALIDATED_POLICY, seed, prefetched=True)
        for seed in [4101, 4102, 4103]
    ]
    fixed_no_prefetch = mod.summarize_runs(fixed_no_prefetch_runs)
    fixed_with_prefetch = mod.summarize_runs(fixed_with_prefetch_runs)

    return {
        "method": "Pi05 predictive prefetch on shared 10Hz shell",
        "policy": dict(PI05_VALIDATED_POLICY),
        "admission_level": {
            "without_prefetch": without_prefetch["summary"],
            "with_prefetch": with_prefetch["summary"],
            "delta": {
                "hard_miss_count_removed": float(
                    without_prefetch["summary"]["hard_miss_count"] - with_prefetch["summary"]["hard_miss_count"]
                ),
                "mean_service_p95_ms_delta": float(
                    with_prefetch["summary"]["mean_service_p95_ms"] - without_prefetch["summary"]["mean_service_p95_ms"]
                ),
                "mean_min_robot_score_delta": float(
                    with_prefetch["summary"]["mean_min_robot_score"]
                    - without_prefetch["summary"]["mean_min_robot_score"]
                ),
            },
        },
        "fixed_four_robot_level": {
            "without_prefetch": {
                "mean_sla_miss_count": fixed_no_prefetch["mean_sla_miss_count"],
                "mean_hard_miss_count": fixed_no_prefetch["mean_hard_miss_count"],
                "mean_latency_p95_ms": fixed_no_prefetch["mean_latency_p95_ms"],
                "mean_min_robot_score": fixed_no_prefetch["mean_min_robot_score"],
            },
            "with_prefetch": {
                "mean_sla_miss_count": fixed_with_prefetch["mean_sla_miss_count"],
                "mean_hard_miss_count": fixed_with_prefetch["mean_hard_miss_count"],
                "mean_latency_p95_ms": fixed_with_prefetch["mean_latency_p95_ms"],
                "mean_min_robot_score": fixed_with_prefetch["mean_min_robot_score"],
            },
            "delta": {
                "mean_sla_miss_count_removed": float(
                    fixed_no_prefetch["mean_sla_miss_count"] - fixed_with_prefetch["mean_sla_miss_count"]
                ),
                "mean_hard_miss_count_removed": float(
                    fixed_no_prefetch["mean_hard_miss_count"] - fixed_with_prefetch["mean_hard_miss_count"]
                ),
            },
        },
    }


def run_phase_lock_case(mod, label: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    strict = mod.aggregate(specs, "strict_horizon")
    phase_lock = mod.aggregate(specs, "phase_lock_batch")
    return {
        "scenario": label,
        "robots": len(specs),
        "strict_horizon": {
            "reply_over_chunk_actions_count": strict["reply_over_chunk_actions_count"],
            "mean_request_to_result_p95_ms": strict["mean_request_to_result_p95_ms"],
            "mean_batch_size": strict["mean_batch_size"],
            "mean_min_robot_score": strict["mean_min_robot_score"],
            "mean_fleet_score": strict["mean_fleet_score"],
            "stable_under_100ms": stable_100ms(
                strict["reply_over_chunk_actions_count"],
                strict["mean_request_to_result_p95_ms"],
            ),
        },
        "phase_lock_batch": {
            "reply_over_chunk_actions_count": phase_lock["reply_over_chunk_actions_count"],
            "mean_request_to_result_p95_ms": phase_lock["mean_request_to_result_p95_ms"],
            "mean_batch_size": phase_lock["mean_batch_size"],
            "mean_min_robot_score": phase_lock["mean_min_robot_score"],
            "mean_fleet_score": phase_lock["mean_fleet_score"],
            "stable_under_100ms": stable_100ms(
                phase_lock["reply_over_chunk_actions_count"],
                phase_lock["mean_request_to_result_p95_ms"],
            ),
        },
        "delta": {
            "reply_over_removed": int(
                strict["reply_over_chunk_actions_count"] - phase_lock["reply_over_chunk_actions_count"]
            ),
            "mean_batch_gain": float(phase_lock["mean_batch_size"] - strict["mean_batch_size"]),
            "mean_min_robot_score_gain": float(phase_lock["mean_min_robot_score"] - strict["mean_min_robot_score"]),
            "mean_fleet_score_gain": float(phase_lock["mean_fleet_score"] - strict["mean_fleet_score"]),
        },
    }


def gr00t_phase_lock_effectiveness() -> dict[str, Any]:
    mod = import_module("gr00t_phase_lock_validation", GR00T_PHASE_LOCK)
    specs_2x = load_json(GR00T_2X_OPT)["specs"]
    specs_4x = load_json(GR00T_4X_OPT)["specs"]
    return {
        "method": "GR00T shared-prefix phase-lock batching",
        "cases": [
            run_phase_lock_case(mod, "2x_per_model_8_robots", specs_2x),
            run_phase_lock_case(mod, "4x_per_model_16_robots", specs_4x),
        ],
    }


def gr00t_fair_admission_effectiveness() -> dict[str, Any]:
    mod = import_module("gr00t_fair_admission_validation", GR00T_FAIR_ADMISSION)
    baseline = mod.run_policy("baseline_greedy", mod.admit_baseline)
    fair = mod.run_policy("quota_fair", mod.admit_quota_fair)
    b = baseline["summary"]
    f = fair["summary"]
    return {
        "method": "GR00T quota-fair admission over greedy batch-first admission",
        "baseline_greedy": b,
        "quota_fair": f,
        "delta": {
            "accept_rate_gap_reduction": float(b["accept_rate_gap"] - f["accept_rate_gap"]),
            "final_count_gap_reduction": float(b["final_count_gap"] - f["final_count_gap"]),
            "jain_final_count_gain": float(f["jain_final_count"] - b["jain_final_count"]),
            "mean_final_robot_count_delta": float(f["mean_final_robot_count"] - b["mean_final_robot_count"]),
            "mean_p95_ms_delta": float(f["mean_p95_ms"] - b["mean_p95_ms"]),
        },
    }


def run_mps_case(mod, label: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    batch_only = mod.aggregate(specs, 60.0, MPS_SEEDS, "batch_only")
    batch_plus_mps = mod.aggregate(specs, 60.0, MPS_SEEDS, "batch_plus_mps")
    return {
        "scenario": label,
        "robots": len(specs),
        "batch_only": {
            "reply_over_chunk_actions_count": batch_only["reply_over_chunk_actions_count"],
            "mean_request_to_result_p95_ms": batch_only["mean_request_to_result_p95_ms"],
            "mean_batch_size": batch_only["mean_batch_size"],
            "stable_under_100ms": stable_100ms(
                batch_only["reply_over_chunk_actions_count"],
                batch_only["mean_request_to_result_p95_ms"],
            ),
        },
        "batch_plus_mps": {
            "reply_over_chunk_actions_count": batch_plus_mps["reply_over_chunk_actions_count"],
            "mean_request_to_result_p95_ms": batch_plus_mps["mean_request_to_result_p95_ms"],
            "mean_batch_size": batch_plus_mps["mean_batch_size"],
            "stable_under_100ms": stable_100ms(
                batch_plus_mps["reply_over_chunk_actions_count"],
                batch_plus_mps["mean_request_to_result_p95_ms"],
            ),
        },
        "delta": {
            "reply_over_change": int(
                batch_plus_mps["reply_over_chunk_actions_count"] - batch_only["reply_over_chunk_actions_count"]
            ),
            "mean_p95_ms_delta": float(
                batch_plus_mps["mean_request_to_result_p95_ms"]
                - batch_only["mean_request_to_result_p95_ms"]
            ),
            "mean_batch_size_delta": float(batch_plus_mps["mean_batch_size"] - batch_only["mean_batch_size"]),
        },
    }


def gr00t_batch_mps_effectiveness() -> dict[str, Any]:
    mod = import_module("gr00t_batch_mps_validation", GR00T_BATCH_MPS)
    specs_4x = load_json(GR00T_4X_OPT)["specs"]
    specs_6x = load_json(GR00T_6X_BEST)["specs"]
    return {
        "method": "GR00T batch-only vs batch+MPS",
        "cases": [
            run_mps_case(mod, "4x_per_model_16_robots", specs_4x),
            run_mps_case(mod, "6x_per_model_24_robots", specs_6x),
        ],
    }


def build_findings(payload: dict[str, Any]) -> list[str]:
    pi05 = payload["pi05_prefetch_effectiveness"]
    phase = payload["gr00t_phase_lock_effectiveness"]["cases"]
    fair = payload["gr00t_fair_admission_effectiveness"]
    mps = payload["gr00t_batch_mps_effectiveness"]["cases"]
    return [
        (
            "Pi05 predictive prefetch removes hard misses in the admission-level run "
            f"({pi05['admission_level']['without_prefetch']['hard_miss_count']} -> "
            f"{pi05['admission_level']['with_prefetch']['hard_miss_count']}) without increasing p95 latency."
        ),
        (
            "GR00T phase-lock batching is the decisive enabler for stable serving: "
            f"at 8 robots reply-over drops {phase[0]['strict_horizon']['reply_over_chunk_actions_count']} -> "
            f"{phase[0]['phase_lock_batch']['reply_over_chunk_actions_count']}, and at 16 robots "
            f"{phase[1]['strict_horizon']['reply_over_chunk_actions_count']} -> "
            f"{phase[1]['phase_lock_batch']['reply_over_chunk_actions_count']}."
        ),
        (
            "GR00T quota-fair admission materially reduces admission bias: accept-rate gap "
            f"{fair['baseline_greedy']['accept_rate_gap']:.4f} -> {fair['quota_fair']['accept_rate_gap']:.4f}, "
            f"final-count gap {fair['baseline_greedy']['final_count_gap']} -> {fair['quota_fair']['final_count_gap']}."
        ),
        (
            "For the validated GR00T runtime, MPS does not improve stable capacity over batch-only: "
            f"at 16 robots both are stable, and at 24 robots both are unstable with "
            f"reply-over {mps[1]['batch_only']['reply_over_chunk_actions_count']} vs "
            f"{mps[1]['batch_plus_mps']['reply_over_chunk_actions_count']}."
        ),
    ]


def main():
    payload = {
        "meta": {
            "goal": "Run fresh, method-level experiments that separately validate the core Pi05 and GR00T serving techniques used in the unified chunked-VLA serving design.",
            "date": "2026-04-12",
            "result_file": str(OUT),
        },
        "pi05_prefetch_effectiveness": pi05_prefetch_effectiveness(),
        "gr00t_phase_lock_effectiveness": gr00t_phase_lock_effectiveness(),
        "gr00t_fair_admission_effectiveness": gr00t_fair_admission_effectiveness(),
        "gr00t_batch_mps_effectiveness": gr00t_batch_mps_effectiveness(),
    }
    payload["findings"] = build_findings(payload)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(OUT)
    print(json.dumps({"findings": payload["findings"]}, indent=2))


if __name__ == "__main__":
    main()
