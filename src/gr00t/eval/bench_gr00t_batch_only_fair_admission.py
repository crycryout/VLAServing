#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/VLAServing")
SRC = ROOT / "src" / "gr00t" / "eval" / "bench_gr00t_shared_prefix_phase_lock_batch_mps.py"
OUT = ROOT / "results" / "gr00t_batch_only_fair_admission_20260412.json"

GROUP_SEEDS = [9101, 9102, 9103, 9104, 9105, 9106]
ARRIVALS_PER_GROUP = 30
PREDICT_DURATION_S = 10.0
TRUTH_DURATION_S = 60.0
PREDICT_SEEDS = [1201]
TRUTH_SEEDS = [1201, 1202, 1203]

TYPE_INFO = [
    {"model": "30hz_bridge", "hz": 30.0, "target_phase_ms": 0.0},
    {"model": "20hz_fractal", "hz": 20.0, "target_phase_ms": 12.5},
    {"model": "10hz_libero", "hz": 10.0, "target_phase_ms": 25.0},
    {"model": "10hz_rel30k", "hz": 10.0, "target_phase_ms": 87.5},
]

TARGET_TOTAL_PER_TYPE = {
    "30Hz::30hz_bridge": 4,
    "20Hz::20hz_fractal": 4,
    "10Hz::10hz_libero": 4,
    "10Hz::10hz_rel30k": 4,
}

BASE_SPECS = [
    {"model": row["model"], "hz": row["hz"], "start_ms": row["target_phase_ms"]}
    for row in TYPE_INFO
]


def load_module():
    spec = importlib.util.spec_from_file_location("bench_gr00t_shared_prefix_phase_lock_batch_mps", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = load_module()


def type_key(model: str, hz: float) -> str:
    return f"{int(hz)}Hz::{model}"


TARGET_PHASE_BY_MODEL = {row["model"]: float(row["target_phase_ms"]) for row in TYPE_INFO}
CANDIDATE_TYPES = [{"model": row["model"], "hz": float(row["hz"])} for row in TYPE_INFO]


def counts_by_type(specs: list[dict[str, float | str]]) -> dict[str, int]:
    ctr = Counter()
    for spec in specs:
        ctr[type_key(str(spec["model"]), float(spec["hz"]))] += 1
    return dict(ctr)


def same_phase_count(specs: list[dict[str, float | str]], cand: dict[str, float | str]) -> int:
    count = 0
    for spec in specs:
        if str(spec["model"]) != str(cand["model"]):
            continue
        if abs(float(spec["hz"]) - float(cand["hz"])) > 1e-9:
            continue
        if abs(float(spec["start_ms"]) - float(cand["start_ms"])) > 1e-9:
            continue
        count += 1
    return count


def predict_metrics(specs: list[dict[str, float | str]]) -> dict:
    return MOD.aggregate(specs, PREDICT_DURATION_S, PREDICT_SEEDS, "batch_only")


def truth_metrics(specs: list[dict[str, float | str]]) -> dict:
    return MOD.aggregate(specs, TRUTH_DURATION_S, TRUTH_SEEDS, "batch_only")


def stable(metrics: dict) -> bool:
    return (
        metrics["reply_over_chunk_actions_count"] == 0
        and metrics["mean_request_to_result_p95_ms"] <= 100.0
    )


def rank_baseline(specs: list[dict[str, float | str]], cand: dict[str, float | str], pred: dict):
    cohort = same_phase_count(specs + [cand], cand)
    return (
        -cohort,
        pred["mean_request_to_result_p95_ms"],
        -pred["mean_batch_size"],
        -pred["mean_min_robot_score"],
        -pred["mean_fleet_score"],
    )


def admit_baseline(specs: list[dict[str, float | str]], offered: dict[str, float | str]):
    best = None
    for phase in MOD.phase_grid(float(offered["hz"])):
        cand = {"model": str(offered["model"]), "hz": float(offered["hz"]), "start_ms": float(phase)}
        pred = predict_metrics(specs + [cand])
        if not stable(pred):
            continue
        rank = rank_baseline(specs, cand, pred)
        if best is None or rank < best[0]:
            best = (rank, cand, pred)
    return None if best is None else {"candidate": best[1], "pred": best[2]}


def admit_quota_fair(specs: list[dict[str, float | str]], offered: dict[str, float | str]):
    key = type_key(str(offered["model"]), float(offered["hz"]))
    current = counts_by_type(specs).get(key, 0)
    if current >= TARGET_TOTAL_PER_TYPE[key]:
        return None

    phases = [TARGET_PHASE_BY_MODEL[str(offered["model"])]]
    for phase in MOD.phase_grid(float(offered["hz"])):
        if all(abs(float(phase) - x) > 1e-9 for x in phases):
            phases.append(float(phase))

    best = None
    for phase in phases:
        cand = {"model": str(offered["model"]), "hz": float(offered["hz"]), "start_ms": float(phase)}
        pred = predict_metrics(specs + [cand])
        if not stable(pred):
            continue
        phase_dist = abs(float(phase) - TARGET_PHASE_BY_MODEL[str(offered["model"])])
        cohort = same_phase_count(specs + [cand], cand)
        rank = (
            phase_dist,
            -cohort,
            pred["mean_request_to_result_p95_ms"],
            -pred["mean_batch_size"],
            -pred["mean_min_robot_score"],
            -pred["mean_fleet_score"],
        )
        if best is None or rank < best[0]:
            best = (rank, cand, pred)
    return None if best is None else {"candidate": best[1], "pred": best[2]}


def jain_index(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    denom = arr.size * float(np.square(arr).sum())
    if denom <= 1e-12:
        return 1.0
    return float((float(arr.sum()) ** 2) / denom)


def summarize_groups(groups: list[dict]) -> dict:
    offered_total = Counter()
    accepted_total = Counter()
    final_total = Counter()
    for group in groups:
        offered_total.update(group["offered_histogram"])
        accepted_total.update(group["accepted_histogram"])
        final_total.update(group["final_histogram"])

    accept_rate_by_type = {}
    for key, offered in offered_total.items():
        accept_rate_by_type[key] = float(accepted_total.get(key, 0) / offered) if offered else 0.0

    accepted_vals = [float(accepted_total.get(key, 0)) for key in sorted(TARGET_TOTAL_PER_TYPE)]
    rate_vals = [float(accept_rate_by_type.get(key, 0.0)) for key in sorted(TARGET_TOTAL_PER_TYPE)]
    final_vals = [float(final_total.get(key, 0)) for key in sorted(TARGET_TOTAL_PER_TYPE)]

    return {
        "offered_total": dict(offered_total),
        "accepted_total": dict(accepted_total),
        "final_total": dict(final_total),
        "accept_rate_by_type": accept_rate_by_type,
        "accept_rate_gap": float(max(rate_vals) - min(rate_vals)),
        "accepted_count_gap": float(max(accepted_vals) - min(accepted_vals)),
        "final_count_gap": float(max(final_vals) - min(final_vals)),
        "jain_accept_rate": jain_index(rate_vals),
        "jain_accepted_count": jain_index(accepted_vals),
        "jain_final_count": jain_index(final_vals),
        "mean_final_robot_count": float(np.mean([group["final_robot_count"] for group in groups])),
        "mean_reply_over": float(np.mean([group["truth"]["reply_over_chunk_actions_count"] for group in groups])),
        "mean_p95_ms": float(np.mean([group["truth"]["mean_request_to_result_p95_ms"] for group in groups])),
        "stable_group_ratio": float(
            np.mean(
                [
                    1.0
                    if (
                        group["truth"]["reply_over_chunk_actions_count"] == 0
                        and group["truth"]["mean_request_to_result_p95_ms"] <= 100.0
                    )
                    else 0.0
                    for group in groups
                ]
            )
        ),
    }


def run_policy(name: str, admit_fn):
    groups = []
    for group_seed in GROUP_SEEDS:
        rng = np.random.default_rng(group_seed)
        specs = [dict(spec) for spec in BASE_SPECS]
        offered_hist = Counter()
        accepted_hist = Counter()
        decision_log = []

        for step in range(ARRIVALS_PER_GROUP):
            offered = dict(CANDIDATE_TYPES[int(rng.integers(0, len(CANDIDATE_TYPES)))])
            key = type_key(str(offered["model"]), float(offered["hz"]))
            offered_hist[key] += 1
            decision = admit_fn(specs, offered)
            if decision is None:
                decision_log.append({"step": step, "key": key, "accepted": False})
                continue
            specs.append(decision["candidate"])
            accepted_hist[key] += 1
            decision_log.append(
                {
                    "step": step,
                    "key": key,
                    "accepted": True,
                    "phase_ms": float(decision["candidate"]["start_ms"]),
                    "predictive_p95_ms": float(decision["pred"]["mean_request_to_result_p95_ms"]),
                    "predictive_mean_batch": float(decision["pred"]["mean_batch_size"]),
                }
            )

        truth = truth_metrics(specs)
        groups.append(
            {
                "seed": int(group_seed),
                "final_robot_count": len(specs),
                "offered_histogram": dict(offered_hist),
                "accepted_histogram": dict(accepted_hist),
                "final_histogram": counts_by_type(specs),
                "truth": truth,
                "decisions_head": decision_log[:24],
            }
        )

    return {"groups": groups, "summary": summarize_groups(groups)}


def main():
    payload = {
        "meta": {
            "predict_duration_s": PREDICT_DURATION_S,
            "truth_duration_s": TRUTH_DURATION_S,
            "predict_seeds": PREDICT_SEEDS,
            "truth_seeds": TRUTH_SEEDS,
            "arrivals_per_group": ARRIVALS_PER_GROUP,
            "group_seeds": GROUP_SEEDS,
            "target_total_per_type": TARGET_TOTAL_PER_TYPE,
            "base_specs": BASE_SPECS,
            "policy_note": "quota_fair reserves 4 total robots per type and prefers the model's dedicated target phase",
        },
        "baseline_greedy": run_policy("baseline_greedy", admit_baseline),
        "quota_fair": run_policy("quota_fair", admit_quota_fair),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
