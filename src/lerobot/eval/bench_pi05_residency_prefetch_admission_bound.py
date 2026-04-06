#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESULTS = Path("/root/autodl-tmp/VLAServing/results")
OUT = RESULTS / "pi05_residency_prefetch_admission_bound_20260406.json"

BASE_ROBOTS = [
    {"model": "30hz_official_ft", "hz": 30.0, "start_ms": 0.0},
    {"model": "20hz_quantiles", "hz": 20.0, "start_ms": 50.0},
    {"model": "10hz_a_logits", "hz": 10.0, "start_ms": 100.0},
    {"model": "10hz_b_autoh", "hz": 10.0, "start_ms": 600.0},
]

# The model-frequency binding is fixed.
CANDIDATE_TYPES = [
    {"model": "30hz_official_ft", "hz": 30.0},
    {"model": "20hz_quantiles", "hz": 20.0},
    {"model": "10hz_a_logits", "hz": 10.0},
    {"model": "10hz_b_autoh", "hz": 10.0},
]

SERVICE_INFER_MS = {
    "30hz_official_ft": 43.198463439941406,
    "20hz_quantiles": 43.18052673339844,
    "10hz_a_logits": 43.21331214904785,
    "10hz_b_autoh": 43.06164741516113,
}
FULL_SWAP_MS = 289.47464376688004
GPU_THREE_SHELLS_GB = 22.455
FULL_SWAP_PAYLOAD_GIB = 6.736735850572586
HORIZON_S_PRED = 20.0
HORIZON_S_TRUTH = 60.0
PHASE_BINS = 8


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "num_samples": int(arr.size),
    }


def request_period_ms(hz: float) -> float:
    return 10.0 / hz * 1000.0


def hard_slack_ms(hz: float) -> float:
    return 40.0 / hz * 1000.0


def shell_of(model: str) -> str:
    if model == "30hz_official_ft":
        return "A"
    if model == "20hz_quantiles":
        return "B"
    return "C"


def resident_fraction_for(model: str, shared_r10: float) -> float:
    return shared_r10 if model in ("10hz_a_logits", "10hz_b_autoh") else 1.0


def generate_requests(robots, horizon_s: float):
    reqs = []
    horizon_ms = horizon_s * 1000.0
    for rid, robot in enumerate(robots):
        t = float(robot["start_ms"])
        period = request_period_ms(float(robot["hz"]))
        while t < horizon_ms:
            reqs.append(
                {
                    "robot_id": rid,
                    "model": robot["model"],
                    "hz": float(robot["hz"]),
                    "arrival_ms": t,
                }
            )
            t += period
    reqs.sort(key=lambda x: (x["arrival_ms"], x["model"], x["robot_id"]))
    return reqs


def simulate(robots, shared_r10: float, horizon_s: float):
    requests = generate_requests(robots, horizon_s)
    compute_free_ms = 0.0
    copy_free_ms = 0.0
    shell_c_loaded = "10hz_a_logits"
    shell_c_ready_ms = 0.0
    samples = []

    future_c_models = [None] * len(requests)
    next_c_model = None
    for i in range(len(requests) - 1, -1, -1):
        future_c_models[i] = next_c_model
        if shell_of(requests[i]["model"]) == "C":
            next_c_model = requests[i]["model"]

    for idx, req in enumerate(requests):
        model = req["model"]
        arrival_ms = req["arrival_ms"]
        swap_wait_ms = 0.0

        if shell_of(model) == "C":
            eff_swap_ms = FULL_SWAP_MS * (1.0 - resident_fraction_for(model, shared_r10))
            if shell_c_loaded != model:
                copy_start = max(arrival_ms, shell_c_ready_ms, copy_free_ms)
                copy_end = copy_start + eff_swap_ms
                shell_c_loaded = model
                shell_c_ready_ms = copy_end
                copy_free_ms = copy_end
            ready_ms = max(arrival_ms, shell_c_ready_ms)
            swap_wait_ms = ready_ms - arrival_ms
        else:
            ready_ms = arrival_ms

        start_ms = max(ready_ms, compute_free_ms)
        infer_ms = SERVICE_INFER_MS[model]
        end_ms = start_ms + infer_ms
        compute_free_ms = end_ms

        if shell_of(model) == "C":
            nxt = future_c_models[idx]
            if nxt is not None and nxt != shell_c_loaded:
                eff_swap_ms = FULL_SWAP_MS * (1.0 - resident_fraction_for(nxt, shared_r10))
                copy_start = max(end_ms, copy_free_ms)
                copy_end = copy_start + eff_swap_ms
                shell_c_loaded = nxt
                shell_c_ready_ms = copy_end
                copy_free_ms = copy_end

        e2e_ms = end_ms - arrival_ms
        samples.append(
            {
                **req,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "infer_ms": infer_ms,
                "swap_wait_ms": swap_wait_ms,
                "backlog_ms": max(0.0, start_ms - ready_ms),
                "e2e_ms": e2e_ms,
                "hard_deadline_ms": hard_slack_ms(req["hz"]),
                "hard_miss": bool(e2e_ms > hard_slack_ms(req["hz"])),
            }
        )

    e2e = [s["e2e_ms"] for s in samples]
    hard_miss_count = sum(int(s["hard_miss"]) for s in samples)
    return {
        "service_e2e_ms": stats(e2e),
        "hard_miss_count": hard_miss_count,
        "samples_head": samples[:12],
    }


def candidate_phase_grid(hz: float):
    period = request_period_ms(hz)
    return [i * period / PHASE_BINS for i in range(PHASE_BINS)]


def gpu_memory_estimate_gb(shared_r10: float):
    extra = 2.0 * shared_r10 * FULL_SWAP_PAYLOAD_GIB
    total = GPU_THREE_SHELLS_GB + extra
    return {
        "three_shells_gb": GPU_THREE_SHELLS_GB,
        "extra_10hz_resident_gb": extra,
        "total_estimated_gb": total,
    }


@dataclass
class SweepCfg:
    groups: int
    candidate_stream_len: int
    shared_r10: float
    service_deadline_ms: float


def run_group(group_seed: int, cfg: SweepCfg):
    rng = np.random.default_rng(group_seed)
    robots = [dict(r) for r in BASE_ROBOTS]
    admission_log = []
    rejects = 0

    for i in range(cfg.candidate_stream_len):
        choice = CANDIDATE_TYPES[int(rng.integers(0, len(CANDIDATE_TYPES)))]
        model = choice["model"]
        hz = float(choice["hz"])
        best = None
        for phase_ms in candidate_phase_grid(hz):
            candidate = {"model": model, "hz": hz, "start_ms": float(phase_ms)}
            trial = robots + [candidate]
            pred = simulate(trial, cfg.shared_r10, HORIZON_S_PRED)
            ok = pred["hard_miss_count"] == 0 and pred["service_e2e_ms"]["p95_ms"] <= cfg.service_deadline_ms
            if not ok:
                continue
            # Prefer smaller p95, then smaller mean.
            rank = (
                pred["service_e2e_ms"]["p95_ms"],
                pred["service_e2e_ms"]["mean_ms"],
            )
            if best is None or rank < best[0]:
                best = (rank, candidate, pred)

        item = {"candidate_idx": i, "model": model, "hz": hz, "accepted": bool(best is not None)}
        if best is not None:
            robots.append(best[1])
            item["start_ms"] = best[1]["start_ms"]
            item["predictive_p95_ms"] = best[2]["service_e2e_ms"]["p95_ms"]
        else:
            rejects += 1
        admission_log.append(item)

    truth = simulate(robots, cfg.shared_r10, HORIZON_S_TRUTH)
    hist = {}
    for r in robots:
        key = f'{int(r["hz"])}Hz::{r["model"]}'
        hist[key] = hist.get(key, 0) + 1
    return {
        "admitted_total": len(robots),
        "rejected_total": rejects,
        "admitted_robots": robots,
        "admitted_histogram": hist,
        "final_metrics": truth,
        "admission_log_head": admission_log[:30],
    }


def run_cfg(cfg: SweepCfg):
    groups = [run_group(2026040600 + i, cfg) for i in range(cfg.groups)]
    total = sum(g["admitted_total"] for g in groups)
    hard = sum(g["final_metrics"]["hard_miss_count"] for g in groups)
    freq_hist = {}
    for g in groups:
        for k, v in g["admitted_histogram"].items():
            freq_hist[k] = freq_hist.get(k, 0) + v
    return {
        "config": {
            "groups": cfg.groups,
            "candidate_stream_len": cfg.candidate_stream_len,
            "shared_r10": cfg.shared_r10,
            "service_deadline_ms": cfg.service_deadline_ms,
            "gpu_memory_estimate_gb": gpu_memory_estimate_gb(cfg.shared_r10),
        },
        "summary": {
            "mean_admitted_total": total / len(groups),
            "total_admitted_robots": total,
            "mean_service_p95_ms": float(np.mean([g["final_metrics"]["service_e2e_ms"]["p95_ms"] for g in groups])),
            "mean_service_max_ms": float(np.mean([g["final_metrics"]["service_e2e_ms"]["max_ms"] for g in groups])),
            "hard_miss_count": hard,
            "admitted_histogram": freq_hist,
        },
        "groups_detail": groups,
    }


def main():
    runs = []
    for shared_r10 in (0.0, 0.2, 0.4, 0.6):
        for deadline_ms in (45.0, 50.0, 55.0, 60.0, 80.0):
            runs.append(
                run_cfg(
                    SweepCfg(
                        groups=3,
                        candidate_stream_len=80,
                        shared_r10=shared_r10,
                        service_deadline_ms=deadline_ms,
                    )
                )
            )

    feasible = [r for r in runs if r["summary"]["hard_miss_count"] == 0]
    feasible.sort(
        key=lambda x: (
            x["config"]["shared_r10"],
            x["config"]["service_deadline_ms"],
            -x["summary"]["mean_admitted_total"],
            x["summary"]["mean_service_p95_ms"],
        )
    )

    result = {
        "setup": {
            "design": "three-shell Pi0.5 serving with random robot admission on top of the fixed 30/20/10/10 base set, with model-frequency binding enforced",
            "base_robots": BASE_ROBOTS,
            "candidate_types": CANDIDATE_TYPES,
        },
        "all_runs": runs,
        "best_feasible": feasible[0] if feasible else None,
        "top5_feasible": feasible[:5],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(OUT)
    if feasible:
        print(json.dumps(feasible[0], indent=2))
    else:
        print("No feasible run found.")


if __name__ == "__main__":
    main()
