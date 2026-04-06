#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


RESULTS = Path("/root/autodl-tmp/VLAServing/results")
OUT = RESULTS / "pi05_four_model_residency_prefetch_system_20260406.json"

MODEL_ORDER = [
    "30hz_official_ft",
    "20hz_quantiles",
    "10hz_a_logits",
    "10hz_b_autoh",
]

FULL_SHELL_ALLOC_GB = 22.455
FULL_SWAP_PAYLOAD_GIB = 6.736735850572586
FULL_SWAP_MS = 289.47464376688004
SERVICE_INFER_MS = {
    "30hz_official_ft": 43.198463439941406,
    "20hz_quantiles": 43.18052673339844,
    "10hz_a_logits": 43.21331214904785,
    "10hz_b_autoh": 43.06164741516113,
}

REQUEST_PERIOD_MS = {
    "30hz_official_ft": 10.0 / 30.0 * 1000.0,
    "20hz_quantiles": 10.0 / 20.0 * 1000.0,
    "10hz_a_logits": 10.0 / 10.0 * 1000.0,
    "10hz_b_autoh": 10.0 / 10.0 * 1000.0,
}
HARD_SLACK_MS = {
    "30hz_official_ft": 40.0 / 30.0 * 1000.0,
    "20hz_quantiles": 40.0 / 20.0 * 1000.0,
    "10hz_a_logits": 40.0 / 10.0 * 1000.0,
    "10hz_b_autoh": 40.0 / 10.0 * 1000.0,
}

HORIZON_S = 60.0
OFFSET20_GRID_MS = np.arange(0.0, 500.0, 25.0)
OFFSET10A_GRID_MS = np.arange(0.0, 500.0, 25.0)
R10_GRID = np.arange(0.0, 0.61, 0.05)
DEADLINES_MS = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 100.0]


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


def generate_requests(offset20_ms: float, offset10a_ms: float):
    offsets = {
        "30hz_official_ft": 0.0,
        "20hz_quantiles": offset20_ms,
        "10hz_a_logits": offset10a_ms,
        "10hz_b_autoh": (offset10a_ms + 500.0) % 1000.0,
    }
    reqs = []
    horizon_ms = HORIZON_S * 1000.0
    for model_name in MODEL_ORDER:
        t = offsets[model_name]
        period = REQUEST_PERIOD_MS[model_name]
        while t < horizon_ms:
            reqs.append({"model": model_name, "arrival_ms": t})
            t += period
    reqs.sort(key=lambda x: (x["arrival_ms"], x["model"]))
    return reqs, offsets


def simulate(offset20_ms: float, offset10a_ms: float, resident_fraction_10: float):
    effective_swap_ms = FULL_SWAP_MS * (1.0 - resident_fraction_10)
    requests, offsets = generate_requests(offset20_ms, offset10a_ms)

    compute_free_ms = 0.0
    copy_free_ms = 0.0
    shell_c_loaded = "10hz_a_logits"
    shell_c_ready_ms = 0.0
    samples = []

    for req in requests:
        model = req["model"]
        arrival_ms = req["arrival_ms"]
        swap_wait_ms = 0.0

        if model in ("10hz_a_logits", "10hz_b_autoh"):
            if shell_c_loaded != model:
                copy_start = max(arrival_ms, shell_c_ready_ms, copy_free_ms)
                copy_end = copy_start + effective_swap_ms
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

        if model in ("10hz_a_logits", "10hz_b_autoh"):
            other = "10hz_b_autoh" if model == "10hz_a_logits" else "10hz_a_logits"
            copy_start = max(end_ms, copy_free_ms)
            copy_end = copy_start + effective_swap_ms
            shell_c_loaded = other
            shell_c_ready_ms = copy_end
            copy_free_ms = copy_end

        e2e_ms = end_ms - arrival_ms
        backlog_ms = max(0.0, start_ms - ready_ms)
        samples.append(
            {
                "model": model,
                "arrival_ms": arrival_ms,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "infer_ms": infer_ms,
                "swap_wait_ms": swap_wait_ms,
                "backlog_ms": backlog_ms,
                "e2e_ms": e2e_ms,
                "hard_deadline_ms": HARD_SLACK_MS[model],
                "hard_miss": bool(e2e_ms > HARD_SLACK_MS[model]),
            }
        )

    e2e = [s["e2e_ms"] for s in samples]
    backlog = [s["backlog_ms"] for s in samples]
    swap_wait = [s["swap_wait_ms"] for s in samples]
    hard_miss_count = sum(int(s["hard_miss"]) for s in samples)
    stable = []
    deadline_sweep = {}
    for d in DEADLINES_MS:
        miss_count = sum(1 for x in e2e if x > d)
        deadline_sweep[str(int(d))] = {
            "deadline_ms": d,
            "pass": miss_count == 0,
            "miss_count": miss_count,
        }
        if miss_count == 0:
            stable.append(d)

    extra_resident_gb = 2.0 * resident_fraction_10 * FULL_SWAP_PAYLOAD_GIB
    total_estimated_gpu_gb = FULL_SHELL_ALLOC_GB + extra_resident_gb

    per_model = {}
    for model in MODEL_ORDER:
        vals = [s for s in samples if s["model"] == model]
        per_model[model] = {
            "e2e_ms": stats([s["e2e_ms"] for s in vals]),
            "swap_wait_ms": stats([s["swap_wait_ms"] for s in vals]),
            "backlog_ms": stats([s["backlog_ms"] for s in vals]),
        }

    return {
        "resident_fraction_10": resident_fraction_10,
        "effective_swap_ms": effective_swap_ms,
        "offsets_ms": offsets,
        "gpu_memory_estimate_gb": {
            "three_shells_gb": FULL_SHELL_ALLOC_GB,
            "extra_10hz_resident_gb": extra_resident_gb,
            "total_estimated_gb": total_estimated_gpu_gb,
            "fits_under_24gb": total_estimated_gpu_gb < 24.0,
        },
        "service_e2e_ms": stats(e2e),
        "swap_wait_ms": stats(swap_wait),
        "backlog_ms": stats(backlog),
        "hard_deadline_miss_count": hard_miss_count,
        "deadline_sweep": deadline_sweep,
        "stable_min_deadline_ms": None if not stable else min(stable),
        "per_model": per_model,
        "samples_head": samples[:12],
    }


def main():
    candidates = []
    for resident_fraction_10 in R10_GRID:
        for offset20_ms in OFFSET20_GRID_MS:
            for offset10a_ms in OFFSET10A_GRID_MS:
                candidates.append(
                    simulate(
                        float(offset20_ms),
                        float(offset10a_ms),
                        float(round(resident_fraction_10, 2)),
                    )
                )

    feasible = [
        c
        for c in candidates
        if c["hard_deadline_miss_count"] == 0 and c["gpu_memory_estimate_gb"]["fits_under_24gb"]
    ]
    feasible.sort(
        key=lambda x: (
            x["resident_fraction_10"],
            float("inf") if x["stable_min_deadline_ms"] is None else x["stable_min_deadline_ms"],
            x["service_e2e_ms"]["p95_ms"],
            x["swap_wait_ms"]["p95_ms"],
        )
    )

    best = feasible[0]
    result = {
        "setup": {
            "design": "three-shell Pi0.5 serving: 30Hz resident shell, 20Hz resident shell, one shared shell for two 10Hz models with predictive prefetch and optional partial 10Hz residency",
            "request_policy": "each robot requests next inference after consuming first 10 of 50 actions",
            "simulation_horizon_s": HORIZON_S,
            "search_grid": {
                "resident_fraction_10": [float(x) for x in R10_GRID],
                "offset20_ms": [float(x) for x in OFFSET20_GRID_MS],
                "offset10a_ms": [float(x) for x in OFFSET10A_GRID_MS],
            },
        },
        "measured_constants": {
            "full_shells_alloc_gb_validated": FULL_SHELL_ALLOC_GB,
            "full_swap_payload_gib": FULL_SWAP_PAYLOAD_GIB,
            "full_swap_ms": FULL_SWAP_MS,
            "infer_ms": SERVICE_INFER_MS,
            "request_period_ms": REQUEST_PERIOD_MS,
            "hard_slack_ms": HARD_SLACK_MS,
        },
        "best": best,
        "top5": feasible[:5],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(OUT)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
