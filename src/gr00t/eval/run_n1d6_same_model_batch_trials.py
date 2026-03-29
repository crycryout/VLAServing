#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tyro


SIM_PATH = "/root/autodl-tmp/VLAServing/src/gr00t/eval/simulate_n1d6_same_model_batch_serving.py"
HZ_OPTIONS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


@dataclass
class Config:
    groups: int = 10
    max_requests_per_group: int = 160
    global_threshold: float = 1.0
    conservative_horizon: float = 5.0
    admit_min_robot_score: float = 0.97
    admit_fleet_score: float = 0.985
    phase_bins: int = 16
    phase_search_horizons: int = 2
    predict_duration_s: float = 20.0
    predict_seeds: int = 2
    truth_duration_s: float = 90.0
    truth_seeds: int = 4
    output_path: str = "/root/autodl-tmp/VLAServing/results/groot_n1d6_same_model_batch_trials.json"


def load_sim_module() -> Any:
    spec = importlib.util.spec_from_file_location("groot_n1d6_same_model_batch_sim", SIM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load simulator module from {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_specs(specs: list[Any]) -> list[dict]:
    return [
        {
            "model": "gr00t_n1d6-libero",
            "hz": float(spec.hz),
            "start_ms": float(spec.start_ms),
            "name": spec.name,
        }
        for spec in specs
    ]


def summarize_breakdown(specs: list[Any]) -> list[dict]:
    counts: dict[int, int] = {}
    for spec in specs:
        key = int(spec.hz)
        counts[key] = counts.get(key, 0) + 1
    return [{"model": "gr00t_n1d6-libero", "hz": float(hz), "count": int(count)} for hz, count in sorted(counts.items())]


def run_trials(cfg: Config) -> dict:
    module = load_sim_module()
    sim_cfg = module.Config(
        predict_duration_s=cfg.predict_duration_s,
        predict_seeds=cfg.predict_seeds,
        truth_duration_s=cfg.truth_duration_s,
        truth_seeds=cfg.truth_seeds,
        phase_bins=cfg.phase_bins,
    )
    horizon = module.AutoHorizonParams(conservative_horizon=cfg.conservative_horizon)
    metric = module.SuccessMetricParams()
    partition = module.ResourcePartition(
        conservative_horizon=cfg.conservative_horizon,
        global_threshold=cfg.global_threshold,
    )

    truth_cache: dict[tuple[tuple[float, float], ...], dict] = {}

    def state_key(specs: list[Any]) -> tuple[tuple[float, float], ...]:
        return tuple(sorted((float(spec.hz), round(float(spec.start_ms), 4)) for spec in specs))

    def evaluate(specs: list[Any], predict: bool) -> dict:
        key = state_key(specs)
        if not predict and key in truth_cache:
            return truth_cache[key]
        if not specs:
            result = {
                "fleet_score": 1.0,
                "min_robot_score": 1.0,
                "max_p95_latency_ms": 0.0,
                "avg_weighted_deviation": 0.0,
                "requests_per_s": 0.0,
                "robot_count": 0,
                "reply_over_chunk_actions_count": 0,
                "non_bootstrap_replies_count": 0,
                "reply_over_chunk_actions_ratio": 0.0,
                "miss_autohorizon_count": 0,
                "chunk_count": 0,
                "miss_autohorizon_ratio": 0.0,
                "consumed_hist": {},
                "horizon_hist": {},
            }
        else:
            if predict:
                result = module.predictive_eval(specs, sim_cfg, horizon, metric)
            else:
                result = module.long_truth(specs, sim_cfg, horizon, metric)
                truth_cache[key] = result
        return result

    def candidate_phases(hz: float) -> list[float]:
        cycle_ms = cfg.phase_search_horizons * 16.0 * (1000.0 / hz)
        bins = max(1, cfg.phase_bins)
        return [i * cycle_ms / bins for i in range(bins)]

    def choose_phase(specs: list[Any], hz: float, idx: int) -> tuple[bool, dict | None, Any | None]:
        if not module.coarse_accept(
            specs + [module.RobotSpec(hz=hz, start_ms=0.0, starts_ready=True, name="tmp")],
            partition,
        ):
            return False, None, None
        best = None
        for phase_ms in candidate_phases(hz):
            candidate = module.RobotSpec(
                hz=hz,
                start_ms=phase_ms,
                starts_ready=True,
                name=f"gr00t_n1d6_libero_{int(hz)}hz_{idx}",
            )
            full_specs = specs + [candidate]
            if not module.coarse_accept(full_specs, partition):
                continue
            pred = evaluate(full_specs, predict=True)
            accepted = bool(
                pred["reply_over_chunk_actions_count"] == 0
                and pred["min_robot_score"] >= cfg.admit_min_robot_score
                and pred["fleet_score"] >= cfg.admit_fleet_score
            )
            if not accepted:
                continue
            rank = (
                pred["min_robot_score"],
                pred["fleet_score"],
                -pred["avg_weighted_deviation"],
                pred["requests_per_s"],
            )
            if best is None or rank > best[0]:
                best = (rank, pred, candidate)
        if best is None:
            return False, None, None
        return True, best[1], best[2]

    groups = []
    for group_idx in range(cfg.groups):
        rng = np.random.default_rng(392_000 + group_idx)
        specs: list[Any] = []
        admission_log = []
        rejected_requests = 0
        next_idx = 0

        for request_idx in range(cfg.max_requests_per_group):
            feasible = []
            for hz in HZ_OPTIONS:
                ok, _pred, _candidate = choose_phase(specs, float(hz), next_idx)
                if ok:
                    feasible.append(float(hz))
            if not feasible:
                break

            candidate_hz = float(HZ_OPTIONS[int(rng.integers(0, len(HZ_OPTIONS)))])
            accepted, pred, candidate = choose_phase(specs, candidate_hz, next_idx)
            item = {
                "request_idx": request_idx,
                "model": "gr00t_n1d6-libero",
                "hz": candidate_hz,
                "accepted": bool(accepted),
            }
            if accepted and candidate is not None:
                specs.append(candidate)
                next_idx += 1
                item["start_ms"] = float(candidate.start_ms)
                item["fleet_size_after_accept"] = len(specs)
                item["predictive_eval"] = pred
            else:
                rejected_requests += 1
            admission_log.append(item)

        final_eval = evaluate(specs, predict=False)
        groups.append(
            {
                "group_idx": group_idx,
                "admitted_total": len(specs),
                "rejected_requests": rejected_requests,
                "admitted_robot_breakdown": summarize_breakdown(specs),
                "admitted_robots": summarize_specs(specs),
                "final_metrics": final_eval,
                "admission_log": admission_log,
            }
        )

    total_admitted = sum(g["admitted_total"] for g in groups)
    hz_hist: dict[int, int] = {}
    reply_over_chunk = 0
    total_non_bootstrap = 0
    miss_autoh = 0
    total_chunks = 0
    for g in groups:
        for row in g["admitted_robots"]:
            hz = int(row["hz"])
            hz_hist[hz] = hz_hist.get(hz, 0) + 1
        fm = g["final_metrics"]
        reply_over_chunk += fm["reply_over_chunk_actions_count"]
        total_non_bootstrap += fm["non_bootstrap_replies_count"]
        miss_autoh += fm["miss_autohorizon_count"]
        total_chunks += fm["chunk_count"]

    summary = {
        "groups": cfg.groups,
        "model": "gr00t_n1d6-libero",
        "mean_admitted_total": float(total_admitted / len(groups)),
        "total_admitted_robots": int(total_admitted),
        "mean_fleet_score": float(sum(g["final_metrics"]["fleet_score"] for g in groups) / len(groups)),
        "mean_min_robot_score": float(sum(g["final_metrics"]["min_robot_score"] for g in groups) / len(groups)),
        "mean_max_p95_latency_ms": float(sum(g["final_metrics"]["max_p95_latency_ms"] for g in groups) / len(groups)),
        "groups_min_robot_ge_threshold": int(sum(g["final_metrics"]["min_robot_score"] >= cfg.admit_min_robot_score for g in groups)),
        "groups_fleet_ge_threshold": int(sum(g["final_metrics"]["fleet_score"] >= cfg.admit_fleet_score for g in groups)),
        "admitted_frequency_histogram": {str(k): int(v) for k, v in sorted(hz_hist.items())},
        "reply_over_chunk_actions_count": int(reply_over_chunk),
        "non_bootstrap_replies_count": int(total_non_bootstrap),
        "reply_over_chunk_actions_ratio": float(reply_over_chunk / total_non_bootstrap) if total_non_bootstrap else 0.0,
        "miss_autohorizon_count": int(miss_autoh),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(miss_autoh / total_chunks) if total_chunks else 0.0,
        "contains_25hz": bool(hz_hist.get(25, 0) > 0),
        "contains_30hz": bool(hz_hist.get(30, 0) > 0),
    }
    return {
        "config": cfg.__dict__,
        "summary": summary,
        "groups_detail": groups,
        "service_curve_ms": module.BATCH_SERVICE_MS,
        "peak_request_throughput": module.PEAK_REQUEST_THROUGHPUT,
    }


def main(cfg: Config) -> None:
    result = run_trials(cfg)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
