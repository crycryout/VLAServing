#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import tyro

SIM_PATH = "/root/autodl-tmp/VLAServing/src/gr00t/eval/simulate_n1d6_global_reservation_serving.py"
HZ_OPTIONS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


@dataclass
class Config:
    groups: int = 10
    max_requests_per_group: int = 120
    global_threshold: float = 0.7
    conservative_horizon: float = 5.0
    admit_min_robot_score: float = 0.97
    admit_fleet_score: float = 0.985
    phase_bins: int = 12
    phase_search_horizons: int = 1
    predict_duration_s: float = 20.0
    predict_seeds: int = 2
    truth_duration_s: float = 90.0
    truth_seeds: int = 4
    output_path: str = "/root/autodl-tmp/groot_n1d6_global_reservation_trials.json"


def load_sim_module() -> Any:
    spec = importlib.util.spec_from_file_location("groot_n1d6_global_reservation_sim", SIM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load simulator module from {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_specs(specs: list[Any], model_names: tuple[str, ...]) -> list[dict]:
    return [
        {
            "model_idx": int(spec.model_idx),
            "model": model_names[int(spec.model_idx)],
            "hz": float(spec.hz),
            "start_ms": float(spec.start_ms),
            "name": spec.name,
        }
        for spec in specs
    ]


def summarize_breakdown(specs: list[Any], model_names: tuple[str, ...]) -> list[dict]:
    counts: dict[tuple[int, int], int] = {}
    for spec in specs:
        key = (int(spec.model_idx), int(spec.hz))
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for (model_idx, hz), count in sorted(counts.items()):
        rows.append(
            {
                "model": model_names[model_idx],
                "hz": float(hz),
                "count": int(count),
            }
        )
    return rows


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
    model_names = tuple(module.MODEL_NAMES)

    truth_cache: dict[tuple[tuple[int, float, float], ...], dict] = {}

    def state_key(specs: list[Any]) -> tuple[tuple[int, float, float], ...]:
        return tuple(sorted((int(spec.model_idx), float(spec.hz), round(float(spec.start_ms), 4)) for spec in specs))

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

    def choose_phase(specs: list[Any], model_idx: int, hz: float, idx: int) -> tuple[bool, dict | None, Any | None]:
        if not module.coarse_accept(
            specs + [module.RobotSpec(model_idx=model_idx, hz=hz, start_ms=0.0, starts_ready=True, name="tmp")],
            partition,
        ):
            return False, None, None

        best = None
        for phase_ms in candidate_phases(hz):
            candidate = module.RobotSpec(
                model_idx=model_idx,
                hz=hz,
                start_ms=phase_ms,
                starts_ready=True,
                name=f"{model_names[model_idx]}_{int(hz)}hz_{idx}",
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
                pred["reply_over_chunk_actions_count"] == 0,
                pred["min_robot_score"],
                pred["fleet_score"],
                -pred["avg_weighted_deviation"],
                -pred["requests_per_s"],
            )
            if best is None or rank > best[0]:
                best = (rank, pred, candidate)
        if best is None:
            return False, None, None
        return True, best[1], best[2]

    groups = []
    for group_idx in range(cfg.groups):
        rng = module.np.random.default_rng(291_000 + group_idx)
        specs: list[Any] = []
        admission_log = []
        rejected_requests = 0
        next_idx = 0

        for request_idx in range(cfg.max_requests_per_group):
            feasible = []
            for model_idx in range(len(model_names)):
                for hz in HZ_OPTIONS:
                    ok, _pred, _candidate = choose_phase(specs, model_idx, float(hz), next_idx)
                    if ok:
                        feasible.append((model_idx, float(hz)))
            if not feasible:
                break

            candidate_model_idx = int(rng.integers(0, len(model_names)))
            candidate_hz = float(HZ_OPTIONS[int(rng.integers(0, len(HZ_OPTIONS)))])
            accepted, pred, candidate = choose_phase(specs, candidate_model_idx, candidate_hz, next_idx)
            item = {
                "request_idx": request_idx,
                "model": model_names[candidate_model_idx],
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
                "admitted_robot_breakdown": summarize_breakdown(specs, model_names),
                "admitted_robots": summarize_specs(specs, model_names),
                "final_metrics": final_eval,
                "admission_log": admission_log,
            }
        )

    total_admitted = sum(g["admitted_total"] for g in groups)
    hz_hist: dict[int, int] = {}
    model_hz_hist: dict[str, dict[str, int]] = {}
    reply_over_chunk = 0
    total_non_bootstrap = 0
    miss_autoh = 0
    total_chunks = 0
    for g in groups:
        for row in g["admitted_robots"]:
            hz = int(row["hz"])
            hz_hist[hz] = hz_hist.get(hz, 0) + 1
            per_model = model_hz_hist.setdefault(row["model"], {})
            per_model[str(hz)] = per_model.get(str(hz), 0) + 1
        fm = g["final_metrics"]
        reply_over_chunk += fm["reply_over_chunk_actions_count"]
        total_non_bootstrap += fm["non_bootstrap_replies_count"]
        miss_autoh += fm["miss_autohorizon_count"]
        total_chunks += fm["chunk_count"]

    summary = {
        "groups": cfg.groups,
        "mean_admitted_total": float(total_admitted / len(groups)),
        "total_admitted_robots": int(total_admitted),
        "mean_fleet_score": float(sum(g["final_metrics"]["fleet_score"] for g in groups) / len(groups)),
        "mean_min_robot_score": float(sum(g["final_metrics"]["min_robot_score"] for g in groups) / len(groups)),
        "mean_max_p95_latency_ms": float(sum(g["final_metrics"]["max_p95_latency_ms"] for g in groups) / len(groups)),
        "groups_min_robot_ge_threshold": int(sum(g["final_metrics"]["min_robot_score"] >= cfg.admit_min_robot_score for g in groups)),
        "groups_fleet_ge_threshold": int(sum(g["final_metrics"]["fleet_score"] >= cfg.admit_fleet_score for g in groups)),
        "admitted_frequency_histogram": {str(k): int(v) for k, v in sorted(hz_hist.items())},
        "admitted_frequency_histogram_by_model": model_hz_hist,
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
        "algorithm": {
            "models": list(model_names),
            "server_model": "single_stage_compiled_full_e2e_global_reservation",
            "denoising_step": 1,
            "torch_compile": True,
            "service_ms": module.SERVICE_MS,
            "global_threshold": cfg.global_threshold,
            "conservative_horizon": cfg.conservative_horizon,
            "phase_bins": cfg.phase_bins,
            "phase_search_horizons": cfg.phase_search_horizons,
            "chunk_size": sim_cfg.chunk_size,
        },
        "summary": summary,
        "groups": groups,
    }


def main() -> None:
    cfg = tyro.cli(Config)
    result = run_trials(cfg)
    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
