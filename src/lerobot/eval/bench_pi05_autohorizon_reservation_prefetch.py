#!/usr/bin/env python3

from __future__ import annotations

import bisect
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


RESULTS = Path("/root/autodl-tmp/VLAServing/results")
OUT = RESULTS / "pi05_autohorizon_reservation_prefetch_20260406.json"

BASE_ROBOTS = [
    {"model": "30hz_official_ft", "hz": 30.0, "start_ms": 0.0},
    {"model": "20hz_quantiles", "hz": 20.0, "start_ms": 50.0},
    {"model": "10hz_a_logits", "hz": 10.0, "start_ms": 100.0},
    {"model": "10hz_b_autoh", "hz": 10.0, "start_ms": 600.0},
]

CANDIDATE_TYPES = [
    {"model": "30hz_official_ft", "hz": 30.0},
    {"model": "20hz_quantiles", "hz": 20.0},
    {"model": "10hz_a_logits", "hz": 10.0},
    {"model": "10hz_b_autoh", "hz": 10.0},
]

INFER_MS = {
    "30hz_official_ft": 43.198463439941406,
    "20hz_quantiles": 43.18052673339844,
    "10hz_a_logits": 43.21331214904785,
    "10hz_b_autoh": 43.06164741516113,
}
COMPRESSED_DELTA_GIB = {
    "10hz_a_logits": 4.010575335472822,
    "10hz_b_autoh": 3.6609760150313377,
}
H2D_GIB_PER_S = 23.27228306738265
DECODE_APPLY_GIB_PER_S = 12.0
THREE_SHELLS_GB = 22.455
PHASE_BINS = 8
PRIORITY_GAMMA = 1.35


@dataclass(frozen=True)
class SuccessMetricParams:
    alpha: float = 0.018
    beta: float = 1.15
    robot_threshold: float = 0.97
    fleet_threshold: float = 0.985


class HorizonProcess:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.current = None
        self._start_states = np.array([16, 17, 18, 50], dtype=np.int64)
        self._start_probs = np.array([0.30, 0.55, 0.025, 0.125], dtype=np.float64)
        self._transition = {
            15: (np.array([15, 16, 18, 19], dtype=np.int64), np.array([0.375, 0.375, 0.125, 0.125], dtype=np.float64)),
            16: (np.array([15, 16, 17, 50], dtype=np.int64), np.array([0.1333333333, 0.1, 0.1666666667, 0.6], dtype=np.float64)),
            17: (np.array([15, 16, 17, 18, 19, 50], dtype=np.int64), np.array([0.015625, 0.0625, 0.1875, 0.15625, 0.015625, 0.5625], dtype=np.float64)),
            18: (np.array([17, 18, 19, 50], dtype=np.int64), np.array([0.1025641026, 0.4615384615, 0.1025641026, 0.3333333333], dtype=np.float64)),
            19: (np.array([18, 19, 50], dtype=np.int64), np.array([0.3333333333, 0.1666666667, 0.5], dtype=np.float64)),
            50: (np.array([16, 17, 18, 19, 50], dtype=np.int64), np.array([0.1617647059, 0.3823529412, 0.1323529412, 0.0294117647, 0.2941176471], dtype=np.float64)),
        }

    def next(self) -> int:
        if self.current is None:
            self.current = int(self.rng.choice(self._start_states, p=self._start_probs))
            return self.current
        states, probs = self._transition[self.current]
        self.current = int(self.rng.choice(states, p=probs))
        return self.current


@dataclass
class Reservation:
    start_ms: float
    finish_ms: float
    robot_idx: int
    model: str
    horizon: int
    consumed: int
    chunk_start_ms: float
    period_ms: float
    prefetch_start_ms: float | None = None
    prefetch_finish_ms: float | None = None


@dataclass
class RobotRuntime:
    model: str
    hz: float
    start_ms: float
    proc: HorizonProcess
    period_ms: float
    chunk_scores: list[float] = field(default_factory=list)
    weighted_deviations: list[float] = field(default_factory=list)
    miss_autohorizon_count: int = 0
    reply_over_chunk_actions: int = 0
    chunk_count: int = 0
    requests_sent: int = 0


def shell_of(model: str) -> str:
    if model == "30hz_official_ft":
        return "A"
    if model == "20hz_quantiles":
        return "B"
    return "C"


def request_period_ms(hz: float) -> float:
    return 1000.0 / hz


def phase_grid(hz: float):
    p = request_period_ms(hz)
    return [i * p / PHASE_BINS for i in range(PHASE_BINS)]


def gpu_memory_estimate_gb(cfg: dict):
    extra = COMPRESSED_DELTA_GIB["10hz_a_logits"] * cfg["r10a"] + COMPRESSED_DELTA_GIB["10hz_b_autoh"] * cfg["r10b"]
    total = THREE_SHELLS_GB + extra
    return {
        "three_shells_gb": THREE_SHELLS_GB,
        "resident_compressed_pages_gb": extra,
        "total_estimated_gb": total,
        "fits_under_24gb": total < 24.0,
    }


def resident_fraction(cfg: dict, model: str) -> float:
    if model == "10hz_a_logits":
        return cfg["r10a"]
    if model == "10hz_b_autoh":
        return cfg["r10b"]
    return 1.0


def prefetch_ms(cfg: dict, model: str) -> float:
    if model not in COMPRESSED_DELTA_GIB:
        return 0.0
    missing_gib = COMPRESSED_DELTA_GIB[model] * (1.0 - resident_fraction(cfg, model))
    return missing_gib / H2D_GIB_PER_S * 1000.0 + missing_gib / DECODE_APPLY_GIB_PER_S * 1000.0


def chunk_success(actual_consumed: int, horizon: int, metric: SuccessMetricParams) -> tuple[float, float]:
    deviation = abs(actual_consumed - horizon)
    weighted = float(deviation * ((50 / horizon) ** metric.beta))
    return float(math.exp(-metric.alpha * weighted)), weighted


def geometric_mean(vals: list[float]) -> float:
    if not vals:
        return 1.0
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / len(vals)))


def _insert_res(res_list: list[Reservation], res: Reservation):
    starts = [r.start_ms for r in res_list]
    idx = bisect.bisect_left(starts, res.start_ms)
    res_list.insert(idx, res)


def _insert_iv(iv_list: list[tuple[float, float]], start: float, finish: float):
    starts = [s for s, _ in iv_list]
    idx = bisect.bisect_left(starts, start)
    iv_list.insert(idx, (start, finish))


def _candidate_gaps(res_list: list[Reservation], release_ms: float, hard_finish_ms: float):
    prev_end = release_ms
    if not res_list:
        yield prev_end, hard_finish_ms, None
        return
    prev = None
    for res in res_list:
        if res.finish_ms <= release_ms + 1e-9:
            prev = res
            continue
        if res.start_ms >= hard_finish_ms - 1e-9:
            break
        gap_start = prev_end
        gap_end = min(res.start_ms, hard_finish_ms)
        if gap_end > gap_start + 1e-9:
            yield gap_start, gap_end, prev
        prev_end = max(prev_end, res.finish_ms)
        prev = res
        if prev_end >= hard_finish_ms - 1e-9:
            return
    if hard_finish_ms > prev_end + 1e-9:
        yield prev_end, hard_finish_ms, prev


def _candidate_copy_slot(copy_res: list[tuple[float, float]], release_ms: float, latest_finish_ms: float, prep_ms: float):
    if prep_ms <= 0:
        return latest_finish_ms, latest_finish_ms
    prev_end = release_ms
    best = None
    for s, f in copy_res:
        if f <= release_ms + 1e-9:
            prev_end = max(prev_end, f)
            continue
        if s >= latest_finish_ms - 1e-9:
            break
        gap_start = prev_end
        gap_end = min(s, latest_finish_ms)
        if gap_end - gap_start >= prep_ms - 1e-9:
            cand_end = gap_end
            cand_start = cand_end - prep_ms
            best = (cand_start, cand_end)
        prev_end = max(prev_end, f)
    if latest_finish_ms - prev_end >= prep_ms - 1e-9:
        best = (latest_finish_ms - prep_ms, latest_finish_ms)
    return best


def _find_slot(shell_res: list[Reservation], copy_res: list[tuple[float, float]], chunk_start_ms: float, period_ms: float, horizon: int, infer_ms: float, model: str, cfg: dict):
    hard_finish_ms = chunk_start_ms + 50.0 * period_ms
    prep = prefetch_ms(cfg, model) if shell_of(model) == "C" else 0.0
    weight = (50 / max(horizon, 1)) ** PRIORITY_GAMMA
    best = None
    for c in sorted(range(1, 51), key=lambda x: (abs(x - horizon) * weight, abs(x - horizon), x)):
        lower = chunk_start_ms + (c - 1) * period_ms
        upper = min(chunk_start_ms + c * period_ms, hard_finish_ms)
        target_finish = min(max(chunk_start_ms + horizon * period_ms, lower + 1e-6), upper)
        for gap_start, gap_end, prev_res in _candidate_gaps(shell_res, chunk_start_ms, hard_finish_ms):
            feasible_finish_lo = max(gap_start + infer_ms, lower + 1e-6)
            feasible_finish_hi = min(gap_end, upper)
            if feasible_finish_lo > feasible_finish_hi + 1e-9:
                continue
            finish = min(max(target_finish, feasible_finish_lo), feasible_finish_hi)
            start = finish - infer_ms
            copy_start = copy_end = None
            if shell_of(model) == "C":
                previous_model = prev_res.model if prev_res is not None else "10hz_a_logits"
                effective_prep = 0.0 if previous_model == model else prep
                slot = _candidate_copy_slot(copy_res, chunk_start_ms, start, effective_prep)
                if slot is None:
                    continue
                copy_start, copy_end = slot
            rank = (
                abs(c - horizon) * weight,
                abs(finish - (chunk_start_ms + horizon * period_ms)),
                finish,
            )
            if best is None or rank < best[0]:
                best = (rank, start, finish, c, copy_start, copy_end)
        if best is not None and best[0][0] == 0:
            break
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5]


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


def simulate(specs, cfg: dict, duration_s: float, seed: int):
    metric = SuccessMetricParams()
    duration_ms = duration_s * 1000.0
    runtimes = [
        RobotRuntime(
            model=s["model"],
            hz=float(s["hz"]),
            start_ms=float(s["start_ms"]),
            proc=HorizonProcess(np.random.default_rng(seed * 1000 + i)),
            period_ms=request_period_ms(float(s["hz"])),
        )
        for i, s in enumerate(specs)
    ]
    shell_res = {"A": [], "B": [], "C": []}
    copy_res: list[tuple[float, float]] = []
    reservation_queue: list[tuple[float, int, Reservation]] = []
    chunk_id = [0] * len(specs)

    def schedule_chunk(robot_idx: int, chunk_start_ms: float):
        if chunk_start_ms > duration_ms + 1e-9:
            return
        rt = runtimes[robot_idx]
        h = rt.proc.next()
        sh = shell_of(rt.model)
        slot = _find_slot(shell_res[sh], copy_res, chunk_start_ms, rt.period_ms, h, INFER_MS[rt.model], rt.model, cfg)
        rt.requests_sent += 1
        if slot is None:
            rt.reply_over_chunk_actions += 1
            return
        start, finish, consumed, pstart, pend = slot
        res = Reservation(
            start_ms=start,
            finish_ms=finish,
            robot_idx=robot_idx,
            model=rt.model,
            horizon=h,
            consumed=consumed,
            chunk_start_ms=chunk_start_ms,
            period_ms=rt.period_ms,
            prefetch_start_ms=pstart,
            prefetch_finish_ms=pend,
        )
        _insert_res(shell_res[sh], res)
        if pstart is not None and pend is not None and pend > pstart:
            _insert_iv(copy_res, pstart, pend)
        heapq.heappush(reservation_queue, (finish, chunk_id[robot_idx], res))
        chunk_id[robot_idx] += 1

    for i, rt in enumerate(runtimes):
        schedule_chunk(i, rt.start_ms)

    samples = []
    while reservation_queue:
        _, _, res = heapq.heappop(reservation_queue)
        rt = runtimes[res.robot_idx]
        rt.chunk_count += 1
        if res.finish_ms > res.chunk_start_ms + 50.0 * rt.period_ms + 1e-9:
            rt.reply_over_chunk_actions += 1
            continue
        if res.consumed != res.horizon:
            rt.miss_autohorizon_count += 1
        score, weighted = chunk_success(res.consumed, res.horizon, metric)
        rt.chunk_scores.append(score)
        rt.weighted_deviations.append(weighted)
        samples.append(
            {
                "robot_id": res.robot_idx,
                "model": res.model,
                "hz": rt.hz,
                "chunk_start_ms": res.chunk_start_ms,
                "horizon": res.horizon,
                "shell": shell_of(res.model),
                "prefetch_wait_ms": 0.0 if res.prefetch_finish_ms is None else max(0.0, res.prefetch_finish_ms - res.chunk_start_ms),
                "queue_wait_ms": max(0.0, res.start_ms - max(res.chunk_start_ms, res.prefetch_finish_ms or res.chunk_start_ms)),
                "e2e_ms": res.finish_ms - res.chunk_start_ms,
                "actual_consumed": res.consumed,
                "hard_deadline_ms": 50.0 * rt.period_ms,
                "hard_miss": False,
                "score": score,
            }
        )
        if res.finish_ms <= duration_ms + 1e-9:
            schedule_chunk(res.robot_idx, res.finish_ms)

    robot_scores = [geometric_mean(rt.chunk_scores) for rt in runtimes]
    total_miss = sum(rt.miss_autohorizon_count for rt in runtimes)
    total_chunks = sum(rt.chunk_count for rt in runtimes)
    total_reply_over = sum(rt.reply_over_chunk_actions for rt in runtimes)
    e2e = [s["e2e_ms"] for s in samples]
    return {
        "service_e2e_ms": stats(e2e),
        "prefetch_wait_ms": stats([s["prefetch_wait_ms"] for s in samples]),
        "queue_wait_ms": stats([s["queue_wait_ms"] for s in samples]),
        "hard_miss_count": 0,
        "reply_over_chunk_actions_count": int(total_reply_over),
        "miss_autohorizon_count": int(total_miss),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(total_miss / total_chunks) if total_chunks else 0.0,
        "fleet_score": geometric_mean(robot_scores),
        "min_robot_score": min(robot_scores) if robot_scores else 1.0,
        "samples_head": samples[:12],
    }


def aggregate(specs, cfg: dict, duration_s: float, seeds):
    outs = [simulate(specs, cfg, duration_s=duration_s, seed=s) for s in seeds]
    return {
        "hard_miss_count": int(sum(o["hard_miss_count"] for o in outs)),
        "reply_over_chunk_actions_count": int(sum(o["reply_over_chunk_actions_count"] for o in outs)),
        "mean_service_p95_ms": float(np.mean([o["service_e2e_ms"]["p95_ms"] for o in outs])),
        "mean_fleet_score": float(np.mean([o["fleet_score"] for o in outs])),
        "mean_min_robot_score": float(np.mean([o["min_robot_score"] for o in outs])),
        "mean_miss_autohorizon_ratio": float(np.mean([o["miss_autohorizon_ratio"] for o in outs])),
        "truth_runs": outs,
    }


def search_fixed4():
    runs = []
    for r10a in (0.0, 0.1, 0.2):
        for r10b in (0.0, 0.1, 0.2):
            cfg = {"r10a": r10a, "r10b": r10b}
            mem = gpu_memory_estimate_gb(cfg)
            if not mem["fits_under_24gb"]:
                continue
            truth = simulate([dict(r) for r in BASE_ROBOTS], cfg, duration_s=60.0, seed=17)
            runs.append({"config": cfg, "gpu_memory": mem, "metrics": truth})
    runs.sort(
        key=lambda x: (
            x["metrics"]["reply_over_chunk_actions_count"],
            -x["metrics"]["fleet_score"],
            -x["metrics"]["min_robot_score"],
            x["metrics"]["miss_autohorizon_ratio"],
            x["gpu_memory"]["total_estimated_gb"],
        )
    )
    return runs[0], runs[:5]


@dataclass
class AdmissionCfg:
    r10a: float
    r10b: float
    groups: int = 3
    candidate_stream_len: int = 40
    predict_seeds: int = 1
    truth_seeds: int = 2


def run_group(seed: int, cfg: AdmissionCfg):
    rng = np.random.default_rng(seed)
    robots = [dict(r) for r in BASE_ROBOTS]
    runtime_cfg = {"r10a": cfg.r10a, "r10b": cfg.r10b}
    admission_log = []
    rejected = 0
    metric = SuccessMetricParams()

    for i in range(cfg.candidate_stream_len):
        choice = CANDIDATE_TYPES[int(rng.integers(0, len(CANDIDATE_TYPES)))]
        model = choice["model"]
        hz = float(choice["hz"])
        best = None
        for phase in phase_grid(hz):
            candidate = {"model": model, "hz": hz, "start_ms": float(phase)}
            trial = robots + [candidate]
            pred = aggregate(trial, runtime_cfg, duration_s=20.0, seeds=range(1, 1 + cfg.predict_seeds))
            ok = (
                pred["hard_miss_count"] == 0
                and pred["reply_over_chunk_actions_count"] == 0
                and pred["mean_min_robot_score"] >= metric.robot_threshold
                and pred["mean_fleet_score"] >= metric.fleet_threshold
            )
            if not ok:
                continue
            rank = (
                -pred["mean_fleet_score"],
                -pred["mean_min_robot_score"],
                pred["mean_miss_autohorizon_ratio"],
                pred["mean_service_p95_ms"],
            )
            if best is None or rank < best[0]:
                best = (rank, candidate, pred)
        item = {"candidate_idx": i, "model": model, "hz": hz, "accepted": bool(best is not None)}
        if best is None:
            rejected += 1
        else:
            robots.append(best[1])
            item["start_ms"] = best[1]["start_ms"]
            item["predictive_fleet_score"] = best[2]["mean_fleet_score"]
            item["predictive_min_robot_score"] = best[2]["mean_min_robot_score"]
        admission_log.append(item)

    truth = aggregate(robots, runtime_cfg, duration_s=60.0, seeds=range(101, 101 + cfg.truth_seeds))
    hist = {}
    for r in robots:
        key = f'{int(r["hz"])}Hz::{r["model"]}'
        hist[key] = hist.get(key, 0) + 1
    return {
        "admitted_total": len(robots),
        "rejected_total": rejected,
        "admitted_histogram": hist,
        "final_metrics": truth,
        "admission_log_head": admission_log[:30],
    }


def search_admission(best_cfg):
    cfg = AdmissionCfg(r10a=best_cfg["r10a"], r10b=best_cfg["r10b"])
    groups = [run_group(20260406200 + i, cfg) for i in range(cfg.groups)]
    total = sum(g["admitted_total"] for g in groups)
    hard = sum(g["final_metrics"]["hard_miss_count"] for g in groups)
    hist = {}
    for g in groups:
        for k, v in g["admitted_histogram"].items():
            hist[k] = hist.get(k, 0) + v
    return {
        "config": {
            "r10a": cfg.r10a,
            "r10b": cfg.r10b,
            "gpu_memory_estimate_gb": gpu_memory_estimate_gb(best_cfg),
        },
        "summary": {
            "mean_admitted_total": total / len(groups),
            "total_admitted_robots": total,
            "hard_miss_count": hard,
            "mean_fleet_score": float(np.mean([g["final_metrics"]["mean_fleet_score"] for g in groups])),
            "mean_min_robot_score": float(np.mean([g["final_metrics"]["mean_min_robot_score"] for g in groups])),
            "mean_service_p95_ms": float(np.mean([g["final_metrics"]["mean_service_p95_ms"] for g in groups])),
            "mean_miss_autohorizon_ratio": float(np.mean([g["final_metrics"]["mean_miss_autohorizon_ratio"] for g in groups])),
            "admitted_histogram": hist,
        },
        "groups_detail": groups,
    }


def main():
    fixed_best, fixed_top5 = search_fixed4()
    adm = search_admission(fixed_best["config"])
    result = {
        "setup": {
            "design": "Pi0.5 p50 AutoHorizon + shell reservation + prefetch stream reservation for the shared 10Hz shell",
            "base_robots": BASE_ROBOTS,
            "candidate_types": CANDIDATE_TYPES,
        },
        "fixed4_best": fixed_best,
        "fixed4_top5": fixed_top5,
        "admission": adm,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(OUT)
    print(json.dumps({"fixed4_best": fixed_best, "admission": adm["summary"]}, indent=2))


if __name__ == "__main__":
    main()
