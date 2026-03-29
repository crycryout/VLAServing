#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
import bisect
import math
import numpy as np

SERVICE_MS = 44.59212875366211
PRIORITY_GAMMA = 1.35


@dataclass(frozen=True)
class AutoHorizonParams:
    states: tuple[int, ...] = (15, 16, 17, 18, 19, 50)
    hmin: int = 15
    hmax: int = 50
    expected_horizon: float = 29.333333333333332
    conservative_horizon: float = 16.0


@dataclass(frozen=True)
class SuccessMetricParams:
    alpha: float = 0.018
    beta: float = 1.15
    robot_threshold: float = 0.97
    fleet_threshold: float = 0.985


@dataclass(frozen=True)
class ResourcePartition:
    conservative_horizon: float = 16.0
    global_threshold: float = 0.5


@dataclass
class Config:
    predict_duration_s: float = 20.0
    predict_seeds: int = 2
    truth_duration_s: float = 90.0
    truth_seeds: int = 4
    chunk_size: int = 50
    phase_bins: int = 12


@dataclass(frozen=True)
class RobotSpec:
    hz: float
    start_ms: float
    starts_ready: bool
    name: str


@dataclass
class Reservation:
    start_ms: float
    finish_ms: float
    robot_idx: int
    chunk_id: int
    chunk_start_ms: float
    period_ms: float
    horizon: int
    next_horizon: int
    consumed: int


class HorizonProcess:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.current: int | None = None
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
class RobotRuntime:
    spec: RobotSpec
    period_ms: float
    chunk_scores: list[float] = field(default_factory=list)
    weighted_deviations: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    non_bootstrap_replies: int = 0
    reply_over_chunk_actions: int = 0
    chunk_count: int = 0
    miss_autohorizon_count: int = 0
    consumed_hist: dict[int, int] = field(default_factory=dict)
    horizon_hist: dict[int, int] = field(default_factory=dict)
    per_horizon_alignment: dict[int, dict[str, int]] = field(default_factory=dict)
    requests_sent: int = 0


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values)))


def chunk_success(actual_consumed: int, optimal_horizon: int, horizon: AutoHorizonParams, metric: SuccessMetricParams) -> tuple[float, float]:
    deviation = abs(actual_consumed - optimal_horizon)
    weighted_dev = float(deviation * ((horizon.hmax / optimal_horizon) ** metric.beta))
    score = float(math.exp(-metric.alpha * weighted_dev))
    return score, weighted_dev


def coarse_load(specs: list[RobotSpec], partition: ResourcePartition) -> float:
    service_s = SERVICE_MS / 1000.0
    return float(sum((spec.hz / partition.conservative_horizon) * service_s for spec in specs))


def coarse_accept(specs: list[RobotSpec], partition: ResourcePartition) -> bool:
    return coarse_load(specs, partition) <= partition.global_threshold


def _insert_reservation(reservations: list[Reservation], reservation: Reservation) -> None:
    starts = [r.start_ms for r in reservations]
    idx = bisect.bisect_left(starts, reservation.start_ms)
    reservations.insert(idx, reservation)


def _candidate_gaps(reservations: list[Reservation], release_ms: float, hard_finish_ms: float):
    prev_end = release_ms
    if not reservations:
        yield prev_end, hard_finish_ms
        return
    for res in reservations:
        if res.finish_ms <= release_ms + 1e-9:
            continue
        if res.start_ms >= hard_finish_ms - 1e-9:
            break
        gap_start = prev_end
        gap_end = min(res.start_ms, hard_finish_ms)
        if gap_end - gap_start >= SERVICE_MS - 1e-9:
            yield gap_start, gap_end
        prev_end = max(prev_end, res.finish_ms)
        if prev_end >= hard_finish_ms - 1e-9:
            return
    if hard_finish_ms - prev_end >= SERVICE_MS - 1e-9:
        yield prev_end, hard_finish_ms


def _find_best_slot(
    reservations: list[Reservation],
    chunk_start_ms: float,
    period_ms: float,
    horizon: int,
    chunk_size: int,
) -> tuple[float, float, int] | None:
    release_ms = chunk_start_ms
    hard_finish_ms = chunk_start_ms + chunk_size * period_ms
    weight = (chunk_size / max(horizon, 1)) ** PRIORITY_GAMMA
    best = None
    for c in sorted(range(1, chunk_size + 1), key=lambda x: (abs(x - horizon) * weight, abs(x - horizon), x)):
        lower = chunk_start_ms + (c - 1) * period_ms
        upper = chunk_start_ms + c * period_ms
        desired_finish = min(max(chunk_start_ms + horizon * period_ms, lower + 1e-6), upper)
        for gap_start, gap_end in _candidate_gaps(reservations, release_ms, hard_finish_ms):
            feasible_start = max(gap_start, release_ms)
            feasible_finish_lo = max(feasible_start + SERVICE_MS, lower + 1e-6)
            feasible_finish_hi = min(gap_end, upper)
            if feasible_finish_lo > feasible_finish_hi + 1e-9:
                continue
            finish = min(max(desired_finish, feasible_finish_lo), feasible_finish_hi)
            start = finish - SERVICE_MS
            rank = (
                abs(c - horizon) * weight,
                abs(c - horizon),
                abs(finish - (chunk_start_ms + horizon * period_ms)),
                finish,
            )
            if best is None or rank < best[0]:
                best = (rank, start, finish, c)
        if best is not None and best[0][0] == 0:
            break
    if best is None:
        return None
    return best[1], best[2], best[3]


def simulate(specs: list[RobotSpec], cfg: Config, horizon_params: AutoHorizonParams, metric: SuccessMetricParams, seed: int, duration_s: float) -> dict:
    duration_ms = duration_s * 1000.0
    processes = [HorizonProcess(np.random.default_rng(seed * 1000 + i)) for i in range(len(specs))]
    runtimes = [RobotRuntime(spec=s, period_ms=1000.0 / s.hz) for i, s in enumerate(specs)]
    reservations: list[Reservation] = []
    chunk_ids = [0 for _ in specs]

    def schedule_chunk(robot_idx: int, chunk_start_ms: float) -> bool:
        if chunk_start_ms > duration_ms + 1e-9:
            return False
        h = processes[robot_idx].next()
        nh = processes[robot_idx].next()
        period_ms = runtimes[robot_idx].period_ms
        slot = _find_best_slot(reservations, chunk_start_ms, period_ms, h, cfg.chunk_size)
        if slot is None:
            runtimes[robot_idx].reply_over_chunk_actions += 1
            return False
        start_ms, finish_ms, consumed = slot
        res = Reservation(
            start_ms=start_ms,
            finish_ms=finish_ms,
            robot_idx=robot_idx,
            chunk_id=chunk_ids[robot_idx],
            chunk_start_ms=chunk_start_ms,
            period_ms=period_ms,
            horizon=h,
            next_horizon=nh,
            consumed=consumed,
        )
        chunk_ids[robot_idx] += 1
        _insert_reservation(reservations, res)
        runtimes[robot_idx].requests_sent += 1
        return True

    for i, spec in enumerate(specs):
        schedule_chunk(i, spec.start_ms)

    while reservations:
        res = reservations.pop(0)
        rt = runtimes[res.robot_idx]
        rt.non_bootstrap_replies += 1
        rt.latencies_ms.append(SERVICE_MS)
        if res.finish_ms > res.chunk_start_ms + cfg.chunk_size * res.period_ms + 1e-9:
            rt.reply_over_chunk_actions += 1
            continue
        actual_consumed = res.consumed
        switch_time_ms = res.finish_ms
        rt.chunk_count += 1
        if actual_consumed != res.horizon:
            rt.miss_autohorizon_count += 1
        score, weighted_dev = chunk_success(actual_consumed, res.horizon, horizon_params, metric)
        rt.chunk_scores.append(score)
        rt.weighted_deviations.append(weighted_dev)
        rt.consumed_hist[actual_consumed] = rt.consumed_hist.get(actual_consumed, 0) + 1
        rt.horizon_hist[res.horizon] = rt.horizon_hist.get(res.horizon, 0) + 1
        bucket = rt.per_horizon_alignment.setdefault(res.horizon, {"early": 0, "exact": 0, "late": 0})
        if actual_consumed < res.horizon:
            bucket["early"] += 1
        elif actual_consumed > res.horizon:
            bucket["late"] += 1
        else:
            bucket["exact"] += 1
        if switch_time_ms <= duration_ms + 1e-9:
            schedule_chunk(res.robot_idx, switch_time_ms)

    robot_scores = [geometric_mean(rt.chunk_scores) for rt in runtimes]
    p95s = [float(np.percentile(rt.latencies_ms, 95)) for rt in runtimes if rt.latencies_ms]
    total_reply_over = sum(rt.reply_over_chunk_actions for rt in runtimes)
    total_non_bootstrap = sum(rt.non_bootstrap_replies for rt in runtimes)
    total_miss = sum(rt.miss_autohorizon_count for rt in runtimes)
    total_chunks = sum(rt.chunk_count for rt in runtimes)
    total_weighted = sum(sum(rt.weighted_deviations) for rt in runtimes)
    total_weighted_count = sum(len(rt.weighted_deviations) for rt in runtimes)
    consumed_hist: dict[int, int] = {}
    horizon_hist: dict[int, int] = {}
    per_horizon_alignment: dict[int, dict[str, int]] = {}
    for rt in runtimes:
        for key, value in rt.consumed_hist.items():
            consumed_hist[key] = consumed_hist.get(key, 0) + value
        for key, value in rt.horizon_hist.items():
            horizon_hist[key] = horizon_hist.get(key, 0) + value
        for key, value in rt.per_horizon_alignment.items():
            bucket = per_horizon_alignment.setdefault(key, {"early": 0, "exact": 0, "late": 0})
            bucket["early"] += int(value["early"])
            bucket["exact"] += int(value["exact"])
            bucket["late"] += int(value["late"])

    return {
        "fleet_score": geometric_mean(robot_scores),
        "min_robot_score": min(robot_scores) if robot_scores else 1.0,
        "max_p95_latency_ms": max(p95s) if p95s else 0.0,
        "avg_weighted_deviation": total_weighted / total_weighted_count if total_weighted_count else 0.0,
        "requests_per_s": float(sum(rt.requests_sent for rt in runtimes) / duration_s) if duration_s > 0 else 0.0,
        "robot_count": len(specs),
        "reply_over_chunk_actions_count": int(total_reply_over),
        "non_bootstrap_replies_count": int(total_non_bootstrap),
        "reply_over_chunk_actions_ratio": float(total_reply_over / total_non_bootstrap) if total_non_bootstrap else 0.0,
        "miss_autohorizon_count": int(total_miss),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(total_miss / total_chunks) if total_chunks else 0.0,
        "consumed_hist": {str(k): int(v) for k, v in sorted(consumed_hist.items())},
        "horizon_hist": {str(k): int(v) for k, v in sorted(horizon_hist.items())},
        "per_horizon_alignment": {
            str(k): {"early": int(v["early"]), "exact": int(v["exact"]), "late": int(v["late"])}
            for k, v in sorted(per_horizon_alignment.items())
        },
    }


def _aggregate(runs: list[dict]) -> dict:
    total_reply_over = sum(r["reply_over_chunk_actions_count"] for r in runs)
    total_non_bootstrap = sum(r["non_bootstrap_replies_count"] for r in runs)
    total_miss = sum(r["miss_autohorizon_count"] for r in runs)
    total_chunks = sum(r["chunk_count"] for r in runs)
    consumed_hist: dict[int, int] = {}
    horizon_hist: dict[int, int] = {}
    per_horizon_alignment: dict[int, dict[str, int]] = {}
    for r in runs:
        for key, value in r.get("consumed_hist", {}).items():
            ikey = int(key)
            consumed_hist[ikey] = consumed_hist.get(ikey, 0) + int(value)
        for key, value in r.get("horizon_hist", {}).items():
            ikey = int(key)
            horizon_hist[ikey] = horizon_hist.get(ikey, 0) + int(value)
        for key, value in r.get("per_horizon_alignment", {}).items():
            ikey = int(key)
            bucket = per_horizon_alignment.setdefault(ikey, {"early": 0, "exact": 0, "late": 0})
            bucket["early"] += int(value["early"])
            bucket["exact"] += int(value["exact"])
            bucket["late"] += int(value["late"])
    return {
        "fleet_score": float(np.mean([r["fleet_score"] for r in runs])),
        "min_robot_score": float(np.mean([r["min_robot_score"] for r in runs])),
        "max_p95_latency_ms": float(np.mean([r["max_p95_latency_ms"] for r in runs])),
        "avg_weighted_deviation": float(np.mean([r["avg_weighted_deviation"] for r in runs])),
        "requests_per_s": float(np.mean([r["requests_per_s"] for r in runs])),
        "robot_count": int(runs[0]["robot_count"]) if runs else 0,
        "reply_over_chunk_actions_count": int(total_reply_over),
        "non_bootstrap_replies_count": int(total_non_bootstrap),
        "reply_over_chunk_actions_ratio": float(total_reply_over / total_non_bootstrap) if total_non_bootstrap else 0.0,
        "miss_autohorizon_count": int(total_miss),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(total_miss / total_chunks) if total_chunks else 0.0,
        "consumed_hist": {str(k): int(v) for k, v in sorted(consumed_hist.items())},
        "horizon_hist": {str(k): int(v) for k, v in sorted(horizon_hist.items())},
        "per_horizon_alignment": {
            str(k): {"early": int(v["early"]), "exact": int(v["exact"]), "late": int(v["late"])}
            for k, v in sorted(per_horizon_alignment.items())
        },
    }


def predictive_eval(specs, cfg, horizon, metric):
    runs = [simulate(specs, cfg, horizon, metric, 1000 + i, cfg.predict_duration_s) for i in range(cfg.predict_seeds)]
    return _aggregate(runs)


def long_truth(specs, cfg, horizon, metric):
    runs = [simulate(specs, cfg, horizon, metric, 2000 + i, cfg.truth_duration_s) for i in range(cfg.truth_seeds)]
    return _aggregate(runs)
