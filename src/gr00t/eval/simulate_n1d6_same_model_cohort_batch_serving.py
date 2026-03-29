#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np


MODEL_NAME = "gr00t_n1d6-libero"
PRIORITY_GAMMA = 1.35
BATCH_CURVE_PATH = Path(
    "/root/autodl-tmp/VLAServing/results/groot_n1d6_same_model_batch_curve_step1_compile_libero.json"
)


@dataclass(frozen=True)
class AutoHorizonParams:
    states: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11, 12)
    hmin: int = 4
    hmax: int = 12
    expected_horizon: float = 5.151944655361512
    conservative_horizon: float = 5.0


@dataclass(frozen=True)
class SuccessMetricParams:
    alpha: float = 0.035
    beta: float = 1.35
    robot_threshold: float = 0.97
    fleet_threshold: float = 0.985


@dataclass(frozen=True)
class ResourcePartition:
    conservative_horizon: float = 5.0
    global_threshold: float = 1.0


@dataclass
class Config:
    predict_duration_s: float = 20.0
    predict_seeds: int = 2
    truth_duration_s: float = 90.0
    truth_seeds: int = 4
    chunk_size: int = 16
    max_batch: int = 8
    slot_start_phase_ms: float = 0.0
    low_hz_max: float = 15.0
    high_hz_min: float = 25.0
    low_phase_bins: int = 4
    mid_phase_bins: int = 3
    high_phase_bins: int = 2
    low_slot_period_ms: float = 200.0
    mid_slot_period_ms: float = 160.0
    high_slot_period_ms: float = 120.0


@dataclass(frozen=True)
class RobotSpec:
    hz: float
    start_ms: float
    starts_ready: bool
    name: str


@dataclass
class ScheduledJob:
    robot_idx: int
    chunk_id: int
    chunk_start_ms: float
    period_ms: float
    horizon: int
    next_horizon: int


@dataclass
class Slot:
    start_ms: float
    finish_ms: float
    jobs: list[ScheduledJob] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.jobs)


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


def _load_batch_curve(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    return {int(row["batch_size"]): float(row["service_ms_for_scheduler"]) for row in payload["results"]}


BATCH_SERVICE_MS = _load_batch_curve(BATCH_CURVE_PATH)
MAX_BATCH = max(BATCH_SERVICE_MS)
PEAK_REQUEST_THROUGHPUT = max(batch / ms * 1000.0 for batch, ms in BATCH_SERVICE_MS.items())


class HorizonProcess:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.current: int | None = None
        self._start_states = np.array((5, 6, 9, 10, 11), dtype=np.int64)
        self._start_probs = np.array((0.966, 0.02, 0.002, 0.01, 0.002), dtype=np.float64)
        self._transition = {
            4: (np.array((4, 5, 6, 8, 9, 10, 11), dtype=np.int64), np.array((0.3445255474, 0.5270072993, 0.0043795620, 0.0014598540, 0.0029197080, 0.0218978102, 0.0978102190), dtype=np.float64)),
            5: (np.array((4, 5, 6, 7, 8, 9, 10, 11, 12), dtype=np.int64), np.array((0.0708333333, 0.8612179487, 0.0481837607, 0.0019230769, 0.0008547009, 0.0006410256, 0.0055555556, 0.0102564103, 0.0005341880), dtype=np.float64)),
            6: (np.array((4, 5, 6, 7, 8, 9, 10, 11), dtype=np.int64), np.array((0.0031982942, 0.5010660981, 0.4797441365, 0.0053304904, 0.0053304904, 0.0031982942, 0.0010660981, 0.0010660981), dtype=np.float64)),
            7: (np.array((5, 6, 7, 8, 10), dtype=np.int64), np.array((0.0566037736, 0.3207547170, 0.3207547170, 0.2641509434, 0.0377358491), dtype=np.float64)),
            8: (np.array((5, 6, 7, 8), dtype=np.int64), np.array((0.1111111111, 0.0694444444, 0.1944444444, 0.6250000000), dtype=np.float64)),
            9: (np.array((5, 6, 7), dtype=np.int64), np.array((0.5555555556, 0.3333333333, 0.1111111111), dtype=np.float64)),
            10: (np.array((4, 5, 6, 7, 8, 11), dtype=np.int64), np.array((0.0821917808, 0.8082191781, 0.0547945205, 0.0273972603, 0.0136986301, 0.0136986301), dtype=np.float64)),
            11: (np.array((4, 5, 6), dtype=np.int64), np.array((0.1761006289, 0.8176100629, 0.0062893082), dtype=np.float64)),
            12: (np.array((4, 5, 10), dtype=np.int64), np.array((0.5, 0.25, 0.25), dtype=np.float64)),
        }

    def next(self) -> int:
        if self.current is None:
            self.current = int(self.rng.choice(self._start_states, p=self._start_probs))
            return self.current
        states, probs = self._transition.get(self.current, self._transition[5])
        self.current = int(self.rng.choice(states, p=probs))
        return self.current


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
    req_rate = sum((spec.hz / partition.conservative_horizon) for spec in specs)
    return float(req_rate / PEAK_REQUEST_THROUGHPUT)


def coarse_accept(specs: list[RobotSpec], partition: ResourcePartition) -> bool:
    return coarse_load(specs, partition) <= partition.global_threshold


def _slot_service_ms(batch_size: int) -> float:
    return BATCH_SERVICE_MS[max(1, min(batch_size, MAX_BATCH))]


def _tier_for_hz(hz: float, cfg: Config) -> str:
    if hz >= cfg.high_hz_min:
        return "high"
    if hz <= cfg.low_hz_max:
        return "low"
    return "mid"


def _slot_period_for_hz(hz: float, cfg: Config) -> float:
    tier = _tier_for_hz(hz, cfg)
    if tier == "high":
        return cfg.high_slot_period_ms
    if tier == "low":
        return cfg.low_slot_period_ms
    return cfg.mid_slot_period_ms


def _consumed_at_finish(job: ScheduledJob, finish_ms: float, chunk_size: int) -> int:
    elapsed = max(0.0, finish_ms - job.chunk_start_ms)
    consumed = int(math.ceil(elapsed / job.period_ms - 1e-9))
    return min(max(consumed, 1), chunk_size)


def _job_rank(job: ScheduledJob, finish_ms: float, chunk_size: int) -> tuple[float, float]:
    consumed = _consumed_at_finish(job, finish_ms, chunk_size)
    weight = (chunk_size / max(job.horizon, 1)) ** PRIORITY_GAMMA
    return abs(consumed - job.horizon) * weight, abs(consumed - job.horizon)


def _slot_start_candidates(release_ms: float, hard_finish_ms: float, hz: float, cfg: Config) -> list[float]:
    slot_period_ms = _slot_period_for_hz(hz, cfg)
    earliest_start = release_ms
    latest_start = hard_finish_ms - _slot_service_ms(1)
    if latest_start < earliest_start - 1e-9:
        return []
    first_idx = math.ceil((earliest_start - cfg.slot_start_phase_ms) / slot_period_ms)
    last_idx = math.floor((latest_start - cfg.slot_start_phase_ms) / slot_period_ms)
    return [cfg.slot_start_phase_ms + i * slot_period_ms for i in range(first_idx, last_idx + 1)]


def _try_assign(slots_by_start: dict[float, Slot], start_ms: float, job: ScheduledJob, cfg: Config) -> tuple[tuple, Slot] | None:
    slot = slots_by_start.get(start_ms)
    if slot is None:
        slot = Slot(start_ms=start_ms, finish_ms=start_ms + _slot_service_ms(1), jobs=[])
    new_batch = slot.batch_size + 1
    if new_batch > cfg.max_batch:
        return None
    new_finish = start_ms + _slot_service_ms(new_batch)
    ordered_starts = sorted(slots_by_start.keys())
    prev_starts = [s for s in ordered_starts if s < start_ms]
    next_starts = [s for s in ordered_starts if s > start_ms]
    if prev_starts:
        prev_slot = slots_by_start[prev_starts[-1]]
        if prev_slot.finish_ms > start_ms + 1e-9:
            return None
    if next_starts:
        next_slot = slots_by_start[next_starts[0]]
        if new_finish > next_slot.start_ms + 1e-9:
            return None
    hard_finish_new = job.chunk_start_ms + cfg.chunk_size * job.period_ms
    if new_finish > hard_finish_new + 1e-9:
        return None
    old_cost = 0.0
    new_cost = 0.0
    for existing in slot.jobs:
        old_cost += _job_rank(existing, slot.finish_ms, cfg.chunk_size)[0]
        hard_finish_existing = existing.chunk_start_ms + cfg.chunk_size * existing.period_ms
        if new_finish > hard_finish_existing + 1e-9:
            return None
        new_cost += _job_rank(existing, new_finish, cfg.chunk_size)[0]
    new_cost += _job_rank(job, new_finish, cfg.chunk_size)[0]
    new_slot = Slot(start_ms=start_ms, finish_ms=new_finish, jobs=list(slot.jobs) + [job])
    consumed = _consumed_at_finish(job, new_finish, cfg.chunk_size)
    rank = (
        new_cost - old_cost,
        _job_rank(job, new_finish, cfg.chunk_size)[0],
        abs(consumed - job.horizon),
        -new_batch,
        new_finish,
    )
    return rank, new_slot


def _find_best_assignment(slots_by_start: dict[float, Slot], job: ScheduledJob, cfg: Config) -> tuple[float, Slot] | None:
    release_ms = job.chunk_start_ms
    hard_finish_ms = job.chunk_start_ms + cfg.chunk_size * job.period_ms
    starts = _slot_start_candidates(release_ms, hard_finish_ms, 1000.0 / job.period_ms, cfg)
    best = None
    for start_ms in starts:
        candidate = _try_assign(slots_by_start, start_ms, job, cfg)
        if candidate is None:
            continue
        rank, slot = candidate
        if best is None or rank < best[0]:
            best = (rank, slot)
    if best is None:
        return None
    return best[1].start_ms, best[1]


def simulate(specs: list[RobotSpec], cfg: Config, horizon_params: AutoHorizonParams, metric: SuccessMetricParams, seed: int, duration_s: float) -> dict:
    duration_ms = duration_s * 1000.0
    processes = [HorizonProcess(np.random.default_rng(seed * 1000 + i)) for i in range(len(specs))]
    runtimes = [RobotRuntime(spec=s, period_ms=1000.0 / s.hz) for s in specs]
    slots_by_start: dict[float, Slot] = {}
    chunk_ids = [0 for _ in specs]

    def schedule_chunk(robot_idx: int, chunk_start_ms: float) -> bool:
        if chunk_start_ms > duration_ms + 1e-9:
            return False
        h = processes[robot_idx].next()
        nh = processes[robot_idx].next()
        period_ms = runtimes[robot_idx].period_ms
        job = ScheduledJob(
            robot_idx=robot_idx,
            chunk_id=chunk_ids[robot_idx],
            chunk_start_ms=chunk_start_ms,
            period_ms=period_ms,
            horizon=h,
            next_horizon=nh,
        )
        assignment = _find_best_assignment(slots_by_start, job, cfg)
        if assignment is None:
            runtimes[robot_idx].reply_over_chunk_actions += 1
            return False
        start_ms, new_slot = assignment
        slots_by_start[start_ms] = new_slot
        chunk_ids[robot_idx] += 1
        runtimes[robot_idx].requests_sent += 1
        return True

    for i, spec in enumerate(specs):
        schedule_chunk(i, spec.start_ms)

    slot_batch_hist: dict[int, int] = {}
    ordered_starts = sorted(slots_by_start.keys())
    while ordered_starts:
        start_ms = ordered_starts.pop(0)
        slot = slots_by_start.pop(start_ms)
        slot_batch_hist[slot.batch_size] = slot_batch_hist.get(slot.batch_size, 0) + 1
        latency_ms = slot.finish_ms - slot.start_ms
        for job in slot.jobs:
            rt = runtimes[job.robot_idx]
            rt.non_bootstrap_replies += 1
            rt.latencies_ms.append(latency_ms)
            if slot.finish_ms > job.chunk_start_ms + cfg.chunk_size * job.period_ms + 1e-9:
                rt.reply_over_chunk_actions += 1
                continue
            actual_consumed = _consumed_at_finish(job, slot.finish_ms, cfg.chunk_size)
            switch_time_ms = slot.finish_ms
            rt.chunk_count += 1
            if actual_consumed != job.horizon:
                rt.miss_autohorizon_count += 1
            score, weighted_dev = chunk_success(actual_consumed, job.horizon, horizon_params, metric)
            rt.chunk_scores.append(score)
            rt.weighted_deviations.append(weighted_dev)
            rt.consumed_hist[actual_consumed] = rt.consumed_hist.get(actual_consumed, 0) + 1
            rt.horizon_hist[job.horizon] = rt.horizon_hist.get(job.horizon, 0) + 1
            bucket = rt.per_horizon_alignment.setdefault(job.horizon, {"early": 0, "exact": 0, "late": 0})
            if actual_consumed < job.horizon:
                bucket["early"] += 1
            elif actual_consumed > job.horizon:
                bucket["late"] += 1
            else:
                bucket["exact"] += 1
            if switch_time_ms <= duration_ms + 1e-9:
                schedule_chunk(job.robot_idx, switch_time_ms)
        ordered_starts = sorted(slots_by_start.keys())

    robot_scores = [geometric_mean(rt.chunk_scores) for rt in runtimes]
    p95s = [float(np.percentile(rt.latencies_ms, 95)) for rt in runtimes if rt.latencies_ms]
    total_reply_over = sum(rt.reply_over_chunk_actions for rt in runtimes)
    total_non_bootstrap = sum(rt.non_bootstrap_replies for rt in runtimes)
    total_miss = sum(rt.miss_autohorizon_count for rt in runtimes)
    total_chunks = sum(rt.chunk_count for rt in runtimes)
    total_weighted = sum(sum(rt.weighted_deviations) for rt in runtimes)
    total_weighted_count = sum(len(rt.weighted_deviations) for rt in runtimes)
    total_slots = sum(slot_batch_hist.values())
    total_jobs_in_slots = sum(batch * count for batch, count in slot_batch_hist.items())
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
        "slot_batch_hist": {str(k): int(v) for k, v in sorted(slot_batch_hist.items())},
        "mean_slot_batch_size": float(total_jobs_in_slots / total_slots) if total_slots else 0.0,
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
    slot_batch_hist: dict[int, int] = {}
    consumed_hist: dict[int, int] = {}
    horizon_hist: dict[int, int] = {}
    per_horizon_alignment: dict[int, dict[str, int]] = {}
    for r in runs:
        for key, value in r.get("slot_batch_hist", {}).items():
            ikey = int(key)
            slot_batch_hist[ikey] = slot_batch_hist.get(ikey, 0) + int(value)
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
    total_slots = sum(slot_batch_hist.values())
    total_jobs_in_slots = sum(batch * count for batch, count in slot_batch_hist.items())
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
        "slot_batch_hist": {str(k): int(v) for k, v in sorted(slot_batch_hist.items())},
        "mean_slot_batch_size": float(total_jobs_in_slots / total_slots) if total_slots else 0.0,
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
