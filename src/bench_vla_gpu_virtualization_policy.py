#!/usr/bin/env python3

from __future__ import annotations

import bisect
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/autodl-tmp/VLAServing")
RESULTS = ROOT / "results"
OUT = RESULTS / "vla_gpu_virtualization_policy_20260412.json"
GR00T_BATCH_CURVE = RESULTS / "groot_n1d6_same_model_batch_curve_step1_compile_libero.json"

PI05_AUTOH = RESULTS / "pi05_autohorizon_simulator_fit_20260329.json"
GR00T_AUTOH = RESULTS / "groot_n15_official_horizon_simulator_fit_20260328.json"

H2D_GIB_PER_S = 23.27228306738265
DECODE_APPLY_GIB_PER_S = 12.0
GPU_MEMORY_LIMIT_GIB = 24.0
SEEDS = [1201, 1202, 1203]
ADMISSION_GROUPS = 3
PREDICT_DURATION_S = 20.0
TRUTH_DURATION_S = 60.0
PHASE_BINS = 8


@dataclass(frozen=True)
class SuccessMetricParams:
    alpha: float
    beta: float
    robot_threshold: float
    fleet_threshold: float


@dataclass
class RequestSpec:
    robot_idx: int
    model: str
    hz: float
    period_ms: float
    chunk_size: int
    chunk_start_ms: float
    horizon: int


@dataclass
class Reservation:
    start_ms: float
    finish_ms: float
    robot_indices: tuple[int, ...]
    model: str
    horizons: tuple[int, ...]
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
    proc: "HorizonProcess"
    period_ms: float
    chunk_scores: list[float] = field(default_factory=list)
    weighted_deviations: list[float] = field(default_factory=list)
    phase_shift_actions: list[int] = field(default_factory=list)
    miss_autohorizon_count: int = 0
    reply_over_chunk_actions: int = 0
    chunk_count: int = 0
    requests_sent: int = 0


class HorizonProcess:
    def __init__(
        self,
        start_states: np.ndarray,
        start_probs: np.ndarray,
        transition: dict[int, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ):
        self.start_states = start_states
        self.start_probs = start_probs
        self.transition = transition
        self.rng = rng
        self.current: int | None = None

    def next(self) -> int:
        if self.current is None:
            self.current = int(self.rng.choice(self.start_states, p=self.start_probs))
            return self.current
        states, probs = self.transition[self.current]
        self.current = int(self.rng.choice(states, p=probs))
        return self.current


def load_pi05_horizon() -> dict[str, Any]:
    with PI05_AUTOH.open() as f:
        d = json.load(f)
    start_states = np.array([int(x) for x in d["start_probs"].keys()], dtype=np.int64)
    start_probs = np.array([float(d["start_probs"][str(x)]) for x in start_states], dtype=np.float64)
    transition = {}
    for src, row in d["transition_probs"].items():
        states = np.array([int(x) for x in row.keys()], dtype=np.int64)
        probs = np.array([float(row[str(x)]) for x in states], dtype=np.float64)
        transition[int(src)] = (states, probs)
    return {
        "start_states": start_states,
        "start_probs": start_probs,
        "transition": transition,
        "mean_horizon": float(d["mean_horizon"]),
    }


def load_gr00t_horizon() -> dict[str, Any]:
    with GR00T_AUTOH.open() as f:
        d = json.load(f)
    start_states = np.array([int(x) for x in d["start_probs"].keys()], dtype=np.int64)
    start_probs = np.array([float(d["start_probs"][str(x)]) for x in start_states], dtype=np.float64)
    transition = {}
    for src, row in d["transition_matrix"].items():
        total = float(sum(row.values()))
        states = np.array([int(x) for x in row.keys()], dtype=np.int64)
        probs = np.array([float(row[str(x)]) / total for x in states], dtype=np.float64)
        transition[int(src)] = (states, probs)
    return {
        "start_states": start_states,
        "start_probs": start_probs,
        "transition": transition,
        "mean_horizon": float(d["mean_horizon"]),
    }


PI05_H = load_pi05_horizon()
GR00T_H = load_gr00t_horizon()


def load_gr00t_batch_service_ms() -> dict[int, float]:
    with GR00T_BATCH_CURVE.open() as f:
        payload = json.load(f)
    out: dict[int, float] = {}
    for row in payload["results"]:
        out[int(row["batch_size"])] = float(row["service_ms_for_scheduler"])
    return out


GR00T_BATCH_SERVICE_MS = load_gr00t_batch_service_ms()


WORKLOADS: dict[str, dict[str, Any]] = {
    "pi05": {
        "base_robots": [
            {"model": "30hz_official_ft", "hz": 30.0, "start_ms": 0.0},
            {"model": "20hz_quantiles", "hz": 20.0, "start_ms": 50.0},
            {"model": "10hz_a_logits", "hz": 10.0, "start_ms": 100.0},
            {"model": "10hz_b_autoh", "hz": 10.0, "start_ms": 600.0},
        ],
        "candidate_types": [
            {"model": "30hz_official_ft", "hz": 30.0},
            {"model": "20hz_quantiles", "hz": 20.0},
            {"model": "10hz_a_logits", "hz": 10.0},
            {"model": "10hz_b_autoh", "hz": 10.0},
        ],
        "chunk_size": 50,
        "horizon": PI05_H,
        "metric": SuccessMetricParams(alpha=0.018, beta=1.15, robot_threshold=0.97, fleet_threshold=0.985),
        "phase_shift_floor_actions": 20,
        "infer_ms": {
            "30hz_official_ft": 43.198463439941406,
            "20hz_quantiles": 43.18052673339844,
            "10hz_a_logits": 43.21331214904785,
            "10hz_b_autoh": 43.06164741516113,
        },
        "shell_gib": {
            "30hz_official_ft": 7.485,
            "20hz_quantiles": 7.485,
            "10hz_a_logits": 7.485,
            "10hz_b_autoh": 7.485,
        },
        "state_gib": {
            "10hz_a_logits": 4.010575335472822,
            "10hz_b_autoh": 3.6609760150313377,
        },
    },
    "gr00t_n1d6": {
        "base_robots": [
            {"model": "30hz_bridge", "hz": 30.0, "start_ms": 0.0},
            {"model": "20hz_fractal", "hz": 20.0, "start_ms": 40.0},
            {"model": "10hz_libero", "hz": 10.0, "start_ms": 120.0},
            {"model": "10hz_rel30k", "hz": 10.0, "start_ms": 620.0},
        ],
        "candidate_types": [
            {"model": "30hz_bridge", "hz": 30.0},
            {"model": "20hz_fractal", "hz": 20.0},
            {"model": "10hz_libero", "hz": 10.0},
            {"model": "10hz_rel30k", "hz": 10.0},
        ],
        "chunk_size": 16,
        "horizon": GR00T_H,
        "metric": SuccessMetricParams(alpha=0.035, beta=1.35, robot_threshold=0.88, fleet_threshold=0.90),
        "phase_shift_floor_actions": 6,
        "infer_ms": {
            "30hz_bridge": 43.8,
            "20hz_fractal": 43.8,
            "10hz_libero": 43.88061095960438,
            "10hz_rel30k": 43.88061095960438,
        },
        "shell_gib": {
            "30hz_bridge": 6573377712 / (1024**3),
            "20hz_fractal": 6573377712 / (1024**3),
            "10hz_libero": 9192043768 / (1024**3),
            "10hz_rel30k": 9192043768 / (1024**3),
        },
        "state_gib": {
            "10hz_libero": 9192043768 / (1024**3),
            "10hz_rel30k": 9192043768 / (1024**3),
        },
    },
}


def request_period_ms(hz: float) -> float:
    return 1000.0 / hz


def phase_grid(hz: float) -> list[float]:
    period = request_period_ms(hz)
    return [i * period / PHASE_BINS for i in range(PHASE_BINS)]


def phase_candidates(
    workload: str,
    hz: float,
    model: str,
    robots: list[dict[str, float | str]],
    phase_mode: str,
) -> list[float]:
    phases = set(float(x) for x in phase_grid(hz))
    if phase_mode == "batch_align" and workload == "gr00t_n1d6":
        period = request_period_ms(hz)
        for robot in robots:
            if str(robot["model"]) != model:
                continue
            if abs(float(robot["hz"]) - hz) > 1e-9:
                continue
            phases.add(float(robot["start_ms"]) % period)
    return sorted(phases)


def geometric_mean(vals: list[float]) -> float:
    if not vals:
        return 1.0
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / len(vals)))


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "num_samples": int(arr.size),
    }


def chunk_success(actual_consumed: int, horizon: int, chunk_size: int, metric: SuccessMetricParams) -> tuple[float, float]:
    deviation = abs(actual_consumed - horizon)
    weighted = deviation * ((chunk_size / max(horizon, 1)) ** metric.beta)
    return float(math.exp(-metric.alpha * weighted)), float(weighted)


def shell_of(model: str) -> str:
    if model.startswith("30hz_"):
        return "A"
    if model.startswith("20hz_"):
        return "B"
    return "C"


def _insert_res(res_list: list[Reservation], res: Reservation):
    starts = [r.start_ms for r in res_list]
    idx = bisect.bisect_left(starts, res.start_ms)
    res_list.insert(idx, res)


def _insert_iv(iv_list: list[tuple[float, float]], start: float, finish: float):
    starts = [s for s, _ in iv_list]
    idx = bisect.bisect_left(starts, start)
    iv_list.insert(idx, (start, finish))


def resident_fraction(cfg: dict[str, float], model: str) -> float:
    if model.startswith("10hz_a_") or model.endswith("_libero"):
        return float(cfg["r10a"])
    if model.startswith("10hz_b_") or model.endswith("_rel30k"):
        return float(cfg["r10b"])
    return 1.0


def gpu_memory_estimate_gb(workload: str, cfg: dict[str, float]) -> dict[str, float | bool]:
    w = WORKLOADS[workload]
    dedicated_mem = w["shell_gib"][w["base_robots"][0]["model"]] + w["shell_gib"][w["base_robots"][1]["model"]]
    shared_shell_mem = max(w["shell_gib"][w["base_robots"][2]["model"]], w["shell_gib"][w["base_robots"][3]["model"]])
    extra = 0.0
    for model, gib in w["state_gib"].items():
        extra += gib * resident_fraction(cfg, model)
    total = dedicated_mem + shared_shell_mem + extra
    return {
        "dedicated_shells_gb": dedicated_mem,
        "shared_shell_gb": shared_shell_mem,
        "resident_state_gb": extra,
        "total_estimated_gb": total,
        "fits_under_24gb": total < GPU_MEMORY_LIMIT_GIB,
    }


def prefetch_ms(workload: str, cfg: dict[str, float], model: str) -> float:
    state_gib = WORKLOADS[workload]["state_gib"].get(model, 0.0)
    missing_gib = state_gib * (1.0 - resident_fraction(cfg, model))
    if missing_gib <= 0.0:
        return 0.0
    h2d_ms = missing_gib / H2D_GIB_PER_S * 1000.0
    decode_ms = missing_gib / DECODE_APPLY_GIB_PER_S * 1000.0
    return h2d_ms + decode_ms


def infer_service_ms(workload: str, model: str, batch_size: int) -> float:
    if workload != "gr00t_n1d6":
        return float(WORKLOADS[workload]["infer_ms"][model])
    batch = max(1, min(int(batch_size), max(GR00T_BATCH_SERVICE_MS)))
    return float(GR00T_BATCH_SERVICE_MS[batch])


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


def consumed_lower_bound(
    workload: str,
    chunk_size: int,
    horizon: int,
    phase_floor_actions: int,
    phase_mode: str,
) -> int:
    if phase_mode == "strict_horizon":
        return horizon
    if phase_mode == "phase_shift":
        return max(1, min(horizon, phase_floor_actions))
    if phase_mode == "batch_align":
        if workload == "gr00t_n1d6" and horizon > phase_floor_actions:
            return 1
        return horizon
    raise ValueError(phase_mode)


def _find_slot(
    workload: str,
    shell_res: list[Reservation],
    copy_res: list[tuple[float, float]],
    chunk_start_ms: float,
    period_ms: float,
    horizons: list[int],
    model: str,
    cfg: dict[str, float],
    phase_mode: str,
) -> tuple[float, float, int, float | None, float | None] | None:
    w = WORKLOADS[workload]
    chunk_size = int(w["chunk_size"])
    infer_ms = infer_service_ms(workload, model, len(horizons))
    hard_finish_ms = chunk_start_ms + chunk_size * period_ms
    prep = prefetch_ms(workload, cfg, model) if shell_of(model) == "C" else 0.0
    c_lo = max(
        consumed_lower_bound(workload, chunk_size, int(h), int(w["phase_shift_floor_actions"]), phase_mode)
        for h in horizons
    )
    candidates = list(range(c_lo, chunk_size + 1))
    candidates.sort(key=lambda c: (sum(abs(c - int(h)) for h in horizons), c))
    best = None
    for c in candidates:
        lower = chunk_start_ms + (c - 1) * period_ms
        upper = min(chunk_start_ms + c * period_ms, hard_finish_ms)
        median_h = int(np.median(np.asarray(horizons, dtype=np.int64)))
        target_finish = min(max(chunk_start_ms + median_h * period_ms, lower + 1e-6), upper)
        for gap_start, gap_end, prev_res in _candidate_gaps(shell_res, chunk_start_ms, hard_finish_ms):
            feasible_finish_lo = max(gap_start + infer_ms, lower + 1e-6)
            feasible_finish_hi = min(gap_end, upper)
            if feasible_finish_lo > feasible_finish_hi + 1e-9:
                continue
            finish = min(max(target_finish, feasible_finish_lo), feasible_finish_hi)
            start = finish - infer_ms
            copy_start = copy_end = None
            if shell_of(model) == "C":
                previous_model = prev_res.model if prev_res is not None else list(w["state_gib"].keys())[0]
                effective_prep = 0.0 if previous_model == model else prep
                slot = _candidate_copy_slot(copy_res, chunk_start_ms, start, effective_prep)
                if slot is None:
                    continue
                copy_start, copy_end = slot
            rank = (
                sum(abs(c - int(h)) for h in horizons),
                max(0.0, finish - (chunk_start_ms + median_h * period_ms)),
                finish,
            )
            if best is None or rank < best[0]:
                best = (rank, start, finish, c, copy_start, copy_end)
        if best is not None and best[0][0] == 0:
            break
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5]


def simulate(
    workload: str,
    specs: list[dict[str, float | str]],
    cfg: dict[str, float],
    duration_s: float,
    seed: int,
    phase_mode: str,
    horizon_floor: int | None = None,
):
    w = WORKLOADS[workload]
    metric: SuccessMetricParams = w["metric"]
    duration_ms = duration_s * 1000.0
    runtimes = [
        RobotRuntime(
            model=str(s["model"]),
            hz=float(s["hz"]),
            start_ms=float(s["start_ms"]),
            proc=HorizonProcess(
                w["horizon"]["start_states"],
                w["horizon"]["start_probs"],
                w["horizon"]["transition"],
                np.random.default_rng(seed * 1000 + i),
            ),
            period_ms=request_period_ms(float(s["hz"])),
        )
        for i, s in enumerate(specs)
    ]
    shell_res = {"A": [], "B": [], "C": []}
    copy_res: list[tuple[float, float]] = []
    reservation_queue: list[tuple[float, int, Reservation]] = []
    next_chunk_start: list[float | None] = [rt.start_ms for rt in runtimes]
    event_id = 0

    def schedule_chunk(robot_idx: int):
        nonlocal event_id
        chunk_start_ms = next_chunk_start[robot_idx]
        if chunk_start_ms is None:
            return
        if chunk_start_ms > duration_ms + 1e-9:
            return
        rt = runtimes[robot_idx]
        batch_members = [robot_idx]
        if workload == "gr00t_n1d6":
            for j, other in enumerate(runtimes):
                if j == robot_idx:
                    continue
                if next_chunk_start[j] is None:
                    continue
                if other.model != rt.model:
                    continue
                if abs(float(next_chunk_start[j]) - float(chunk_start_ms)) > 1e-6:
                    continue
                batch_members.append(j)
        batch_members = sorted(batch_members)
        horizons: list[int] = []
        for j in batch_members:
            h = runtimes[j].proc.next()
            if horizon_floor is not None:
                h = max(h, horizon_floor)
            horizons.append(int(h))
            runtimes[j].requests_sent += 1
            next_chunk_start[j] = None

        sh = shell_of(rt.model)
        slot = _find_slot(
            workload,
            shell_res[sh],
            copy_res,
            float(chunk_start_ms),
            rt.period_ms,
            horizons,
            rt.model,
            cfg,
            phase_mode,
        )
        if slot is None:
            for j in batch_members:
                runtimes[j].reply_over_chunk_actions += 1
            return
        start, finish, consumed, pstart, pend = slot
        res = Reservation(
            start_ms=start,
            finish_ms=finish,
            robot_indices=tuple(batch_members),
            model=rt.model,
            horizons=tuple(horizons),
            consumed=consumed,
            chunk_start_ms=float(chunk_start_ms),
            period_ms=rt.period_ms,
            prefetch_start_ms=pstart,
            prefetch_finish_ms=pend,
        )
        _insert_res(shell_res[sh], res)
        if pstart is not None and pend is not None and pend > pstart:
            _insert_iv(copy_res, pstart, pend)
        heapq.heappush(reservation_queue, (finish, event_id, res))
        event_id += 1

    for i, _ in enumerate(runtimes):
        schedule_chunk(i)

    samples = []
    while reservation_queue:
        _, _, res = heapq.heappop(reservation_queue)
        for robot_idx, horizon in zip(res.robot_indices, res.horizons):
            rt = runtimes[robot_idx]
            rt.chunk_count += 1
            if res.finish_ms > res.chunk_start_ms + w["chunk_size"] * rt.period_ms + 1e-9:
                rt.reply_over_chunk_actions += 1
                continue
            if res.consumed != horizon:
                rt.miss_autohorizon_count += 1
            score, weighted = chunk_success(res.consumed, horizon, int(w["chunk_size"]), metric)
            rt.chunk_scores.append(score)
            rt.weighted_deviations.append(weighted)
            rt.phase_shift_actions.append(res.consumed - horizon)
            samples.append(
                {
                    "robot_id": robot_idx,
                    "batch_size": len(res.robot_indices),
                    "model": res.model,
                    "hz": rt.hz,
                    "chunk_start_ms": res.chunk_start_ms,
                    "horizon": horizon,
                    "consumed": res.consumed,
                    "phase_shift_actions": res.consumed - horizon,
                    "shell": shell_of(res.model),
                    "prefetch_wait_ms": 0.0 if res.prefetch_finish_ms is None else max(0.0, res.prefetch_finish_ms - res.chunk_start_ms),
                    "queue_wait_ms": max(0.0, res.start_ms - max(res.chunk_start_ms, res.prefetch_finish_ms or res.chunk_start_ms)),
                    "chunk_elapsed_ms": res.finish_ms - res.chunk_start_ms,
                    "request_to_result_ms": res.finish_ms - res.start_ms,
                    "hard_deadline_ms": w["chunk_size"] * rt.period_ms,
                    "score": score,
                }
            )
            if res.finish_ms <= duration_ms + 1e-9:
                next_chunk_start[robot_idx] = res.finish_ms
        for robot_idx in res.robot_indices:
            schedule_chunk(robot_idx)

    robot_scores = [geometric_mean(rt.chunk_scores) for rt in runtimes]
    total_miss = sum(rt.miss_autohorizon_count for rt in runtimes)
    total_chunks = sum(rt.chunk_count for rt in runtimes)
    total_reply_over = sum(rt.reply_over_chunk_actions for rt in runtimes)
    all_chunk_elapsed = [s["chunk_elapsed_ms"] for s in samples]
    all_r2r = [s["request_to_result_ms"] for s in samples]
    phase_shifts = [s["phase_shift_actions"] for s in samples]
    return {
        "chunk_elapsed_ms": stats(all_chunk_elapsed),
        "request_to_result_ms": stats(all_r2r),
        "prefetch_wait_ms": stats([s["prefetch_wait_ms"] for s in samples]),
        "queue_wait_ms": stats([s["queue_wait_ms"] for s in samples]),
        "phase_shift_actions": {
            "mean": float(np.mean(np.asarray(phase_shifts, dtype=np.float64))) if phase_shifts else 0.0,
            "mean_abs": float(np.mean(np.abs(np.asarray(phase_shifts, dtype=np.float64)))) if phase_shifts else 0.0,
            "p95_abs": float(np.percentile(np.abs(np.asarray(phase_shifts, dtype=np.float64)), 95)) if phase_shifts else 0.0,
        },
        "hard_miss_count": 0,
        "reply_over_chunk_actions_count": int(total_reply_over),
        "miss_autohorizon_count": int(total_miss),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(total_miss / total_chunks) if total_chunks else 0.0,
        "fleet_score": geometric_mean(robot_scores),
        "min_robot_score": min(robot_scores) if robot_scores else 1.0,
        "samples_head": samples[:16],
    }


def aggregate(
    workload: str,
    specs: list[dict[str, float | str]],
    cfg: dict[str, float],
    duration_s: float,
    seeds: list[int],
    phase_mode: str,
    horizon_floor: int | None = None,
):
    outs = [simulate(workload, specs, cfg, duration_s, s, phase_mode, horizon_floor=horizon_floor) for s in seeds]
    return {
        "hard_miss_count": int(sum(o["hard_miss_count"] for o in outs)),
        "reply_over_chunk_actions_count": int(sum(o["reply_over_chunk_actions_count"] for o in outs)),
        "mean_request_to_result_p95_ms": float(np.mean([o["request_to_result_ms"]["p95_ms"] for o in outs])),
        "mean_chunk_elapsed_p95_ms": float(np.mean([o["chunk_elapsed_ms"]["p95_ms"] for o in outs])),
        "mean_fleet_score": float(np.mean([o["fleet_score"] for o in outs])),
        "mean_min_robot_score": float(np.mean([o["min_robot_score"] for o in outs])),
        "mean_miss_autohorizon_ratio": float(np.mean([o["miss_autohorizon_ratio"] for o in outs])),
        "mean_phase_shift_abs_actions": float(np.mean([o["phase_shift_actions"]["mean_abs"] for o in outs])),
        "truth_runs": outs,
    }


def config_grid(workload: str):
    if workload == "pi05":
        vals = (0.0, 0.1, 0.2, 0.3)
    else:
        vals = (0.0, 0.1, 0.2)
    for r10a in vals:
        for r10b in vals:
            cfg = {"r10a": r10a, "r10b": r10b}
            mem = gpu_memory_estimate_gb(workload, cfg)
            if mem["fits_under_24gb"]:
                yield cfg, mem


def search_fixed4(workload: str, phase_mode: str, horizon_floor: int | None = None):
    runs = []
    base = [dict(r) for r in WORKLOADS[workload]["base_robots"]]
    for cfg, mem in config_grid(workload):
        truth = aggregate(workload, base, cfg, TRUTH_DURATION_S, SEEDS, phase_mode, horizon_floor=horizon_floor)
        runs.append({"config": cfg, "gpu_memory": mem, "metrics": truth})
    runs.sort(
        key=lambda x: (
            x["metrics"]["reply_over_chunk_actions_count"],
            -x["metrics"]["mean_fleet_score"],
            -x["metrics"]["mean_min_robot_score"],
            x["metrics"]["mean_miss_autohorizon_ratio"],
            x["metrics"]["mean_request_to_result_p95_ms"],
            x["gpu_memory"]["total_estimated_gb"],
        )
    )
    return runs[0], runs[:5]


def run_group(workload: str, seed: int, cfg: dict[str, float], phase_mode: str, horizon_floor: int | None = None):
    rng = np.random.default_rng(seed)
    robots = [dict(r) for r in WORKLOADS[workload]["base_robots"]]
    metric: SuccessMetricParams = WORKLOADS[workload]["metric"]
    admission_log = []
    rejected = 0

    for i in range(40):
        choice = WORKLOADS[workload]["candidate_types"][int(rng.integers(0, len(WORKLOADS[workload]["candidate_types"])))]
        model = str(choice["model"])
        hz = float(choice["hz"])
        best = None
        for phase in phase_candidates(workload, hz, model, robots, phase_mode):
            candidate = {"model": model, "hz": hz, "start_ms": float(phase)}
            trial = robots + [candidate]
            pred = aggregate(workload, trial, cfg, PREDICT_DURATION_S, [1], phase_mode, horizon_floor=horizon_floor)
            ok = (
                pred["hard_miss_count"] == 0
                and pred["reply_over_chunk_actions_count"] == 0
                and pred["mean_min_robot_score"] >= metric.robot_threshold
                and pred["mean_fleet_score"] >= metric.fleet_threshold
                and pred["mean_request_to_result_p95_ms"] <= 100.0
            )
            if not ok:
                continue
            rank = (
                -pred["mean_fleet_score"],
                -pred["mean_min_robot_score"],
                pred["mean_miss_autohorizon_ratio"],
                pred["mean_request_to_result_p95_ms"],
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

    truth = aggregate(workload, robots, cfg, TRUTH_DURATION_S, [101, 102], phase_mode, horizon_floor=horizon_floor)
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


def search_admission(workload: str, best_cfg: dict[str, float], phase_mode: str, horizon_floor: int | None = None):
    groups = [run_group(workload, 2026041200 + i, best_cfg, phase_mode, horizon_floor=horizon_floor) for i in range(ADMISSION_GROUPS)]
    total = sum(g["admitted_total"] for g in groups)
    hard = sum(g["final_metrics"]["hard_miss_count"] for g in groups)
    hist = {}
    for g in groups:
        for k, v in g["admitted_histogram"].items():
            hist[k] = hist.get(k, 0) + v
    return {
        "config": {**best_cfg, "gpu_memory_estimate_gb": gpu_memory_estimate_gb(workload, best_cfg)},
        "summary": {
            "mean_admitted_total": total / len(groups),
            "total_admitted_robots": total,
            "hard_miss_count": hard,
            "mean_fleet_score": float(np.mean([g["final_metrics"]["mean_fleet_score"] for g in groups])),
            "mean_min_robot_score": float(np.mean([g["final_metrics"]["mean_min_robot_score"] for g in groups])),
            "mean_request_to_result_p95_ms": float(np.mean([g["final_metrics"]["mean_request_to_result_p95_ms"] for g in groups])),
            "mean_chunk_elapsed_p95_ms": float(np.mean([g["final_metrics"]["mean_chunk_elapsed_p95_ms"] for g in groups])),
            "mean_miss_autohorizon_ratio": float(np.mean([g["final_metrics"]["mean_miss_autohorizon_ratio"] for g in groups])),
            "mean_phase_shift_abs_actions": float(np.mean([g["final_metrics"]["mean_phase_shift_abs_actions"] for g in groups])),
            "admitted_histogram": hist,
        },
        "groups_detail": groups,
    }


def evaluate_workload(workload: str, horizon_floor: int | None = None):
    strict_best, strict_top5 = search_fixed4(workload, "strict_horizon", horizon_floor=horizon_floor)
    phase_best, phase_top5 = search_fixed4(workload, "phase_shift", horizon_floor=horizon_floor)
    batch_best, batch_top5 = search_fixed4(workload, "batch_align", horizon_floor=horizon_floor)
    strict_adm = search_admission(workload, strict_best["config"], "strict_horizon", horizon_floor=horizon_floor)
    phase_adm = search_admission(workload, phase_best["config"], "phase_shift", horizon_floor=horizon_floor)
    batch_adm = search_admission(workload, batch_best["config"], "batch_align", horizon_floor=horizon_floor)
    return {
        "setup": {
            "base_robots": WORKLOADS[workload]["base_robots"],
            "candidate_types": WORKLOADS[workload]["candidate_types"],
            "chunk_size": WORKLOADS[workload]["chunk_size"],
            "phase_shift_floor_actions": WORKLOADS[workload]["phase_shift_floor_actions"],
            "horizon_floor": horizon_floor,
            "design": "frequency-aware GPU virtualization with two dedicated shells, one shared shell for low-frequency models, predictive prefetch, action-consumption-aware phase shifting, and GR00T batch-aware same-model phase alignment",
        },
        "strict_horizon": {
            "fixed4_best": strict_best,
            "fixed4_top5": strict_top5,
            "admission": strict_adm,
        },
        "phase_shift": {
            "fixed4_best": phase_best,
            "fixed4_top5": phase_top5,
            "admission": phase_adm,
        },
        "batch_align": {
            "fixed4_best": batch_best,
            "fixed4_top5": batch_top5,
            "admission": batch_adm,
        },
        "improvement": {
            "fixed4_delta_fleet_score": phase_best["metrics"]["mean_fleet_score"] - strict_best["metrics"]["mean_fleet_score"],
            "fixed4_delta_min_robot_score": phase_best["metrics"]["mean_min_robot_score"] - strict_best["metrics"]["mean_min_robot_score"],
            "fixed4_delta_miss_autohorizon_ratio": phase_best["metrics"]["mean_miss_autohorizon_ratio"] - strict_best["metrics"]["mean_miss_autohorizon_ratio"],
            "fixed4_delta_request_to_result_p95_ms": phase_best["metrics"]["mean_request_to_result_p95_ms"] - strict_best["metrics"]["mean_request_to_result_p95_ms"],
            "admission_delta_mean_admitted_total": phase_adm["summary"]["mean_admitted_total"] - strict_adm["summary"]["mean_admitted_total"],
            "admission_delta_fleet_score": phase_adm["summary"]["mean_fleet_score"] - strict_adm["summary"]["mean_fleet_score"],
            "batch_align_delta_mean_admitted_total_vs_strict": batch_adm["summary"]["mean_admitted_total"] - strict_adm["summary"]["mean_admitted_total"],
            "batch_align_delta_mean_admitted_total_vs_phase_shift": batch_adm["summary"]["mean_admitted_total"] - phase_adm["summary"]["mean_admitted_total"],
        },
    }


def main():
    result = {
        "meta": {
            "seeds": SEEDS,
            "predict_duration_s": PREDICT_DURATION_S,
            "truth_duration_s": TRUTH_DURATION_S,
            "phase_modes": ["strict_horizon", "phase_shift", "batch_align"],
            "notes": [
                "strict_horizon means each chunk must be served at the sampled AutoHorizon point.",
                "phase_shift allows request service to slide within a safe action-consumption window [min(horizon, floor), chunk_size].",
                "batch_align allows GR00T requests with horizon > floor to trigger after any consumed action and aligns new same-model robots to existing same-model phases to create batch opportunities.",
                "The policy keeps 30Hz and 20Hz models on dedicated shells and shares one shell between two 10Hz models.",
            ],
        },
        "pi05": evaluate_workload("pi05"),
        "gr00t_n1d6": evaluate_workload("gr00t_n1d6"),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
