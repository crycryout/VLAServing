#!/usr/bin/env python3

from __future__ import annotations

import heapq
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/VLAServing")
SRC = ROOT / "src" / "bench_vla_gpu_virtualization_policy.py"
OUT = ROOT / "results" / "gr00t_shared_prefix_phase_lock_batch_mps_20260412.json"

SEEDS = [1201, 1202, 1203]
PREDICT_DURATION_S = 20.0
TRUTH_DURATION_S = 60.0
CHUNK_SIZE = 16
HORIZON_FLOOR = 8
MAX_BATCH = 8
PENALTY_SCALE = 8.0 / 25.0
PHASE_BINS = 8
SCENARIOS = [1, 2, 4, 6, 8]
TOPK_FINALISTS = 8

MODEL_INFO = [
    ("30hz_bridge", 30.0, 0.0),
    ("20hz_fractal", 20.0, 6.25),
    ("10hz_libero", 10.0, 25.0),
    ("10hz_rel30k", 10.0, 62.5),
]


@dataclass
class RobotRuntime:
    model: str
    hz: float
    start_ms: float
    proc: any
    period_ms: float
    chunk_scores: list[float] = field(default_factory=list)
    phase_shift_actions: list[int] = field(default_factory=list)
    miss_autohorizon_count: int = 0
    reply_over_chunk_actions: int = 0
    chunk_count: int = 0


@dataclass
class PendingRequest:
    robot_idx: int
    model: str
    hz: float
    period_ms: float
    chunk_start_ms: float
    horizon: int
    lower_c: int
    upper_c: int
    earliest_finish_ms: float
    latest_finish_ms: float


@dataclass
class BatchCandidate:
    model: str
    requests: list[PendingRequest]
    work_ms: float
    common_low_ms: float
    common_high_ms: float
    earliest_start_ms: float
    latest_start_ms: float


@dataclass
class RunningBatch:
    model: str
    requests: list[PendingRequest]
    work_ms: float
    remaining_work_ms: float
    common_low_ms: float
    common_high_ms: float
    start_ms: float


def load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_vla_gpu_virtualization_policy", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = load_bench_module()
GR00T_BATCH_SERVICE_MS = MOD.GR00T_BATCH_SERVICE_MS
GR00T_H = MOD.GR00T_H
BASE_METRIC = MOD.WORKLOADS["gr00t_n1d6"]["metric"]
METRIC = MOD.SuccessMetricParams(
    alpha=BASE_METRIC.alpha * PENALTY_SCALE,
    beta=BASE_METRIC.beta,
    robot_threshold=BASE_METRIC.robot_threshold,
    fleet_threshold=BASE_METRIC.fleet_threshold,
)


def request_period_ms(hz: float) -> float:
    return 1000.0 / hz


def phase_grid(hz: float) -> list[float]:
    period = request_period_ms(hz)
    return [i * period / PHASE_BINS for i in range(PHASE_BINS)]


def build_same_phase_specs(copies_per_model: int, p20: float, p10a: float, p10b: float) -> list[dict[str, float | str]]:
    phases = {
        "30hz_bridge": 0.0,
        "20hz_fractal": float(p20),
        "10hz_libero": float(p10a),
        "10hz_rel30k": float(p10b),
    }
    specs: list[dict[str, float | str]] = []
    for model, hz, _ in MODEL_INFO:
        for _copy in range(copies_per_model):
            specs.append({"model": model, "hz": hz, "start_ms": phases[model]})
    return specs


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


def infer_service_ms(batch_size: int) -> float:
    batch = max(1, min(int(batch_size), max(GR00T_BATCH_SERVICE_MS)))
    return float(GR00T_BATCH_SERVICE_MS[batch])


def fit_batch_scaling_beta() -> float:
    xs = []
    ys = []
    for batch_size, service_ms in sorted(GR00T_BATCH_SERVICE_MS.items()):
        xs.append(math.log(float(batch_size)))
        ys.append(math.log(float(service_ms)))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    a = np.vstack([x, np.ones_like(x)]).T
    beta, _ = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(beta)


MPS_SCALE_BETA = fit_batch_scaling_beta()


def allowed_range(horizon: int) -> tuple[int, int]:
    horizon = max(int(horizon), HORIZON_FLOOR)
    if horizon <= HORIZON_FLOOR:
        return HORIZON_FLOOR, HORIZON_FLOOR
    return HORIZON_FLOOR, CHUNK_SIZE


def make_pending(rt: RobotRuntime, robot_idx: int, chunk_start_ms: float) -> PendingRequest:
    horizon = max(int(rt.proc.next()), HORIZON_FLOOR)
    lower_c, upper_c = allowed_range(horizon)
    eps = 1e-6
    return PendingRequest(
        robot_idx=robot_idx,
        model=rt.model,
        hz=rt.hz,
        period_ms=rt.period_ms,
        chunk_start_ms=chunk_start_ms,
        horizon=horizon,
        lower_c=lower_c,
        upper_c=upper_c,
        earliest_finish_ms=chunk_start_ms + (lower_c - 1) * rt.period_ms + eps,
        latest_finish_ms=chunk_start_ms + upper_c * rt.period_ms,
    )


def chunk_success(actual_consumed: int, req: PendingRequest) -> float:
    if req.lower_c <= actual_consumed <= req.upper_c:
        return 1.0
    deviation = abs(actual_consumed - req.horizon)
    weighted = deviation * ((CHUNK_SIZE / max(req.horizon, 1)) ** METRIC.beta)
    return float(math.exp(-METRIC.alpha * weighted))


def best_candidate_for_model(reqs: list[PendingRequest]) -> BatchCandidate | None:
    if not reqs:
        return None
    highs = sorted({req.latest_finish_ms for req in reqs})
    best = None
    for high in highs:
        eligible = [req for req in reqs if req.earliest_finish_ms <= high + 1e-9 and req.latest_finish_ms >= high - 1e-9]
        if not eligible:
            continue
        eligible.sort(key=lambda req: (req.latest_finish_ms, req.earliest_finish_ms, req.robot_idx))
        chosen = eligible[:MAX_BATCH]
        common_low = max(req.earliest_finish_ms for req in chosen)
        work_ms = infer_service_ms(len(chosen))
        earliest_start = common_low - work_ms
        latest_start = high - work_ms
        if earliest_start > latest_start + 1e-9:
            continue
        rank = (
            -len(chosen),
            high,
            earliest_start,
        )
        cand = BatchCandidate(
            model=chosen[0].model,
            requests=chosen,
            work_ms=work_ms,
            common_low_ms=common_low,
            common_high_ms=high,
            earliest_start_ms=earliest_start,
            latest_start_ms=latest_start,
        )
        if best is None or rank < best[0]:
            best = (rank, cand)
    return None if best is None else best[1]


def aggregate_weights(specs: list[dict[str, float | str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in specs:
        out[str(spec["model"])] = out.get(str(spec["model"]), 0.0) + float(spec["hz"])
    return out


def batch_hist(samples: list[dict]) -> dict[str, int]:
    ctr = Counter()
    for sample in samples:
        ctr[str(sample["batch_size"])] += 1
    return dict(sorted(ctr.items(), key=lambda kv: int(kv[0])))


def mps_share_rates(active: dict[str, RunningBatch], weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights[model] for model in active)
    rates = {}
    for model in active:
        share = weights[model] / total
        rates[model] = share ** MPS_SCALE_BETA
    return rates


def project_wave_finish_times(candidates: list[BatchCandidate], weights: dict[str, float]) -> dict[str, float]:
    remaining = {cand.model: cand.work_ms for cand in candidates}
    finish = {}
    elapsed = 0.0
    while remaining:
        total = sum(weights[model] for model in remaining)
        rates = {model: (weights[model] / total) ** MPS_SCALE_BETA for model in remaining}
        dt = min(remaining[model] / rates[model] for model in remaining)
        elapsed += dt
        done = []
        for model in list(remaining):
            remaining[model] = max(0.0, remaining[model] - rates[model] * dt)
            if remaining[model] <= 1e-6:
                finish[model] = elapsed
                done.append(model)
        for model in done:
            remaining.pop(model, None)
    return finish


def choose_mps_wave(
    ready: list[BatchCandidate],
    now_ms: float,
    weights: dict[str, float],
) -> list[BatchCandidate]:
    if not ready:
        return []
    best_ok = None
    best_any = None
    for r in range(1, len(ready) + 1):
        for subset in itertools.combinations(ready, r):
            finish_rel = project_wave_finish_times(list(subset), weights)
            ok = True
            total_priority = 0.0
            total_requests = 0
            max_finish = 0.0
            lateness = 0.0
            for cand in subset:
                finish_abs = now_ms + finish_rel[cand.model]
                total_priority += weights[cand.model] * len(cand.requests)
                total_requests += len(cand.requests)
                max_finish = max(max_finish, finish_rel[cand.model])
                for req in cand.requests:
                    consumed = int(math.ceil((finish_abs - req.chunk_start_ms) / req.period_ms - 1e-12))
                    if finish_rel[cand.model] > 100.0 + 1e-9:
                        ok = False
                    if finish_abs > req.latest_finish_ms + 1e-9:
                        ok = False
                    if finish_abs > req.chunk_start_ms + CHUNK_SIZE * req.period_ms + 1e-9:
                        ok = False
                    if not (req.lower_c <= consumed <= req.upper_c):
                        ok = False
                    lateness += max(0.0, finish_abs - req.latest_finish_ms)
            rank = (
                -total_priority,
                -total_requests,
                max_finish,
                lateness,
            )
            if ok:
                if best_ok is None or rank < best_ok[0]:
                    best_ok = (rank, list(subset))
            if best_any is None or (lateness, max_finish, -total_priority) < best_any[0]:
                best_any = ((lateness, max_finish, -total_priority), list(subset))
    if best_ok is not None:
        return best_ok[1]
    return [] if best_any is None else best_any[1]


def simulate_single_queue(specs: list[dict[str, float | str]], duration_s: float, seed: int) -> dict:
    duration_ms = duration_s * 1000.0
    runtimes = [
        RobotRuntime(
            model=str(spec["model"]),
            hz=float(spec["hz"]),
            start_ms=float(spec["start_ms"]),
            proc=MOD.HorizonProcess(
                GR00T_H["start_states"],
                GR00T_H["start_probs"],
                GR00T_H["transition"],
                np.random.default_rng(seed * 1000 + idx),
            ),
            period_ms=request_period_ms(float(spec["hz"])),
        )
        for idx, spec in enumerate(specs)
    ]
    activation_heap: list[tuple[float, int]] = []
    for idx, rt in enumerate(runtimes):
        heapq.heappush(activation_heap, (rt.start_ms, idx))
    pending: dict[int, PendingRequest] = {}
    current_time = 0.0
    samples: list[dict] = []
    running: RunningBatch | None = None
    steps = 0

    def activate_until(limit_ms: float):
        while activation_heap and activation_heap[0][0] <= limit_ms + 1e-9:
            chunk_start_ms, robot_idx = heapq.heappop(activation_heap)
            if chunk_start_ms > duration_ms + 1e-9:
                continue
            if robot_idx in pending:
                continue
            pending[robot_idx] = make_pending(runtimes[robot_idx], robot_idx, chunk_start_ms)

    def drop_expired(now_ms: float):
        expired = [req for req in pending.values() if now_ms + infer_service_ms(1) > req.latest_finish_ms + 1e-9]
        for req in expired:
            runtimes[req.robot_idx].reply_over_chunk_actions += 1
            pending.pop(req.robot_idx, None)

    def best_ready_candidate(now_ms: float) -> BatchCandidate | None:
        by_model: dict[str, list[PendingRequest]] = {}
        for req in pending.values():
            by_model.setdefault(req.model, []).append(req)
        ready = []
        future = []
        for reqs in by_model.values():
            cand = best_candidate_for_model(reqs)
            if cand is None:
                continue
            if cand.earliest_start_ms <= now_ms + 1e-9:
                ready.append(cand)
            else:
                future.append(cand)
        if ready:
            ready.sort(key=lambda cand: (cand.latest_start_ms, -len(cand.requests), cand.common_high_ms, cand.model))
            return ready[0]
        if future:
            future.sort(key=lambda cand: (cand.earliest_start_ms, cand.latest_start_ms, cand.model))
            return future[0]
        return None

    while activation_heap or pending or running is not None:
        steps += 1
        if steps > 2_000_000:
            raise RuntimeError("single_queue simulation made no progress")
        activate_until(current_time)
        if running is None:
            drop_expired(current_time)
            if not pending:
                if not activation_heap:
                    break
                current_time = max(current_time, activation_heap[0][0])
                continue
            cand = best_ready_candidate(current_time)
            if cand is None:
                break
            if cand.earliest_start_ms > current_time + 1e-9:
                next_activation = activation_heap[0][0] if activation_heap else math.inf
                current_time = min(cand.earliest_start_ms, next_activation)
                continue
            for req in cand.requests:
                pending.pop(req.robot_idx, None)
            running = RunningBatch(
                model=cand.model,
                requests=cand.requests,
                work_ms=cand.work_ms,
                remaining_work_ms=cand.work_ms,
                common_low_ms=cand.common_low_ms,
                common_high_ms=cand.common_high_ms,
                start_ms=current_time,
            )
            finish_ms = max(current_time + cand.work_ms, cand.common_low_ms)
            running.remaining_work_ms = max(0.0, finish_ms - current_time)
        next_activation = activation_heap[0][0] if activation_heap else math.inf
        finish_ms = current_time + running.remaining_work_ms
        if next_activation < finish_ms - 1e-9:
            current_time = next_activation
            activate_until(current_time)
            continue
        current_time = finish_ms
        finished = running
        running = None
        for req in finished.requests:
            rt = runtimes[req.robot_idx]
            consumed = int(math.ceil((current_time - req.chunk_start_ms) / req.period_ms - 1e-12))
            rt.chunk_count += 1
            if not (req.lower_c <= consumed <= req.upper_c):
                rt.miss_autohorizon_count += 1
            if current_time > req.chunk_start_ms + CHUNK_SIZE * req.period_ms + 1e-9:
                rt.reply_over_chunk_actions += 1
                continue
            rt.chunk_scores.append(chunk_success(consumed, req))
            rt.phase_shift_actions.append(consumed - req.horizon)
            samples.append(
                {
                    "robot_id": req.robot_idx,
                    "model": req.model,
                    "hz": req.hz,
                    "chunk_start_ms": req.chunk_start_ms,
                    "horizon": req.horizon,
                    "consumed": consumed,
                    "phase_shift_actions": consumed - req.horizon,
                    "chunk_elapsed_ms": current_time - req.chunk_start_ms,
                    "request_to_result_ms": finished.work_ms,
                    "batch_size": len(finished.requests),
                    "finish_ms": current_time,
                }
            )
            if current_time <= duration_ms + 1e-9:
                heapq.heappush(activation_heap, (current_time, req.robot_idx))

    robot_scores = [geometric_mean(rt.chunk_scores) for rt in runtimes]
    total_miss = sum(rt.miss_autohorizon_count for rt in runtimes)
    total_chunks = sum(rt.chunk_count for rt in runtimes)
    phase_shifts = [sample["phase_shift_actions"] for sample in samples]
    return {
        "request_to_result_ms": stats([sample["request_to_result_ms"] for sample in samples]),
        "chunk_elapsed_ms": stats([sample["chunk_elapsed_ms"] for sample in samples]),
        "phase_shift_actions": {
            "mean": float(np.mean(np.asarray(phase_shifts, dtype=np.float64))) if phase_shifts else 0.0,
            "mean_abs": float(np.mean(np.abs(np.asarray(phase_shifts, dtype=np.float64)))) if phase_shifts else 0.0,
            "p95_abs": float(np.percentile(np.abs(np.asarray(phase_shifts, dtype=np.float64)), 95)) if phase_shifts else 0.0,
        },
        "hard_miss_count": 0,
        "reply_over_chunk_actions_count": int(sum(rt.reply_over_chunk_actions for rt in runtimes)),
        "miss_autohorizon_count": int(total_miss),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(total_miss / total_chunks) if total_chunks else 0.0,
        "fleet_score": geometric_mean(robot_scores),
        "min_robot_score": min(robot_scores) if robot_scores else 1.0,
        "batch_histogram": batch_hist(samples),
        "mean_batch_size": float(np.mean(np.asarray([sample["batch_size"] for sample in samples], dtype=np.float64))) if samples else 0.0,
        "samples_head": samples[:24],
    }


def simulate_mps(specs: list[dict[str, float | str]], duration_s: float, seed: int) -> dict:
    duration_ms = duration_s * 1000.0
    runtimes = [
        RobotRuntime(
            model=str(spec["model"]),
            hz=float(spec["hz"]),
            start_ms=float(spec["start_ms"]),
            proc=MOD.HorizonProcess(
                GR00T_H["start_states"],
                GR00T_H["start_probs"],
                GR00T_H["transition"],
                np.random.default_rng(seed * 1000 + idx),
            ),
            period_ms=request_period_ms(float(spec["hz"])),
        )
        for idx, spec in enumerate(specs)
    ]
    model_weights = aggregate_weights(specs)
    activation_heap: list[tuple[float, int]] = []
    for idx, rt in enumerate(runtimes):
        heapq.heappush(activation_heap, (rt.start_ms, idx))
    pending_by_model: dict[str, dict[int, PendingRequest]] = {}
    active: dict[str, RunningBatch] = {}
    current_time = 0.0
    samples: list[dict] = []
    steps = 0

    def activate_until(limit_ms: float):
        while activation_heap and activation_heap[0][0] <= limit_ms + 1e-9:
            chunk_start_ms, robot_idx = heapq.heappop(activation_heap)
            if chunk_start_ms > duration_ms + 1e-9:
                continue
            req = make_pending(runtimes[robot_idx], robot_idx, chunk_start_ms)
            pending_by_model.setdefault(req.model, {})[robot_idx] = req

    def drop_expired(now_ms: float):
        for model, reqs in list(pending_by_model.items()):
            expired = [req for req in reqs.values() if now_ms + infer_service_ms(1) > req.latest_finish_ms + 1e-9]
            for req in expired:
                runtimes[req.robot_idx].reply_over_chunk_actions += 1
                reqs.pop(req.robot_idx, None)
            if not reqs:
                pending_by_model.pop(model, None)

    def launch_ready_batches(now_ms: float):
        if active:
            return False
        ready = []
        for model, reqs in sorted(list(pending_by_model.items())):
            cand = best_candidate_for_model(list(reqs.values()))
            if cand is None or cand.earliest_start_ms > now_ms + 1e-9:
                continue
            ready.append(cand)
        chosen = choose_mps_wave(ready, now_ms, model_weights)
        if not chosen:
            return False
        for cand in chosen:
            reqs = pending_by_model.get(cand.model, {})
            for req in cand.requests:
                reqs.pop(req.robot_idx, None)
            if not reqs and cand.model in pending_by_model:
                pending_by_model.pop(cand.model, None)
            active[cand.model] = RunningBatch(
                model=cand.model,
                requests=cand.requests,
                work_ms=cand.work_ms,
                remaining_work_ms=cand.work_ms,
                common_low_ms=cand.common_low_ms,
                common_high_ms=cand.common_high_ms,
                start_ms=now_ms,
            )
        return True

    while activation_heap or any(pending_by_model.values()) or active:
        steps += 1
        if steps > 2_000_000:
            raise RuntimeError("mps simulation made no progress")
        activate_until(current_time)
        drop_expired(current_time)
        launch_ready_batches(current_time)

        if not active:
            next_activation = activation_heap[0][0] if activation_heap else math.inf
            next_ready = math.inf
            for model, reqs in pending_by_model.items():
                if model in active:
                    continue
                cand = best_candidate_for_model(list(reqs.values()))
                if cand is None:
                    continue
                next_ready = min(next_ready, cand.earliest_start_ms)
            jump = min(next_activation, next_ready)
            if jump is math.inf:
                break
            current_time = max(current_time, jump)
            continue

        rates = mps_share_rates(active, model_weights)
        next_activation = activation_heap[0][0] if activation_heap else math.inf
        dt_finish = min(batch.remaining_work_ms / rates[model] for model, batch in active.items())
        dt_event = min(
            dt_finish,
            max(0.0, next_activation - current_time) if next_activation < math.inf else math.inf,
        )
        if dt_event is math.inf:
            break
        if dt_event < 1e-9:
            if next_activation <= current_time + 1e-9 or next_ready <= current_time + 1e-9:
                current_time += 1e-6
                continue
        for model, batch in active.items():
            batch.remaining_work_ms = max(0.0, batch.remaining_work_ms - rates[model] * dt_event)
        current_time += dt_event
        activate_until(current_time)

        finished_models = [model for model, batch in active.items() if batch.remaining_work_ms <= 1e-6]
        for model in finished_models:
            finished = active.pop(model)
            for req in finished.requests:
                rt = runtimes[req.robot_idx]
                consumed = int(math.ceil((current_time - req.chunk_start_ms) / req.period_ms - 1e-12))
                rt.chunk_count += 1
                if not (req.lower_c <= consumed <= req.upper_c):
                    rt.miss_autohorizon_count += 1
                if current_time > req.chunk_start_ms + CHUNK_SIZE * req.period_ms + 1e-9:
                    rt.reply_over_chunk_actions += 1
                    continue
                rt.chunk_scores.append(chunk_success(consumed, req))
                rt.phase_shift_actions.append(consumed - req.horizon)
                samples.append(
                    {
                        "robot_id": req.robot_idx,
                        "model": req.model,
                        "hz": req.hz,
                        "chunk_start_ms": req.chunk_start_ms,
                        "horizon": req.horizon,
                        "consumed": consumed,
                        "phase_shift_actions": consumed - req.horizon,
                        "chunk_elapsed_ms": current_time - req.chunk_start_ms,
                        "request_to_result_ms": current_time - finished.start_ms,
                        "batch_size": len(finished.requests),
                        "finish_ms": current_time,
                    }
                )
                if current_time <= duration_ms + 1e-9:
                    heapq.heappush(activation_heap, (current_time, req.robot_idx))

    robot_scores = [geometric_mean(rt.chunk_scores) for rt in runtimes]
    total_miss = sum(rt.miss_autohorizon_count for rt in runtimes)
    total_chunks = sum(rt.chunk_count for rt in runtimes)
    phase_shifts = [sample["phase_shift_actions"] for sample in samples]
    return {
        "request_to_result_ms": stats([sample["request_to_result_ms"] for sample in samples]),
        "chunk_elapsed_ms": stats([sample["chunk_elapsed_ms"] for sample in samples]),
        "phase_shift_actions": {
            "mean": float(np.mean(np.asarray(phase_shifts, dtype=np.float64))) if phase_shifts else 0.0,
            "mean_abs": float(np.mean(np.abs(np.asarray(phase_shifts, dtype=np.float64)))) if phase_shifts else 0.0,
            "p95_abs": float(np.percentile(np.abs(np.asarray(phase_shifts, dtype=np.float64)), 95)) if phase_shifts else 0.0,
        },
        "hard_miss_count": 0,
        "reply_over_chunk_actions_count": int(sum(rt.reply_over_chunk_actions for rt in runtimes)),
        "miss_autohorizon_count": int(total_miss),
        "chunk_count": int(total_chunks),
        "miss_autohorizon_ratio": float(total_miss / total_chunks) if total_chunks else 0.0,
        "fleet_score": geometric_mean(robot_scores),
        "min_robot_score": min(robot_scores) if robot_scores else 1.0,
        "batch_histogram": batch_hist(samples),
        "mean_batch_size": float(np.mean(np.asarray([sample["batch_size"] for sample in samples], dtype=np.float64))) if samples else 0.0,
        "samples_head": samples[:24],
    }


def aggregate(specs: list[dict[str, float | str]], duration_s: float, seeds: list[int], mode: str) -> dict:
    fn = simulate_mps if mode == "batch_plus_mps" else simulate_single_queue
    outs = [fn(specs, duration_s, seed) for seed in seeds]
    batch_ctr = Counter()
    for out in outs:
        for batch_size, count in out["batch_histogram"].items():
            batch_ctr[str(batch_size)] += int(count)
    reply_over = int(sum(out["reply_over_chunk_actions_count"] for out in outs))
    p95 = float(np.mean([out["request_to_result_ms"]["p95_ms"] for out in outs]))
    return {
        "hard_miss_count": int(sum(out["hard_miss_count"] for out in outs)),
        "reply_over_chunk_actions_count": reply_over,
        "mean_request_to_result_p95_ms": p95,
        "mean_chunk_elapsed_p95_ms": float(np.mean([out["chunk_elapsed_ms"]["p95_ms"] for out in outs])),
        "mean_fleet_score": float(np.mean([out["fleet_score"] for out in outs])),
        "mean_min_robot_score": float(np.mean([out["min_robot_score"] for out in outs])),
        "mean_miss_autohorizon_ratio": float(np.mean([out["miss_autohorizon_ratio"] for out in outs])),
        "mean_phase_shift_abs_actions": float(np.mean([out["phase_shift_actions"]["mean_abs"] for out in outs])),
        "mean_batch_size": float(np.mean([out["mean_batch_size"] for out in outs])),
        "batch_histogram": dict(sorted(batch_ctr.items(), key=lambda kv: int(kv[0]))),
        "stable_under_100ms": bool(reply_over == 0 and p95 <= 100.0),
        "truth_runs": outs,
    }


def search_scenario(copies_per_model: int, mode: str) -> dict:
    phase20 = phase_grid(20.0)
    phase10 = phase_grid(10.0)
    rows = []
    for p20 in phase20:
        for p10a in phase10:
            for p10b in phase10:
                specs = build_same_phase_specs(copies_per_model, p20, p10a, p10b)
                pred = aggregate(specs, PREDICT_DURATION_S, [SEEDS[0]], mode)
                rows.append(
                    {
                        "specs": specs,
                        "phases": {"20hz_fractal": p20, "10hz_libero": p10a, "10hz_rel30k": p10b},
                        "pred": pred,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["pred"]["reply_over_chunk_actions_count"],
            row["pred"]["mean_request_to_result_p95_ms"],
            -row["pred"]["mean_batch_size"],
            -row["pred"]["mean_min_robot_score"],
            -row["pred"]["mean_fleet_score"],
        )
    )
    finalists = []
    for row in rows[:TOPK_FINALISTS]:
        truth = aggregate(row["specs"], TRUTH_DURATION_S, SEEDS, mode)
        finalists.append({"specs": row["specs"], "phases": row["phases"], "metrics": truth})
    finalists.sort(
        key=lambda row: (
            row["metrics"]["reply_over_chunk_actions_count"],
            row["metrics"]["mean_request_to_result_p95_ms"],
            -row["metrics"]["mean_batch_size"],
            -row["metrics"]["mean_min_robot_score"],
            -row["metrics"]["mean_fleet_score"],
        )
    )
    return {"best": finalists[0], "finalists": finalists}


def main():
    results = {
        "settings": {
            "shared_prefix_resident": True,
            "swap_cost_mode": "none",
            "chunk_size": CHUNK_SIZE,
            "horizon_floor": HORIZON_FLOOR,
            "phase_rule": "if horizon>8, may infer after any consumed action in [8,16]",
            "max_batch": MAX_BATCH,
            "penalty_scale_vs_pi05": PENALTY_SCALE,
            "mps_scaling_beta": MPS_SCALE_BETA,
            "truth_duration_s": TRUTH_DURATION_S,
            "predict_duration_s": PREDICT_DURATION_S,
            "seeds": SEEDS,
        },
        "scenarios": {},
    }
    for copies_per_model in SCENARIOS:
        scenario_key = f"{copies_per_model}x_per_model"
        results["scenarios"][scenario_key] = {
            "robot_count": 4 * copies_per_model,
            "batch_only": search_scenario(copies_per_model, "batch_only"),
            "batch_plus_mps": search_scenario(copies_per_model, "batch_plus_mps"),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(results, f, indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
