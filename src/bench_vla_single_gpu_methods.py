#!/usr/bin/env python3

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/autodl-tmp/VLAServing")
RESULTS = ROOT / "results"
OUT = RESULTS / "vla_single_gpu_methods_20260411.json"

H2D_GIB_PER_S = 23.27228306738265
PI05_INFER_MS = {
    "30hz_official_ft": 43.198463439941406,
    "20hz_quantiles": 43.18052673339844,
    "10hz_a_logits": 43.21331214904785,
    "10hz_b_autoh": 43.06164741516113,
}
PI05_SHELL_GIB = {
    "30hz_official_ft": 7.485,
    "20hz_quantiles": 7.485,
    "10hz_a_logits": 7.485,
    "10hz_b_autoh": 7.485,
}
GR00T_INFER_MS = {
    "30hz_bridge": 43.8,
    "20hz_fractal": 43.8,
    "10hz_libero": 43.88061095960438,
    "10hz_rel30k": 43.88061095960438,
}
GR00T_SHELL_GIB = {
    "30hz_bridge": 6573377712 / (1024**3),
    "20hz_fractal": 6573377712 / (1024**3),
    "10hz_libero": 9192043768 / (1024**3),
    "10hz_rel30k": 9192043768 / (1024**3),
}
PARTITIONS = [1.0, 0.75, 0.5, 0.375, 0.25]
GPU_MEMORY_LIMIT_GIB = 24.0
PI05_BATCH_PROFILE = RESULTS / "lerobot_p50_step1_full_e2e_batch_sweep_compile_dynamic_20260327_summary.json"
GR00T_BATCH_PROFILE = RESULTS / "groot_n1d6_same_model_batch_curve_step1_compile_libero.json"
PI05_AUTOH = RESULTS / "pi05_autohorizon_simulator_fit_20260329.json"
GR00T_AUTOH = RESULTS / "groot_n15_official_horizon_simulator_fit_20260328.json"
SLA_MS = 100.0
SIM_DURATION_S = 90.0
SEEDS = [1101, 1102, 1103]


@dataclass(frozen=True)
class SuccessMetricParams:
    alpha: float
    beta: float


@dataclass
class RequestSpec:
    robot_id: int
    model: str
    hz: float
    period_ms: float
    chunk_size: int
    chunk_start_ms: float
    request_ms: float
    target_finish_ms: float
    hard_deadline_ms: float
    horizon: int


@dataclass
class RobotRuntime:
    model: str
    hz: float
    chunk_size: int
    start_ms: float
    proc: "HorizonProcess"
    period_ms: float
    requests_sent: int = 0
    sla_miss_count: int = 0
    hard_miss_count: int = 0
    miss_autohorizon_count: int = 0
    chunk_scores: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    actual_consumed: list[int] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    next_horizon: int | None = None


class HorizonProcess:
    def __init__(self, start_states: np.ndarray, start_probs: np.ndarray, transition: dict[int, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator):
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


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values)))


def chunk_success(actual_consumed: int, horizon: int, chunk_size: int, metric: SuccessMetricParams) -> float:
    deviation = abs(actual_consumed - horizon)
    weighted = deviation * ((chunk_size / max(horizon, 1)) ** metric.beta)
    return float(math.exp(-metric.alpha * weighted))


def latest_feasible_start(req: RequestSpec, infer_ms: float, resource_ready_ms: float, compute_ready_ms: float) -> float:
    earliest = max(req.request_ms, resource_ready_ms, compute_ready_ms)
    latest = req.target_finish_ms - infer_ms
    if latest >= earliest - 1e-9:
        return latest
    return earliest


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


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
        "mean_horizon": float(d["mean_horizon"]),
        "start_states": start_states,
        "start_probs": start_probs,
        "transition": transition,
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
        "mean_horizon": float(d["mean_horizon"]),
        "start_states": start_states,
        "start_probs": start_probs,
        "transition": transition,
    }


def load_batch_profile_pi05() -> dict[int, float]:
    with PI05_BATCH_PROFILE.open() as f:
        d = json.load(f)
    out = {}
    for item in d["results"]:
        b = int(item["batch_size"])
        out[b] = float(item["full_e2e"]["p50_ms"])
    return out


def load_batch_profile_gr00t() -> dict[int, float]:
    with GR00T_BATCH_PROFILE.open() as f:
        d = json.load(f)
    out = {}
    for item in d["results"]:
        out[int(item["batch_size"])] = float(item["service_ms_for_scheduler"])
    return out


PI05_H = load_pi05_horizon()
GR00T_H = load_gr00t_horizon()
PI05_BATCH = load_batch_profile_pi05()
GR00T_BATCH = load_batch_profile_gr00t()


def make_workload(name: str, seed: int) -> tuple[list[RobotRuntime], dict[str, Any]]:
    if name == "pi05":
        base = [
            ("30hz_official_ft", 30.0, 0.0),
            ("20hz_quantiles", 20.0, 50.0),
            ("10hz_a_logits", 10.0, 100.0),
            ("10hz_b_autoh", 10.0, 600.0),
        ]
        chunk_size = 50
        horizon = PI05_H
        metric = SuccessMetricParams(alpha=0.018, beta=1.15)
        infer_ms = PI05_INFER_MS
        shell_gib = PI05_SHELL_GIB
    elif name == "gr00t":
        base = [
            ("30hz_bridge", 30.0, 0.0),
            ("20hz_fractal", 20.0, 40.0),
            ("10hz_libero", 10.0, 120.0),
            ("10hz_rel30k", 10.0, 620.0),
        ]
        chunk_size = 16
        horizon = GR00T_H
        metric = SuccessMetricParams(alpha=0.035, beta=1.35)
        infer_ms = GR00T_INFER_MS
        shell_gib = GR00T_SHELL_GIB
    else:
        raise ValueError(name)
    robots = []
    for i, (model, hz, start_ms) in enumerate(base):
        proc = HorizonProcess(
            horizon["start_states"],
            horizon["start_probs"],
            horizon["transition"],
            np.random.default_rng(seed * 100 + i),
        )
        robots.append(
            RobotRuntime(
                model=model,
                hz=hz,
                chunk_size=chunk_size,
                start_ms=start_ms,
                proc=proc,
                period_ms=1000.0 / hz,
            )
        )
    meta = {
        "chunk_size": chunk_size,
        "metric": {"alpha": metric.alpha, "beta": metric.beta},
        "infer_ms": infer_ms,
        "shell_gib": shell_gib,
        "mean_horizon": horizon["mean_horizon"],
    }
    return robots, meta


def build_request(rt: RobotRuntime, chunk_start_ms: float) -> RequestSpec:
    h = rt.proc.next()
    req_ms = max(chunk_start_ms + h * rt.period_ms - SLA_MS, chunk_start_ms)
    return RequestSpec(
        robot_id=-1,
        model=rt.model,
        hz=rt.hz,
        period_ms=rt.period_ms,
        chunk_size=rt.chunk_size,
        chunk_start_ms=chunk_start_ms,
        request_ms=req_ms,
        target_finish_ms=chunk_start_ms + h * rt.period_ms,
        hard_deadline_ms=chunk_start_ms + rt.chunk_size * rt.period_ms,
        horizon=h,
    )


class MethodBase:
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float]):
        self.infer_ms = infer_ms
        self.shell_gib = shell_gib
        self.compute_available_ms = 0.0
        self.copy_available_ms = 0.0
        self.total_h2d_gib = 0.0
        self.swap_count = 0
        self.prefetch_count = 0

    def register_future(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]):
        return None

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

    def method_stats(self) -> dict[str, Any]:
        return {
            "total_h2d_gib": self.total_h2d_gib,
            "swap_count": self.swap_count,
            "prefetch_count": self.prefetch_count,
        }


class OracleResident(MethodBase):
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float]):
        super().__init__(infer_ms, shell_gib)
        self.gpu_memory_gib = sum(shell_gib.values())

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        start = latest_feasible_start(req, self.infer_ms[req.model], now_ms, self.compute_available_ms)
        finish = start + self.infer_ms[req.model]
        self.compute_available_ms = finish
        return finish, {"shell": req.model, "copy_ms": 0.0}

    def method_stats(self) -> dict[str, Any]:
        out = super().method_stats()
        out["gpu_memory_gib"] = self.gpu_memory_gib
        return out


class ReefLike(MethodBase):
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float]):
        super().__init__(infer_ms, shell_gib)
        self.current_model: str | None = None
        self.gpu_memory_gib = max(shell_gib.values())

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        ready = now_ms
        copy_ms = 0.0
        if self.current_model != req.model:
            swap_gib = self.shell_gib[req.model]
            copy_ms = swap_gib / H2D_GIB_PER_S * 1000.0
            copy_start = max(now_ms, self.copy_available_ms)
            ready = copy_start + copy_ms
            self.copy_available_ms = ready
            self.current_model = req.model
            self.total_h2d_gib += swap_gib
            self.swap_count += 1
        start = latest_feasible_start(req, self.infer_ms[req.model], ready, self.compute_available_ms)
        finish = start + self.infer_ms[req.model]
        self.compute_available_ms = finish
        return finish, {"shell": "temporal", "copy_ms": copy_ms}

    def method_stats(self) -> dict[str, Any]:
        out = super().method_stats()
        out["gpu_memory_gib"] = self.gpu_memory_gib
        return out


class ShellPoolNoPrefetch(MethodBase):
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float], resident_models: list[str], overlap_copy: bool):
        super().__init__(infer_ms, shell_gib)
        self.resident_models = list(resident_models)
        self.overlap_copy = overlap_copy
        self.next_use: dict[str, float] = {}
        self.gpu_memory_gib = sum(shell_gib[m] for m in self.resident_models)

    def register_future(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]):
        self.next_use[req.model] = req.request_ms

    def _evict_model(self, requested: str) -> str:
        candidates = [m for m in self.resident_models if m != requested]
        if not candidates:
            return requested
        return max(candidates, key=lambda m: self.next_use.get(m, math.inf))

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        ready = now_ms
        copy_ms = 0.0
        if req.model not in self.resident_models:
            evicted = self._evict_model(req.model)
            swap_gib = self.shell_gib[req.model]
            copy_ms = swap_gib / H2D_GIB_PER_S * 1000.0
            copy_start = max(now_ms, self.copy_available_ms)
            copy_finish = copy_start + copy_ms
            self.copy_available_ms = copy_finish
            ready = copy_finish
            self.resident_models.remove(evicted)
            self.resident_models.append(req.model)
            self.total_h2d_gib += swap_gib
            self.swap_count += 1
        if self.overlap_copy:
            resource_ready = max(now_ms, ready)
        else:
            resource_ready = ready
        start = latest_feasible_start(req, self.infer_ms[req.model], resource_ready, self.compute_available_ms)
        finish = start + self.infer_ms[req.model]
        self.compute_available_ms = finish
        return finish, {"resident_models": list(self.resident_models), "copy_ms": copy_ms}

    def method_stats(self) -> dict[str, Any]:
        out = super().method_stats()
        out["gpu_memory_gib"] = self.gpu_memory_gib
        out["resident_models_final"] = list(self.resident_models)
        return out


class UsherLikeSpatial(MethodBase):
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float], slots: list[tuple[str, float]]):
        super().__init__(infer_ms, shell_gib)
        self.slot_models = {i: model for i, (model, _p) in enumerate(slots)}
        self.slot_partitions = {i: p for i, (_model, p) in enumerate(slots)}
        self.slot_compute_ms = {i: 0.0 for i in self.slot_models}
        self.next_use: dict[str, float] = {}
        self.gpu_memory_gib = sum(shell_gib[m] for m in self.slot_models.values())

    def register_future(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]):
        self.next_use[req.model] = req.request_ms

    def _find_slot(self, model: str) -> int | None:
        for slot, resident in self.slot_models.items():
            if resident == model:
                return slot
        return None

    def _evict_slot(self, requested: str) -> int:
        candidates = [(self.next_use.get(model, math.inf), self.slot_partitions[slot], slot)
                      for slot, model in self.slot_models.items() if model != requested]
        _next_use, _part, slot = max(candidates)
        return slot

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        copy_ms = 0.0
        ready = now_ms
        slot = self._find_slot(req.model)
        if slot is None:
            slot = self._evict_slot(req.model)
            swap_gib = self.shell_gib[req.model]
            copy_ms = swap_gib / H2D_GIB_PER_S * 1000.0
            copy_start = max(now_ms, self.copy_available_ms)
            ready = copy_start + copy_ms
            self.copy_available_ms = ready
            self.total_h2d_gib += swap_gib
            self.swap_count += 1
            self.slot_models[slot] = req.model
        part = self.slot_partitions[slot]
        infer_ms = self.infer_ms[req.model] / (part ** 0.90)
        start = latest_feasible_start(req, infer_ms, ready, self.slot_compute_ms[slot])
        finish = start + infer_ms
        self.slot_compute_ms[slot] = finish
        return finish, {"slot": slot, "partition": part, "copy_ms": copy_ms}

    def method_stats(self) -> dict[str, Any]:
        out = super().method_stats()
        out["gpu_memory_gib"] = self.gpu_memory_gib
        out["resident_models_final"] = list(self.slot_models.values())
        return out


class DistServeLikeDisagg(MethodBase):
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float], stage_partitions: tuple[float, float] = (0.5, 0.5)):
        super().__init__(infer_ms, shell_gib)
        self.stage_partitions = stage_partitions
        self.stage_compute_ms = [0.0, 0.0]
        self.stage_models: list[str | None] = [None, None]
        self.stage_handoff_ms = 2.0
        self.gpu_memory_gib = 2.0 * max(shell_gib.values())

    def _stage_time(self, model: str, frac: float, part: float) -> float:
        return (self.infer_ms[model] * frac) / (part ** 0.90)

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        copy_ms_total = 0.0
        part0, part1 = self.stage_partitions
        stage0_ready = now_ms
        if self.stage_models[0] != req.model:
            copy_ms = self.shell_gib[req.model] / H2D_GIB_PER_S * 1000.0
            copy_start = max(now_ms, self.copy_available_ms)
            stage0_ready = copy_start + copy_ms
            self.copy_available_ms = stage0_ready
            self.total_h2d_gib += self.shell_gib[req.model]
            self.swap_count += 1
            self.stage_models[0] = req.model
            copy_ms_total += copy_ms
        stage0_ms = self._stage_time(req.model, 0.7, part0)
        start0 = max(stage0_ready, self.stage_compute_ms[0])
        finish0 = start0 + stage0_ms
        self.stage_compute_ms[0] = finish0

        stage1_ready = finish0
        if self.stage_models[1] != req.model:
            copy_ms = self.shell_gib[req.model] / H2D_GIB_PER_S * 1000.0
            copy_start = max(stage1_ready, self.copy_available_ms)
            stage1_ready = copy_start + copy_ms
            self.copy_available_ms = stage1_ready
            self.total_h2d_gib += self.shell_gib[req.model]
            self.swap_count += 1
            self.stage_models[1] = req.model
            copy_ms_total += copy_ms
        stage1_ms = self._stage_time(req.model, 0.3, part1) + self.stage_handoff_ms
        start1 = max(stage1_ready, self.stage_compute_ms[1])
        finish1 = start1 + stage1_ms
        self.stage_compute_ms[1] = finish1
        return finish1, {"copy_ms": copy_ms_total}

    def method_stats(self) -> dict[str, Any]:
        out = super().method_stats()
        out["gpu_memory_gib"] = self.gpu_memory_gib
        out["stage_partitions"] = list(self.stage_partitions)
        return out


class VLAAwareSharedShell(MethodBase):
    def __init__(self, infer_ms: dict[str, float], shell_gib: dict[str, float], dedicated: list[str], shared: list[str], shared_fraction: float = 0.0):
        super().__init__(infer_ms, shell_gib)
        self.dedicated = set(dedicated)
        self.shared = list(shared)
        self.shared_fraction = shared_fraction
        self.shared_resident: str = shared[0]
        self.prefetch_ready: dict[str, float] = {}
        self.planned_copy_end: dict[str, float] = {}
        self.gpu_memory_gib = sum(shell_gib[m] for m in dedicated) + max(shell_gib[m] for m in shared)

    def register_future(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]):
        if req.model in self.dedicated:
            return
        # Re-plan based on the earliest upcoming shared-model request.
        pending = [r for m, r in all_requests.items() if m in self.shared]
        if not pending:
            return
        pending.sort(key=lambda r: r.request_ms)
        first = pending[0]
        if first.model == self.shared_resident:
            self.prefetch_ready[first.model] = now_ms
            return
        missing_gib = self.shell_gib[first.model] * (1.0 - self.shared_fraction)
        copy_ms = missing_gib / H2D_GIB_PER_S * 1000.0
        latest_end = first.request_ms
        start = max(now_ms, self.copy_available_ms)
        if start + copy_ms > latest_end:
            start = max(self.copy_available_ms, latest_end - copy_ms)
        finish = start + copy_ms
        self.copy_available_ms = finish
        self.prefetch_ready[first.model] = finish
        self.planned_copy_end[first.model] = finish
        self.total_h2d_gib += missing_gib
        self.prefetch_count += 1
        self.shared_resident = first.model

    def serve(self, now_ms: float, req: RequestSpec, all_requests: dict[str, RequestSpec]) -> tuple[float, dict[str, Any]]:
        copy_ms = 0.0
        ready = now_ms
        if req.model in self.shared:
            if req.model in self.prefetch_ready and self.prefetch_ready[req.model] <= now_ms + 1e-9:
                ready = now_ms
            elif req.model == self.shared_resident:
                ready = now_ms
            else:
                missing_gib = self.shell_gib[req.model] * (1.0 - self.shared_fraction)
                copy_ms = missing_gib / H2D_GIB_PER_S * 1000.0
                start = max(now_ms, self.copy_available_ms)
                ready = start + copy_ms
                self.copy_available_ms = ready
                self.total_h2d_gib += missing_gib
                self.swap_count += 1
                self.shared_resident = req.model
        start = latest_feasible_start(req, self.infer_ms[req.model], ready, self.compute_available_ms)
        finish = start + self.infer_ms[req.model]
        self.compute_available_ms = finish
        return finish, {"copy_ms": copy_ms, "shared_resident": self.shared_resident}

    def method_stats(self) -> dict[str, Any]:
        out = super().method_stats()
        out["gpu_memory_gib"] = self.gpu_memory_gib
        out["shared_fraction"] = self.shared_fraction
        return out


def sim_request_level(workload: str, method_name: str, seed: int, shared_fraction: float = 0.0) -> dict[str, Any]:
    robots, meta = make_workload(workload, seed)
    infer_ms = meta["infer_ms"]
    shell_gib = meta["shell_gib"]
    metric = SuccessMetricParams(**meta["metric"])

    if method_name == "oracle_full_resident":
        method: MethodBase = OracleResident(infer_ms, shell_gib)
    elif method_name == "reef_like_temporal":
        method = ReefLike(infer_ms, shell_gib)
    elif method_name == "clockwork_like":
        resident = [robots[0].model, robots[1].model, robots[2].model]
        method = ShellPoolNoPrefetch(infer_ms, shell_gib, resident, overlap_copy=False)
    elif method_name == "paella_like":
        resident = [robots[0].model, robots[1].model, robots[2].model]
        method = ShellPoolNoPrefetch(infer_ms, shell_gib, resident, overlap_copy=True)
    elif method_name == "usher_like":
        method = UsherLikeSpatial(
            infer_ms,
            shell_gib,
            slots=[
                (robots[0].model, 0.375),
                (robots[1].model, 0.375),
                (robots[2].model, 0.25),
            ],
        )
    elif method_name == "distserve_like":
        method = DistServeLikeDisagg(infer_ms, shell_gib)
    elif method_name == "vla_aware":
        method = VLAAwareSharedShell(
            infer_ms,
            shell_gib,
            dedicated=[robots[0].model, robots[1].model],
            shared=[robots[2].model, robots[3].model],
            shared_fraction=shared_fraction,
        )
    else:
        raise ValueError(method_name)

    events: list[tuple[float, int]] = []
    requests: dict[str, RequestSpec] = {}
    for idx, rt in enumerate(robots):
        req = build_request(rt, rt.start_ms)
        req.robot_id = idx
        requests[rt.model] = req
        method.register_future(rt.start_ms, req, requests)
    for req in requests.values():
        heapq.heappush(events, (req.request_ms, req.robot_id))

    duration_ms = SIM_DURATION_S * 1000.0
    while events:
        now_ms, robot_idx = heapq.heappop(events)
        rt = robots[robot_idx]
        req = requests[rt.model]
        if abs(req.request_ms - now_ms) > 1e-6:
            continue
        if now_ms > duration_ms:
            continue
        finish, _detail = method.serve(now_ms, req, requests)
        latency = finish - now_ms
        actual = max(1, min(req.chunk_size, int(math.ceil((finish - req.chunk_start_ms) / req.period_ms))))
        score = chunk_success(actual, req.horizon, req.chunk_size, metric)
        rt.requests_sent += 1
        rt.latencies_ms.append(latency)
        rt.actual_consumed.append(actual)
        rt.horizons.append(req.horizon)
        rt.chunk_scores.append(score)
        if latency > SLA_MS + 1e-9:
            rt.sla_miss_count += 1
        if finish > req.hard_deadline_ms + 1e-9:
            rt.hard_miss_count += 1
        if actual != req.horizon:
            rt.miss_autohorizon_count += 1

        next_req = build_request(rt, finish)
        next_req.robot_id = robot_idx
        requests[rt.model] = next_req
        method.register_future(finish, next_req, requests)
        if next_req.request_ms <= duration_ms + 1e-9:
            heapq.heappush(events, (next_req.request_ms, robot_idx))

    all_scores = [s for r in robots for s in r.chunk_scores]
    per_robot = []
    for r in robots:
        miss_ratio = (r.miss_autohorizon_count / len(r.chunk_scores)) if r.chunk_scores else 0.0
        per_robot.append(
            {
                "model": r.model,
                "hz": r.hz,
                "requests": r.requests_sent,
                "sla_miss_count": r.sla_miss_count,
                "hard_miss_count": r.hard_miss_count,
                "miss_autohorizon_ratio": miss_ratio,
                "latency_p50_ms": percentile(r.latencies_ms, 50),
                "latency_p95_ms": percentile(r.latencies_ms, 95),
                "latency_max_ms": max(r.latencies_ms) if r.latencies_ms else 0.0,
                "fleet_score": geometric_mean(r.chunk_scores),
            }
        )
    return {
        "per_robot": per_robot,
        "summary": {
            "requests_total": int(sum(r.requests_sent for r in robots)),
            "sla_miss_count": int(sum(r.sla_miss_count for r in robots)),
            "hard_miss_count": int(sum(r.hard_miss_count for r in robots)),
            "sla_miss_rate": float(sum(r.sla_miss_count for r in robots) / max(sum(r.requests_sent for r in robots), 1)),
            "hard_miss_rate": float(sum(r.hard_miss_count for r in robots) / max(sum(r.requests_sent for r in robots), 1)),
            "miss_autohorizon_ratio": float(sum(r.miss_autohorizon_count for r in robots) / max(sum(len(r.chunk_scores) for r in robots), 1)),
            "fleet_score": geometric_mean(all_scores),
            "min_robot_score": min((geometric_mean(r.chunk_scores) for r in robots), default=1.0),
            "latency_p50_ms": percentile([x for r in robots for x in r.latencies_ms], 50),
            "latency_p95_ms": percentile([x for r in robots for x in r.latencies_ms], 95),
            "latency_p99_ms": percentile([x for r in robots for x in r.latencies_ms], 99),
            "latency_max_ms": max((max(r.latencies_ms) for r in robots if r.latencies_ms), default=0.0),
        },
        "method_stats": method.method_stats(),
    }


def fit_partition_requirement(rate_rps: float, latency_profile: dict[int, float], swap_ms: float = 0.0) -> float | None:
    for p in PARTITIONS:
        for b, l_ms in latency_profile.items():
            exec_ms = l_ms / (p ** 0.90)
            cycle_ms = exec_ms + swap_ms
            throughput = 1000.0 * b / cycle_ms
            if cycle_ms <= SLA_MS and throughput >= rate_rps:
                return p
    return None


def analytic_gpulet(workload: str) -> dict[str, Any]:
    if workload == "pi05":
        batch = {b: PI05_BATCH[b] for b in range(1, 9)}
        models = [
            ("30hz_official_ft", 30.0, PI05_H["mean_horizon"], 7.485, 289.47464376688004),
            ("20hz_quantiles", 20.0, PI05_H["mean_horizon"], 7.485, 289.47464376688004),
            ("10hz_a_logits", 10.0, PI05_H["mean_horizon"], 7.485, 289.47464376688004),
            ("10hz_b_autoh", 10.0, PI05_H["mean_horizon"], 7.485, 289.47464376688004),
        ]
    else:
        batch = {b: GR00T_BATCH[b] for b in range(1, 9)}
        models = [
            ("30hz_bridge", 30.0, GR00T_H["mean_horizon"], GR00T_SHELL_GIB["30hz_bridge"], GR00T_SHELL_GIB["30hz_bridge"] / H2D_GIB_PER_S * 1000.0),
            ("20hz_fractal", 20.0, GR00T_H["mean_horizon"], GR00T_SHELL_GIB["20hz_fractal"], GR00T_SHELL_GIB["20hz_fractal"] / H2D_GIB_PER_S * 1000.0),
            ("10hz_libero", 10.0, GR00T_H["mean_horizon"], GR00T_SHELL_GIB["10hz_libero"], GR00T_SHELL_GIB["10hz_libero"] / H2D_GIB_PER_S * 1000.0),
            ("10hz_rel30k", 10.0, GR00T_H["mean_horizon"], GR00T_SHELL_GIB["10hz_rel30k"], GR00T_SHELL_GIB["10hz_rel30k"] / H2D_GIB_PER_S * 1000.0),
        ]

    rates = {m: hz / mean_h for m, hz, mean_h, *_ in models}
    spatial_only = []
    used_p = 0.0
    used_mem = 0.0
    spatial_ok = True
    for m, hz, mean_h, mem_gib, swap_ms in models:
        p = fit_partition_requirement(rates[m], batch)
        if p is None or used_p + p > 1.0 + 1e-9 or used_mem + mem_gib > GPU_MEMORY_LIMIT_GIB + 1e-9:
            spatial_ok = False
            break
        spatial_only.append({"model": m, "partition": p, "cycle_ms": batch[1] / (p ** 0.90)})
        used_p += p
        used_mem += mem_gib

    temporal_ok = False
    temporal_cycle_ms = None
    for batch_size, l_ms in batch.items():
        cycle = 4 * (l_ms + models[0][4])
        throughput = 1000.0 * batch_size / cycle
        if cycle <= SLA_MS and throughput >= sum(rates.values()):
            temporal_ok = True
            temporal_cycle_ms = cycle
            break

    groups = [[models[0], models[1]], [models[2], models[3]]]
    st_gpulets = []
    st_mem = 0.0
    st_ok = True
    for grp in groups:
        mem_gib = max(x[3] for x in grp)
        swap_ms = max(x[4] for x in grp)
        best = None
        for p in PARTITIONS:
            for b, l_ms in batch.items():
                cycle = len(grp) * (l_ms / (p ** 0.90) + swap_ms)
                throughput = 1000.0 * b / cycle
                if cycle <= SLA_MS and all(throughput >= rates[x[0]] for x in grp):
                    cand = (p, cycle, b)
                    if best is None or cand < best:
                        best = cand
        if best is None:
            st_ok = False
            break
        p, cycle, b = best
        st_gpulets.append({"models": [x[0] for x in grp], "partition": p, "cycle_ms": cycle, "batch": b})
        st_mem += mem_gib
    if st_mem > GPU_MEMORY_LIMIT_GIB + 1e-9:
        st_ok = False

    return {
        "temporal_only": {
            "feasible_under_100ms": temporal_ok,
            "representative_cycle_ms": temporal_cycle_ms,
            "reason": None if temporal_ok else "queue/cycle plus repeated model switching exceeds 100ms",
        },
        "spatial_only": {
            "feasible_under_100ms": spatial_ok,
            "gpu_memory_gib": used_mem,
            "used_partition": used_p,
            "schedule": spatial_only if spatial_ok else None,
            "reason": None if spatial_ok else "per-model partitions or shell memory exceed single-GPU budget",
        },
        "spatio_temporal": {
            "feasible_under_100ms": st_ok,
            "gpu_memory_gib": st_mem,
            "schedule": st_gpulets if st_ok else None,
            "reason": None if st_ok else "gpulet cycle/duty-window exceeds 100ms once swap-aware shells are included",
        },
    }


def summarize_method_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["requests_total", "sla_miss_count", "hard_miss_count", "sla_miss_rate", "hard_miss_rate", "miss_autohorizon_ratio", "fleet_score", "min_robot_score", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms"]
    summary = {}
    for k in keys:
        vals = [r["summary"][k] for r in runs]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"max_{k}"] = float(np.max(vals))
        summary[f"min_{k}"] = float(np.min(vals))
    method_keys = sorted(set().union(*(r["method_stats"].keys() for r in runs)))
    method_summary = {}
    for k in method_keys:
        vals = [r["method_stats"].get(k, 0.0) for r in runs if isinstance(r["method_stats"].get(k, 0.0), (int, float))]
        if vals:
            method_summary[f"mean_{k}"] = float(np.mean(vals))
    return {"summary": summary, "method_summary": method_summary}


def run_request_methods(workload: str) -> dict[str, Any]:
    methods = [
        ("oracle_full_resident", {}),
        ("reef_like_temporal", {}),
        ("clockwork_like", {}),
        ("paella_like", {}),
        ("usher_like", {}),
        ("distserve_like", {}),
    ]
    vla_fracs = [0.0, 0.1, 0.2, 0.3]
    out = {}
    for method, kwargs in methods:
        runs = [sim_request_level(workload, method, seed, **kwargs) for seed in SEEDS]
        out[method] = {"runs": runs, **summarize_method_runs(runs)}
    best = None
    for frac in vla_fracs:
        runs = [sim_request_level(workload, "vla_aware", seed, shared_fraction=frac) for seed in SEEDS]
        agg = summarize_method_runs(runs)
        cand = (
            agg["summary"]["mean_hard_miss_count"],
            agg["summary"]["mean_sla_miss_count"],
            agg["summary"]["mean_miss_autohorizon_ratio"],
            agg["summary"]["mean_latency_p95_ms"],
            frac,
        )
        if best is None or cand < best[0]:
            best = (cand, frac, runs, agg)
    _, frac, runs, agg = best
    out["vla_aware"] = {"best_shared_fraction": frac, "runs": runs, **agg}
    return out


def main():
    result = {
        "meta": {
            "deadline_ms": SLA_MS,
            "duration_s": SIM_DURATION_S,
            "seeds": SEEDS,
            "methods": [
                "oracle_full_resident",
                "reef_like_temporal",
                "clockwork_like",
                "paella_like",
                "usher_like",
                "distserve_like",
                "gpulet_like",
                "vla_aware",
            ],
            "notes": [
                "Clockwork/REEF/Paella are paper-faithful reproductions of core scheduling ideas, not source-level artifact replays.",
                "USHER-like uses fixed spatial partitions with reactive cold-model replacement and no predictive model-state management.",
                "DistServe-like uses a two-stage disaggregated pipeline on one GPU; this is intentionally conservative because VLA inference is not naturally prefill/decode separable.",
                "GPUlet uses a swap-aware coarse schedule search with batch and partition candidates.",
                "Request time is derived from AutoHorizon target minus the 100ms response budget.",
            ],
        },
        "pi05": {
            "request_level": run_request_methods("pi05"),
            "gpulet_like": analytic_gpulet("pi05"),
        },
        "gr00t_n1d6": {
            "request_level": run_request_methods("gr00t"),
            "gpulet_like": analytic_gpulet("gr00t"),
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "out": str(OUT),
        "pi05_methods": list(result["pi05"]["request_level"].keys()),
        "gr00t_methods": list(result["gr00t_n1d6"]["request_level"].keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
