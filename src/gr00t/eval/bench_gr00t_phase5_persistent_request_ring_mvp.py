#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path("/root/autodl-tmp/VLAServing")
OUT = ROOT / "results" / "gr00t_phase5_persistent_request_ring_mvp_20260520.json"
EXT_DIR = ROOT / "results" / "torch_extensions" / "phase5_request_ring"
ADMISSION_SRC = ROOT / "src" / "gr00t" / "eval" / "bench_gr00t_mpk_synthetic_admission_profile.py"


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t err__ = (call);                                                                 \
        if (err__ != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err__));       \
        }                                                                                           \
    } while (0)

struct HostDesc {
    volatile int valid;
    int lane;
    int request_idx;
    long long release_ns;
};

struct HostCompletion {
    volatile int done;
    int lane;
    int request_idx;
    unsigned long long gpu_first;
    unsigned long long gpu_done;
    unsigned long long compute_start;
    unsigned long long compute_end;
    unsigned long long hbm_start;
    unsigned long long hbm_end;
    long long host_visible_ns;
};

__device__ __forceinline__ unsigned long long read_timer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ int atomic_load_i(int* addr) {
    return atomicAdd(addr, 0);
}

__device__ void do_compute_chunk(
    float* out,
    long lane_offset,
    int local_worker,
    int workers_per_lane,
    int tiles_per_stage,
    int tile_span,
    int iters
) {
    int tid = threadIdx.x;
    float x0 = 0.001f * (float)((tid & 255) + 1);
    float x1 = x0 + 0.11f;
    float x2 = x0 + 0.23f;
    float x3 = x0 + 0.37f;
    for (int tile = local_worker; tile < tiles_per_stage; tile += workers_per_lane) {
        long offset = lane_offset + (long)tile * (long)tile_span;
        for (int i = tid; i < tile_span; i += blockDim.x) {
            float v = out[offset + i] + x0;
            #pragma unroll 4
            for (int k = 0; k < iters; ++k) {
                v = fmaf(v, 1.00017f, x1);
                x1 = fmaf(x1, 0.99991f, x2);
                x2 = fmaf(x2, 1.00003f, x3);
                x3 = fmaf(x3, 0.99997f, v);
            }
            out[offset + i] = v + x1 + x2 + x3;
        }
    }
}

__device__ void do_hbm_chunk(
    float* out,
    const float* inp,
    long lane_offset,
    long lane_span,
    int local_worker,
    int workers_per_lane,
    int tiles_per_stage,
    int tile_span,
    int iters,
    int stride_words
) {
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int tile = local_worker; tile < tiles_per_stage; tile += workers_per_lane) {
        long local_tile_offset = (long)tile * (long)tile_span;
        long offset = lane_offset + local_tile_offset;
        for (int i = tid; i < tile_span; i += blockDim.x) {
            long local = local_tile_offset + i;
            long base = offset + i;
            #pragma unroll 1
            for (int k = 0; k < iters; ++k) {
                long j = lane_offset + ((local + (long)k * stride_words) % lane_span);
                acc += inp[j] * 1.0001f;
                out[base] = acc + out[base] * 0.9999f;
            }
        }
    }
}

__device__ void lane_barrier(int* barrier, int idx, int workers_per_lane) {
    if (threadIdx.x == 0) {
        atomicAdd(&barrier[idx], 1);
        while (atomic_load_i(&barrier[idx]) < workers_per_lane) {
            __nanosleep(128);
        }
    }
    __syncthreads();
}

__device__ void acquire_stage_credit(
    int* counters,
    int* flags,
    int flag_idx,
    int stage,
    int credit,
    int local_worker
) {
    if (threadIdx.x == 0) {
        if (local_worker == 0) {
            while (true) {
                int now_active = atomicAdd(&counters[stage], 1) + 1;
                if (now_active <= credit) {
                    atomicMax(&counters[2 + stage], now_active);
                    atomicExch(&flags[flag_idx], 1);
                    break;
                }
                atomicSub(&counters[stage], 1);
                atomicAdd(&counters[6 + stage], 1);
                __nanosleep(128);
            }
        } else {
            while (atomic_load_i(&flags[flag_idx]) == 0) {
                __nanosleep(128);
            }
        }
    }
    __syncthreads();
}

__device__ void release_stage_credit(
    int* counters,
    int* flags,
    int flag_idx,
    int stage,
    int local_worker
) {
    if (threadIdx.x == 0 && local_worker == 0) {
        atomicSub(&counters[stage], 1);
        atomicExch(&flags[flag_idx], 2);
    }
    __syncthreads();
}

__global__ void phase5_request_ring_kernel(
    float* out,
    const float* inp,
    volatile HostDesc* descs,
    volatile HostCompletion* completions,
    int* counters,
    int* barrier,
    int* credit_flags,
    unsigned long long* summary_times,
    int lane_count,
    int requests_per_lane,
    long lane_span,
    int tiles_per_stage,
    int tile_span,
    int workers_per_lane,
    int threads,
    int compute_iters,
    int hbm_iters,
    int stride_words,
    int compute_credit,
    int hbm_credit
) {
    int lane = blockIdx.x / workers_per_lane;
    int local_worker = blockIdx.x - lane * workers_per_lane;
    if (lane >= lane_count) {
        return;
    }
    long lane_offset = (long)lane * lane_span;
    if (threadIdx.x == 0) {
        atomicMin(&summary_times[0], read_timer());
    }

    for (int req = 0; req < requests_per_lane; ++req) {
        int global_req = req * lane_count + lane;
        while (descs[global_req].valid == 0) {
            __nanosleep(128);
        }
        __syncthreads();

        unsigned long long first = read_timer();
        int compute_flag = global_req * 2;
        int hbm_flag = global_req * 2 + 1;

        acquire_stage_credit(counters, credit_flags, compute_flag, 0, compute_credit, local_worker);
        unsigned long long cs = read_timer();
        do_compute_chunk(out, lane_offset, local_worker, workers_per_lane, tiles_per_stage, tile_span, compute_iters);
        unsigned long long ce = read_timer();
        lane_barrier(barrier, compute_flag, workers_per_lane);
        release_stage_credit(counters, credit_flags, compute_flag, 0, local_worker);

        acquire_stage_credit(counters, credit_flags, hbm_flag, 1, hbm_credit, local_worker);
        unsigned long long hs = read_timer();
        do_hbm_chunk(out, inp, lane_offset, lane_span, local_worker, workers_per_lane, tiles_per_stage, tile_span, hbm_iters, stride_words);
        unsigned long long he = read_timer();
        lane_barrier(barrier, hbm_flag, workers_per_lane);
        release_stage_credit(counters, credit_flags, hbm_flag, 1, local_worker);

        if (threadIdx.x == 0 && local_worker == 0) {
            completions[global_req].lane = lane;
            completions[global_req].request_idx = req;
            completions[global_req].gpu_first = first;
            completions[global_req].compute_start = cs;
            completions[global_req].compute_end = ce;
            completions[global_req].hbm_start = hs;
            completions[global_req].hbm_end = he;
            completions[global_req].gpu_done = read_timer();
            __threadfence_system();
            completions[global_req].done = 1;
            atomicAdd(&counters[10], 1);
        }
    }

    if (threadIdx.x == 0) {
        atomicMax(&summary_times[1], read_timer());
    }
}

static inline long long ns_since(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1
) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

void wait_until_ns(std::chrono::high_resolution_clock::time_point t0, long long target_ns) {
    auto target = t0 + std::chrono::nanoseconds(target_ns);
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto remain = std::chrono::duration_cast<std::chrono::microseconds>(target - now).count();
        if (remain <= 80) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(std::min<long long>(remain - 40, 200)));
    }
    while (std::chrono::high_resolution_clock::now() < target) {
    }
}

py::dict run_request_ring(
    int lane_count,
    int requests_per_lane,
    int workers_per_lane,
    int tiles_per_stage,
    int tile_span,
    int threads,
    int compute_iters,
    int hbm_iters,
    int stride_words,
    int compute_credit,
    int hbm_credit,
    double phase_gap_ms,
    int release_mode,
    double initial_delay_ms
) {
    int total_requests = lane_count * requests_per_lane;
    long lane_span = (long)tiles_per_stage * (long)tile_span;
    long total_words = (long)lane_count * lane_span;
    int blocks = lane_count * workers_per_lane;
    size_t bytes = (size_t)total_words * sizeof(float);

    std::vector<HostDesc> h_desc((size_t)total_requests);
    HostCompletion* h_comp = nullptr;
    HostDesc* d_desc = nullptr;
    HostCompletion* d_comp = nullptr;
    memset(h_desc.data(), 0, (size_t)total_requests * sizeof(HostDesc));
    CUDA_CHECK(cudaHostAlloc((void**)&h_comp, (size_t)total_requests * sizeof(HostCompletion), cudaHostAllocMapped));
    memset(h_comp, 0, (size_t)total_requests * sizeof(HostCompletion));

    float* d_out = nullptr;
    float* d_inp = nullptr;
    int* d_counters = nullptr;
    int* d_barrier = nullptr;
    int* d_credit_flags = nullptr;
    unsigned long long* d_summary = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_desc, (size_t)total_requests * sizeof(HostDesc)));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_comp, h_comp, 0));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inp, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_counters, 12 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_barrier, (size_t)total_requests * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_credit_flags, (size_t)total_requests * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_summary, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_desc, 0, (size_t)total_requests * sizeof(HostDesc)));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    CUDA_CHECK(cudaMemset(d_inp, 0, bytes));
    CUDA_CHECK(cudaMemset(d_counters, 0, 12 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_barrier, 0, (size_t)total_requests * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_credit_flags, 0, (size_t)total_requests * 2 * sizeof(int)));
    unsigned long long init_summary[2] = {0x7fffffffffffffffULL, 0ULL};
    CUDA_CHECK(cudaMemcpy(d_summary, init_summary, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStream_t copy_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event, stream));
    phase5_request_ring_kernel<<<blocks, threads, 0, stream>>>(
        d_out,
        d_inp,
        d_desc,
        d_comp,
        d_counters,
        d_barrier,
        d_credit_flags,
        d_summary,
        lane_count,
        requests_per_lane,
        lane_span,
        tiles_per_stage,
        tile_span,
        workers_per_lane,
        threads,
        compute_iters,
        hbm_iters,
        stride_words,
        compute_credit,
        hbm_credit
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event, stream));

    auto host_start = std::chrono::high_resolution_clock::now();
    long long initial_ns = (long long)(initial_delay_ms * 1000000.0);
    long long phase_gap_ns = (long long)(phase_gap_ms * 1000000.0);
    int completed = 0;
    std::vector<char> seen((size_t)total_requests, 0);

    auto scan_completions = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        long long now_ns = ns_since(host_start, now);
        for (int i = 0; i < total_requests; ++i) {
            if (!seen[(size_t)i] && h_comp[i].done != 0) {
                seen[(size_t)i] = 1;
                h_comp[i].host_visible_ns = now_ns;
                completed++;
            }
        }
    };

    for (int req = 0; req < requests_per_lane; ++req) {
        for (int lane = 0; lane < lane_count; ++lane) {
            int global_req = req * lane_count + lane;
            long long target_ns = initial_ns;
            if (release_mode == 1) {
                target_ns += (long long)(req * lane_count + lane) * phase_gap_ns;
            } else if (release_mode == 0) {
                target_ns += (long long)req * (long long)lane_count * phase_gap_ns;
            }
            if (release_mode != 2) {
                wait_until_ns(host_start, target_ns);
            }
            auto release_now = std::chrono::high_resolution_clock::now();
            h_desc[(size_t)global_req].lane = lane;
            h_desc[(size_t)global_req].request_idx = req;
            h_desc[(size_t)global_req].release_ns = ns_since(host_start, release_now);
            std::atomic_thread_fence(std::memory_order_release);
            h_desc[(size_t)global_req].valid = 1;
            CUDA_CHECK(cudaMemcpyAsync(
                d_desc + global_req,
                &h_desc[(size_t)global_req],
                sizeof(HostDesc),
                cudaMemcpyHostToDevice,
                copy_stream
            ));
            CUDA_CHECK(cudaStreamSynchronize(copy_stream));
            scan_completions();
        }
    }

    while (completed < total_requests) {
        scan_completions();
        if (completed < total_requests) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float event_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&event_ms, start_event, stop_event));

    int h_counters[12];
    unsigned long long h_summary[2];
    CUDA_CHECK(cudaMemcpy(h_counters, d_counters, 12 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_summary, d_summary, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    py::list release_ns;
    py::list host_visible_ns;
    py::list gpu_first;
    py::list gpu_done;
    py::list compute_start;
    py::list compute_end;
    py::list hbm_start;
    py::list hbm_end;
    py::list lanes;
    py::list reqs;
    for (int i = 0; i < total_requests; ++i) {
        release_ns.append(h_desc[(size_t)i].release_ns);
        host_visible_ns.append(h_comp[i].host_visible_ns);
        gpu_first.append(h_comp[i].gpu_first);
        gpu_done.append(h_comp[i].gpu_done);
        compute_start.append(h_comp[i].compute_start);
        compute_end.append(h_comp[i].compute_end);
        hbm_start.append(h_comp[i].hbm_start);
        hbm_end.append(h_comp[i].hbm_end);
        lanes.append(h_comp[i].lane);
        reqs.append(h_comp[i].request_idx);
    }
    py::list counters;
    for (int i = 0; i < 12; ++i) {
        counters.append(h_counters[i]);
    }
    py::list summary;
    summary.append(h_summary[0]);
    summary.append(h_summary[1]);

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaFree(d_desc));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_counters));
    CUDA_CHECK(cudaFree(d_barrier));
    CUDA_CHECK(cudaFree(d_credit_flags));
    CUDA_CHECK(cudaFree(d_summary));
    CUDA_CHECK(cudaFreeHost(h_comp));
    py::dict result;
    result["event_ms"] = event_ms;
    result["lane_count"] = lane_count;
    result["requests_per_lane"] = requests_per_lane;
    result["workers_per_lane"] = workers_per_lane;
    result["tiles_per_stage"] = tiles_per_stage;
    result["tile_span"] = tile_span;
    result["threads"] = threads;
    result["compute_iters"] = compute_iters;
    result["hbm_iters"] = hbm_iters;
    result["compute_credit"] = compute_credit;
    result["hbm_credit"] = hbm_credit;
    result["phase_gap_ms"] = phase_gap_ms;
    result["release_mode"] = release_mode;
    result["release_ns"] = release_ns;
    result["host_visible_ns"] = host_visible_ns;
    result["gpu_first"] = gpu_first;
    result["gpu_done"] = gpu_done;
    result["compute_start"] = compute_start;
    result["compute_end"] = compute_end;
    result["hbm_start"] = hbm_start;
    result["hbm_end"] = hbm_end;
    result["lanes"] = lanes;
    result["request_indices"] = reqs;
    result["counters"] = counters;
    result["summary_times"] = summary;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_request_ring", &run_request_ring, "Phase-5 persistent request-ring MVP");
}
"""


@dataclass
class RingCfg:
    lane_count: int = 5
    requests_per_lane: int = 24
    workers_per_lane: int = 0
    tiles_per_stage: int = 72
    tile_span: int = 16_384
    threads: int = 128
    compute_iters: int = 180
    hbm_iters: int = 8
    stride_words: int = 4099
    initial_delay_ms: float = 2.0
    repeat: int = 3


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(module_name, path)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_ext():
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="gr00t_phase5_persistent_request_ring_mvp_ext_v3",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(EXT_DIR),
        verbose=False,
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def stat(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.fmean(values)) if values else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "num_samples": len(values),
    }


def as_int_list(value: Any) -> list[int]:
    return [int(x) for x in value]


def parse_raw(raw: dict[str, Any], trim: int = 3) -> dict[str, Any]:
    release_ns = as_int_list(raw["release_ns"])
    visible_ns = as_int_list(raw["host_visible_ns"])
    gpu_first = as_int_list(raw["gpu_first"])
    gpu_done = as_int_list(raw["gpu_done"])
    compute_start = as_int_list(raw["compute_start"])
    compute_end = as_int_list(raw["compute_end"])
    hbm_start = as_int_list(raw["hbm_start"])
    hbm_end = as_int_list(raw["hbm_end"])
    lanes = as_int_list(raw["lanes"])
    reqs = as_int_list(raw["request_indices"])
    counters = as_int_list(raw["counters"])
    summary = as_int_list(raw["summary_times"])
    event_ms = float(raw["event_ms"])
    timer_units_per_ms = float(max(1, summary[1] - summary[0])) / max(event_ms, 1e-9)
    rows = []
    for idx in range(len(release_ns)):
        active_ms = max(0.0, (gpu_done[idx] - gpu_first[idx]) / timer_units_per_ms)
        compute_ms = max(0.0, (compute_end[idx] - compute_start[idx]) / timer_units_per_ms)
        hbm_ms = max(0.0, (hbm_end[idx] - hbm_start[idx]) / timer_units_per_ms)
        host_e2e_ms = max(0.0, (visible_ns[idx] - release_ns[idx]) / 1_000_000.0)
        rows.append(
            {
                "global_request": idx,
                "lane": lanes[idx],
                "request_index": reqs[idx],
                "release_ms": release_ns[idx] / 1_000_000.0,
                "host_visible_ms": visible_ns[idx] / 1_000_000.0,
                "host_e2e_ms": host_e2e_ms,
                "gpu_active_ms": active_ms,
                "queue_plus_visibility_ms": max(0.0, host_e2e_ms - active_ms),
                "compute_ms": compute_ms,
                "hbm_ms": hbm_ms,
            }
        )
    steady = [
        row
        for row in rows
        if trim <= int(row["request_index"]) < int(raw["requests_per_lane"]) - trim
    ]
    done_times = sorted(float(row["host_visible_ms"]) for row in steady)
    done_intervals = [b - a for a, b in zip(done_times, done_times[1:])]
    return {
        "event_ms": event_ms,
        "timer_units_per_ms": timer_units_per_ms,
        "completed_requests": len(rows),
        "steady_requests": len(steady),
        "host_e2e_ms": stat([float(row["host_e2e_ms"]) for row in steady]),
        "gpu_active_ms": stat([float(row["gpu_active_ms"]) for row in steady]),
        "queue_plus_visibility_ms": stat([float(row["queue_plus_visibility_ms"]) for row in steady]),
        "compute_ms": stat([float(row["compute_ms"]) for row in steady]),
        "hbm_ms": stat([float(row["hbm_ms"]) for row in steady]),
        "done_interval_ms": stat(done_intervals),
        "max_same_stage_lanes": {
            "compute": counters[2],
            "hbm": counters[3],
        },
        "credit_denials": {
            "compute": counters[6],
            "hbm": counters[7],
        },
        "samples": steady[:12],
    }


def summarize_runs(name: str, raws: list[dict[str, Any]], trim: int = 3) -> dict[str, Any]:
    parsed = [parse_raw(raw, trim=trim) for raw in raws]
    return {
        "name": name,
        "config": {
            "lane_count": int(raws[0]["lane_count"]),
            "requests_per_lane": int(raws[0]["requests_per_lane"]),
            "workers_per_lane": int(raws[0]["workers_per_lane"]),
            "tiles_per_stage": int(raws[0]["tiles_per_stage"]),
            "tile_span": int(raws[0]["tile_span"]),
            "threads": int(raws[0]["threads"]),
            "compute_iters": int(raws[0]["compute_iters"]),
            "hbm_iters": int(raws[0]["hbm_iters"]),
            "compute_credit": int(raws[0]["compute_credit"]),
            "hbm_credit": int(raws[0]["hbm_credit"]),
            "phase_gap_ms": float(raws[0]["phase_gap_ms"]),
            "release_mode": int(raws[0]["release_mode"]),
        },
        "event_ms": stat([float(row["event_ms"]) for row in parsed]),
        "host_e2e_ms": stat([float(row["host_e2e_ms"]["p50"]) for row in parsed]),
        "host_e2e_p95_ms": stat([float(row["host_e2e_ms"]["p95"]) for row in parsed]),
        "gpu_active_ms": stat([float(row["gpu_active_ms"]["p50"]) for row in parsed]),
        "gpu_active_p95_ms": stat([float(row["gpu_active_ms"]["p95"]) for row in parsed]),
        "queue_plus_visibility_ms": stat([float(row["queue_plus_visibility_ms"]["p50"]) for row in parsed]),
        "queue_plus_visibility_p95_ms": stat(
            [float(row["queue_plus_visibility_ms"]["p95"]) for row in parsed]
        ),
        "compute_ms": stat([float(row["compute_ms"]["p50"]) for row in parsed]),
        "hbm_ms": stat([float(row["hbm_ms"]["p50"]) for row in parsed]),
        "done_interval_ms": stat([float(row["done_interval_ms"]["p50"]) for row in parsed]),
        "max_same_stage_lanes": {
            "compute": stat([float(row["max_same_stage_lanes"]["compute"]) for row in parsed]),
            "hbm": stat([float(row["max_same_stage_lanes"]["hbm"]) for row in parsed]),
        },
        "credit_denials": {
            "compute": stat([float(row["credit_denials"]["compute"]) for row in parsed]),
            "hbm": stat([float(row["credit_denials"]["hbm"]) for row in parsed]),
        },
        "runs": parsed,
    }


def run_mode(
    ext: Any,
    cfg: RingCfg,
    *,
    name: str,
    lane_count: int,
    release_mode: int,
    phase_gap_ms: float,
    compute_credit: int,
    hbm_credit: int,
) -> dict[str, Any]:
    raws = []
    for _ in range(cfg.repeat):
        raw = ext.run_request_ring(
            int(lane_count),
            int(cfg.requests_per_lane),
            int(cfg.workers_per_lane),
            int(cfg.tiles_per_stage),
            int(cfg.tile_span),
            int(cfg.threads),
            int(cfg.compute_iters),
            int(cfg.hbm_iters),
            int(cfg.stride_words),
            int(compute_credit),
            int(hbm_credit),
            float(phase_gap_ms),
            int(release_mode),
            float(cfg.initial_delay_ms),
        )
        raws.append(dict(raw))
    return summarize_runs(name, raws)


def load_admission_module():
    spec = importlib.util.spec_from_file_location("phase5_ring_admission", ADMISSION_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_admission(runtime: dict[str, Any]) -> dict[str, Any]:
    base = load_admission_module()
    base.install_exact_horizon_policy()
    curves = base.build_synthetic_curves()
    no_mpk_curve = {int(k): float(v) for k, v in curves["baseline_fused_curve_ms"].items()}
    service_fn = base.make_service_curve_fn(no_mpk_curve)

    no_mpk = base.run_policy("no_mpk_fused_conservative", service_fn)
    base.add_batch_histogram_to_summary(no_mpk)

    solo = runtime["runtime_summary"]["solo_immediate"]
    phase_credit = runtime["runtime_summary"]["phase_credit3"]
    phase_nocredit = runtime["runtime_summary"]["phase_no_credit"]

    solo_active = float(solo["gpu_active_ms"]["p50"])
    p50_slowdown = float(phase_credit["gpu_active_ms"]["p50"]) / max(solo_active, 1e-9)
    p95_slowdown = float(phase_credit["gpu_active_p95_ms"]["p50"]) / max(
        float(solo["gpu_active_p95_ms"]["p50"]), 1e-9
    )
    no_credit_slowdown = float(phase_nocredit["gpu_active_ms"]["p50"]) / max(solo_active, 1e-9)

    def run_phase(label: str, service_ms: float) -> dict[str, Any]:
        agg = lambda specs, duration_s, seeds: base.aggregate_phase_lanes(
            specs,
            duration_s,
            seeds,
            lane_count=5,
            lane_service_ms=float(service_ms),
        )
        result = base.run_policy(label, aggregate_fn=agg)
        base.add_batch_histogram_to_summary(result)
        return result

    phase_p50 = run_phase(
        "phase5_persistent_ring_measured_p50_slowdown",
        no_mpk_curve[1] * p50_slowdown,
    )
    phase_p95 = run_phase(
        "phase5_persistent_ring_measured_p95_slowdown",
        no_mpk_curve[1] * p95_slowdown,
    )
    phase_no_credit = run_phase(
        "phase5_persistent_ring_no_credit_slowdown",
        no_mpk_curve[1] * no_credit_slowdown,
    )
    phase_batch4 = run_phase("phase5_equal_latency_batch4_reference", no_mpk_curve[4])
    phase_batch5 = run_phase("phase5_equal_latency_batch5_reference", no_mpk_curve[5])

    for result in [phase_p50, phase_p95, phase_no_credit, phase_batch4, phase_batch5]:
        base.add_batch_histogram_to_summary(result)

    return {
        "service_curve_ms": {str(k): float(v) for k, v in sorted(no_mpk_curve.items())},
        "measured_slowdown": {
            "phase_credit3_gpu_active_p50_over_solo": p50_slowdown,
            "phase_credit3_gpu_active_p95_over_solo": p95_slowdown,
            "phase_no_credit_gpu_active_p50_over_solo": no_credit_slowdown,
            "phase_credit3_lane_service_p50_scaled_ms": no_mpk_curve[1] * p50_slowdown,
            "phase_credit3_lane_service_p95_scaled_ms": no_mpk_curve[1] * p95_slowdown,
            "phase_no_credit_lane_service_scaled_ms": no_mpk_curve[1] * no_credit_slowdown,
        },
        "admission_results": {
            "no_mpk_fused_conservative": no_mpk,
            "phase5_persistent_ring_measured_p50_slowdown": phase_p50,
            "phase5_persistent_ring_measured_p95_slowdown": phase_p95,
            "phase5_persistent_ring_no_credit_slowdown": phase_no_credit,
            "phase5_equal_latency_batch4_reference": phase_batch4,
            "phase5_equal_latency_batch5_reference": phase_batch5,
        },
        "summary": {
            "no_mpk_fused_conservative": no_mpk["summary"],
            "phase5_persistent_ring_measured_p50_slowdown": phase_p50["summary"],
            "phase5_persistent_ring_measured_p95_slowdown": phase_p95["summary"],
            "phase5_persistent_ring_no_credit_slowdown": phase_no_credit["summary"],
            "phase5_equal_latency_batch4_reference": phase_batch4["summary"],
            "phase5_equal_latency_batch5_reference": phase_batch5["summary"],
            "p50_slowdown_vs_no_mpk": base.compare_summaries(
                phase_p50["summary"], no_mpk["summary"]
            ),
            "p95_slowdown_vs_no_mpk": base.compare_summaries(
                phase_p95["summary"], no_mpk["summary"]
            ),
            "no_credit_slowdown_vs_no_mpk": base.compare_summaries(
                phase_no_credit["summary"], no_mpk["summary"]
            ),
        },
    }


def compact_admission(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "mean_final_robot_count",
        "mean_p95_ms",
        "mean_queued_request_to_result_p95_ms",
        "mean_queue_wait_after_request_p95_ms",
        "mean_batch_size",
        "accept_rate_gap",
        "mean_lane_utilization",
    ]
    return {
        name: {key: value.get(key) for key in keys}
        for name, value in summary.items()
        if isinstance(value, dict) and "mean_final_robot_count" in value
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    sm_count = int(props.multi_processor_count)
    cfg = RingCfg()
    cfg.requests_per_lane = int(os.environ.get("RING_REQS_PER_LANE", str(cfg.requests_per_lane)))
    cfg.tiles_per_stage = int(os.environ.get("RING_TILES_PER_STAGE", str(cfg.tiles_per_stage)))
    cfg.tile_span = int(os.environ.get("RING_TILE_SPAN", str(cfg.tile_span)))
    cfg.threads = int(os.environ.get("RING_THREADS", str(cfg.threads)))
    cfg.compute_iters = int(os.environ.get("RING_COMPUTE_ITERS", str(cfg.compute_iters)))
    cfg.hbm_iters = int(os.environ.get("RING_HBM_ITERS", str(cfg.hbm_iters)))
    cfg.repeat = int(os.environ.get("RING_REPEAT", str(cfg.repeat)))
    cfg.workers_per_lane = int(
        os.environ.get("RING_WORKERS_PER_LANE", str(max(1, math.ceil(sm_count * 0.30))))
    )
    out_path = Path(os.environ.get("RING_OUT", str(OUT)))

    ext = load_ext()
    # A tiny warmup keeps the measured modes from paying lazy CUDA setup costs.
    _ = run_mode(
        ext,
        cfg,
        name="warmup",
        lane_count=1,
        release_mode=2,
        phase_gap_ms=0.0,
        compute_credit=1,
        hbm_credit=1,
    )

    solo = run_mode(
        ext,
        cfg,
        name="solo_immediate",
        lane_count=1,
        release_mode=2,
        phase_gap_ms=0.0,
        compute_credit=1,
        hbm_credit=1,
    )
    solo_active_ms = float(solo["gpu_active_ms"]["p50"])
    phase_gap_ms = float(os.environ.get("RING_PHASE_GAP_MS", str(max(0.05, solo_active_ms / 5.0))))

    modes = {
        "burst_no_credit": run_mode(
            ext,
            cfg,
            name="burst_no_credit",
            lane_count=5,
            release_mode=0,
            phase_gap_ms=phase_gap_ms,
            compute_credit=5,
            hbm_credit=5,
        ),
        "burst_credit3": run_mode(
            ext,
            cfg,
            name="burst_credit3",
            lane_count=5,
            release_mode=0,
            phase_gap_ms=phase_gap_ms,
            compute_credit=3,
            hbm_credit=3,
        ),
        "phase_no_credit": run_mode(
            ext,
            cfg,
            name="phase_no_credit",
            lane_count=5,
            release_mode=1,
            phase_gap_ms=phase_gap_ms,
            compute_credit=5,
            hbm_credit=5,
        ),
        "phase_credit3": run_mode(
            ext,
            cfg,
            name="phase_credit3",
            lane_count=5,
            release_mode=1,
            phase_gap_ms=phase_gap_ms,
            compute_credit=3,
            hbm_credit=3,
        ),
        "phase_credit2": run_mode(
            ext,
            cfg,
            name="phase_credit2",
            lane_count=5,
            release_mode=1,
            phase_gap_ms=phase_gap_ms,
            compute_credit=2,
            hbm_credit=2,
        ),
    }
    runtime_summary = {"solo_immediate": solo, **modes}
    runtime_payload = {
        "meta": {
            "date": "2026-05-20",
            "scope": (
                "Long-running five-phase persistent request-ring MVP. CPU writes "
                "small descriptors into a device-memory ring with H2D descriptor updates; "
                "one GPU persistent kernel polls the ring, runs phase-shifted lanes "
                "with dummy compute/HBM stages, and writes completion records. This "
                "is a real GPU persistent runtime mechanism with no per-request kernel "
                "launch, not a GR00T VLM/DiT compute implementation."
            ),
            "gpu": props.name,
            "sm_count": sm_count,
            "config": cfg.__dict__,
            "phase_gap_ms": phase_gap_ms,
            "release_modes": {
                "0": "burst: all lanes released together each service period",
                "1": "phase: one lane released every phase gap",
                "2": "immediate: descriptor available immediately",
            },
        },
        "runtime_summary": runtime_summary,
    }
    admission = run_admission(runtime_payload)
    payload = {**runtime_payload, "admission_from_measured_slowdown": admission}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    compact_runtime = {
        name: {
            "host_e2e_p50_ms": value["host_e2e_ms"]["p50"],
            "gpu_active_p50_ms": value["gpu_active_ms"]["p50"],
            "queue_plus_visibility_p50_ms": value["queue_plus_visibility_ms"]["p50"],
            "done_interval_p50_ms": value["done_interval_ms"]["p50"],
            "max_compute": value["max_same_stage_lanes"]["compute"]["p50"],
            "max_hbm": value["max_same_stage_lanes"]["hbm"]["p50"],
        }
        for name, value in runtime_summary.items()
    }
    print(out_path)
    print(json.dumps({"runtime": compact_runtime}, indent=2))
    print(json.dumps({"admission": compact_admission(admission["summary"])}, indent=2))


if __name__ == "__main__":
    main()
