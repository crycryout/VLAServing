from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tg4perfetto import TraceGenerator

MIRAGE_PYTHON = "/root/autodl-tmp/mirage/python"
STEADY_BENCH = (
    "/root/autodl-tmp/VLAServing/src/gr00t/eval/"
    "bench_gr00t_mpk_steady_state_runtime.py"
)
RESULT_JSON = (
    "/root/autodl-tmp/VLAServing/results/"
    "gr00t_mpk_runtime_breakdown_20260423.json"
)
TRACE_ONLINE = (
    "/root/autodl-tmp/VLAServing/results/"
    "gr00t_mpk_online_notoken_conservative_20260423.perfetto-trace"
)
TRACE_OFFLINE = (
    "/root/autodl-tmp/VLAServing/results/"
    "gr00t_mpk_offline_r8_conservative_20260423.perfetto-trace"
)

if MIRAGE_PYTHON not in sys.path:
    sys.path.insert(0, MIRAGE_PYTHON)

from mirage.mpk.profiler_persistent import decode_tag  # noqa: E402
from mirage.mpk.profiler_persistent import event_name_list as BASE_EVENT_NAME_LIST  # noqa: E402


STEADY_SPEC = importlib.util.spec_from_file_location("steady_bench", STEADY_BENCH)
STEADY_MOD = importlib.util.module_from_spec(STEADY_SPEC)
assert STEADY_SPEC.loader is not None
sys.modules[STEADY_SPEC.name] = STEADY_MOD
STEADY_SPEC.loader.exec_module(STEADY_MOD)

GraphContext = STEADY_MOD.GraphContext
build_gr00t_minidit_pk = STEADY_MOD.build_gr00t_minidit_pk
torch_twoblock_reference = STEADY_MOD.torch_twoblock_reference


EVENT_NAME_LIST = dict(BASE_EVENT_NAME_LIST)
EVENT_NAME_LIST.update(
    {
        122: "TASK_LAYER_NORM",
        123: "TASK_ADA_LAYER_NORM",
        124: "TASK_GELU",
        125: "TASK_BIAS_ADD",
        126: "TASK_ELEMENTWISE_ADD",
        127: "TASK_FULL_ATTENTION",
        128: "TASK_LINEAR_GENERIC",
    }
)

SCHEDULER_EVENT_NAMES = {
    "TASK_SCHD_TASKS",
    "TASK_SCHD_EVENTS",
    "TASK_GET_EVENT",
    "TASK_GET_NEXT_TASK",
}


def make_profiler_tensor() -> torch.Tensor:
    return torch.zeros((3000 * 128,), dtype=torch.int64, device="cuda").contiguous()


def get_stream_ptr() -> int:
    stream = torch.cuda.current_stream()
    if hasattr(stream, "cuda_stream"):
        try:
            return int(stream.cuda_stream)
        except Exception:
            return int(stream.cuda_stream.value)
    raise ValueError("Unable to obtain CUDA stream pointer")


def cuda_elapsed_ms(launch_fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    launch_fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def host_elapsed_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def parse_profiler_tensor(profiler_tensor: torch.Tensor) -> dict[str, Any]:
    profiler_np = profiler_tensor.cpu().numpy().view(np.uint32).reshape(-1, 2)
    num_blocks = int(profiler_np[0, 0])
    num_groups = int(profiler_np[0, 1])

    start_times: dict[tuple[int, int, int, int], int] = {}
    event_totals_us: defaultdict[str, float] = defaultdict(float)
    event_counts: defaultdict[str, int] = defaultdict(int)
    global_min = None
    global_max = None

    for i in range(1, len(profiler_np)):
        raw_tag = int(profiler_np[i, 0])
        raw_ts = int(profiler_np[i, 1])
        if raw_tag == 0 and raw_ts == 0:
            continue
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            raw_tag, num_blocks, num_groups
        )
        event_name = EVENT_NAME_LIST.get(event_idx, f"TASK_{event_idx}")
        key = (block_idx, group_idx, event_idx, event_no)

        if global_min is None or raw_ts < global_min:
            global_min = raw_ts
        if global_max is None or raw_ts > global_max:
            global_max = raw_ts

        if event_type == 0:
            start_times[key] = raw_ts
        elif event_type == 1:
            start = start_times.pop(key, None)
            if start is None:
                continue
            duration_us = max(0, raw_ts - start) / 1000.0
            event_totals_us[event_name] += duration_us
            event_counts[event_name] += 1

    total_trace_span_us = (
        0.0 if global_min is None or global_max is None else (global_max - global_min) / 1000.0
    )
    scheduler_total_us = sum(
        v for k, v in event_totals_us.items() if k in SCHEDULER_EVENT_NAMES
    )
    worker_total_us = sum(
        v for k, v in event_totals_us.items() if k not in SCHEDULER_EVENT_NAMES
    )

    top_events = sorted(
        (
            {
                "event": name,
                "total_us": total_us,
                "count": event_counts[name],
                "avg_us": total_us / max(event_counts[name], 1),
            }
            for name, total_us in event_totals_us.items()
        ),
        key=lambda x: x["total_us"],
        reverse=True,
    )[:20]

    return {
        "num_blocks": num_blocks,
        "num_groups": num_groups,
        "trace_span_us": total_trace_span_us,
        "scheduler_total_us": scheduler_total_us,
        "worker_total_us": worker_total_us,
        "top_events_by_total_us": top_events,
    }


def write_perfetto_trace(profiler_tensor: torch.Tensor, file_name: str) -> None:
    profiler_np = profiler_tensor.cpu().numpy().view(np.uint32).reshape(-1, 2)
    num_blocks = int(profiler_np[0, 0])
    num_groups = int(profiler_np[0, 1])
    tgen = TraceGenerator(file_name)
    tid_map = {}
    track_map = {}

    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_np)):
        tag = int(profiler_np[i, 0])
        timestamp = int(profiler_np[i, 1])
        if tag == 0 and timestamp == 0:
            continue
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        event = EVENT_NAME_LIST.get(event_idx, f"TASK_{event_idx}") + f"_{event_no}"
        tid = tid_map[(block_idx, group_idx)]
        track_key = (block_idx, group_idx, event_idx)
        if track_key not in track_map:
            track_map[track_key] = tid.create_track()
        track = track_map[track_key]
        if event_type == 0:
            track.open(timestamp, event)
        elif event_type == 1:
            track.close(timestamp)
        elif event_type == 2:
            track.instant(timestamp, event)

    tgen.flush()


def profile_case(
    *,
    ctx: GraphContext,
    mode: str,
    total_num_requests: int,
    max_num_batched_requests: int,
    max_num_batched_tokens: int,
    max_num_pages: int,
    num_workers: int,
    num_local_schedulers: int,
    trace_path: str,
):
    profiler_tensor = make_profiler_tensor()
    (
        pk,
        out_buf,
        model,
        hidden_states,
        encoder_hidden_states,
        backbone_attention_mask,
        temb,
    ) = build_gr00t_minidit_pk(
        ctx,
        mode=mode,
        total_num_requests=total_num_requests,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        num_workers=num_workers,
        num_local_schedulers=num_local_schedulers,
        profiler_tensor=profiler_tensor,
        trace_name=None,
    )
    try:
        init_ms = host_elapsed_ms(pk.init_request_func)
        stream_ptr = get_stream_ptr()
        enqueue_ms = host_elapsed_ms(lambda: pk.launch_func(stream_ptr))
        torch.cuda.synchronize()

        # Re-initialize and run the measured/profiled launch.
        pk.init_request_func()
        profiler_tensor.zero_()
        gpu_elapsed_ms = cuda_elapsed_ms(lambda: pk.launch_func(stream_ptr))
        write_perfetto_trace(profiler_tensor, trace_path)
        breakdown = parse_profiler_tensor(profiler_tensor)

        with torch.no_grad():
            ref = torch_twoblock_reference(
                model,
                hidden_states,
                encoder_hidden_states,
                backbone_attention_mask,
                temb,
            ).squeeze(0)
        diff = (out_buf.float() - ref.float()).abs()

        return {
            "mode": mode,
            "total_num_requests": total_num_requests,
            "num_workers": num_workers,
            "num_local_schedulers": num_local_schedulers,
            "init_request_func_wall_ms": init_ms,
            "launch_enqueue_wall_ms": enqueue_ms,
            "gpu_elapsed_ms": gpu_elapsed_ms,
            "correctness": {
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "out_sum": float(out_buf.float().sum().item()),
            },
            "profiler_breakdown": breakdown,
            "trace_path": trace_path,
        }
    finally:
        pk.finalize()


def main():
    ctx = GraphContext()
    auto_workers, auto_schedulers = STEADY_MOD.mirage.get_configurations_from_gpu(0)

    result = {
        "meta": {
            "date": "2026-04-23",
            "device": "RTX 4090",
            "graph": "GR00T N1.6 mini AlternateVLDiT two-block core",
            "notes": [
                "launch_enqueue_wall_ms is host-side submit overhead only.",
                "gpu_elapsed_ms is measured with CUDA events around launch_func().",
                "profiler totals are aggregated across block/group tracks and may exceed wall time.",
            ],
        },
        "online_notoken_conservative": profile_case(
            ctx=ctx,
            mode="online_notoken",
            total_num_requests=1,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=1,
            num_workers=1,
            num_local_schedulers=1,
            trace_path=TRACE_ONLINE,
        ),
        "online_notoken_auto": profile_case(
            ctx=ctx,
            mode="online_notoken",
            total_num_requests=1,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=1,
            num_workers=auto_workers,
            num_local_schedulers=auto_schedulers,
            trace_path=TRACE_ONLINE.replace(
                "conservative", "auto"
            ),
        ),
        "offline_r8_conservative": profile_case(
            ctx=ctx,
            mode="offline",
            total_num_requests=8,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=8,
            num_workers=1,
            num_local_schedulers=1,
            trace_path=TRACE_OFFLINE,
        ),
    }

    Path(RESULT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
