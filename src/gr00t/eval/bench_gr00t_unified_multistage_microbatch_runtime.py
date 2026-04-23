#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path("/root/autodl-tmp/VLAServing")
ISAAC_ROOT = Path("/root/autodl-tmp/Isaac-GR00T")
DEPLOYMENT_SCRIPTS_DIR = ISAAC_ROOT / "scripts" / "deployment"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_ROOT))
if str(DEPLOYMENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_SCRIPTS_DIR))

from benchmark_segmented_batch_n1d6 import Config as SegmentConfig
from benchmark_segmented_batch_n1d6 import _load_model
from benchmark_segmented_batch_n1d6 import _make_dummy_inputs
from benchmark_segmented_batch_n1d6 import _maybe_override_inference_steps
from benchmark_segmented_batch_n1d6 import _move_batch_to_device
from benchmark_segmented_batch_n1d6 import _torch_dtype


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "num_samples": int(arr.size),
    }


def _mark_step_begin() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def _cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    ts = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - ts) * 1000.0


def pixel_shuffle_back(
    vit_embeds: torch.Tensor,
    spatial_shapes: torch.Tensor,
    downsample_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, channels = vit_embeds.shape
    assert batch == 1, f"expected batch=1 before pixel shuffle, got {batch}"
    shapes = spatial_shapes.tolist()
    lengths = [h * w for h, w in shapes]
    slices = torch.split(vit_embeds.view(-1, channels), lengths, dim=0)
    features = [
        sl.transpose(0, 1).reshape(channels, h, w)
        for sl, (h, w) in zip(slices, shapes)
    ]
    down_feats = [None] * len(features)
    grouped: dict[tuple[int, int], list[int]] = {}
    for pos, (h, w) in enumerate(shapes):
        grouped.setdefault((h, w), []).append(pos)
    for (h, w), idxs in grouped.items():
        grp = torch.stack([features[i] for i in idxs], dim=0)
        out = F.pixel_unshuffle(grp, downscale_factor=int(1 / downsample_ratio))
        out = out.flatten(start_dim=2).transpose(1, 2)
        for i, feat in zip(idxs, out):
            down_feats[i] = feat
    down_feats = torch.cat(down_feats, dim=0).unsqueeze(0)
    return down_feats, (spatial_shapes * downsample_ratio).to(torch.int32)


@dataclass
class PreparedTask:
    request_id: int
    slot_id: int
    arrival_ms: float
    backbone_features: torch.Tensor
    backbone_attention_mask: torch.Tensor
    image_mask: torch.Tensor
    state_features: torch.Tensor
    embodiment_id: torch.Tensor
    current_actions: torch.Tensor
    step_idx: int = 0
    service_accum_ms: float = 0.0


class UnifiedRuntime:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str,
        *,
        num_slots: int,
        inference_steps: int,
        local_files_only: bool,
    ) -> None:
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        self.device = device
        self.dtype = _torch_dtype(dtype)
        self.num_slots = num_slots
        self.inference_steps = inference_steps

        bench_cfg = SegmentConfig(
            model_paths=[model_path],
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            single_model_batch_size=num_slots,
            segmented_batch_size=num_slots,
            inference_steps=inference_steps,
        )
        self.loaded = _load_model(model_path, bench_cfg, move_to_device=True, device=device)
        _maybe_override_inference_steps(self.loaded.model, inference_steps)

        self.model = self.loaded.model
        self.backbone = self.model.backbone
        self.action_head = self.model.action_head
        eagle = self.backbone.model
        self.vision_model = eagle.vision_model
        self.projector = eagle.mlp1
        self.embed_tokens = eagle.language_model.get_input_embeddings()
        self.llm_body = eagle.language_model.model
        self.image_token_index = int(eagle.image_token_index)
        self.downsample_ratio = float(eagle.downsample_ratio)

        dummy_inputs = _move_batch_to_device(
            _make_dummy_inputs(self.model, bench_cfg, batch_size=num_slots),
            device,
            self.dtype,
        )
        _, action_inputs = self.model.prepare_input(dict(dummy_inputs))

        self.input_pool = dummy_inputs
        self.action_input_pool = {
            "state": action_inputs["state"].contiguous(),
            "embodiment_id": action_inputs["embodiment_id"].contiguous(),
        }
        self.dt = 1.0 / self.action_head.num_inference_timesteps
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(20260423)

    def enable_compilation(
        self,
        *,
        compile_llm: bool,
        compile_dit: bool,
        compile_backend: str,
        compile_mode: str,
    ) -> None:
        eagle = self.backbone.model
        if compile_llm:
            self.llm_body = torch.compile(
                self.llm_body,
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            eagle.language_model.model = self.llm_body
        if compile_dit:
            self.action_head.model = torch.compile(
                self.action_head.model,
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )

    def slot_inputs(self, slot_id: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.input_pool.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_slots:
                out[key] = value[slot_id : slot_id + 1]
            else:
                out[key] = value
        return out

    def gather_inputs(self, slot_ids: list[int]) -> dict[str, Any]:
        idx = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        out: dict[str, Any] = {}
        for key, value in self.input_pool.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.num_slots:
                out[key] = value.index_select(0, idx)
            else:
                out[key] = value
        return out

    def action_input_for_slot(self, slot_id: int) -> BatchFeature:
        return BatchFeature(
            data={
                "state": self.action_input_pool["state"][slot_id : slot_id + 1],
                "embodiment_id": self.action_input_pool["embodiment_id"][slot_id : slot_id + 1],
            }
        )

    @torch.inference_mode()
    def whole_request_batch(self, slot_ids: list[int]) -> torch.Tensor:
        _mark_step_begin()
        inputs = self.gather_inputs(slot_ids)
        out = self.model.get_action(dict(inputs))
        return out["action_pred"]

    @torch.inference_mode()
    def staged_backbone_single(self, slot_id: int) -> tuple[BatchFeature, dict[str, float]]:
        inputs = self.slot_inputs(slot_id)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        stage_ms: dict[str, float] = {}

        _mark_step_begin()
        torch.cuda.synchronize()
        ts = time.perf_counter()
        vision_out = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        vit_embeds = vision_out.last_hidden_state
        spatial_shapes = vision_out.spatial_shapes
        vit_embeds, _ = pixel_shuffle_back(vit_embeds, spatial_shapes, self.downsample_ratio)
        torch.cuda.synchronize()
        stage_ms["vision"] = (time.perf_counter() - ts) * 1000.0

        torch.cuda.synchronize()
        ts = time.perf_counter()
        vit_embeds = self.projector(vit_embeds)
        batch, tokens, hidden = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(batch * tokens, hidden)
        torch.cuda.synchronize()
        stage_ms["projector"] = (time.perf_counter() - ts) * 1000.0

        torch.cuda.synchronize()
        ts = time.perf_counter()
        input_embeds = self.embed_tokens(input_ids)
        batch, seq, hidden = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch * seq, hidden)
        selected = input_ids.reshape(batch * seq) == self.image_token_index
        n = int(selected.sum().item())
        flat_embeds[selected] = vit_embeds[:n]
        input_embeds = flat_embeds.reshape(batch, seq, hidden)
        torch.cuda.synchronize()
        stage_ms["fuse"] = (time.perf_counter() - ts) * 1000.0

        _mark_step_begin()
        torch.cuda.synchronize()
        ts = time.perf_counter()
        hidden = self.llm_body(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        ).last_hidden_state
        torch.cuda.synchronize()
        stage_ms["llm"] = (time.perf_counter() - ts) * 1000.0

        backbone_output = BatchFeature(
            data={
                "backbone_features": hidden,
                "backbone_attention_mask": attention_mask == 1,
                "image_mask": input_ids == self.image_token_index,
            }
        )
        return backbone_output, stage_ms

    @torch.inference_mode()
    def prepare_task(self, request_id: int, slot_id: int, arrival_ms: float) -> tuple[PreparedTask, dict[str, float]]:
        backbone_output, stage_ms = self.staged_backbone_single(slot_id)
        action_input = self.action_input_for_slot(slot_id)
        features = self.action_head._encode_features(backbone_output, action_input)
        current_actions = torch.randn(
            (1, self.action_head.config.action_horizon, self.action_head.action_dim),
            device=self.device,
            dtype=features.backbone_features.dtype,
            generator=self.generator,
        )
        task = PreparedTask(
            request_id=request_id,
            slot_id=slot_id,
            arrival_ms=arrival_ms,
            backbone_features=features.backbone_features,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
            image_mask=backbone_output.image_mask,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            current_actions=current_actions,
        )
        return task, stage_ms

    @torch.inference_mode()
    def run_step_batch(self, tasks: list[PreparedTask]) -> dict[str, float]:
        if not tasks:
            raise ValueError("tasks must be non-empty")
        step_idx = tasks[0].step_idx
        if any(task.step_idx != step_idx for task in tasks):
            raise ValueError("all tasks in a step batch must share the same step index")

        backbone_features = torch.cat([task.backbone_features for task in tasks], dim=0)
        state_features = torch.cat([task.state_features for task in tasks], dim=0)
        embodiment_id = torch.cat([task.embodiment_id for task in tasks], dim=0)
        image_mask = torch.cat([task.image_mask for task in tasks], dim=0)
        backbone_attention_mask = torch.cat([task.backbone_attention_mask for task in tasks], dim=0)
        actions = torch.cat([task.current_actions for task in tasks], dim=0)

        t_cont = step_idx / float(self.action_head.num_inference_timesteps)
        t_discretized = int(t_cont * self.action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            (len(tasks),),
            fill_value=t_discretized,
            device=self.device,
            dtype=torch.long,
        )

        _mark_step_begin()
        torch.cuda.synchronize()
        ts = time.perf_counter()
        action_features = self.action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if self.action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
            pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = torch.cat((state_features, action_features), dim=1)
        torch.cuda.synchronize()
        encode_ms = (time.perf_counter() - ts) * 1000.0

        _mark_step_begin()
        torch.cuda.synchronize()
        ts = time.perf_counter()
        model_output = self.action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=backbone_features,
            timestep=timesteps_tensor,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )
        torch.cuda.synchronize()
        dit_ms = (time.perf_counter() - ts) * 1000.0

        torch.cuda.synchronize()
        ts = time.perf_counter()
        pred = self.action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -self.action_head.action_horizon :]
        next_actions = actions + self.dt * pred_velocity
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - ts) * 1000.0

        offset = 0
        for task in tasks:
            task.current_actions = next_actions[offset : offset + 1]
            task.step_idx += 1
            offset += 1

        return {
            "encode_ms": encode_ms,
            "dit_ms": dit_ms,
            "decode_ms": decode_ms,
            "total_ms": encode_ms + dit_ms + decode_ms,
        }

    @torch.inference_mode()
    def reference_full_actions(
        self,
        tasks: list[PreparedTask],
        initial_actions: torch.Tensor,
    ) -> torch.Tensor:
        backbone_features = torch.cat([task.backbone_features for task in tasks], dim=0)
        state_features = torch.cat([task.state_features for task in tasks], dim=0)
        embodiment_id = torch.cat([task.embodiment_id for task in tasks], dim=0)
        image_mask = torch.cat([task.image_mask for task in tasks], dim=0)
        backbone_attention_mask = torch.cat([task.backbone_attention_mask for task in tasks], dim=0)
        actions = initial_actions.clone()
        for step_idx in range(self.action_head.num_inference_timesteps):
            t_cont = step_idx / float(self.action_head.num_inference_timesteps)
            t_discretized = int(t_cont * self.action_head.num_timestep_buckets)
            timesteps_tensor = torch.full(
                (len(tasks),),
                fill_value=t_discretized,
                device=self.device,
                dtype=torch.long,
            )
            action_features = self.action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
                pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_output = self.action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=backbone_features,
                timestep=timesteps_tensor,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
            pred = self.action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_head.action_horizon :]
            actions = actions + self.dt * pred_velocity
        return actions


def _build_phase_locked_requests(
    *,
    cohort_size: int,
    num_cohorts: int,
    waves: int,
    period_ms: float,
    phase_stride_ms: float,
) -> list[dict[str, float | int]]:
    events: list[dict[str, float | int]] = []
    request_id = 0
    for cohort in range(num_cohorts):
        slot_start = cohort * cohort_size
        for wave in range(waves):
            arrival_ms = cohort * phase_stride_ms + wave * period_ms
            for inner in range(cohort_size):
                events.append(
                    {
                        "request_id": request_id,
                        "slot_id": slot_start + inner,
                        "arrival_ms": float(arrival_ms),
                    }
                )
                request_id += 1
    events.sort(key=lambda item: (float(item["arrival_ms"]), int(item["slot_id"])))
    return events


def _run_whole_request_replay(
    runtime: UnifiedRuntime,
    requests: list[dict[str, float | int]],
    *,
    deadline_ms: float,
) -> dict[str, Any]:
    current_ms = 0.0
    cursor = 0
    pending: list[dict[str, float | int]] = []
    latency_ms: list[float] = []
    queue_wait_ms: list[float] = []
    service_ms: list[float] = []
    miss_count = 0

    while cursor < len(requests) or pending:
        while cursor < len(requests) and float(requests[cursor]["arrival_ms"]) <= current_ms + 1e-9:
            pending.append(requests[cursor])
            cursor += 1
        if not pending:
            current_ms = float(requests[cursor]["arrival_ms"])
            continue

        slot_ids = [int(item["slot_id"]) for item in pending]
        arrivals = [float(item["arrival_ms"]) for item in pending]
        svc_ms = _cuda_ms(lambda: runtime.whole_request_batch(slot_ids))
        start_ms = current_ms
        finish_ms = start_ms + svc_ms
        current_ms = finish_ms
        service_ms.append(svc_ms)
        for arrival_ms in arrivals:
            queue_wait = start_ms - arrival_ms
            latency = finish_ms - arrival_ms
            queue_wait_ms.append(queue_wait)
            latency_ms.append(latency)
            if latency > deadline_ms:
                miss_count += 1
        pending.clear()

    return {
        "service_ms": _stats(service_ms),
        "request_to_result_ms": _stats(latency_ms),
        "queue_wait_ms": _stats(queue_wait_ms),
        "deadline_miss_count": int(miss_count),
        "deadline_miss_ratio": float(miss_count / max(1, len(latency_ms))),
        "num_requests": int(len(latency_ms)),
        "makespan_ms": float(current_ms),
    }


def _select_ready_step(
    ready: dict[int, list[PreparedTask]],
    *,
    step_batch_trigger: int,
    has_pending_vlm: bool,
) -> int | None:
    best_step = None
    best_size = -1
    for step_idx, tasks in ready.items():
        if len(tasks) > best_size:
            best_step = step_idx
            best_size = len(tasks)
    if best_step is None:
        return None
    if best_size >= step_batch_trigger:
        return best_step
    if not has_pending_vlm:
        return best_step
    return None


def _run_unified_replay(
    runtime: UnifiedRuntime,
    requests: list[dict[str, float | int]],
    *,
    deadline_ms: float,
    step_batch_trigger: int,
) -> dict[str, Any]:
    current_ms = 0.0
    cursor = 0
    pending_vlm: list[dict[str, float | int]] = []
    ready_by_step: dict[int, list[PreparedTask]] = {}
    latency_ms: list[float] = []
    queue_wait_ms: list[float] = []
    vlm_service_ms: list[float] = []
    step_service_ms: list[float] = []
    step_batch_sizes: list[int] = []
    vlm_stage_breakdown = {"vision": [], "projector": [], "fuse": [], "llm": []}
    step_breakdown = {"encode_ms": [], "dit_ms": [], "decode_ms": [], "total_ms": []}
    miss_count = 0

    def add_arrivals(limit_ms: float) -> None:
        nonlocal cursor
        while cursor < len(requests) and float(requests[cursor]["arrival_ms"]) <= limit_ms + 1e-9:
            pending_vlm.append(requests[cursor])
            cursor += 1

    while cursor < len(requests) or pending_vlm or ready_by_step:
        add_arrivals(current_ms)
        chosen_step = _select_ready_step(
            ready_by_step,
            step_batch_trigger=step_batch_trigger,
            has_pending_vlm=bool(pending_vlm),
        )
        if chosen_step is not None:
            tasks = ready_by_step.pop(chosen_step)
            service = runtime.run_step_batch(tasks)
            step_service_ms.append(service["total_ms"])
            step_batch_sizes.append(len(tasks))
            for key, val in service.items():
                step_breakdown[key].append(val)
            current_ms += service["total_ms"]
            add_arrivals(current_ms)
            for task in tasks:
                task.service_accum_ms += service["total_ms"]
                if task.step_idx < runtime.action_head.num_inference_timesteps:
                    ready_by_step.setdefault(task.step_idx, []).append(task)
                else:
                    latency = current_ms - task.arrival_ms
                    queue_wait = max(0.0, latency - task.service_accum_ms)
                    latency_ms.append(latency)
                    queue_wait_ms.append(queue_wait)
                    if latency > deadline_ms:
                        miss_count += 1
            continue

        if pending_vlm:
            req = pending_vlm.pop(0)
            slot_id = int(req["slot_id"])
            arrival_ms = float(req["arrival_ms"])
            start_ms = max(current_ms, arrival_ms)
            if start_ms > current_ms:
                current_ms = start_ms
            torch.cuda.synchronize()
            ts = time.perf_counter()
            task, stage_ms = runtime.prepare_task(
                int(req["request_id"]),
                slot_id,
                arrival_ms,
            )
            torch.cuda.synchronize()
            svc_ms = (time.perf_counter() - ts) * 1000.0
            task.service_accum_ms += svc_ms
            current_ms += svc_ms
            vlm_service_ms.append(svc_ms)
            for key, val in stage_ms.items():
                vlm_stage_breakdown[key].append(val)
            ready_by_step.setdefault(0, []).append(task)
            add_arrivals(current_ms)
            continue

        if cursor < len(requests):
            current_ms = float(requests[cursor]["arrival_ms"])

    return {
        "vlm_service_ms": _stats(vlm_service_ms) if vlm_service_ms else {},
        "dit_step_service_ms": _stats(step_service_ms) if step_service_ms else {},
        "dit_step_batch_size": _stats(step_batch_sizes) if step_batch_sizes else {},
        "vlm_stage_breakdown_ms": {
            key: _stats(vals) for key, vals in vlm_stage_breakdown.items() if vals
        },
        "dit_step_breakdown_ms": {
            key: _stats(vals) for key, vals in step_breakdown.items() if vals
        },
        "request_to_result_ms": _stats(latency_ms),
        "queue_wait_ms": _stats(queue_wait_ms),
        "deadline_miss_count": int(miss_count),
        "deadline_miss_ratio": float(miss_count / max(1, len(latency_ms))),
        "num_requests": int(len(latency_ms)),
        "makespan_ms": float(current_ms),
    }


def _run_burst_curves(
    runtime: UnifiedRuntime,
    batch_sizes: list[int],
    *,
    step_batch_trigger: int,
    include_whole: bool,
    include_unified: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for batch_size in batch_sizes:
        requests = [
            {
                "request_id": idx,
                "slot_id": idx,
                "arrival_ms": 0.0,
            }
            for idx in range(batch_size)
        ]
        row: dict[str, Any] = {}
        if include_whole:
            row["whole_request_batch"] = _run_whole_request_replay(runtime, requests, deadline_ms=1e9)
        if include_unified:
            row["unified_multistage_step_microbatch"] = _run_unified_replay(
                runtime,
                requests,
                deadline_ms=1e9,
                step_batch_trigger=step_batch_trigger,
            )
        out[str(batch_size)] = row
    return out


def _warmup_whole_request(runtime: UnifiedRuntime, batch_sizes: list[int]) -> None:
    for batch_size in sorted(set(batch_sizes)):
        slot_ids = list(range(batch_size))
        _ = runtime.whole_request_batch(slot_ids)


def _warmup_unified(runtime: UnifiedRuntime, batch_sizes: list[int]) -> None:
    for batch_size in sorted(set(batch_sizes)):
        tasks = []
        for slot_id in range(batch_size):
            task, _ = runtime.prepare_task(slot_id, slot_id, 0.0)
            tasks.append(task)
        for _ in range(runtime.action_head.num_inference_timesteps):
            runtime.run_step_batch(tasks)


def _run_equivalence_probe(runtime: UnifiedRuntime, batch_size: int) -> dict[str, float]:
    tasks = []
    for slot_id in range(batch_size):
        task, _ = runtime.prepare_task(slot_id, slot_id, 0.0)
        tasks.append(task)
    initial_actions = torch.cat([task.current_actions.clone() for task in tasks], dim=0)
    reference = runtime.reference_full_actions(tasks, initial_actions)

    replay_tasks = []
    for idx, task in enumerate(tasks):
        replay_tasks.append(
            PreparedTask(
                request_id=task.request_id,
                slot_id=task.slot_id,
                arrival_ms=task.arrival_ms,
                backbone_features=task.backbone_features,
                backbone_attention_mask=task.backbone_attention_mask,
                image_mask=task.image_mask,
                state_features=task.state_features,
                embodiment_id=task.embodiment_id,
                current_actions=initial_actions[idx : idx + 1].clone(),
                step_idx=0,
            )
        )
    for _ in range(runtime.action_head.num_inference_timesteps):
        runtime.run_step_batch(replay_tasks)
    actual = torch.cat([task.current_actions for task in replay_tasks], dim=0)
    diff = (reference - actual).abs()
    return {
        "batch_size": float(batch_size),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-bridge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--burst-batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--phase-cohort-size", type=int, default=2)
    parser.add_argument("--phase-num-cohorts", type=int, default=2)
    parser.add_argument("--phase-waves", type=int, default=8)
    parser.add_argument("--phase-period-ms", type=float, default=80.0)
    parser.add_argument("--phase-stride-ms", type=float, default=20.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--step-batch-trigger", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--compile-llm", action="store_true")
    parser.add_argument("--compile-dit", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-json",
        default="/root/autodl-tmp/VLAServing/results/gr00t_unified_multistage_microbatch_runtime_20260423.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    runtime = UnifiedRuntime(
        args.model_path,
        args.device,
        args.dtype,
        num_slots=args.num_slots,
        inference_steps=args.inference_steps,
        local_files_only=args.local_files_only,
    )

    warmup_batch_sizes = list(args.burst_batch_sizes) + [args.phase_cohort_size]
    _warmup_whole_request(runtime, warmup_batch_sizes)

    burst_whole = _run_burst_curves(
        runtime,
        args.burst_batch_sizes,
        step_batch_trigger=args.step_batch_trigger,
        include_whole=True,
        include_unified=False,
    )
    phase_requests = _build_phase_locked_requests(
        cohort_size=args.phase_cohort_size,
        num_cohorts=args.phase_num_cohorts,
        waves=args.phase_waves,
        period_ms=args.phase_period_ms,
        phase_stride_ms=args.phase_stride_ms,
    )
    phase_whole = _run_whole_request_replay(runtime, phase_requests, deadline_ms=args.deadline_ms)

    runtime.enable_compilation(
        compile_llm=args.compile_llm,
        compile_dit=args.compile_dit,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )

    _warmup_unified(runtime, warmup_batch_sizes)
    _ = _run_equivalence_probe(runtime, batch_size=2)

    burst_unified = _run_burst_curves(
        runtime,
        args.burst_batch_sizes,
        step_batch_trigger=args.step_batch_trigger,
        include_whole=False,
        include_unified=True,
    )
    phase_unified = _run_unified_replay(
        runtime,
        phase_requests,
        deadline_ms=args.deadline_ms,
        step_batch_trigger=args.step_batch_trigger,
    )
    equivalence = _run_equivalence_probe(runtime, batch_size=min(4, args.num_slots))

    burst: dict[str, Any] = {}
    for key in sorted(set(burst_whole) | set(burst_unified), key=int):
        burst[key] = {}
        burst[key].update(burst_whole.get(key, {}))
        burst[key].update(burst_unified.get(key, {}))

    payload = {
        "meta": {
            "model_path": args.model_path,
            "device": args.device,
            "dtype": args.dtype,
            "num_slots": args.num_slots,
            "inference_steps": args.inference_steps,
            "compile_llm": args.compile_llm,
            "compile_dit": args.compile_dit,
            "compile_backend": args.compile_backend,
            "compile_mode": args.compile_mode,
            "whole_request_baseline_mode": "eager",
            "unified_runtime_mode": "compiled" if (args.compile_llm or args.compile_dit) else "eager",
            "phase_cohort_size": args.phase_cohort_size,
            "phase_num_cohorts": args.phase_num_cohorts,
            "phase_waves": args.phase_waves,
            "phase_period_ms": args.phase_period_ms,
            "phase_stride_ms": args.phase_stride_ms,
            "deadline_ms": args.deadline_ms,
            "step_batch_trigger": args.step_batch_trigger,
        },
        "validation": {
            "step_microbatch_vs_reference": equivalence,
        },
        "burst_curves": burst,
        "phase_locked_replay": {
            "whole_request_batch": phase_whole,
            "unified_multistage_step_microbatch": phase_unified,
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
