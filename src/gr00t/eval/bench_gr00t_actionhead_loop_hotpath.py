#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import torch
import tyro
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead


@dataclass
class Config:
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    seq_len: int = 256
    batch_sizes: list[int] = field(default_factory=lambda: [4, 8])
    warmup: int = 2
    iterations: int = 6
    output_json: str = (
        "/root/autodl-tmp/VLAServing/results/gr00t_actionhead_loop_hotpath_20260416.json"
    )


def _stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "num_samples": int(arr.size),
    }


def _cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    ts = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - ts) * 1000.0


@torch.no_grad()
def baseline_get_action(
    action_head: Gr00tN1d6ActionHead,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
) -> BatchFeature:
    features = action_head._encode_features(backbone_output, action_input)
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_input.embodiment_id

    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    actions = torch.randn(
        size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )
    dt = 1.0 / action_head.num_inference_timesteps

    for t in range(action_head.num_inference_timesteps):
        t_cont = t / float(action_head.num_inference_timesteps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = torch.cat((state_features, action_features), dim=1)

        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )

        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


@torch.no_grad()
def candidate_reuse_buffers_get_action(
    action_head: Gr00tN1d6ActionHead,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
) -> BatchFeature:
    features = action_head._encode_features(backbone_output, action_input)
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_input.embodiment_id

    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    state_horizon = state_features.shape[1]
    actions = torch.randn(
        size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )
    dt = 1.0 / action_head.num_inference_timesteps
    timesteps_tensor = torch.empty(size=(batch_size,), device=device, dtype=torch.long)
    sa_embs = torch.empty(
        size=(
            batch_size,
            state_horizon + action_head.config.action_horizon,
            action_head.input_embedding_dim,
        ),
        device=device,
        dtype=state_features.dtype,
    )
    sa_embs[:, :state_horizon].copy_(state_features)
    pos_embs = None
    if action_head.config.add_pos_embed:
        pos_embs = action_head.position_embedding.weight[
            : action_head.config.action_horizon
        ].unsqueeze(0)

    for t in range(action_head.num_inference_timesteps):
        t_cont = t / float(action_head.num_inference_timesteps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor.fill_(t_discretized)
        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if pos_embs is not None:
            action_features.add_(pos_embs)
        sa_embs[:, state_horizon:].copy_(action_features)

        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )

        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


def _make_inputs(
    cfg: Gr00tN1d6Config,
    batch_size: int,
    seq_len: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[BatchFeature, BatchFeature]:
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(
                batch_size,
                seq_len,
                cfg.backbone_embedding_dim,
                device=device,
                dtype=dtype,
            ),
            "backbone_attention_mask": torch.ones(
                batch_size, seq_len, device=device, dtype=torch.bool
            ),
            "image_mask": torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.randn(
                batch_size, 1, cfg.max_state_dim, device=device, dtype=dtype
            ),
            "embodiment_id": torch.zeros(batch_size, device=device, dtype=torch.long),
        }
    )
    return backbone_output, action_input


def _torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def run(cfg: Config) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dtype = _torch_dtype(cfg.dtype)
    model_cfg = Gr00tN1d6Config()
    action_head = Gr00tN1d6ActionHead(model_cfg).eval().to(device=cfg.device, dtype=dtype)

    rows = []
    for batch_size in cfg.batch_sizes:
        backbone_output, action_input = _make_inputs(
            model_cfg, batch_size, cfg.seq_len, cfg.device, dtype
        )

        for _ in range(cfg.warmup):
            baseline_get_action(action_head, backbone_output, action_input)
            candidate_reuse_buffers_get_action(action_head, backbone_output, action_input)

        baseline_times = [
            _cuda_ms(lambda: baseline_get_action(action_head, backbone_output, action_input))
            for _ in range(cfg.iterations)
        ]
        candidate_times = [
            _cuda_ms(
                lambda: candidate_reuse_buffers_get_action(
                    action_head, backbone_output, action_input
                )
            )
            for _ in range(cfg.iterations)
        ]

        row = {
            "batch_size": int(batch_size),
            "baseline": _stats(baseline_times),
            "candidate_reuse_buffers": _stats(candidate_times),
        }
        row["candidate_delta_p50_ms"] = (
            row["candidate_reuse_buffers"]["p50_ms"] - row["baseline"]["p50_ms"]
        )
        row["candidate_speedup_p50"] = (
            row["baseline"]["p50_ms"] / row["candidate_reuse_buffers"]["p50_ms"]
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    payload = {
        "config": asdict(cfg),
        "results": rows,
    }
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
