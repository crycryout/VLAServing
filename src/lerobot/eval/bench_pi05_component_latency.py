#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _cuda_ms(fn) -> float:
    if not torch.cuda.is_available():
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000.0

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
        "num_samples": int(arr.size),
    }


def _make_fake_batch(policy: PI05Policy, batch_size: int) -> dict[str, torch.Tensor]:
    device = next(policy.parameters()).device
    batch: dict[str, torch.Tensor] = {}
    input_features = policy.config.input_features
    for key, feature in input_features.items():
        shape = tuple(feature.shape)
        if key.startswith("observation.images.") and "empty_camera" not in key:
            batch[key] = torch.rand((batch_size, *shape), device=device, dtype=torch.float32)

    token_len = int(policy.config.tokenizer_max_length)
    batch[OBS_LANGUAGE_TOKENS] = torch.full((batch_size, token_len), 1, device=device, dtype=torch.long)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones((batch_size, token_len), device=device, dtype=torch.bool)
    return batch


def _prepare_prefix_context(policy: PI05Policy, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)
    policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = policy.model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return {
        "images": images,
        "img_masks": img_masks,
        "tokens": tokens,
        "masks": masks,
        "prefix_pad_masks": prefix_pad_masks,
        "past_key_values": past_key_values,
    }


def _bench_prefix(policy: PI05Policy, batch: dict[str, torch.Tensor]) -> float:
    def fn():
        _prepare_prefix_context(policy, batch)

    return _cuda_ms(fn)


def _make_suffix_step_fn(policy: PI05Policy, denoise_step_partial):
    if getattr(policy.model.config, "compile_model", False):
        return torch.compile(lambda x: denoise_step_partial(x), mode=policy.model.config.compile_mode)
    return denoise_step_partial


def _bench_suffix(
    policy: PI05Policy,
    batch: dict[str, torch.Tensor],
    *,
    num_steps: int,
    noise: torch.Tensor,
    step_fn,
) -> float:
    ctx = _prepare_prefix_context(policy, batch)
    prefix_pad_masks = ctx["prefix_pad_masks"]
    past_key_values = ctx["past_key_values"]
    denoise_step_partial = policy.model._denoise_step_partial
    denoise_step_partial.set_context(prefix_pad_masks, past_key_values)
    dt = -1.0 / num_steps
    base_noise = noise.clone()

    def fn():
        x_t = base_noise.clone()
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.full(
                (x_t.shape[0],),
                float(t),
                device=x_t.device,
                dtype=torch.float32,
            )
            denoise_step_partial.timestep = time_tensor
            v_t = step_fn(x_t)
            x_t = x_t + dt * v_t

    return _cuda_ms(fn)


def _bench_full(policy: PI05Policy, batch: dict[str, torch.Tensor], *, num_steps: int, noise: torch.Tensor) -> float:
    def fn():
        policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    return _cuda_ms(fn)


def _run_once(policy: PI05Policy, batch: dict[str, torch.Tensor], *, num_steps: int, warmup: int, iters: int) -> dict[str, Any]:
    device = next(policy.parameters()).device
    noise = policy.model.sample_noise(
        (batch[OBS_LANGUAGE_TOKENS].shape[0], policy.config.chunk_size, policy.config.max_action_dim),
        device,
    )
    suffix_step_fn = _make_suffix_step_fn(policy, policy.model._denoise_step_partial)

    for _ in range(warmup):
        _bench_full(policy, batch, num_steps=num_steps, noise=noise)
        _bench_prefix(policy, batch)
        _bench_suffix(policy, batch, num_steps=num_steps, noise=noise, step_fn=suffix_step_fn)

    full_ms = []
    prefix_ms = []
    suffix_ms = []
    for _ in range(iters):
        full_ms.append(_bench_full(policy, batch, num_steps=num_steps, noise=noise))
        prefix_ms.append(_bench_prefix(policy, batch))
        suffix_ms.append(_bench_suffix(policy, batch, num_steps=num_steps, noise=noise, step_fn=suffix_step_fn))

    return {
        "num_steps": int(num_steps),
        "batch_size": int(batch[OBS_LANGUAGE_TOKENS].shape[0]),
        "full_e2e": _stats(full_ms),
        "vlm_prefix": _stats(prefix_ms),
        "dit_suffix": _stats(suffix_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 1])
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = None
    if args.disable_compile or args.disable_gradient_checkpointing:
        config = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
        if args.disable_compile:
            config.compile_model = False
        if args.disable_gradient_checkpointing:
            config.gradient_checkpointing = False

    policy = PI05Policy.from_pretrained(args.model_path, config=config, local_files_only=True)
    policy.eval()
    batch = _make_fake_batch(policy, args.batch_size)

    summary: dict[str, Any] = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "disable_compile": bool(args.disable_compile),
        "disable_gradient_checkpointing": bool(args.disable_gradient_checkpointing),
        "device": str(next(policy.parameters()).device),
        "results": {},
    }

    for step in args.steps:
        result = _run_once(policy, batch, num_steps=step, warmup=args.warmup, iters=args.iters)
        summary["results"][str(step)] = result
        with (out_dir / f"step{step}_summary.json").open("w") as f:
            json.dump(result, f, indent=2)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
