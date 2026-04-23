from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

MIRAGE_PYTHON = "/root/autodl-tmp/mirage/python"
ISAAC_GR00T_ROOT = "/root/autodl-tmp/Isaac-GR00T"
RESULT_PATH = (
    "/root/autodl-tmp/VLAServing/results/"
    "gr00t_mpk_steady_state_runtime_20260423.json"
)
COMPILE_COPY_DIR = "/root/autodl-tmp/mirage/tests/runtime_python/test_mode"

if MIRAGE_PYTHON not in sys.path:
    sys.path.insert(0, MIRAGE_PYTHON)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402


DIT_PY = os.path.join(ISAAC_GR00T_ROOT, "gr00t", "model", "modules", "dit.py")
DIT_SPEC = importlib.util.spec_from_file_location("gr00t_dit_module", DIT_PY)
DIT_MODULE = importlib.util.module_from_spec(DIT_SPEC)
assert DIT_SPEC.loader is not None
DIT_SPEC.loader.exec_module(DIT_MODULE)
AlternateVLDiT = DIT_MODULE.AlternateVLDiT


def summarize(samples: list[float]) -> dict[str, Any]:
    xs = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(xs.mean()),
        "std_ms": float(xs.std()),
        "p50_ms": float(np.percentile(xs, 50)),
        "p95_ms": float(np.percentile(xs, 95)),
        "min_ms": float(xs.min()),
        "max_ms": float(xs.max()),
        "num_samples": int(xs.size),
    }


def event_timed_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def attach_const(pk: PersistentKernel, tensor: torch.Tensor, name: str):
    return pk.attach_input(tensor.contiguous(), name=name)


def alloc_buffer(
    pk: PersistentKernel,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    name: str,
):
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    return tensor, pk.attach_input(tensor, name=name)


def linear_bias(
    pk: PersistentKernel,
    input_dt,
    weight: torch.Tensor,
    bias: torch.Tensor,
    seq_len: int,
    out_dim: int,
    block_dim: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
    prefix: str,
):
    weight_dt = attach_const(pk, weight, f"{prefix}_weight")
    bias_dt = attach_const(pk, bias, f"{prefix}_bias")
    _, lin_dt = alloc_buffer(pk, (seq_len, out_dim), dtype, device, f"{prefix}_lin")
    out_buf, out_dt = alloc_buffer(pk, (seq_len, out_dim), dtype, device, f"{prefix}_out")
    pk.linear_generic_layer(
        input=input_dt,
        weight=weight_dt,
        output=lin_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )
    pk.bias_add_layer(
        input=lin_dt,
        bias=bias_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )
    return out_buf, out_dt


def elementwise_add(
    pk: PersistentKernel,
    a_dt,
    b_dt,
    shape: tuple[int, int],
    block_dim: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
    prefix: str,
):
    out_buf, out_dt = alloc_buffer(pk, shape, dtype, device, prefix)
    pk.elementwise_add_layer(
        input_a=a_dt,
        input_b=b_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )
    return out_buf, out_dt


def layernorm_no_affine(
    pk: PersistentKernel,
    input_dt,
    shape: tuple[int, int],
    eps: float,
    block_dim: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
    prefix: str,
):
    out_buf, out_dt = alloc_buffer(pk, shape, dtype, device, prefix)
    pk.layernorm_layer(
        input=input_dt,
        output=out_dt,
        weight=None,
        bias=None,
        eps=eps,
        grid_dim=(shape[0], 1, 1),
        block_dim=block_dim,
    )
    return out_buf, out_dt


def adaln_no_affine(
    pk: PersistentKernel,
    input_dt,
    temb_dt,
    mod_weight: torch.Tensor,
    mod_bias: torch.Tensor,
    shape: tuple[int, int],
    eps: float,
    block_dim: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
    prefix: str,
):
    mod_weight_dt = attach_const(pk, mod_weight, f"{prefix}_mod_weight")
    mod_bias_dt = attach_const(pk, mod_bias, f"{prefix}_mod_bias")
    out_buf, out_dt = alloc_buffer(pk, shape, dtype, device, prefix)
    pk.adalayernorm_layer(
        input=input_dt,
        temb=temb_dt,
        mod_weight=mod_weight_dt,
        mod_bias=mod_bias_dt,
        output=out_dt,
        eps=eps,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )
    return out_buf, out_dt


def gelu_approx(
    pk: PersistentKernel,
    input_dt,
    shape: tuple[int, int],
    block_dim: tuple[int, int, int],
    dtype: torch.dtype,
    device: str,
    prefix: str,
):
    out_buf, out_dt = alloc_buffer(pk, shape, dtype, device, prefix)
    pk.gelu_layer(
        input=input_dt,
        output=out_dt,
        approximate_tanh=True,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )
    return out_buf, out_dt


@dataclass
class GraphContext:
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    q_len: int = 17
    kv_len: int = 64
    dim: int = 1536
    cross_dim: int = 2048
    output_dim: int = 1024
    num_heads: int = 32
    head_dim: int = 48
    input_scale: float = 0.1

    @property
    def ff_dim(self) -> int:
        return 4 * self.dim


def make_model_and_inputs(ctx: GraphContext):
    torch.manual_seed(0)
    model = AlternateVLDiT(
        positional_embeddings=None,
        num_layers=2,
        num_attention_heads=ctx.num_heads,
        attention_head_dim=ctx.head_dim,
        norm_type="ada_norm",
        dropout=0.2,
        final_dropout=True,
        output_dim=ctx.output_dim,
        interleave_self_attention=True,
        cross_attention_dim=ctx.cross_dim,
        attend_text_every_n_blocks=2,
    ).to(device=ctx.device, dtype=ctx.dtype)
    model.eval()
    hidden_states = (
        torch.randn(1, ctx.q_len, ctx.dim, device=ctx.device, dtype=ctx.dtype)
        * ctx.input_scale
    )
    encoder_hidden_states = (
        torch.randn(1, ctx.kv_len, ctx.cross_dim, device=ctx.device, dtype=ctx.dtype)
        * ctx.input_scale
    )
    backbone_attention_mask = torch.ones(
        1, ctx.kv_len, device=ctx.device, dtype=torch.bool
    )
    timestep = torch.tensor([137], device=ctx.device, dtype=torch.long)
    with torch.no_grad():
        temb = model.timestep_encoder(timestep)
    return model, hidden_states, encoder_hidden_states, backbone_attention_mask, timestep, temb


def torch_twoblock_reference(
    model,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    backbone_attention_mask: torch.Tensor,
    temb: torch.Tensor,
):
    block0 = model.transformer_blocks[0]
    block1 = model.transformer_blocks[1]
    x = block0(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=backbone_attention_mask,
        temb=temb,
    )
    x = block1(
        x,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=temb,
    )
    return x


def build_meta_tensors(
    device: str,
    total_num_requests: int,
    max_seq_length: int,
    max_num_batched_requests: int,
    max_num_batched_tokens: int,
    max_num_pages: int,
):
    return {
        "step": torch.zeros(total_num_requests, dtype=torch.int32, device=device),
        "tokens": torch.zeros(
            (total_num_requests, max_seq_length), dtype=torch.int64, device=device
        ),
        "input_tokens": torch.zeros(
            (max_num_batched_tokens,), dtype=torch.int64, device=device
        ),
        "output_tokens": torch.zeros(
            (max_num_batched_tokens,), dtype=torch.int64, device=device
        ),
        "num_new_tokens": torch.ones((1,), dtype=torch.int32, device=device),
        "prompt_lengths": torch.ones(
            (total_num_requests,), dtype=torch.int32, device=device
        ),
        "qo_indptr_buffer": torch.zeros(
            (max_num_batched_requests + 1,), dtype=torch.int32, device=device
        ),
        "paged_kv_indptr_buffer": torch.zeros(
            (max_num_batched_requests + 1,), dtype=torch.int32, device=device
        ),
        "paged_kv_indices_buffer": torch.zeros(
            (max_num_pages,), dtype=torch.int32, device=device
        ),
        "paged_kv_last_page_len_buffer": torch.zeros(
            (max_num_batched_requests,), dtype=torch.int32, device=device
        ),
    }


def build_gr00t_minidit_pk(
    ctx: GraphContext,
    mode: str,
    total_num_requests: int,
    max_num_batched_requests: int,
    max_num_batched_tokens: int,
    max_num_pages: int,
    num_workers: int,
    num_local_schedulers: int,
):
    model, hidden_states, encoder_hidden_states, backbone_attention_mask, _, temb = (
        make_model_and_inputs(ctx)
    )
    meta_tensors = build_meta_tensors(
        device=ctx.device,
        total_num_requests=total_num_requests,
        max_seq_length=2,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
    )
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        dict(
            mode=mode,
            world_size=1,
            mpi_rank=0,
            num_workers=num_workers,
            num_local_schedulers=num_local_schedulers,
            num_remote_schedulers=0,
            max_seq_length=2,
            max_num_batched_requests=max_num_batched_requests,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_pages=max_num_pages,
            page_size=1,
            meta_tensors=meta_tensors,
            test_mode=False,
            use_cutlass_kernel=False,
        )
    )
    pk = PersistentKernel(**params)
    block0 = model.transformer_blocks[0]
    block1 = model.transformer_blocks[1]
    block_dim = (128, 1, 1)

    hidden_dt = attach_const(pk, hidden_states.squeeze(0).contiguous(), "hidden_in")
    encoder_dt = attach_const(
        pk, encoder_hidden_states.squeeze(0).contiguous(), "encoder_in"
    )
    temb_dt = attach_const(pk, temb.contiguous(), "temb")
    cross_mask_dt = attach_const(
        pk,
        torch.ones(ctx.kv_len, device=ctx.device, dtype=torch.uint8),
        "cross_mask",
    )

    _, norm0_dt = adaln_no_affine(
        pk,
        hidden_dt,
        temb_dt,
        block0.norm1.linear.weight,
        block0.norm1.linear.bias,
        (ctx.q_len, ctx.dim),
        block0.norm1.norm.eps,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_norm1",
    )
    _, q0_dt = linear_bias(
        pk,
        norm0_dt,
        block0.attn1.to_q.weight,
        block0.attn1.to_q.bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_q",
    )
    _, k0_dt = linear_bias(
        pk,
        encoder_dt,
        block0.attn1.to_k.weight,
        block0.attn1.to_k.bias,
        ctx.kv_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_k",
    )
    _, v0_dt = linear_bias(
        pk,
        encoder_dt,
        block0.attn1.to_v.weight,
        block0.attn1.to_v.bias,
        ctx.kv_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_v",
    )
    _, attn0_dt = alloc_buffer(
        pk, (ctx.q_len, ctx.dim), ctx.dtype, ctx.device, "block0_attn_ctx"
    )
    pk.full_attention_layer(
        q=q0_dt,
        k=k0_dt,
        v=v0_dt,
        attention_mask=cross_mask_dt,
        output=attn0_dt,
        num_heads=ctx.num_heads,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
        causal=False,
    )
    _, attn0_out_dt = linear_bias(
        pk,
        attn0_dt,
        block0.attn1.to_out[0].weight,
        block0.attn1.to_out[0].bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_attn_out",
    )
    _, res0_dt = elementwise_add(
        pk,
        hidden_dt,
        attn0_out_dt,
        (ctx.q_len, ctx.dim),
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_residual",
    )
    _, ff0_norm_dt = layernorm_no_affine(
        pk,
        res0_dt,
        (ctx.q_len, ctx.dim),
        block0.norm3.eps,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_norm3",
    )
    _, ff0_mid_dt = linear_bias(
        pk,
        ff0_norm_dt,
        block0.ff.net[0].proj.weight,
        block0.ff.net[0].proj.bias,
        ctx.q_len,
        ctx.ff_dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_ff_mid",
    )
    _, ff0_act_dt = gelu_approx(
        pk,
        ff0_mid_dt,
        (ctx.q_len, ctx.ff_dim),
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_ff_act",
    )
    _, ff0_out_dt = linear_bias(
        pk,
        ff0_act_dt,
        block0.ff.net[2].weight,
        block0.ff.net[2].bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_ff_out",
    )
    _, hid1_dt = elementwise_add(
        pk,
        res0_dt,
        ff0_out_dt,
        (ctx.q_len, ctx.dim),
        block_dim,
        ctx.dtype,
        ctx.device,
        "block0_hidden_out",
    )

    _, norm1_dt = adaln_no_affine(
        pk,
        hid1_dt,
        temb_dt,
        block1.norm1.linear.weight,
        block1.norm1.linear.bias,
        (ctx.q_len, ctx.dim),
        block1.norm1.norm.eps,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_norm1",
    )
    _, q1_dt = linear_bias(
        pk,
        norm1_dt,
        block1.attn1.to_q.weight,
        block1.attn1.to_q.bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_q",
    )
    _, k1_dt = linear_bias(
        pk,
        norm1_dt,
        block1.attn1.to_k.weight,
        block1.attn1.to_k.bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_k",
    )
    _, v1_dt = linear_bias(
        pk,
        norm1_dt,
        block1.attn1.to_v.weight,
        block1.attn1.to_v.bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_v",
    )
    _, attn1_dt = alloc_buffer(
        pk, (ctx.q_len, ctx.dim), ctx.dtype, ctx.device, "block1_attn_ctx"
    )
    pk.full_attention_layer(
        q=q1_dt,
        k=k1_dt,
        v=v1_dt,
        attention_mask=None,
        output=attn1_dt,
        num_heads=ctx.num_heads,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
        causal=False,
    )
    _, attn1_out_dt = linear_bias(
        pk,
        attn1_dt,
        block1.attn1.to_out[0].weight,
        block1.attn1.to_out[0].bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_attn_out",
    )
    _, res1_dt = elementwise_add(
        pk,
        hid1_dt,
        attn1_out_dt,
        (ctx.q_len, ctx.dim),
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_residual",
    )
    _, ff1_norm_dt = layernorm_no_affine(
        pk,
        res1_dt,
        (ctx.q_len, ctx.dim),
        block1.norm3.eps,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_norm3",
    )
    _, ff1_mid_dt = linear_bias(
        pk,
        ff1_norm_dt,
        block1.ff.net[0].proj.weight,
        block1.ff.net[0].proj.bias,
        ctx.q_len,
        ctx.ff_dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_ff_mid",
    )
    _, ff1_act_dt = gelu_approx(
        pk,
        ff1_mid_dt,
        (ctx.q_len, ctx.ff_dim),
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_ff_act",
    )
    out_buf, ff1_out_dt = linear_bias(
        pk,
        ff1_act_dt,
        block1.ff.net[2].weight,
        block1.ff.net[2].bias,
        ctx.q_len,
        ctx.dim,
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_ff_out",
    )
    out_buf, _ = elementwise_add(
        pk,
        res1_dt,
        ff1_out_dt,
        (ctx.q_len, ctx.dim),
        block_dim,
        ctx.dtype,
        ctx.device,
        "block1_hidden_out",
    )

    pk.compile(output_dir=COMPILE_COPY_DIR)
    return pk, out_buf, model, hidden_states, encoder_hidden_states, backbone_attention_mask, temb


def benchmark_torch_eager_twoblock(ctx: GraphContext, warmup: int, iters: int):
    model, hidden_states, encoder_hidden_states, backbone_attention_mask, _, temb = (
        make_model_and_inputs(ctx)
    )

    def run_once():
        torch_twoblock_reference(
            model,
            hidden_states,
            encoder_hidden_states,
            backbone_attention_mask,
            temb,
        )

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    samples = [event_timed_ms(run_once) for _ in range(iters)]
    return summarize(samples)


def benchmark_online_notoken(
    ctx: GraphContext,
    num_workers: int,
    num_local_schedulers: int,
    launches: int,
):
    pk, out_buf, model, hidden_states, encoder_hidden_states, backbone_attention_mask, temb = (
        build_gr00t_minidit_pk(
            ctx,
            mode="online_notoken",
            total_num_requests=1,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=1,
            num_workers=num_workers,
            num_local_schedulers=num_local_schedulers,
        )
    )
    try:
        correctness = None
        samples = []
        for i in range(launches):
            pk.init_request_func()
            elapsed_ms = event_timed_ms(lambda: pk())
            if i == 0:
                with torch.no_grad():
                    ref = torch_twoblock_reference(
                        model,
                        hidden_states,
                        encoder_hidden_states,
                        backbone_attention_mask,
                        temb,
                    ).squeeze(0)
                diff = (out_buf.float() - ref.float()).abs()
                correctness = {
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "out_sum": float(out_buf.float().sum().item()),
                }
            samples.append(elapsed_ms)
        return {"latency_ms": summarize(samples), "correctness": correctness}
    finally:
        pk.finalize()


def benchmark_offline(
    ctx: GraphContext,
    total_num_requests: int,
    launches: int,
):
    pk, out_buf, model, hidden_states, encoder_hidden_states, backbone_attention_mask, temb = (
        build_gr00t_minidit_pk(
            ctx,
            mode="offline",
            total_num_requests=total_num_requests,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=total_num_requests,
            num_workers=1,
            num_local_schedulers=1,
        )
    )
    try:
        elapsed_samples = []
        per_request_samples = []
        throughput_samples = []
        correctness = None
        for i in range(launches):
            pk.init_request_func()
            elapsed_ms = event_timed_ms(lambda: pk())
            if i == 0:
                with torch.no_grad():
                    ref = torch_twoblock_reference(
                        model,
                        hidden_states,
                        encoder_hidden_states,
                        backbone_attention_mask,
                        temb,
                    ).squeeze(0)
                diff = (out_buf.float() - ref.float()).abs()
                correctness = {
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "out_sum": float(out_buf.float().sum().item()),
                }
            elapsed_samples.append(elapsed_ms)
            per_request_samples.append(elapsed_ms / total_num_requests)
            throughput_samples.append(total_num_requests * 1000.0 / elapsed_ms)
        return {
            "elapsed_ms": summarize(elapsed_samples),
            "per_request_ms": summarize(per_request_samples),
            "throughput_rps": summarize(throughput_samples),
            "correctness": correctness,
        }
    finally:
        pk.finalize()


def main():
    ctx = GraphContext()
    auto_workers, auto_schedulers = mirage.get_configurations_from_gpu(0)
    result = {
        "meta": {
            "date": "2026-04-23",
            "device": "RTX 4090",
            "dtype": "bfloat16",
            "graph": "GR00T N1.6 mini AlternateVLDiT two-block core",
            "shape": {
                "q_len": ctx.q_len,
                "kv_len": ctx.kv_len,
                "hidden_dim": ctx.dim,
                "cross_dim": ctx.cross_dim,
                "num_heads": ctx.num_heads,
                "head_dim": ctx.head_dim,
            },
            "notes": [
                "This benchmark uses non-test-mode PersistentKernel request paths.",
                "online_notoken is used for single-request latency.",
                "offline(total_num_requests=N, max_num_batched_requests=1) is used for steady-state throughput.",
            ],
        },
        "torch_eager_twoblock": benchmark_torch_eager_twoblock(ctx, warmup=10, iters=50),
        "mpk_online_notoken_conservative": benchmark_online_notoken(
            ctx, num_workers=1, num_local_schedulers=1, launches=5
        ),
        "mpk_online_notoken_auto": benchmark_online_notoken(
            ctx,
            num_workers=auto_workers,
            num_local_schedulers=auto_schedulers,
            launches=5,
        ),
        "mpk_offline_conservative_r8": benchmark_offline(
            ctx, total_num_requests=8, launches=2
        ),
        "mpk_offline_conservative_r64": benchmark_offline(
            ctx, total_num_requests=64, launches=2
        ),
    }

    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
