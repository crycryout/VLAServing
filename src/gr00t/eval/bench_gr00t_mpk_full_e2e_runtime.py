from __future__ import annotations

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

MIRAGE_PYTHON = "/root/autodl-tmp/mirage/python"
ISAAC_GR00T_ROOT = "/root/autodl-tmp/Isaac-GR00T"
RESULT_PATH = (
    "/root/autodl-tmp/VLAServing/results/"
    "gr00t_mpk_full_e2e_runtime_20260423.json"
)
COMPILE_OUTPUT_ROOT = "/root/autodl-tmp/mirage/tests/runtime_python/test_mode/gr00t_full_e2e"

if MIRAGE_PYTHON not in sys.path:
    sys.path.insert(0, MIRAGE_PYTHON)
if ISAAC_GR00T_ROOT not in sys.path:
    sys.path.insert(0, ISAAC_GR00T_ROOT)

import mirage  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402


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


def perf_ms(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0


def attach_tensor(pk: PersistentKernel, tensor: torch.Tensor, name: str):
    return pk.attach_input(tensor.contiguous(), name=name)


def alloc_input(
    pk: PersistentKernel,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    name: str,
):
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    return tensor, pk.attach_input(tensor, name=name)


def alloc_output(
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
    weight_dt = attach_tensor(pk, weight, f"{prefix}_weight")
    bias_dt = attach_tensor(pk, bias, f"{prefix}_bias")
    _, lin_dt = alloc_output(pk, (seq_len, out_dim), dtype, device, f"{prefix}_lin")
    out_buf, out_dt = alloc_output(pk, (seq_len, out_dim), dtype, device, f"{prefix}_out")
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
    out_buf, out_dt = alloc_output(pk, shape, dtype, device, prefix)
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
    out_buf, out_dt = alloc_output(pk, shape, dtype, device, prefix)
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
    mod_weight_dt = attach_tensor(pk, mod_weight, f"{prefix}_mod_weight")
    mod_bias_dt = attach_tensor(pk, mod_bias, f"{prefix}_mod_bias")
    out_buf, out_dt = alloc_output(pk, shape, dtype, device, prefix)
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
    out_buf, out_dt = alloc_output(pk, shape, dtype, device, prefix)
    pk.gelu_layer(
        input=input_dt,
        output=out_dt,
        approximate_tanh=True,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )
    return out_buf, out_dt


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


@dataclass
class RuntimeShape:
    q_len: int
    kv_len: int
    hidden_dim: int
    cross_dim: int
    num_heads: int
    head_dim: int
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    @property
    def ff_dim(self) -> int:
        return 4 * self.hidden_dim


class MirageTwoBlockStage:
    def __init__(
        self,
        model,
        stage_index: int,
        shape: RuntimeShape,
        output_dir: str,
    ):
        self.stage_index = stage_index
        self.shape = shape
        self.block_pair_index = stage_index

        meta_tensors = build_meta_tensors(
            device=shape.device,
            total_num_requests=1,
            max_seq_length=2,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=1,
        )
        params = PersistentKernel.get_default_init_parameters()
        params.update(
            dict(
                mode="online_notoken",
                world_size=1,
                mpi_rank=0,
                num_workers=1,
                num_local_schedulers=1,
                num_remote_schedulers=0,
                max_seq_length=2,
                max_num_batched_requests=1,
                max_num_batched_tokens=1,
                max_num_pages=1,
                page_size=1,
                meta_tensors=meta_tensors,
                test_mode=False,
                use_cutlass_kernel=False,
            )
        )
        self.pk = PersistentKernel(**params)

        self.hidden_buf, hidden_dt = alloc_input(
            self.pk,
            (shape.q_len, shape.hidden_dim),
            shape.dtype,
            shape.device,
            f"stage{stage_index}_hidden_in",
        )
        self.encoder_buf, encoder_dt = alloc_input(
            self.pk,
            (shape.kv_len, shape.cross_dim),
            shape.dtype,
            shape.device,
            f"stage{stage_index}_encoder_in",
        )
        self.temb_buf, temb_dt = alloc_input(
            self.pk,
            (1, shape.hidden_dim),
            shape.dtype,
            shape.device,
            f"stage{stage_index}_temb",
        )
        self.mask_buf, mask_dt = alloc_input(
            self.pk,
            (shape.kv_len,),
            torch.uint8,
            shape.device,
            f"stage{stage_index}_cross_mask",
        )

        block0 = model.transformer_blocks[2 * stage_index]
        block1 = model.transformer_blocks[2 * stage_index + 1]
        block_dim = (128, 1, 1)

        _, norm0_dt = adaln_no_affine(
            self.pk,
            hidden_dt,
            temb_dt,
            block0.norm1.linear.weight,
            block0.norm1.linear.bias,
            (shape.q_len, shape.hidden_dim),
            block0.norm1.norm.eps,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_norm1",
        )
        _, q0_dt = linear_bias(
            self.pk,
            norm0_dt,
            block0.attn1.to_q.weight,
            block0.attn1.to_q.bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_q",
        )
        _, k0_dt = linear_bias(
            self.pk,
            encoder_dt,
            block0.attn1.to_k.weight,
            block0.attn1.to_k.bias,
            shape.kv_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_k",
        )
        _, v0_dt = linear_bias(
            self.pk,
            encoder_dt,
            block0.attn1.to_v.weight,
            block0.attn1.to_v.bias,
            shape.kv_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_v",
        )
        _, attn0_dt = alloc_output(
            self.pk,
            (shape.q_len, shape.hidden_dim),
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_attn_ctx",
        )
        self.pk.full_attention_layer(
            q=q0_dt,
            k=k0_dt,
            v=v0_dt,
            attention_mask=mask_dt,
            output=attn0_dt,
            num_heads=shape.num_heads,
            grid_dim=(1, 1, 1),
            block_dim=block_dim,
            causal=False,
        )
        _, attn0_out_dt = linear_bias(
            self.pk,
            attn0_dt,
            block0.attn1.to_out[0].weight,
            block0.attn1.to_out[0].bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_attn_out",
        )
        _, res0_dt = elementwise_add(
            self.pk,
            hidden_dt,
            attn0_out_dt,
            (shape.q_len, shape.hidden_dim),
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_residual",
        )
        _, ff0_norm_dt = layernorm_no_affine(
            self.pk,
            res0_dt,
            (shape.q_len, shape.hidden_dim),
            block0.norm3.eps,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_norm3",
        )
        _, ff0_mid_dt = linear_bias(
            self.pk,
            ff0_norm_dt,
            block0.ff.net[0].proj.weight,
            block0.ff.net[0].proj.bias,
            shape.q_len,
            shape.ff_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_ff_mid",
        )
        _, ff0_act_dt = gelu_approx(
            self.pk,
            ff0_mid_dt,
            (shape.q_len, shape.ff_dim),
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_ff_act",
        )
        _, ff0_out_dt = linear_bias(
            self.pk,
            ff0_act_dt,
            block0.ff.net[2].weight,
            block0.ff.net[2].bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_ff_out",
        )
        _, hid1_dt = elementwise_add(
            self.pk,
            res0_dt,
            ff0_out_dt,
            (shape.q_len, shape.hidden_dim),
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block0_hidden_out",
        )

        _, norm1_dt = adaln_no_affine(
            self.pk,
            hid1_dt,
            temb_dt,
            block1.norm1.linear.weight,
            block1.norm1.linear.bias,
            (shape.q_len, shape.hidden_dim),
            block1.norm1.norm.eps,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_norm1",
        )
        _, q1_dt = linear_bias(
            self.pk,
            norm1_dt,
            block1.attn1.to_q.weight,
            block1.attn1.to_q.bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_q",
        )
        _, k1_dt = linear_bias(
            self.pk,
            norm1_dt,
            block1.attn1.to_k.weight,
            block1.attn1.to_k.bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_k",
        )
        _, v1_dt = linear_bias(
            self.pk,
            norm1_dt,
            block1.attn1.to_v.weight,
            block1.attn1.to_v.bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_v",
        )
        _, attn1_dt = alloc_output(
            self.pk,
            (shape.q_len, shape.hidden_dim),
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_attn_ctx",
        )
        self.pk.full_attention_layer(
            q=q1_dt,
            k=k1_dt,
            v=v1_dt,
            attention_mask=None,
            output=attn1_dt,
            num_heads=shape.num_heads,
            grid_dim=(1, 1, 1),
            block_dim=block_dim,
            causal=False,
        )
        _, attn1_out_dt = linear_bias(
            self.pk,
            attn1_dt,
            block1.attn1.to_out[0].weight,
            block1.attn1.to_out[0].bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_attn_out",
        )
        _, res1_dt = elementwise_add(
            self.pk,
            hid1_dt,
            attn1_out_dt,
            (shape.q_len, shape.hidden_dim),
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_residual",
        )
        _, ff1_norm_dt = layernorm_no_affine(
            self.pk,
            res1_dt,
            (shape.q_len, shape.hidden_dim),
            block1.norm3.eps,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_norm3",
        )
        _, ff1_mid_dt = linear_bias(
            self.pk,
            ff1_norm_dt,
            block1.ff.net[0].proj.weight,
            block1.ff.net[0].proj.bias,
            shape.q_len,
            shape.ff_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_ff_mid",
        )
        _, ff1_act_dt = gelu_approx(
            self.pk,
            ff1_mid_dt,
            (shape.q_len, shape.ff_dim),
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_ff_act",
        )
        _, ff1_out_dt = linear_bias(
            self.pk,
            ff1_act_dt,
            block1.ff.net[2].weight,
            block1.ff.net[2].bias,
            shape.q_len,
            shape.hidden_dim,
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_ff_out",
        )
        self.out_buf, _ = elementwise_add(
            self.pk,
            res1_dt,
            ff1_out_dt,
            (shape.q_len, shape.hidden_dim),
            block_dim,
            shape.dtype,
            shape.device,
            f"stage{stage_index}_block1_hidden_out",
        )

        self.compile_ms = perf_ms(lambda: self.pk.compile(output_dir=output_dir))

    def run(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.shape[0] == 1
        assert encoder_hidden_states.shape[0] == 1
        self.hidden_buf.copy_(hidden_states.squeeze(0).contiguous())
        self.encoder_buf.copy_(encoder_hidden_states.squeeze(0).contiguous())
        self.temb_buf.copy_(temb.contiguous())
        self.mask_buf.copy_(cross_mask.to(dtype=torch.uint8).contiguous())
        self.pk.init_request_func()
        self.pk()
        torch.cuda.synchronize()
        return self.out_buf.unsqueeze(0).clone()

    def finalize(self):
        self.pk.finalize()


class MirageFullDiTCoreRuntime:
    def __init__(self, model, shape: RuntimeShape):
        self.model = model
        self.shape = shape

        meta_tensors = build_meta_tensors(
            device=shape.device,
            total_num_requests=1,
            max_seq_length=2,
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=1,
        )
        params = PersistentKernel.get_default_init_parameters()
        params.update(
            dict(
                mode="online_notoken",
                world_size=1,
                mpi_rank=0,
                num_workers=1,
                num_local_schedulers=1,
                num_remote_schedulers=0,
                max_seq_length=2,
                max_num_batched_requests=1,
                max_num_batched_tokens=1,
                max_num_pages=1,
                page_size=1,
                meta_tensors=meta_tensors,
                test_mode=False,
                use_cutlass_kernel=False,
            )
        )
        self.pk = PersistentKernel(**params)

        self.hidden_buf, hidden_dt = alloc_input(
            self.pk,
            (shape.q_len, shape.hidden_dim),
            shape.dtype,
            shape.device,
            "full_hidden_in",
        )
        self.encoder_buf, encoder_dt = alloc_input(
            self.pk,
            (shape.kv_len, shape.cross_dim),
            shape.dtype,
            shape.device,
            "full_encoder_in",
        )
        self.temb_buf, temb_dt = alloc_input(
            self.pk,
            (1, shape.hidden_dim),
            shape.dtype,
            shape.device,
            "full_temb",
        )
        self.image_mask_buf, image_mask_dt = alloc_input(
            self.pk,
            (shape.kv_len,),
            torch.uint8,
            shape.device,
            "full_image_mask",
        )
        self.non_image_mask_buf, non_image_mask_dt = alloc_input(
            self.pk,
            (shape.kv_len,),
            torch.uint8,
            shape.device,
            "full_non_image_mask",
        )

        block_dim = (128, 1, 1)
        current_hidden_dt = hidden_dt
        current_out_buf = self.hidden_buf
        attend_cycle = 2 * self.model.attend_text_every_n_blocks

        for block_idx, block in enumerate(self.model.transformer_blocks):
            _, norm_dt = adaln_no_affine(
                self.pk,
                current_hidden_dt,
                temb_dt,
                block.norm1.linear.weight,
                block.norm1.linear.bias,
                (shape.q_len, shape.hidden_dim),
                block.norm1.norm.eps,
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_norm1",
            )

            _, q_dt = linear_bias(
                self.pk,
                norm_dt,
                block.attn1.to_q.weight,
                block.attn1.to_q.bias,
                shape.q_len,
                shape.hidden_dim,
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_q",
            )

            if block_idx % 2 == 1:
                _, k_dt = linear_bias(
                    self.pk,
                    norm_dt,
                    block.attn1.to_k.weight,
                    block.attn1.to_k.bias,
                    shape.q_len,
                    shape.hidden_dim,
                    block_dim,
                    shape.dtype,
                    shape.device,
                    f"block{block_idx}_k",
                )
                _, v_dt = linear_bias(
                    self.pk,
                    norm_dt,
                    block.attn1.to_v.weight,
                    block.attn1.to_v.bias,
                    shape.q_len,
                    shape.hidden_dim,
                    block_dim,
                    shape.dtype,
                    shape.device,
                    f"block{block_idx}_v",
                )
                attn_mask_dt = None
            else:
                _, k_dt = linear_bias(
                    self.pk,
                    encoder_dt,
                    block.attn1.to_k.weight,
                    block.attn1.to_k.bias,
                    shape.kv_len,
                    shape.hidden_dim,
                    block_dim,
                    shape.dtype,
                    shape.device,
                    f"block{block_idx}_k",
                )
                _, v_dt = linear_bias(
                    self.pk,
                    encoder_dt,
                    block.attn1.to_v.weight,
                    block.attn1.to_v.bias,
                    shape.kv_len,
                    shape.hidden_dim,
                    block_dim,
                    shape.dtype,
                    shape.device,
                    f"block{block_idx}_v",
                )
                use_non_image = block_idx % attend_cycle == 0
                attn_mask_dt = non_image_mask_dt if use_non_image else image_mask_dt

            _, attn_ctx_dt = alloc_output(
                self.pk,
                (shape.q_len, shape.hidden_dim),
                shape.dtype,
                shape.device,
                f"block{block_idx}_attn_ctx",
            )
            self.pk.full_attention_layer(
                q=q_dt,
                k=k_dt,
                v=v_dt,
                attention_mask=attn_mask_dt,
                output=attn_ctx_dt,
                num_heads=shape.num_heads,
                grid_dim=(1, 1, 1),
                block_dim=block_dim,
                causal=False,
            )
            _, attn_out_dt = linear_bias(
                self.pk,
                attn_ctx_dt,
                block.attn1.to_out[0].weight,
                block.attn1.to_out[0].bias,
                shape.q_len,
                shape.hidden_dim,
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_attn_out",
            )
            _, residual_dt = elementwise_add(
                self.pk,
                current_hidden_dt,
                attn_out_dt,
                (shape.q_len, shape.hidden_dim),
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_residual",
            )
            _, ff_norm_dt = layernorm_no_affine(
                self.pk,
                residual_dt,
                (shape.q_len, shape.hidden_dim),
                block.norm3.eps,
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_norm3",
            )
            _, ff_mid_dt = linear_bias(
                self.pk,
                ff_norm_dt,
                block.ff.net[0].proj.weight,
                block.ff.net[0].proj.bias,
                shape.q_len,
                shape.ff_dim,
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_ff_mid",
            )
            _, ff_act_dt = gelu_approx(
                self.pk,
                ff_mid_dt,
                (shape.q_len, shape.ff_dim),
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_ff_act",
            )
            _, ff_out_dt = linear_bias(
                self.pk,
                ff_act_dt,
                block.ff.net[2].weight,
                block.ff.net[2].bias,
                shape.q_len,
                shape.hidden_dim,
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_ff_out",
            )
            current_out_buf, current_hidden_dt = elementwise_add(
                self.pk,
                residual_dt,
                ff_out_dt,
                (shape.q_len, shape.hidden_dim),
                block_dim,
                shape.dtype,
                shape.device,
                f"block{block_idx}_hidden_out",
            )

        self.out_buf = current_out_buf
        self.compile_ms = perf_ms(lambda: self.pk.compile(output_dir=COMPILE_OUTPUT_ROOT))

    def run(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_mask: torch.Tensor,
        non_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.shape[0] == 1
        assert encoder_hidden_states.shape[0] == 1
        self.hidden_buf.copy_(hidden_states.squeeze(0).contiguous())
        self.encoder_buf.copy_(encoder_hidden_states.squeeze(0).contiguous())
        self.temb_buf.copy_(temb.contiguous())
        self.image_mask_buf.copy_(image_mask.to(dtype=torch.uint8).contiguous())
        self.non_image_mask_buf.copy_(non_image_mask.to(dtype=torch.uint8).contiguous())
        self.pk.init_request_func()
        self.pk()
        torch.cuda.synchronize()
        return self.out_buf.unsqueeze(0).clone()

    def finalize(self):
        self.pk.finalize()


class MirageActionHeadRuntime:
    def __init__(self, action_head, backbone_output: BatchFeature, action_input: BatchFeature):
        self.action_head = action_head
        self.model = action_head.model
        with torch.inference_mode():
            features = action_head._encode_features(backbone_output, action_input)

        self.shape = RuntimeShape(
            q_len=features.state_features.shape[1] + action_head.action_horizon,
            kv_len=features.backbone_features.shape[1],
            hidden_dim=features.state_features.shape[2],
            cross_dim=features.backbone_features.shape[2],
            num_heads=self.model.config.num_attention_heads,
            head_dim=self.model.config.attention_head_dim,
            device=str(features.backbone_features.device),
            dtype=features.backbone_features.dtype,
        )
        Path(COMPILE_OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
        self.core = MirageFullDiTCoreRuntime(self.model, self.shape)
        self.compile_ms_total = self.core.compile_ms
        self.stage_compile_ms = [self.core.compile_ms]

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        initial_actions: torch.Tensor | None = None,
    ) -> BatchFeature:
        features = self.action_head._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            initial_actions=initial_actions,
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        initial_actions: torch.Tensor | None = None,
    ) -> BatchFeature:
        assert backbone_features.shape[0] == 1, "Current Mirage E2E path supports batch=1 only."
        image_mask = backbone_output.image_mask.to(dtype=torch.bool)
        backbone_attention_mask = backbone_output.backbone_attention_mask.to(dtype=torch.bool)
        image_attention_mask = (image_mask & backbone_attention_mask).squeeze(0)
        non_image_attention_mask = ((~image_mask) & backbone_attention_mask).squeeze(0)

        batch_size = backbone_features.shape[0]
        device = backbone_features.device
        dtype = backbone_features.dtype
        if initial_actions is None:
            actions = torch.randn(
                size=(batch_size, self.action_head.action_horizon, self.action_head.action_dim),
                dtype=dtype,
                device=device,
            )
        else:
            actions = initial_actions.clone()

        dt = 1.0 / self.action_head.num_inference_timesteps
        for step_idx in range(self.action_head.num_inference_timesteps):
            t_cont = step_idx / float(self.action_head.num_inference_timesteps)
            t_discretized = int(t_cont * self.action_head.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
            )
            action_features = self.action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            hidden_states = torch.cat((state_features, action_features), dim=1)
            temb = self.model.timestep_encoder(timesteps_tensor)
            hidden_states = self.core.run(
                hidden_states=hidden_states,
                encoder_hidden_states=backbone_features,
                temb=temb,
                image_mask=image_attention_mask,
                non_image_mask=non_image_attention_mask,
            )

            shift, scale = self.model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
            hidden_states = self.model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            model_output = self.model.proj_out_2(hidden_states)
            pred = self.action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_head.action_horizon :]
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": backbone_features,
                "state_features": state_features,
            }
        )

    def finalize(self):
        self.core.finalize()


def build_synthetic_observation() -> dict[str, Any]:
    return {
        "video": {
            "ego_view_bg_crop_pad_res256_freq20": np.zeros(
                (1, 1, 256, 256, 3), dtype=np.uint8
            )
        },
        "state": {
            "left_arm": np.zeros((1, 1, 7), dtype=np.float32),
            "right_arm": np.zeros((1, 1, 7), dtype=np.float32),
            "left_hand": np.zeros((1, 1, 6), dtype=np.float32),
            "right_hand": np.zeros((1, 1, 6), dtype=np.float32),
            "waist": np.zeros((1, 1, 3), dtype=np.float32),
        },
        "language": {"task": [["pick and place the object"]]},
    }


@torch.no_grad()
def official_action_head_rollout(
    action_head,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    initial_actions: torch.Tensor | None = None,
) -> BatchFeature:
    features = action_head._encode_features(backbone_output, action_input)
    backbone_features = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_input.embodiment_id

    batch_size = backbone_features.shape[0]
    device = backbone_features.device
    dtype = backbone_features.dtype
    if initial_actions is None:
        actions = torch.randn(
            size=(batch_size, action_head.action_horizon, action_head.action_dim),
            dtype=dtype,
            device=device,
        )
    else:
        actions = initial_actions.clone()

    dt = 1.0 / action_head.num_inference_timesteps
    for step_idx in range(action_head.num_inference_timesteps):
        t_cont = step_idx / float(action_head.num_inference_timesteps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
        )
        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=backbone_features,
            timestep=timesteps_tensor,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": backbone_features,
            "state_features": state_features,
        }
    )


def build_prepared_inputs(policy: Gr00tPolicy, observation: dict[str, Any]):
    collated_inputs, _ = policy.prepare_inference_inputs(observation)
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
        backbone_outputs = policy.model.backbone(backbone_inputs)
    return collated_inputs, backbone_outputs, action_inputs


def benchmark_e2e(
    policy: Gr00tPolicy,
    observation: dict[str, Any],
    action_runner,
    warmup: int,
    iters: int,
):
    data_ms = []
    backbone_ms = []
    action_ms = []
    e2e_ms = []

    def run_once(seed: int):
        collated_inputs, _ = policy.prepare_inference_inputs(observation)
        backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
        backbone_outputs = policy.model.backbone(backbone_inputs)
        torch.manual_seed(seed)
        _ = action_runner(backbone_outputs, action_inputs)

    for i in range(warmup):
        run_once(i)

    for i in range(iters):
        seed = 1000 + i
        start_e2e = time.perf_counter()
        collated_inputs, _ = policy.prepare_inference_inputs(observation)
        end_data = time.perf_counter()

        torch.cuda.synchronize()
        start_backbone = time.perf_counter()
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
            backbone_outputs = policy.model.backbone(backbone_inputs)
        torch.cuda.synchronize()
        end_backbone = time.perf_counter()

        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start_action = time.perf_counter()
        with torch.inference_mode():
            _ = action_runner(backbone_outputs, action_inputs)
        torch.cuda.synchronize()
        end_action = time.perf_counter()

        data_ms.append((end_data - start_e2e) * 1000.0)
        backbone_ms.append((end_backbone - start_backbone) * 1000.0)
        action_ms.append((end_action - start_action) * 1000.0)
        e2e_ms.append((end_action - start_e2e) * 1000.0)

    return {
        "data_processing_ms": summarize(data_ms),
        "backbone_ms": summarize(backbone_ms),
        "action_head_ms": summarize(action_ms),
        "e2e_ms": summarize(e2e_ms),
    }


def main():
    device = "cuda:0"
    observation = build_synthetic_observation()

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.GR1,
        model_path="nvidia/GR00T-N1.6-3B",
        device=device,
        strict=True,
    )

    _, backbone_outputs, action_inputs = build_prepared_inputs(policy, observation)
    mirage_runtime = MirageActionHeadRuntime(
        action_head=policy.model.action_head,
        backbone_output=backbone_outputs,
        action_input=action_inputs,
    )

    policy.model.action_head.model.forward = torch.compile(
        policy.model.action_head.model.forward, mode="max-autotune"
    )

    # Warm up compiled official path once before measurement.
    _ = benchmark_e2e(
        policy,
        observation,
        action_runner=lambda backbone_output, action_input: official_action_head_rollout(
            policy.model.action_head, backbone_output, action_input
        ),
        warmup=1,
        iters=1,
    )

    _, correctness_backbone_outputs, correctness_action_inputs = build_prepared_inputs(
        policy, observation
    )
    initial_actions = torch.randn(
        (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
        device=policy.model.device,
        dtype=policy.model.dtype,
    )
    with torch.inference_mode():
        official_pred = official_action_head_rollout(
            policy.model.action_head,
            correctness_backbone_outputs,
            correctness_action_inputs,
            initial_actions=initial_actions,
        )["action_pred"]
        mirage_pred = mirage_runtime.get_action(
            correctness_backbone_outputs,
            correctness_action_inputs,
            initial_actions=initial_actions,
        )["action_pred"]
    diff = (mirage_pred.float() - official_pred.float()).abs()

    official_compiled = benchmark_e2e(
        policy,
        observation,
        action_runner=lambda backbone_output, action_input: official_action_head_rollout(
            policy.model.action_head, backbone_output, action_input
        ),
        warmup=1,
        iters=2,
    )
    mirage_e2e = benchmark_e2e(
        policy,
        observation,
        action_runner=lambda backbone_output, action_input: mirage_runtime.get_action(
            backbone_output, action_input
        ),
        warmup=0,
        iters=1,
    )

    result = {
        "meta": {
            "date": "2026-04-23",
            "device": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "mode": "synthetic_observation_end_to_end",
            "mirage_scope": "processor + backbone + Mirage-backed single 32-layer online_notoken action-head core",
            "official_compile_scope": "matches official benchmark style: torch.compile on action_head.model.forward",
            "synthetic_observation": {
                "video_key": "ego_view_bg_crop_pad_res256_freq20",
                "video_shape": [1, 1, 256, 256, 3],
                "state_dims": {
                    "left_arm": 7,
                    "right_arm": 7,
                    "left_hand": 6,
                    "right_hand": 6,
                    "waist": 3,
                },
            },
            "prepared_shapes": {
                "backbone_features": list(correctness_backbone_outputs.backbone_features.shape),
                "backbone_attention_mask": list(
                    correctness_backbone_outputs.backbone_attention_mask.shape
                ),
                "image_mask": list(correctness_backbone_outputs.image_mask.shape),
                "action_state": list(correctness_action_inputs.state.shape),
                "action_pred": list(official_pred.shape),
            },
            "mirage_compile_ms_total": mirage_runtime.compile_ms_total,
            "mirage_stage_compile_ms": mirage_runtime.stage_compile_ms,
        },
        "correctness_vs_official_compiled": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "official_sum": float(official_pred.float().sum().item()),
            "mirage_sum": float(mirage_pred.float().sum().item()),
        },
        "official_compiled_e2e": official_compiled,
        "mirage_e2e": mirage_e2e,
    }

    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    mirage_runtime.finalize()


if __name__ == "__main__":
    main()
