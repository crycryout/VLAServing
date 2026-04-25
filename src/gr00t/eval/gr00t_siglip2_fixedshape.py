from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from typing import Any

import torch


def convert_images_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, num_channels, image_height, image_width = image.shape
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError("image height/width must be divisible by patch size")
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched = image.reshape(
        batch_size,
        num_channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched = patched.permute(0, 2, 4, 3, 5, 1)
    return patched.reshape(batch_size * num_patches_height * num_patches_width, -1)


def _pixel_groups(pixel_values: Any) -> list[torch.Tensor]:
    if isinstance(pixel_values, torch.Tensor):
        return [group for group in pixel_values]
    if isinstance(pixel_values, (list, tuple)):
        return [group for group in pixel_values]
    raise TypeError(f"Unsupported pixel_values type: {type(pixel_values)!r}")


@dataclass
class Siglip2FixedShapePatch:
    vision_model: torch.nn.Module
    resized_positional_embeddings: torch.Tensor
    spatial_shapes: torch.Tensor
    reverse_mapping: torch.Tensor
    window_order: torch.Tensor
    win_meta_list: list[dict[str, Any]]
    rope_freqs_cis: torch.Tensor
    original_embeddings_forward: Any
    original_rope_get_freqs_cis: Any
    window_cu_seqlens: torch.Tensor
    full_cu_seqlens: torch.Tensor
    window_max_seq_len: int
    full_max_seq_len: int
    original_flash_attention_forward_for_packing: Any
    original_attention_forwards: list[Any]

    @classmethod
    def build(cls, vision_model: torch.nn.Module, sample_pixel_values: Any) -> "Siglip2FixedShapePatch":
        transformer = vision_model.vision_model
        embeddings = transformer.embeddings
        rope_2d = transformer.encoder.rope_2d

        with torch.inference_mode():
            _, win_meta_list, spatial_shapes, reverse_mapping = embeddings(sample_pixel_values)
            positional_embeddings = embeddings.position_embedding.weight.reshape(
                embeddings.position_embedding_size,
                embeddings.position_embedding_size,
                -1,
            )
            resized_positional_embeddings = embeddings.resize_positional_embeddings(
                positional_embeddings,
                spatial_shapes,
            ).clone()
            spatial_shapes_out = spatial_shapes.clone()
            reverse_mapping = reverse_mapping.to(
                device=resized_positional_embeddings.device,
                dtype=torch.long,
            ).clone()
            window_order = torch.argsort(reverse_mapping).clone()
            rope_freqs_cis = rope_2d.get_freqs_cis(
                win_meta_list=win_meta_list,
                device=resized_positional_embeddings.device,
            ).clone()
            window_seq_len_list = [meta["win_hw"][0] * meta["win_hw"][1] for meta in win_meta_list]
            full_mapper: dict[int, int] = {}
            for meta in win_meta_list:
                full_mapper[meta["img_idx"]] = full_mapper.get(meta["img_idx"], 0) + (
                    meta["win_hw"][0] * meta["win_hw"][1]
                )
            full_seq_len_list = [full_mapper[idx] for idx in range(len(full_mapper))]
            window_seq_tensor = torch.tensor(
                window_seq_len_list,
                device=resized_positional_embeddings.device,
                dtype=torch.int32,
            )
            full_seq_tensor = torch.tensor(
                full_seq_len_list,
                device=resized_positional_embeddings.device,
                dtype=torch.int32,
            )
            window_cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(window_seq_tensor, dim=0),
                (1, 0),
            ).to(torch.int32)
            full_cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(full_seq_tensor, dim=0),
                (1, 0),
            ).to(torch.int32)

        siglip_module = sys.modules[vision_model.__class__.__module__]
        original_attention_forwards = [
            layer.self_attn.forward for layer in transformer.encoder.layers
        ]

        return cls(
            vision_model=vision_model,
            resized_positional_embeddings=resized_positional_embeddings,
            spatial_shapes=spatial_shapes_out,
            reverse_mapping=reverse_mapping,
            window_order=window_order,
            win_meta_list=list(win_meta_list),
            rope_freqs_cis=rope_freqs_cis,
            original_embeddings_forward=embeddings.forward,
            original_rope_get_freqs_cis=rope_2d.get_freqs_cis,
            window_cu_seqlens=window_cu_seqlens,
            full_cu_seqlens=full_cu_seqlens,
            window_max_seq_len=max(window_seq_len_list),
            full_max_seq_len=max(full_seq_len_list),
            original_flash_attention_forward_for_packing=siglip_module.flash_attention_forward_for_packing,
            original_attention_forwards=original_attention_forwards,
        )

    def apply(self) -> None:
        transformer = self.vision_model.vision_model
        embeddings = transformer.embeddings
        rope_2d = transformer.encoder.rope_2d
        siglip_module = sys.modules[self.vision_model.__class__.__module__]
        patch = self

        def fixedshape_embeddings_forward(module_self, pixel_values):
            patch_groups = _pixel_groups(pixel_values)
            patched = torch.cat(
                [
                    convert_images_to_patches(group, module_self.patch_size)
                    for group in patch_groups
                ],
                dim=0,
            )
            patch_embeds = module_self.patch_embedding(
                patched.to(dtype=module_self.patch_embedding.weight.dtype)
            )
            full_embeddings = patch_embeds + patch.resized_positional_embeddings
            windows_tensor = full_embeddings.index_select(1, patch.window_order)
            return (
                windows_tensor,
                patch.win_meta_list,
                patch.spatial_shapes,
                patch.reverse_mapping,
            )

        def fixedshape_get_freqs_cis(module_self, win_meta_list, device):
            del win_meta_list, device
            return patch.rope_freqs_cis

        def fixedshape_flash_attention_forward_for_packing(
            module_self,
            query,
            key,
            value,
            attention_mask=None,
            dropout: float = 0.0,
            scaling=None,
            sliding_window=None,
            softcap=None,
            seq_len_list=None,
            **kwargs,
        ):
            del seq_len_list
            seq_len = query.shape[2]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            target_dtype = None
            if query.dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                elif hasattr(module_self.config, "_pre_quantization_dtype"):
                    target_dtype = module_self.config._pre_quantization_dtype
                else:
                    target_dtype = next(
                        layer
                        for layer in module_self.modules()
                        if isinstance(layer, torch.nn.Linear)
                    ).weight.dtype

            kwargs.pop("is_causal", None)
            cu_seqlens = getattr(module_self, "_fixedshape_cu_seqlens")
            max_seq_len = int(getattr(module_self, "_fixedshape_max_seq_len"))
            attn_output = siglip_module._flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query_length=seq_len,
                is_causal=module_self.is_causal,
                dropout=dropout,
                softmax_scale=scaling,
                sliding_window=sliding_window,
                softcap=softcap,
                use_top_left_mask=siglip_module._use_top_left_mask,
                target_dtype=target_dtype,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seq_len,
                max_length_k=max_seq_len,
                **kwargs,
            )
            return attn_output.squeeze(0), None

        def make_fixedshape_attn_forward(original_forward):
            def fixedshape_attn_forward(
                module_self,
                hidden_states,
                output_attentions: bool = False,
                rope_freqs_cis=None,
                win_meta_list=None,
                windows_attn: bool = False,
            ):
                del win_meta_list
                if windows_attn and module_self.use_windows_attn:
                    module_self._fixedshape_cu_seqlens = patch.window_cu_seqlens
                    module_self._fixedshape_max_seq_len = patch.window_max_seq_len
                else:
                    module_self._fixedshape_cu_seqlens = patch.full_cu_seqlens
                    module_self._fixedshape_max_seq_len = patch.full_max_seq_len
                return original_forward(
                    hidden_states,
                    output_attentions=output_attentions,
                    rope_freqs_cis=rope_freqs_cis,
                    win_meta_list=patch.win_meta_list,
                    windows_attn=windows_attn,
                )

            return fixedshape_attn_forward

        embeddings.forward = types.MethodType(fixedshape_embeddings_forward, embeddings)
        rope_2d.get_freqs_cis = types.MethodType(fixedshape_get_freqs_cis, rope_2d)
        siglip_module.flash_attention_forward_for_packing = fixedshape_flash_attention_forward_for_packing
        for layer, original_forward in zip(transformer.encoder.layers, patch.original_attention_forwards):
            layer.self_attn.forward = types.MethodType(
                make_fixedshape_attn_forward(original_forward),
                layer.self_attn,
            )

    def restore(self) -> None:
        transformer = self.vision_model.vision_model
        siglip_module = sys.modules[self.vision_model.__class__.__module__]
        transformer.embeddings.forward = self.original_embeddings_forward
        transformer.encoder.rope_2d.get_freqs_cis = self.original_rope_get_freqs_cis
        siglip_module.flash_attention_forward_for_packing = (
            self.original_flash_attention_forward_for_packing
        )
        for layer, original_forward in zip(transformer.encoder.layers, self.original_attention_forwards):
            layer.self_attn.forward = original_forward
