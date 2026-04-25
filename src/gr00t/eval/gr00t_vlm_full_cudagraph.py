from __future__ import annotations

from typing import Any

import torch

try:
    from gr00t_siglip2_fixedshape import Siglip2FixedShapePatch
except ModuleNotFoundError:
    from src.gr00t.eval.gr00t_siglip2_fixedshape import Siglip2FixedShapePatch


def _empty_like_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value)
    if isinstance(value, list):
        return [_empty_like_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_empty_like_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _empty_like_tree(item) for key, item in value.items()}
    raise TypeError(f"Unsupported tree leaf type: {type(value)!r}")


def _copy_tree(dst: Any, src: Any) -> None:
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
        return
    if isinstance(dst, list):
        for dst_item, src_item in zip(dst, src):
            _copy_tree(dst_item, src_item)
        return
    if isinstance(dst, tuple):
        for dst_item, src_item in zip(dst, src):
            _copy_tree(dst_item, src_item)
        return
    if isinstance(dst, dict):
        for key in dst:
            _copy_tree(dst[key], src[key])
        return
    raise TypeError(f"Unsupported tree leaf type: {type(dst)!r}")


class VLMFullCudaGraphExecutor:
    def __init__(
        self,
        runtime,
        *,
        sample_slot_id: int = 0,
        llm_attn_impl: str = "sdpa",
        vision_attn_impl: str = "sdpa",
    ) -> None:
        self.runtime = runtime
        sample_inputs = runtime.slot_inputs(sample_slot_id)
        sample_action = runtime.action_input_for_slot(sample_slot_id)

        self.pixel_values = _empty_like_tree(sample_inputs["pixel_values"])
        self.input_ids = torch.empty_like(sample_inputs["input_ids"])
        self.attention_mask = torch.empty_like(sample_inputs["attention_mask"])
        self.state = torch.empty_like(sample_action["state"])
        self.embodiment_id = torch.empty_like(sample_action["embodiment_id"])
        self.sample_input_ids = sample_inputs["input_ids"].clone()
        selected = self.sample_input_ids.reshape(-1) == runtime.image_token_index
        self.image_selected_idx = selected.nonzero(as_tuple=False).flatten().to(device=runtime.device)
        self.use_attention_mask = runtime._attention_mask_arg(sample_inputs["attention_mask"]) is not None

        self.prev_llm_attn_impl = runtime.llm_body.config._attn_implementation
        self.prev_vision_attn_impl = runtime.vision_model.config._attn_implementation
        self.prev_vision_transformer_attn_impl = runtime.vision_model.vision_model.config._attn_implementation
        self.fixedshape_patch = Siglip2FixedShapePatch.build(runtime.vision_model, sample_inputs["pixel_values"])

        runtime.llm_body.config._attn_implementation = llm_attn_impl
        runtime.vision_model.config._attn_implementation = vision_attn_impl
        runtime.vision_model.vision_model.config._attn_implementation = vision_attn_impl
        self.fixedshape_patch.apply()
        try:
            self._load_inputs(sample_slot_id, sample_inputs=sample_inputs, sample_action=sample_action)
            sample_backbone, sample_state = self._forward_outputs()
            self.backbone_features = torch.empty_like(sample_backbone)
            self.state_features = torch.empty_like(sample_state)
            self._body()
            self.graph = torch.cuda.CUDAGraph()
            self._capture()
        finally:
            self.fixedshape_patch.restore()
            runtime.llm_body.config._attn_implementation = self.prev_llm_attn_impl
            runtime.vision_model.config._attn_implementation = self.prev_vision_attn_impl
            runtime.vision_model.vision_model.config._attn_implementation = (
                self.prev_vision_transformer_attn_impl
            )

    def _load_inputs(
        self,
        slot_id: int,
        *,
        sample_inputs: dict[str, Any] | None = None,
        sample_action: Any | None = None,
    ) -> None:
        inputs = sample_inputs if sample_inputs is not None else self.runtime.slot_inputs(slot_id)
        action_input = sample_action if sample_action is not None else self.runtime.action_input_for_slot(slot_id)
        _copy_tree(self.pixel_values, inputs["pixel_values"])
        self.input_ids.copy_(inputs["input_ids"])
        self.attention_mask.copy_(inputs["attention_mask"])
        self.state.copy_(action_input["state"])
        self.embodiment_id.copy_(action_input["embodiment_id"])

    def _forward_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        runtime = self.runtime
        vision_out = runtime.vision_model(
            pixel_values=self.pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        vit_embeds, _ = runtime.backbone.model.pixel_shuffle_back(
            vision_out.last_hidden_state,
            vision_out.spatial_shapes,
        )
        vit_embeds = runtime.projector(vit_embeds)
        batch, tokens, hidden = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(batch * tokens, hidden)
        input_embeds = runtime.embed_tokens(self.input_ids)
        input_embeds = self._fuse_image_tokens(input_embeds, vit_embeds)
        hidden_states = runtime.llm_body(
            inputs_embeds=input_embeds,
            attention_mask=self.attention_mask if self.use_attention_mask else None,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        ).last_hidden_state
        backbone_features = runtime.action_head.vlln(hidden_states)
        state_features = runtime.action_head.state_encoder(self.state, self.embodiment_id)
        return backbone_features, state_features

    def _fuse_image_tokens(
        self,
        input_embeds: torch.Tensor,
        vit_embeds: torch.Tensor,
    ) -> torch.Tensor:
        flat_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
        count = min(int(self.image_selected_idx.numel()), int(vit_embeds.shape[0]))
        if count > 0:
            flat_embeds[self.image_selected_idx[:count]] = vit_embeds[:count]
        return flat_embeds.reshape_as(input_embeds)

    def _body(self) -> None:
        backbone_features, state_features = self._forward_outputs()
        self.backbone_features.copy_(backbone_features)
        self.state_features.copy_(state_features)

    def _capture(self) -> None:
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._body()
        warmup_stream.synchronize()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._body()

    def launch(self, slot_id: int) -> None:
        self._load_inputs(slot_id)
        self.graph.replay()

    def materialize_task(
        self,
        request_id: int,
        slot_id: int,
        arrival_ms: float,
    ):
        slot_inputs = self.runtime.slot_inputs(slot_id)
        current_actions = torch.randn(
            (1, self.runtime.action_head.config.action_horizon, self.runtime.action_head.action_dim),
            device=self.runtime.device,
            dtype=self.backbone_features.dtype,
            generator=self.runtime.generator,
        )
        return self.runtime.PreparedTaskCls(
            request_id=request_id,
            slot_id=slot_id,
            arrival_ms=arrival_ms,
            backbone_features=self.backbone_features.clone(),
            backbone_attention_mask=(slot_inputs["attention_mask"] == 1).clone(),
            image_mask=(slot_inputs["input_ids"] == self.runtime.image_token_index).clone(),
            state_features=self.state_features.clone(),
            embodiment_id=self.embodiment_id.clone(),
            current_actions=current_actions,
        )
