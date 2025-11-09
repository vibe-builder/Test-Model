"""Hugging Face compatible Nano XYZ modeling classes."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
try:
    from transformers.generation.utils import GenerationMixin
except Exception:  # pragma: no cover
    try:
        from transformers.generation import GenerationMixin  # type: ignore
    except Exception:  # pragma: no cover
        class GenerationMixin:  # minimal fallback to avoid import errors
            pass
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
try:
    from transformers.utils import register_for_auto_class
except ImportError:  # pragma: no cover
    def register_for_auto_class(_auto_class: str):
        def decorator(cls):
            return cls
        return decorator

from .configuration_nano import NanoConfig
from .model import ModelArchitecture, ModelSettings


class NanoPreTrainedModel(PreTrainedModel):
    """Base class to integrate with the HF checkpoint ecosystem."""

    config_class = NanoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        # ModelArchitecture already initializes all modules.
        return

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None) -> None:  # type: ignore[override]
        try:
            super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        except TypeError:
            super().gradient_checkpointing_enable()
        self._set_activation_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:  # type: ignore[override]
        super().gradient_checkpointing_disable()
        self._set_activation_checkpointing(False)

    def _set_activation_checkpointing(self, enabled: bool) -> None:
        target = getattr(self, "inner", None)
        if target is None and hasattr(self, "model"):
            target = getattr(self.model, "inner", None)  # type: ignore[attr-defined]
        if target is not None:
            target.config.use_activation_checkpointing = enabled
            target.gradient_checkpointing = enabled


@register_for_auto_class("AutoModel")
class NanoModel(NanoPreTrainedModel):
    """Expose the Nano XYZ stack as a HF model."""

    def __init__(self, config: NanoConfig) -> None:
        super().__init__(config)
        settings: ModelSettings = config.to_model_settings()
        self.inner = ModelArchitecture(settings)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **_: Union[torch.Tensor, None],
    ) -> Union[BaseModelOutputWithPast, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        request_hidden_states = bool(output_hidden_states)
        skip_lm_head_flag = True
        capture_hidden = request_hidden_states or skip_lm_head_flag
        outputs = self.inner(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            output_hidden_states=capture_hidden,
            output_attentions=bool(output_attentions),
            skip_lm_head=skip_lm_head_flag,
        )

        presents = None
        _, _, presents, hidden_states, attentions = outputs
        if not use_cache:
            presents = None

        hidden_states_tuple = (hidden_states,) if request_hidden_states else None
        attentions_tuple = attentions if output_attentions else None

        model_output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=hidden_states_tuple,
            attentions=attentions_tuple,
        )
        if not return_dict:
            return model_output.to_tuple()
        return model_output

    def get_input_embeddings(self) -> nn.Embedding:
        return self.inner.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.inner.set_input_embeddings(new_embeddings)


@register_for_auto_class("AutoModelForCausalLM")
class NanoForCausalLM(NanoPreTrainedModel, GenerationMixin):
    """Nano XYZ causal LM head compatible with transformers.generate."""

    _keys_to_ignore_on_save = ["lm_head.weight", "model.inner.language_model_head.weight"]

    def __init__(self, config: NanoConfig) -> None:
        super().__init__(config)
        self.model = NanoModel(config)
        self.lm_head = self.model.inner.language_model_head
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        position_ids = kwargs.pop("position_ids", None)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        output_attentions = kwargs.pop("output_attentions", False)
        return_dict = kwargs.pop("return_dict", self.config.use_return_dict)

        allow_hybrid_cache = getattr(self.config, "allow_hybrid_cache", False)
        hybrid_enabled = bool(self.config.use_lcr or self.config.use_gtr)
        supports_cache = not (hybrid_enabled and not allow_hybrid_cache)
        effective_use_cache = use_cache and supports_cache
        if not supports_cache:
            past_key_values = None

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=effective_use_cache,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        hidden_states = model_outputs.hidden_states
        last_hidden_state = model_outputs.last_hidden_state
        presents = model_outputs.past_key_values
        logits = self.lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        causal_outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
            attentions=model_outputs.attentions,
        )

        if return_dict:
            return causal_outputs

        output = (logits, presents, hidden_states, model_outputs.attentions)
        if loss is not None:
            output = (loss,) + output
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        new_tokens_len = input_ids.size(1)
        past_length = (
            past_key_values[0][0].shape[2]
            if past_key_values and past_key_values[0][0] is not None
            else 0
        )
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None and attention_mask.size(1) == new_tokens_len:
                prefix = attention_mask.new_ones(attention_mask.size(0), past_length)
                attention_mask = torch.cat([prefix, attention_mask], dim=1)
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    past_length + input_ids.size(1),
                    device=input_ids.device,
                ).unsqueeze(0)
            position_ids = position_ids[:, -1:]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        inputs.update(kwargs)
        return inputs

    def _reorder_cache(self, past: list, beam_idx: torch.Tensor) -> list:
        return [tuple(p.index_select(0, beam_idx) for p in layer) for layer in past]

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(new_embeddings)
        self.lm_head = self.model.inner.language_model_head

    def tie_weights(self) -> None:
        """Ensure embedding and LM head weights stay shared."""
        embedding = self.get_input_embeddings()
        self._tie_or_clone_weights(self.lm_head, embedding)
