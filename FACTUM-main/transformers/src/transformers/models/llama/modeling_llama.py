# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union, List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

# --- START OF GPA MODIFICATION: Define a new, clean output class for our modified decoder layer ---
@dataclass
class LlamaDecoderLayerOutput(object):
    """
    Custom output class for our modified LlamaDecoderLayer, including states for Geometric Pathway Analysis.
    """
    hidden_states: torch.FloatTensor
    attention_weights: Optional[torch.FloatTensor] = None
    past_key_value: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None
    gpa_input_state: Optional[torch.FloatTensor] = None
    gpa_pre_ffn_state: Optional[torch.FloatTensor] = None

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    # Custom arguments for intervention
    disturb_head_ids: Optional[torch.LongTensor] = None,
    add_attention_weight: Optional[torch.FloatTensor] = None,
    select_heads_ids: Optional[torch.LongTensor] = None,
    layer_id=None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # START: Your custom modifications
    if disturb_head_ids is not None:
        bsz, num_heads, q_len, _ = attn_weights.shape

        lower_triangular_mask = torch.tril(torch.ones(q_len, q_len, device=attn_weights.device), diagonal=0).bool()
        lower_triangular_count = lower_triangular_mask.sum().item()
        equal_value = 1.0 / lower_triangular_count if lower_triangular_count > 0 else 0
        equal_values = equal_value * lower_triangular_mask.unsqueeze(0).unsqueeze(1).float()

        for head_id in disturb_head_ids:
            attn_weights[:, head_id, :, :] = equal_values

    if select_heads_ids is not None and add_attention_weight is not None:
        for head_id in select_heads_ids:
            attn_weights[:, head_id, :, :] = attn_weights[:, head_id, :, :] * add_attention_weight
    # END: Your custom modifications

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Custom arguments for intervention
        disturb_head_ids: Optional[torch.LongTensor] = None,
        add_attention_weight: Optional[torch.FloatTensor] = None,
        select_heads_ids: Optional[torch.LongTensor] = None,
        layer_id: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape[:2], self.config.num_attention_heads, self.head_dim)
        key_value_shape = (*input_shape[:2], self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(key_value_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            logger.warning_once("Your custom attention interventions are only compatible with `attn_implementation='eager'`. They will be ignored.")
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # Pass custom args to the attention function
            disturb_head_ids=disturb_head_ids,
            add_attention_weight=add_attention_weight,
            select_heads_ids=select_heads_ids,
            layer_id=layer_id,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        # Custom arguments
        output_residual: Optional[bool] = False,
        disturb_head_ids: Optional[torch.LongTensor] = None,
        path_inter: Optional[bool] = False,
        add_attention_weight: Optional[torch.FloatTensor] = None,
        select_heads_ids: Optional[torch.LongTensor] = None,
        reduce_ffn_weight: Optional[torch.FloatTensor] = None,
        weight_reduce: Optional[bool] = False,
        layer_id: Optional[int] = None,
        # --- START OF GPA MODIFICATION ---
        output_gpa_states: Optional[bool] = False,
        # --- END OF GPA MODIFICATION ---
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # --- GPA: Capture the input to the layer ---
        x_input = hidden_states if output_gpa_states else None
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states_attn, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            disturb_head_ids=disturb_head_ids,
            select_heads_ids=select_heads_ids,
            add_attention_weight=add_attention_weight,
            layer_id=layer_id,
            **kwargs,
        )
        
        if path_inter:
            hidden_states = residual # Path intervention bypasses attention output
        else:
            hidden_states = residual + hidden_states_attn

        # --- GPA: Capture the state after the attention block and its residual connection ---
        x_pre_ffn = hidden_states if output_gpa_states else None
        
        # Fully Connected
        residual_for_mlp = hidden_states 
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # --- GPA: We are only interested in the FFN's *contribution*, so we capture it here before the final residual connection ---
        ffn_contribution = hidden_states
        
        # Custom FFN weight reduction
        if weight_reduce and reduce_ffn_weight is not None:
            hidden_states = residual_for_mlp + reduce_ffn_weight * ffn_contribution
        else:
            hidden_states = residual_for_mlp + ffn_contribution

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_residual:
            outputs += (residual_for_mlp,)
        if output_gpa_states:
            gpa_dict = {"input": x_input, "pre_ffn": x_pre_ffn, "ffn_update": ffn_contribution}
            outputs += (gpa_dict,)

        return outputs


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Custom arguments
        output_residual: Optional[bool] = False,
        disturb_layer_heads_ids: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        path_inter_list: Optional[List[torch.FloatTensor]] = None,
        add_attention_weight: Optional[torch.FloatTensor] = None,
        reduce_ffn_weight: Optional[torch.FloatTensor] = None,
        select_heads: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        select_layers: Optional[List[torch.FloatTensor]] = None,
        # --- START OF GPA MODIFICATION ---
        output_gpa_states: Optional[bool] = False,
        # --- END OF GPA MODIFICATION ---
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            output_residual (`bool`, *optional*):
                Whether or not to return the residual states.
            disturb_layer_heads_ids (`Dict[int, List[torch.FloatTensor]]`, *optional*):
                A dictionary mapping layer indices to a list of head IDs to apply a uniform attention distribution.
            path_inter_list (`List[torch.FloatTensor]`, *optional*):
                A list of layer indices where path intervention (bypassing the attention block output) should be applied.
            add_attention_weight (`torch.FloatTensor`, *optional*):
                A weight to multiply with the attention scores of selected heads.
            reduce_ffn_weight (`torch.FloatTensor`, *optional*):
                A weight to multiply with the FFN block output for selected layers.
            select_heads (`Dict[int, List[torch.FloatTensor]]`, *optional*):
                A dictionary mapping layer indices to a list of head IDs for attention weight amplification.
            select_layers (`List[torch.FloatTensor]`, *optional*):
                A list of layer indices where FFN output weight reduction should be applied.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The past_key_values should be either a Cache object or None.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_residuals = () if output_residual else None
        # GPA: Initialize our tuple
        all_gpa_states = () if output_gpa_states else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            disturb_head_ids = disturb_layer_heads_ids.get(i) if disturb_layer_heads_ids is not None else None
            path_inter = True if path_inter_list is not None and i in path_inter_list else False
            select_head_ids = select_heads.get(i) if select_heads is not None else None
            reduce_weight = True if select_layers is not None and i in select_layers else False

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_residual=output_residual,
                disturb_head_ids=disturb_head_ids,
                path_inter=path_inter,
                add_attention_weight=add_attention_weight,
                select_heads_ids=select_head_ids,
                reduce_ffn_weight=reduce_ffn_weight,
                weight_reduce=reduce_weight,
                layer_id=i,
                # --- GPA: Pass our flag down ---
                output_gpa_states=output_gpa_states,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            # Unpack the rest of the outputs carefully
            output_idx = 1
            if output_attentions:
                all_self_attns += (layer_outputs[output_idx],)
                output_idx += 1
            if output_residual:
                all_residuals += (layer_outputs[output_idx],)
                output_idx += 1
            ## GPA Change ##
            if output_gpa_states:
                all_gpa_states += (layer_outputs[output_idx],)
                
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        if output_residual:
            output.residuals = all_residuals

        ## GPA Change ##
        if output_gpa_states:
            # Attach the collected states to the final output object
            output.gpa_states = all_gpa_states
            
        return output


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, select_layers: list=None, select_heads: list=None, layers_max_min: list=None, head_max_min: list=None, weight: float=None, final_max_min: list=None):
        r"""
        Args:
            select_layers (`list`, *optional*):
                List of layer indices for which to perform special processing and analysis.
            select_heads (`list`, *optional*):
                List of [layer_id, head_id] pairs for which to perform special attention processing.
            layers_max_min (`list`, *optional*):
                A list containing the [max, min] values for normalizing layer-level scores.
            head_max_min (`list`, *optional*):
                A list containing the [max, min] values for normalizing head-level scores.
            weight (`float`, *optional*):
                A weight factor used in combining knowledge scores.
            final_max_min (`list`, *optional*):
                A list containing the [max, min] values for normalizing the final combined score.
        """
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.select_layers = select_layers
        self.select_heads = select_heads
        self.layers_max_min = layers_max_min
        self.head_max_min = head_max_min
        self.weight = weight
        self.final_max_min = final_max_min
        self.prefix_len = -1
        self.prefix_hidden_state = None
        self.add_attention_weight = 1.5
        self.reduce_ffn_weight = 0.5
        self.threshold = 0.5

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def calculate_dist(self, sep_vocabulary_dist, sep_attention_dist):
        softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
        softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

        M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

        log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
        log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

        kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)
        kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)
        js_divs = 0.5 * (kl1 + kl2) * 10e5
        return js_divs

    def normalize(self, max_min, score_sum):
        if max_min is None: return 0.0
        max_value, min_value = max_min
        if max_value - min_value != 0:
            normalized_value = (score_sum - min_value) / (max_value - min_value)
        else:
            normalized_value = 0.0
        return normalized_value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Custom arguments
        knowledge_layers: Optional[List[int]] = None,
        disturb_layer_heads_ids: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        path_inter_list: Optional[List[torch.FloatTensor]] = None,
        ## GPA Change ##
        output_gpa_states: Optional[bool] = False,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            knowledge_layers (`List[int]`, *optional*):
                List of layer indices for which to compute and return knowledge-related logits.
            disturb_layer_heads_ids (`Dict[int, List[torch.FloatTensor]]`, *optional*):
                A dictionary mapping layer indices to a list of head IDs to apply a uniform attention distribution.
            path_inter_list (`List[torch.FloatTensor]`, *optional*):
                A list of layer indices where path intervention (bypassing the attention block output) should be applied.
        """
        from torch.nn import CrossEntropyLoss

        is_gpa_mode = output_gpa_states
        is_pks_mode = knowledge_layers is not None
        is_intervention_mode = self.select_layers is not None and self.select_heads is not None


        # New GPA (CAS/POS) mode
        if is_gpa_mode:
            output_attentions = True # CAS needs final layer attentions
            output_hidden_states = True # POS needs all hidden states
            output_residual = False # GPA does not need the PKS-specific residual
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_attentions = True if self.select_heads is not None else output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_hidden_states = True if self.select_layers is not None else output_hidden_states
        
        output_residual = True if is_pks_mode or self.select_layers is not None else False
        
        
        model_kwargs = {key: value for key, value in kwargs.items() if key not in LossKwargs.get_keys()}

        # Initial forward pass to get all necessary states
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            output_residual=output_residual,
            disturb_layer_heads_ids=disturb_layer_heads_ids,
            path_inter_list=path_inter_list,
            ## GPA Change ##
            output_gpa_states=output_gpa_states,
            **model_kwargs,
        )


        # Case 1: NEW Combined GPA and PKS Mode
        if is_gpa_mode and is_pks_mode:
            custom_outputs = {}
            # GPA data
            gpa_states = getattr(outputs, "gpa_states", None)
            if gpa_states is None: 
                raise ValueError("GPA states requested but not found `gpa_states` requires `gpa_states=True`")
            custom_outputs['gpa_states'] = gpa_states
            # PKS data
            logits_dict = {}
            hidden_states_tuple = outputs.hidden_states
            residuals_tuple = getattr(outputs, "residuals", None)
            if residuals_tuple is None: 
                raise ValueError("`select_layers` requires `output_residual=True` but no residuals were returned.")
            for layer in knowledge_layers:
                is_last_hidden = (layer + 1) == len(hidden_states_tuple) -1
                current_hs = hidden_states_tuple[layer + 1]
                logits = self.lm_head(current_hs if is_last_hidden else self.model.norm(current_hs))
                residual_logits = self.lm_head(self.model.norm(residuals_tuple[layer]))
                logits_dict[layer] = (logits, residual_logits)
            custom_outputs['pks_logits'] = logits_dict
            
            # Standard final output processing (using standard causal LM loss)
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            final_outputs = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
            return custom_outputs, final_outputs
        
        ## GPA Change: New branch for GPA metrics ##
        elif is_gpa_mode and not is_pks_mode:
            gpa_states = getattr(outputs, "gpa_states", None)
            if gpa_states is None:
                raise ValueError("`output_gpa_states=True` was passed, but no GPA states were returned from the model.")
            
            # Standard output processing
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            final_outputs = CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )

            # Correct Return Signature: (gpa_states_tuple, standard_output_object)
            return gpa_states, final_outputs
        
        # Case 2: PKS/ECS Metrics Requested (baseline)
        elif is_pks_mode and not is_gpa_mode:
            logits_dict = {}
            hidden_states_tuple = outputs.hidden_states
            residuals_tuple = getattr(outputs, "residuals", None)
            if residuals_tuple is None:
                raise ValueError("`knowledge_layers` requires `output_residual=True` but no residuals were returned.")

            for knowledge_layer in knowledge_layers:
                is_last_hidden = (knowledge_layer + 1) == len(hidden_states_tuple) -1
                current_hs = hidden_states_tuple[knowledge_layer + 1]

                logits = self.lm_head(current_hs if is_last_hidden else self.model.norm(current_hs))
                residual_logits = self.lm_head(self.model.norm(residuals_tuple[knowledge_layer]))
                logits_dict[knowledge_layer] = (logits, residual_logits)

            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            final_outputs = CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
            return logits_dict, final_outputs

        # Case 3: Other intervention (original select_layers logic)
        elif is_intervention_mode:
            logits_dict = {}
            hidden_states_tuple = outputs.hidden_states
            residuals_tuple = getattr(outputs, "residuals", None)
            if residuals_tuple is None:
                raise ValueError("`select_layers` requires `output_residual=True` but no residuals were returned.")

            for knowledge_layer in self.select_layers:
                is_last_hidden = (knowledge_layer + 1) == len(hidden_states_tuple) - 1
                current_hs = hidden_states_tuple[knowledge_layer + 1]
                logits = self.lm_head(current_hs if is_last_hidden else self.model.norm(current_hs))
                residual_logits = self.lm_head(self.model.norm(residuals_tuple[knowledge_layer]))
                logits_dict[knowledge_layer] = (logits[:, -1, :], residual_logits[:, -1, :])

            js_divs_list = [self.calculate_dist(logits_dict[layer][0], logits_dict[layer][1]) for layer in self.select_layers]
            js_divs_score_normalized = self.normalize(self.layers_max_min, sum(js_divs_list) / len(js_divs_list))

            attentions_list = []
            for layer_id, head_id in self.select_heads:
                if outputs.attentions is not None and len(outputs.attentions) > layer_id:
                    attentions_list.append(outputs.attentions[layer_id][:, head_id, :, :])

            pointer_scores_list = [attn_scores[:, -1, :] for attn_scores in attentions_list]
            if not pointer_scores_list:
                final_score_normalized = js_divs_score_normalized
            else:
                pointer_probs_list = torch.cat([scores[:, :self.prefix_len] for scores in pointer_scores_list], dim=0)

                top_k = int(pointer_probs_list.shape[-1] * 0.1)
                top_k_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)[:, :top_k]
                flattened_indices = top_k_indices.flatten()

                current_hidden_state = outputs.last_hidden_state[:, -1, :].squeeze(0)
                if outputs.last_hidden_state.shape[1] != 1:
                    self.prefix_hidden_state = outputs.last_hidden_state[0]

                if self.prefix_hidden_state is not None:
                    selected_hidden_states = self.prefix_hidden_state[flattened_indices.to(self.prefix_hidden_state.device)]
                    top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)
                    attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1)
                    expanded_current_hs = current_hidden_state.unsqueeze(0).expand_as(attend_token_hidden_state)
                    cosine_similarity = F.cosine_similarity(attend_token_hidden_state, expanded_current_hs, dim=1)
                    cosine_similarity_normalized = self.normalize(self.head_max_min, torch.mean(cosine_similarity))
                    final_score = js_divs_score_normalized - self.weight * cosine_similarity_normalized
                    final_score_normalized = self.normalize(self.final_max_min, final_score)
                else:
                    final_score_normalized = js_divs_score_normalized

            if final_score_normalized > self.threshold:
                select_head_dict = {}
                for layer, head in self.select_heads:
                    select_head_dict.setdefault(layer, []).append(head)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states, cache_position=cache_position,
                    add_attention_weight=self.add_attention_weight,
                    reduce_ffn_weight=self.reduce_ffn_weight,
                    select_heads=select_head_dict,
                    select_layers=self.select_layers,
                    **model_kwargs
                )

            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            return CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )

        # Case 4: Standard Forward Pass
        else: 
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                logits_to_keep = kwargs.get("logits_to_keep", 0)
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                loss_logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
                loss = self.loss_function(logits=loss_logits, labels=labels, **kwargs)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


@auto_docstring(
    custom_intro="""..."""
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # Custom arguments
        disturb_layer_heads_ids: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        path_inter_list: Optional[List[torch.FloatTensor]] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        Args:
            disturb_layer_heads_ids (`Dict[int, List[torch.FloatTensor]]`, *optional*):
                A dictionary mapping layer indices to a list of head IDs to apply a uniform attention distribution.
            path_inter_list (`List[torch.FloatTensor]`, *optional*):
                A list of layer indices where path intervention (bypassing the attention block output) should be applied.
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            disturb_layer_heads_ids=disturb_layer_heads_ids,
            path_inter_list=path_inter_list,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in inputs_embeds. Results may be "
                "unexpected if using padding tokens in conjunction with inputs_embeds."
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # Custom arguments
        disturb_layer_heads_ids: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        path_inter_list: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        Args:
            disturb_layer_heads_ids (`Dict[int, List[torch.FloatTensor]]`, *optional*):
                A dictionary mapping layer indices to a list of head IDs to apply a uniform attention distribution.
            path_inter_list (`List[torch.FloatTensor]`, *optional*):
                A list of layer indices where path intervention (bypassing the attention block output) should be applied.
        """
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            disturb_layer_heads_ids=disturb_layer_heads_ids,
            path_inter_list=path_inter_list,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # Custom arguments
        disturb_layer_heads_ids: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        path_inter_list: Optional[List[torch.FloatTensor]] = None,
    ) -> TokenClassifierOutput:
        r"""
        Args:
            disturb_layer_heads_ids (`Dict[int, List[torch.FloatTensor]]`, *optional*):
                A dictionary mapping layer indices to a list of head IDs to apply a uniform attention distribution.
            path_inter_list (`List[torch.FloatTensor]`, *optional*):
                A list of layer indices where path intervention (bypassing the attention block output) should be applied.
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            disturb_layer_heads_ids=disturb_layer_heads_ids,
            path_inter_list=path_inter_list,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]