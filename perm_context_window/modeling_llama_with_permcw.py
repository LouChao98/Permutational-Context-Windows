import math
from abc import ABC
from typing import Optional, Tuple, Dict, Union, List

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    rotate_half,
    _make_causal_mask,
    _expand_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
import logging

"""
The following code is mainly copy+paste from the original modelling_llama.py:
LlamaAttention uses a caching mechanism for the positional rotation vectors (using LlamaRotaryEmbedding). 
This mechanism forces us to override LLaMa attention layer, which in turn forces us to override the decoder, 
and model (so that the correct forward function would be called).
"""

logger = logging.getLogger(__name__)


class LlamaForCausalLMPermCW(LlamaForCausalLM, ABC):
    _no_split_modules = ["LlamaDecoderLayerPermCW"]

    def __init__(self, config: LlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        # using our Llama model variant:
        self.model = LlamaModelPermCW(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None,
        windows_key_values=None, attention_mask=None, sum_windows_size=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1)
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            elif sum_windows_size:
                 position_ids = position_ids[:, sum_windows_size:]

        if windows_key_values and not past_key_values:
            past_key_values = windows_key_values

        return  {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }


class LlamaModelPermCW(LlamaModel, ABC):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # using the alternative decoder layer:
        self.layers = nn.ModuleList([LlamaDecoderLayerPermCW(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if isinstance(position_ids, tuple):
                *position_ids, mask, version = position_ids
                position_ids = tuple(map(lambda t: t.view(-1, seq_length).long(), position_ids)) + (mask, version)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            is_enumerating=isinstance(position_ids, tuple),
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length, is_enumerating=False
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1 and not is_enumerating:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class LlamaDecoderLayerPermCW(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # overriding attention:
        self.self_attn = LlamaAttentionPermCW(config=config)


def apply_one_rope(state, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    state = (state * cos) + (rotate_half(state) * sin)
    return state


class LlamaAttentionPermCW(LlamaAttention):
    # we have to override the forward attention due to the rotary embeddings caching mechanism
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # *** changes to the original code to accommodate PermCW:
        # making sure that the model generates rotary embeddings in the correct length:

        if isinstance(position_ids, tuple):
            assert past_key_value is None
            
            if position_ids[-1] == 1:
                context_position_ids, suffix_position_ids, prefix_position_ids, cont_mask, _ = position_ids
                seq_len = suffix_position_ids.max() + 1
                cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
                key_states = apply_one_rope(key_states, cos, sin, context_position_ids) 
                
                if past_key_value is not None:
                    # reuse k, v, self_attention
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)

                past_key_value = (key_states, value_states) if use_cache else None

                key_states = key_states / math.sqrt(self.head_dim)
                suf_q_states = apply_one_rope(query_states, cos, sin, suffix_position_ids)
                pre_q_states = apply_one_rope(query_states, cos, sin, prefix_position_ids)

                suf_attn_weights = torch.matmul(suf_q_states, key_states.transpose(2, 3))
                pre_attn_weights = torch.matmul(pre_q_states, key_states.transpose(2, 3))
                del suf_q_states, pre_q_states

                m1 = suf_attn_weights.triu_(diagonal=1) + pre_attn_weights.tril_(diagonal=-1)
                del suf_attn_weights, pre_attn_weights

                con_q_states = apply_one_rope(query_states, cos, sin, context_position_ids)
                con_attn_weights = torch.matmul(con_q_states, key_states.transpose(2, 3))
                attn_weights = torch.where(cont_mask, con_attn_weights, m1)

                con_rp = context_position_ids[:, :, None] - context_position_ids[:, None]
                suf_rp = suffix_position_ids[:, :, None] - context_position_ids[:, None]
                pre_rp = prefix_position_ids[:, :, None] - context_position_ids[:, None]
                rp = torch.where(
                    cont_mask,
                    con_rp,
                    suf_rp.triu(diagonal=1) + pre_rp.tril(diagonal=-1),
                )
                attention_mask[rp.unsqueeze(1) < 0] = torch.finfo(attention_mask.dtype).min

            elif position_ids[-1] == 3:
                context_position_ids, suffix_position_ids, prefix_position_ids, cont_mask, _ = position_ids
                seq_len = suffix_position_ids.max() + 1
                cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
                key_states = apply_one_rope(key_states, cos, sin, context_position_ids) 
                
                if past_key_value is not None:
                    # reuse k, v, self_attention
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)

                past_key_value = (key_states, value_states) if use_cache else None

                key_states = key_states / math.sqrt(self.head_dim)
                con_q_states = apply_one_rope(query_states, cos, sin, context_position_ids)
                suf_q_states = apply_one_rope(query_states, cos, sin, suffix_position_ids)
                pre_q_states = apply_one_rope(query_states, cos, sin, prefix_position_ids)

                con_attn_weights = torch.matmul(con_q_states, key_states.transpose(2, 3))
                suf_attn_weights = torch.matmul(suf_q_states, key_states.transpose(2, 3))

                attn_weights = torch.where(cont_mask, suf_attn_weights, con_attn_weights)

                zero_weight =  torch.matmul(pre_q_states, key_states[:, :, 0, None].transpose(2, 3))
                attn_weights[..., 0] = zero_weight[..., 0]

                con_rp = context_position_ids[:, :, None] - context_position_ids[:, None]
                suf_rp = suffix_position_ids[:, :, None] - context_position_ids[:, None]
                rp = torch.where(cont_mask, suf_rp, con_rp)
                zero_rp = prefix_position_ids[:, :] - context_position_ids[:, 0, None]
                rp[0, :, 0] = zero_rp

                attention_mask[rp.unsqueeze(1) < 0] = torch.finfo(attention_mask.dtype).min
            
            rp[attention_mask[0] < -1] = -1
            rp
            # import matplotlib.pyplot as plt
            # plt.imshow(attention_mask[0,0].detach().cpu().numpy() < 0)
            # plt.savefig('att_mask.png')

        else:
            seq_len = kv_seq_len if position_ids is None else int(torch.max(position_ids) + 1)
            cos, sin = self.rotary_emb(value_states, seq_len=seq_len)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # [bsz, nh, t, hd]

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).to(query_states.dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
