# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, attn_params=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = attn_params['num_tokens']
        self.num_objects = attn_params['num_objects']

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        # New added
        self.text_to_weight = nn.Linear(hidden_size, 1, bias=False)
        self.layer_name = attn_params['layer_name']

        self.entity_text_embeds = None  # These params are used in the `forward` function
        self.tokenizer_entity_indexes = None

    def cross_attention(self, attn, query, text_embeds):
        key = attn.to_k(text_embeds)
        value = attn.to_v(text_embeds)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, None)
        res_states = torch.bmm(attention_probs, value)
        res_states = attn.batch_to_head_dim(res_states)

        return res_states

    def cross_attention_attn(self, attn, query, text_embeds):
        key = attn.to_k(text_embeds)
        value = attn.to_v(text_embeds)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(key, query, None)
        batch_size, fea_size = attention_probs.shape[0] // attn.heads, int(attention_probs.shape[-1] ** (1/2))
        # cur_attn_map = attn_ip_attention_probs.reshape(batch_size, attn.heads, self.num_tokens, fea_size, fea_size)
        # attention_probs = attention_probs.reshape(batch_size, attn.heads, -1, fea_size, fea_size)
        attention_probs = attention_probs.reshape(batch_size, attn.heads, -1, fea_size * fea_size)

        return attention_probs

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)   # None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        entity_text_embeds = self.entity_text_embeds
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            # end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            end_pos = encoder_hidden_states.shape[1] - self.num_objects * self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                entity_text_embeds = attn.norm_encoder_hidden_states(entity_text_embeds)   # New added

        query = attn.head_to_batch_dim(query)
        hidden_states = self.cross_attention(attn, query, encoder_hidden_states)

        # For ip-adapter
        ip_hidden_states = ip_hidden_states.reshape(batch_size, self.num_objects, self.num_tokens, -1)

        all_entity_hidden_states, all_word_attn_maps = [], []
        for cur_idx in range(self.num_objects):
            cur_ip_hidden_states = ip_hidden_states[:, cur_idx]
            cur_ip_key = self.to_k_ip(cur_ip_hidden_states)
            cur_ip_value = self.to_v_ip(cur_ip_hidden_states)

            cur_ip_key = attn.head_to_batch_dim(cur_ip_key)
            cur_ip_value = attn.head_to_batch_dim(cur_ip_value)

            cur_ip_attention_probs = attn.get_attention_scores(query, cur_ip_key, None)
            cur_ip_hidden_states = torch.bmm(cur_ip_attention_probs, cur_ip_value)
            cur_ip_hidden_states = attn.batch_to_head_dim(cur_ip_hidden_states)
            
            cur_entity_text_embeds = entity_text_embeds[:, cur_idx]
            cur_entity_hidden_states = self.scale * cur_ip_hidden_states
            all_entity_hidden_states.append(cur_entity_hidden_states)

            # Save the word attention maps
            cur_word_attn_map = self.cross_attention_attn(attn, query, cur_entity_text_embeds).detach() # (batch_size, attn.heads, num_text_tokens, fea_len)
            all_word_attn_maps.append(cur_word_attn_map)

        # Merge the text features and multiple image features
        all_entity_hidden_states = torch.stack(all_entity_hidden_states, dim=1) # (batch_size, num_objects, fea_len, dim)
        all_word_attn_maps = torch.stack(all_word_attn_maps, dim=1) # (batch_size, num_objects, attn.heads, num_text_tokens, fea_len)
        all_word_attn_maps = all_word_attn_maps.mean(dim=2) # (batch_size, num_objects, num_text_tokens, fea_len)
        tokenizer_entity_indexes = self.tokenizer_entity_indexes
        select_word_attn_maps = []
        for batch_idx in range(batch_size):
            for object_idx in range(self.num_objects):
                batch_object_token_len = tokenizer_entity_indexes[batch_idx][object_idx][1]
                batch_object_attn_map = all_word_attn_maps[batch_idx, object_idx, 1:1+batch_object_token_len].mean(dim=0)   # (fea_len)
                select_word_attn_maps.append(batch_object_attn_map)
        select_word_attn_maps = torch.stack(select_word_attn_maps, dim=0).reshape(batch_size, self.num_objects, -1) # (batch_size, num_objects, fea_len)
        # ip attn & text weight
        select_word_attn_maps = select_word_attn_maps / select_word_attn_maps.sum(dim=1, keepdim=True)
        to_object_weights = select_word_attn_maps.unsqueeze(dim=-1) # (batch_size, num_objects, fea_len, 1)
        text_weight = F.sigmoid(self.text_to_weight(hidden_states))    # (batch_size, fea_len, 1)
        text_weight = (text_weight / text_weight.mean(dim=1, keepdim=True)).unsqueeze(dim=1)  # (batch_size, 1, fea_len, 1)
        to_object_weights = to_object_weights * text_weight

        all_entity_hidden_states = (to_object_weights * all_entity_hidden_states).sum(dim=1)    # (batch_size, fea_len, dim)
        hidden_states = hidden_states + all_entity_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states