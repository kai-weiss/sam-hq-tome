# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_attention import MemoryAttentionLayer

# ──────────────────────────────────────────────────────────────────────────────
#  Token‑merging back‑ends
# ──────────────────────────────────────────────────────────────────────────────
from tome_sam.tome_algo.tome.merge import bipartite_soft_matching
from tome_sam.tome_algo.grad_tome.merge import grad_bipartite_soft_matching
from tome_sam.tome_algo.pitome.merge import pitome_vision
from tome_sam.tome_algo.pitome.merge_v1 import pitome_vision_v1
from tome_sam.tome_algo.pitome.merge_v2 import pitome_vision_v2
from tome_sam.utils.tome_presets import SAMToMeSetting, ToMeConfig


# Helper ───────────────────────────────────────────────────────────────────────
def _select_merger(metric: torch.Tensor, tome_setting):
    """Return `(merge_fn, unmerge_fn)` according to *tome_setting*."""
    r = tome_setting.params.r
    n_remove = int(metric.shape[1] * r)
    mode = tome_setting.mode

    if n_remove == 0:
        return None, None  # no merging requested

    if mode == "tome":
        return bipartite_soft_matching(metric, r=n_remove)
    if mode == "grad_tome":
        return grad_bipartite_soft_matching(metric, r=n_remove)
    if mode == "pitome":
        return pitome_vision(metric, ratio=r, margin=torch.tensor(tome_setting.params.margin), alpha=tome_setting.params.alpha)
    if mode == "pitome_v1":
        return pitome_vision_v1(metric, ratio=r, margin=torch.tensor(tome_setting.params.margin), alpha=tome_setting.params.alpha)
    if mode == "pitome_v2":
        return pitome_vision_v2(metric, ratio=r, margin=torch.tensor(tome_setting.params.margin), alpha=tome_setting.params.alpha)

    raise ValueError(f"Unsupported ToMe mode: {mode}")


class ToMeMemoryAttentionLayer(MemoryAttentionLayer):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__(activation,
                         cross_attention,
                         d_model,
                         dim_feedforward,
                         dropout,
                         pos_enc_at_attn,
                         pos_enc_at_cross_attn_keys,
                         pos_enc_at_cross_attn_queries,
                         self_attention)

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        tome_setting: ToMeConfig = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        def _merge(tokens: Tensor, pos_tok: Optional[Tensor]):
            """Merge spatial query tokens"""
            # tome_setting = getattr(self, "tome_setting", None)

            if tome_setting is None or tome_setting.params.r == 0:
                return tokens, pos_tok, None  # no merging requested

            ptr = num_k_exclude_rope  # number of pointer / obj‑tokens to keep

            # Split pointer tokens (prefix) and spatial body tokens
            ptr_tok = tokens[:, :ptr, :]
            body_tok = tokens[:, ptr:, :]
            ptr_pos = None if pos_tok is None else pos_tok[:, :ptr, :]
            body_pos = None if pos_tok is None else pos_tok[:, ptr:, :]

            merge_fn, unmerge_fn = _select_merger(body_tok, tome_setting)
            if merge_fn is None:
                return tokens, pos_tok, None  # nothing merged (too few tokens)

            body_tok_red, _ = merge_fn(body_tok)
            if body_pos is not None:
                body_pos_red, _ = merge_fn(body_pos)
                merged_pos = torch.cat([ptr_pos, body_pos_red], dim=1)
            else:
                merged_pos = None

            merged_tok = torch.cat([ptr_tok, body_tok_red], dim=1)

            # Build closure -> unmerge back to original length
            def _unmerge(proc: Tensor):
                ptr_proc = proc[:, :ptr, :]
                body_proc = proc[:, ptr:, :]
                body_full = unmerge_fn(body_proc)
                return torch.cat([ptr_proc, body_full], dim=1)

            # print("test1", tome_setting, tokens, merged_tok)
            return merged_tok, merged_pos, _unmerge

        # Self-Attn
        m_tgt, m_qpos, un_sa = _merge(tgt, query_pos)
        sa_out = self._forward_sa(m_tgt, m_qpos)
        tgt = un_sa(sa_out) if un_sa is not None else sa_out

        # Cross-Attn
        # tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        m_tgt, m_qpos, un_ca = _merge(tgt, query_pos)
        ca_out = self._forward_ca(
            m_tgt, memory, m_qpos, pos, num_k_exclude_rope
        )
        tgt = un_ca(ca_out) if un_ca is not None else ca_out

        # MLP
        m_tgt, _, un_mlp = _merge(tgt, query_pos)

        tgt2 = self.norm3(m_tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        mlp_out = m_tgt + self.dropout3(tgt2)
        tgt = un_mlp(mlp_out) if un_mlp is not None else mlp_out
        return tgt


class ToMeMemoryAttention(MemoryAttention):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
        tome_setting: ToMeConfig = None,
    ):
        print(f"Memory ToMe: {tome_setting}")
        super().__init__(d_model, pos_enc_at_input, layer, num_layers, batch_first)
        self.tome_setting = tome_setting

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        if self.tome_setting is None:
            self.tome_setting = dict()

        for idx, layer in enumerate(self.layers):
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            # print(idx)
            mem_tome_param = self.tome_setting.get(idx)

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                tome_setting=mem_tome_param,
                **kwds,
            )

        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)  # ???? Why is this here

        return normed_output
