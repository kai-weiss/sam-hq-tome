# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union, Optional
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
from sam2.modeling.backbones.hieradet import MultiScaleAttention

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

from tome_sam.tome_algo.grad_tome.merge import grad_bipartite_soft_matching
from tome_sam.tome_algo.pitome.merge import pitome_vision
from tome_sam.tome_algo.pitome.merge_v1 import pitome_vision_v1
from tome_sam.tome_algo.pitome.merge_v2 import pitome_vision_v2
from tome_sam.tome_algo.tome.merge import bipartite_soft_matching
from tome_sam.tome_algo.tomesd.merge import bipartite_soft_matching_random2d, random_25_bipartite_soft_matching
from tome_sam.utils.tome_presets import ToMeConfig, SAMToMeSetting


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class EfficientMultiScaleAttention(MultiScaleAttention):
    def __init__(
        self,
        tome_setting,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__(dim, dim_out, num_heads, q_pool)
        self.tome_setting = tome_setting

    def forward(self, x: torch.Tensor, merge_operations=None) -> torch.Tensor:
        B, H, W, _ = x.shape
        C = self.dim_out // self.num_heads  # Matches the qkv output dimension

        x = x.reshape(B, H * W, -1)  # (B, N, C * nHeads)
        # token merging on x
        x_merge, x_unmerge = Callable, Callable

        # mean aggregation over multiple heads to reduce dimensions for similarity comparison
        metric = aggregate_over_head(x, self.num_heads, option='mean')
        if self.tome_setting.mode == 'tomesd':
            generator = None
            if not self.tome_setting.params.no_rand:
                generator = torch.Generator(device=x.device).manual_seed(42)

            x_merge, x_unmerge = bipartite_soft_matching_random2d(
                metric=metric, w=W, h=H,
                r=int(H * W * self.tome_setting.params.r),
                sx=self.tome_setting.params.sx, sy=self.tome_setting.params.sy,
                no_rand=self.tome_setting.params.no_rand,
                generator=generator,
            )

        if self.tome_setting.mode == 'tome25':
            generator = torch.Generator(device=x.device).manual_seed(42)
            x_merge, x_unmerge = random_25_bipartite_soft_matching(
                metric=metric, r=int(H * W * self.tome_setting.params.r),
                generator=generator,
            )

        if self.tome_setting.mode == 'tome':
            x_merge, x_unmerge = bipartite_soft_matching(
                metric=metric, r=int(H * W * self.tome_setting.params.r),
            )

        if self.tome_setting.mode == 'grad_tome':
            x_merge, x_unmerge = grad_bipartite_soft_matching(
                metric=metric, r=int(H * W * self.tome_setting.params.r),
            )

        if self.tome_setting.mode == 'pitome':
            x_merge, x_unmerge = pitome_vision(
                metric=metric, ratio=self.tome_setting.params.r,
                margin=torch.tensor(self.tome_setting.params.margin),
                alpha=self.tome_setting.params.alpha,
            )

        if self.tome_setting.mode == 'pitome_v1':
            x_merge, x_unmerge = pitome_vision_v1(
                metric=metric, ratio=self.tome_setting.params.r,
                margin=torch.tensor(self.tome_setting.params.margin),
                alpha=self.tome_setting.params.alpha,
            )

        if self.tome_setting.mode == 'pitome_v2':
            x_merge, x_unmerge = pitome_vision_v2(
                metric=metric, ratio=self.tome_setting.params.r,
                margin=torch.tensor(self.tome_setting.params.margin),
                alpha=self.tome_setting.params.alpha,
            )

        x_reduced, merged_indices = x_merge(x)
        _, N_reduced, _ = x_reduced.shape
        # qkv in shape of (B, N_reduced, 3*nHeads*C)
        qkv = self.qkv(x_reduced)
        # qkv in shape of (3, B*nHeads, N_reduced, C)
        qkv = qkv.view(B, N_reduced, 3, self.num_heads, C).permute(2, 0, 3, 1, 4).reshape(3, B*self.num_heads, N_reduced, C)
        # q,k,v in shape of (B*nHeads, N_reduced, C)
        q, k, v = qkv.unbind(0)

        # q, k, v originally [B*nHeads, N_reduced, C]
        q = q.view(B, self.num_heads, N_reduced, C)
        k = k.view(B, self.num_heads, N_reduced, C)
        v = v.view(B, self.num_heads, N_reduced, C)

        # Q,K,V pooling (for downsample at stage changes)
        if self.q_pool:
            q = q.reshape(B, H, W, -1)
            k = k.reshape(B, H, W, -1)
            v = v.reshape(B, H, W, -1)

            q = do_pool(q, self.q_pool)
            k = do_pool(k, self.q_pool)
            v = do_pool(v, self.q_pool)

            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)
            k = k.reshape(B, H * W, self.num_heads, -1)
            v = v.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(q, k, v)
        # Transpose back
        x = x.transpose(1, 2)

        # reshape x from (B*nHeads, N_reduced, C) to (B, N_reduced, C*nHeads) for unmerging
        x = x.reshape(B, self.num_heads, N_reduced, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N_reduced, -1)

        # token unmerge
        x = x_unmerge(x)  # (B, N, C*nHeads)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        tome_setting: ToMeConfig = None,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        # Choose attention module based on token merging configuration
        if tome_setting is not None:
            self.attn = EfficientMultiScaleAttention(
                tome_setting=tome_setting,
                dim=dim,
                dim_out=dim_out,
                num_heads=num_heads,
                q_pool=self.pool,
            )
        else:
            self.attn = MultiScaleAttention(
                dim,
                dim_out,
                num_heads=num_heads,
                q_pool=self.pool,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        tome_setting: Optional[SAMToMeSetting] = None,
    ):
        print(f"ToMe: {tome_setting}")
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if tome_setting is None:
            tome_setting = dict()

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):

            if i in tome_setting:
                vit_tome_param = tome_setting[i]
            else:
                vit_tome_param = None

            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                tome_setting=vit_tome_param
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)

def aggregate_over_head(x: torch.Tensor, num_heads: int, option: str=None) -> torch.Tensor:
    """
    Aggregates over multiple heads
    Args:
        x(tensor): input tokens with [B, N, C*num_heads]
        num_heads(int): number of heads
        option: how to aggregate over heads

    Returns:
        tensor: aggregated tokens with [B, N, C]
    """
    B, N, _ = x.shape
    metric = x.view(B, N, num_heads, -1)

    if option == 'max':
        metric = metric.max(dim=2).values
    elif option == 'mean':
        metric = metric.mean(dim=2)
    elif option == 'sum':
        metric = metric.sum(dim=2)
    else:
        metric = x

    return metric