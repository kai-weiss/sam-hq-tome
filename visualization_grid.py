# grid_visualization.py
"""Utility functions to reproduce the coloured token‑merging grids from the PiToMe


* ``img_vis`` – a **PIL.Image** (or ``torch.Tensor`` H×W×3 uint8) already
  resized to the ViT patch resolution that was fed into the backbone.
* ``source``  – tensor **[B, L, HW]** with the *origin index* of every token
  after merging (the Tensor you stored in ``model._info['source']``).
* ``attn_score`` – tensor **[B, HW]** (last‑layer attention summed over heads;
  e.g. ``blocks[-1].attn.attention_map.sum(1)[:, 0, :]``).

The function returns a **PIL.Image** with exactly the same resolution as the
input image and the coloured grid overlay.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion

__all__ = [
    "generate_colormap",
    "make_token_merge_grid",
]


# -----------------------------------------------------------------------------
# Colour map helper
# -----------------------------------------------------------------------------

def _adjust_lightness(base_rgb: Tuple[float, float, float], alpha: float) -> Tuple[float, float, float]:
    """Linearly blend *base_rgb* towards white as *alpha*→1."""
    return tuple((1.0 - alpha) * b + alpha for b in base_rgb)


def generate_colormap(num_groups: int, attention_score: torch.Tensor, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Return a list of ``num_groups`` RGB tuples (0‑1 float).

    * Groups with **higher** attention receive **brighter** colours.
    * We start from a blue base‑hue to match Fig. 1 of the PiToMe paper.
    """
    rng = random.Random(seed)
    base_colors = [(0.0, 0.0, 1.0),  # start‑palette (blue)
                   (0.0, 0.7, 1.0),
                   (0.1, 0.2, 1.0),
                   (0.0, 0.5, 1.0)]

    attn = attention_score.detach().flatten().float()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    cmap: List[Tuple[float, float, float]] = []
    for i in range(num_groups):
        base = rng.choice(base_colors)
        cmap.append(_adjust_lightness(base, attn[i % attn.numel()].item()))
    return cmap


# -----------------------------------------------------------------------------
# Main visualisation
# -----------------------------------------------------------------------------

def make_token_merge_grid(
    img: Image.Image | np.ndarray | torch.Tensor,
    source: torch.Tensor,
    attention_score: torch.Tensor,
    *,
    patch_size: int = 16,
    class_token: bool = False,
) -> Image.Image:
    """Return a *PIL.Image* with the coloured token‑merge grid overlay.

    Parameters
    ----------
    img
        Input image at the same resolution used for the ViT backbone.
    source
        Tensor of shape ``[B, L, HW]`` – provenance indices recorded during the
        token‑merging forward pass (one entry per layer, per token).
    attention_score
        Tensor of shape ``[B, HW]`` – summed attention weights of the last
        layer (*before* softmax head‑reduction suggested by the PiToMe paper).
    patch_size
        ViT patch‑size (typically 16 for SAM‑ViT‑L).
    class_token
        Set *True* if your backbone prepends a CLS token (e.g. BLIP).  SAM and
        HQ‑SAM/SAM 2 *do not* use a CLS token → keep *False* (default).
    """

    # ------------------------------------------------------------------
    # Prepare image
    # ------------------------------------------------------------------
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    elif isinstance(img, Image.Image):
        img_np = np.asarray(img, dtype=np.uint8)
    else:  # np.ndarray
        img_np = img.astype(np.uint8)

    img_np = img_np / 255.0  # normalise to 0‑1
    h, w, _ = img_np.shape
    ph, pw = h // patch_size, w // patch_size

    # ------------------------------------------------------------------
    # Prepare *source* (choose *best* layer)
    # ------------------------------------------------------------------
    if class_token:
        source = source[:, :, 1:]  # drop CLS token
    vis = source.argmax(dim=1)  # [B, HW] group‑id per token (select best layer)

    num_groups = int(vis.max().item() + 1)

    # pick colormap once (first image in batch)
    cmap = generate_colormap(num_groups, attention_score[0])

    vis_img = np.zeros_like(img_np)
    for gid in range(num_groups):
        # mask tokens that belong to *gid*
        mask = (vis[0] == gid).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest").view(h, w, 1).numpy()

        # interior / edge separation (erode by 1 pixel)
        mask_eroded = binary_erosion(mask[..., 0], iterations=1)[..., None]
        mask_edge = mask - mask_eroded

        # region colour (average underlying pixels) – fallback to 0 if empty
        denom = mask.sum()
        mean_color = (mask * img_np).sum(axis=(0, 1)) / denom if denom > 0 else np.zeros(3)

        vis_img += mask_eroded * mean_color.reshape(1, 1, 3)
        vis_img += mask_edge * (np.array(cmap[gid]) * 4).reshape(1, 1, 3)

    vis_img = (vis_img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(vis_img)
