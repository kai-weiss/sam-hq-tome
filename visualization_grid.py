from __future__ import annotations

import math
from typing import Literal, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion


# ----- PiToMe-like colormap (blue base, brightness by attention) -----
def _generate_colormap(N: int, attention_score: torch.Tensor, seed: int = 0):
    torch.manual_seed(seed)
    a = attention_score.detach().flatten().float()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    base = (0.2, 0.4, 0.8)

    def adj(alpha):
        # lightness adjustment against base
        return tuple(float(np.clip(c + 0.3 * alpha if i != 1 else c, 0.0, 1.0))
                     for i, c in enumerate(base))

    return [adj(a[i % a.numel()].item()) for i in range(N)]


# ----- helper to remap arbitrary labels
def _compact_labels_1d(ids: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
    uniq = torch.unique(ids, sorted=True)
    lut = {int(v): i for i, v in enumerate(uniq.tolist())}
    mapped = ids.clone()
    for v, i in lut.items():
        mapped[ids == v] = i
    return mapped, uniq.cpu().numpy()


# ----- optional small-region filtering on the patch grid -----
def _filter_small_regions(labels: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return labels
    vals, counts = np.unique(labels, return_counts=True)
    small = set(vals[counts < min_area].tolist())
    if not small:
        return labels
    lab = labels.copy()
    for _ in range(2):
        pad = np.pad(lab, 1, mode="edge")
        neigh = np.stack([
            pad[0:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, 0:-2], pad[1:-1, 2:],
            pad[0:-2, 0:-2], pad[0:-2, 2:], pad[2:, 0:-2], pad[2:, 2:]
        ], axis=0)  # [8, H, W]
        is_small = np.isin(lab, list(small))
        if is_small.any():
            for k in range(8):
                cand = neigh[k]
                repl = is_small & (~np.isin(cand, list(small)))
                lab[repl] = cand[repl]
    return lab


def _attn_to_grid(attn_flat: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """
    attn_flat: [HW_attn] 1D attention
    Returns a [ph, pw] grid
    If HW_attn != ph*pw, we infer a factorization
    (ha, wa) of HW_attn close to pw/ph and bilinearly upsample to (ph, pw)
    """
    attn_flat = attn_flat.flatten()
    HWt = attn_flat.numel()
    if HWt == ph * pw:
        return attn_flat.view(ph, pw)

    target_ratio = pw / max(ph, 1)
    best = (1, HWt);
    best_err = float("inf")
    for ha in range(1, int(math.sqrt(HWt)) + 1):
        if HWt % ha == 0:
            wa = HWt // ha
            err = abs((wa / ha) - target_ratio)
            if err < best_err:
                best_err, best = err, (ha, wa)
    ha, wa = best

    grid = attn_flat.view(1, 1, ha, wa)  # [1,1,ha,wa]
    grid_up = F.interpolate(grid, size=(ph, pw), mode="bilinear", align_corners=False)
    return grid_up[0, 0]  # [ph, pw]


def make_token_merge_grid(
        img: Image.Image | np.ndarray | torch.Tensor,
        source: torch.Tensor,
        attention_score: torch.Tensor,
        *,
        patch_size: int = 16,
        class_token: bool = False,
        layer: int | Literal["first", "last", "auto"] = "last",
        min_area_patches: int = 8,
        underlay_opacity: float = 0.25,
        edge_gain: float = 1.6,
) -> Image.Image:
    """visualization with original image faintly under the grid"""

    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # C,H,W
            arr = arr.permute(1, 2, 0)
        img_np = arr.numpy().astype(np.float32)
        if img_np.max() > 1.0: img_np /= 255.0
    elif isinstance(img, Image.Image):
        img_np = np.asarray(img, dtype=np.float32) / 255.0
    else:
        img_np = img.astype(np.float32)
        if img_np.max() > 1.0: img_np /= 255.0

    h, w, _ = img_np.shape
    ph, pw = h // patch_size, w // patch_size

    src = source
    if src.ndim != 3:
        vis_ids = src.long()
        if vis_ids.ndim != 2:
            raise ValueError(f"Unsupported source shape {tuple(src.shape)}")
    else:
        B, A, HW = src.shape
        if class_token:
            src = src[:, :, 1:]
            HW -= 1
        if torch.is_floating_point(src):
            vis_ids = src.argmax(dim=1).long()
        else:
            L = A
            if layer == "first":
                li = 0
            elif layer == "last":
                li = L - 1
            elif isinstance(layer, int):
                li = int(layer) % L
            else:
                uniq_counts = [int(torch.unique(src[0, i]).numel()) for i in range(L)]
                li = int(np.argmax(uniq_counts))
            vis_ids = src[:, li, :].long()

    assert vis_ids.shape[-1] == ph * pw, f"HW={vis_ids.shape[-1]} != {ph}*{pw}"

    ids0 = vis_ids[0]
    ids0_compact, _ = _compact_labels_1d(ids0)
    labels = ids0_compact.view(ph, pw).cpu().numpy()

    if min_area_patches and min_area_patches > 1:
        labels = _filter_small_regions(labels, min_area_patches)
        vals = np.unique(labels)
        lut = {int(v): i for i, v in enumerate(vals.tolist())}
        labels = np.vectorize(lambda x: lut[int(x)], otypes=[np.int64])(labels)

    uniq_labels = np.unique(labels)
    num_groups = int(uniq_labels.size)
    label2idx = {int(g): i for i, g in enumerate(uniq_labels.tolist())}
    cmap = _generate_colormap(num_groups, attention_score)

    # --- build interior and edge layers separately
    interior_img = np.zeros_like(img_np, dtype=np.float32)
    edge_img = np.zeros_like(img_np, dtype=np.float32)
    edge_mask_union = np.zeros((h, w, 1), dtype=np.float32)

    labels_t = torch.from_numpy(labels).view(1, 1, ph, pw).float()

    for g in uniq_labels.tolist():
        ci = label2idx[int(g)]
        mask = (labels_t == g).float()  # [1,1,ph,pw]
        mask = F.interpolate(mask, size=(h, w), mode="nearest").view(h, w, 1).numpy()

        # 1px outline via erosion
        inner = binary_erosion(mask[..., 0], iterations=1)[..., None].astype(np.float32)
        edge = (mask - inner).astype(np.float32)

        # mean interior color from original image
        denom = float(mask.sum())
        mean_color = (mask * img_np).sum(axis=(0, 1)) / denom if denom > 0 else np.zeros(3, dtype=np.float32)

        interior_img += inner * mean_color.reshape(1, 1, 3)

        col = np.clip(np.array(cmap[ci], dtype=np.float32) * edge_gain, 0.0, 1.0).reshape(1, 1, 3)
        edge_img = edge_img * (1.0 - edge) + edge * col
        edge_mask_union = np.maximum(edge_mask_union, edge)

    underlay_opacity = float(np.clip(underlay_opacity, 0.0, 1.0))
    base = interior_img * (1.0 - underlay_opacity) + img_np * underlay_opacity
    out = base * (1.0 - edge_mask_union) + edge_img

    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)
