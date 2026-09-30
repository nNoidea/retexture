"""Seamless tiling generator for retexture."""

from __future__ import annotations

import numpy as np


def enforce_tile_edges(image: np.ndarray) -> np.ndarray:
    """Make the outer pixel borders agree exactly for a repeating texture."""
    if image.shape[0] < 2 or image.shape[1] < 2:
        return image
    result = np.array(image, copy=True)
    result[:, -1, ...] = result[:, 0, ...]
    result[-1, :, ...] = result[0, :, ...]
    return result


def make_seamless(
    image: np.ndarray,
    seam_size: float = 0.2,
    method: str = "offset_wrap",
) -> np.ndarray:
    """Transform an image into a seamless tileable texture.

    ``offset_wrap`` cross-fades opposite edges directly and preserves their
    orientation. ``mirror`` retains the older reflected-edge treatment for
    legacy presets.
    """
    if seam_size <= 0.01:
        return image

    h, w, c = image.shape
    seam_size = min(max(seam_size, 0.02), 0.48)

    blend_w = max(int(w * seam_size), 2)
    blend_h = max(int(h * seam_size), 2)

    img = image.astype(np.float32)
    res = img.copy()

    if str(method).lower().replace("-", "_") in ("offset_wrap", "offset", "wrap"):
        # Opposite edge pixels are blended with their wrapped counterparts,
        # not mirrored copies. At the border both sides converge to the same
        # average, making the repeat boundary exactly continuous.
        t_x = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, blend_w, endpoint=False))
        left = img[:, :blend_w, :]
        right = img[:, -blend_w:, :][:, ::-1, :]
        edge_average = 0.5 * (left + right)
        left_weight = t_x.reshape(1, blend_w, 1)
        res[:, :blend_w, :] = edge_average * (1.0 - left_weight) + left * left_weight
        res[:, -blend_w:, :] = edge_average[:, ::-1, :] * (1.0 - left_weight[:, ::-1, :]) + img[:, -blend_w:, :] * left_weight[:, ::-1, :]

        t_y = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, blend_h, endpoint=False))
        top = res[:blend_h, :, :]
        bottom = res[-blend_h:, :, :][::-1, :, :]
        edge_average = 0.5 * (top + bottom)
        top_weight = t_y.reshape(blend_h, 1, 1)
        res[:blend_h, :, :] = edge_average * (1.0 - top_weight) + top * top_weight
        res[-blend_h:, :, :] = edge_average[::-1, :, :] * (1.0 - top_weight[::-1, :, :]) + res[-blend_h:, :, :] * top_weight[::-1, :, :]
        return np.clip(res, 0, 255).astype(np.uint8)

    # Legacy mirror method: reflected opposite edges.
    # 1. Horizontal Seam Blending (Left <-> Right)
    t_x = np.linspace(0.0, 1.0, blend_w, endpoint=False)
    alpha_x = 0.5 + 0.5 * np.sin(np.pi * 0.5 * t_x)
    alpha_x = alpha_x.reshape(1, blend_w, 1)

    left_original = img[:, :blend_w, :]
    right_flip = img[:, :-blend_w-1:-1, :] if blend_w < w else img[:, ::-1, :]
    right_flip = right_flip[:, :blend_w, :]

    res[:, :blend_w, :] = left_original * alpha_x + right_flip * (1.0 - alpha_x)

    right_original = img[:, -blend_w:, :]
    left_flip = img[:, blend_w-1::-1, :]
    left_flip = left_flip[:, -blend_w:, :]
    alpha_x_rev = alpha_x[:, ::-1, :]
    res[:, -blend_w:, :] = right_original * alpha_x_rev + left_flip * (1.0 - alpha_x_rev)

    # 2. Vertical Seam Blending (Top <-> Bottom)
    t_y = np.linspace(0.0, 1.0, blend_h, endpoint=False)
    alpha_y = 0.5 + 0.5 * np.sin(np.pi * 0.5 * t_y)
    alpha_y = alpha_y.reshape(blend_h, 1, 1)

    top_original = res[:blend_h, :, :]
    bottom_flip = res[:-blend_h-1:-1, :, :] if blend_h < h else res[::-1, :, :]
    bottom_flip = bottom_flip[:blend_h, :, :]
    res[:blend_h, :, :] = top_original * alpha_y + bottom_flip * (1.0 - alpha_y)

    bottom_original = res[-blend_h:, :, :]
    top_flip = res[blend_h-1::-1, :, :]
    top_flip = top_flip[-blend_h:, :, :]
    alpha_y_rev = alpha_y[::-1, :, :]
    res[-blend_h:, :, :] = bottom_original * alpha_y_rev + top_flip * (1.0 - alpha_y_rev)

    return np.clip(res, 0, 255).astype(np.uint8)
