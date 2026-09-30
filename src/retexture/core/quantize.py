"""Perceptual color conversion and palette matching for retexture."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def srgb_to_oklab(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert an RGB float32 (0..1 or 0..255) array to Oklab perceptual color space (L, a, b).

    Oklab is a modern 2020 perceptual color space designed to align with human retina perception,
    preventing muddy shadow banding and blue/purple hue distortions.
    """
    arr = rgb_arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # 1. Linearize sRGB
    mask = arr > 0.04045
    linear = np.empty_like(arr)
    linear[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] = arr[~mask] / 12.92

    # 2. Linear sRGB to LMS
    r = linear[..., 0]
    g = linear[..., 1]
    b = linear[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # 3. Non-linear cube root
    l_ = np.cbrt(np.maximum(l, 0.0))
    m_ = np.cbrt(np.maximum(m, 0.0))
    s_ = np.cbrt(np.maximum(s, 0.0))

    # 4. LMS to Oklab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ok = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b_ok], axis=-1)


def srgb_to_cielab(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert an RGB float32 (0..1 or 0..255) array to CIE L*a*b* color space."""
    arr = rgb_arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # 1. Linearize sRGB
    mask = arr > 0.04045
    linear = np.empty_like(arr)
    linear[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] = arr[~mask] / 12.92

    # 2. Linear sRGB to XYZ (D65 standard illuminant)
    r = linear[..., 0]
    g = linear[..., 1]
    b = linear[..., 2]

    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        delta = 6.0 / 29.0
        mask_t = t > (delta ** 3)
        res = np.empty_like(t)
        res[mask_t] = np.cbrt(t[mask_t])
        res[~mask_t] = (t[~mask_t] / (3.0 * delta ** 2)) + (4.0 / 29.0)
        return res

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)

    return np.stack([L, a, b_lab], axis=-1)


# Rec. 709 luma weights; sqrt scaling turns the luma-weighted RGB distance into
# a plain Euclidean distance so every metric can share one matching code path.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _as_unit_srgb(rgb: np.ndarray) -> np.ndarray:
    """Normalize sRGB input (uint8 0-255, or float on either scale) to float32 0-1."""
    arr = np.asarray(rgb)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.size and arr.max() > 1.5:
        return arr / 255.0
    return arr


def to_metric_space(rgb: np.ndarray, metric: str) -> np.ndarray:
    """Map sRGB colors to the working space of the given distance metric.

    Returns float32 coordinates of shape (..., 3) in which plain Euclidean
    distance equals the metric's color distance:
    - 'oklab': perceptual Oklab (sRGB is linearized internally, in float)
    - 'cielab': CIE L*a*b* Delta-E space
    - 'luma': RGB scaled by sqrt(Rec. 709 weights)
    - 'euclidean': plain 0-1 RGB

    This is the single conversion point for all palette matching and error
    diffusion: callers convert the image and palette once up front and never
    inside a pixel loop.
    """
    metric = str(metric).lower().strip()
    if metric == "oklab":
        return srgb_to_oklab(_as_unit_srgb(rgb))
    if metric == "cielab":
        return srgb_to_cielab(_as_unit_srgb(rgb))
    if metric == "luma":
        return _as_unit_srgb(rgb) * np.sqrt(_LUMA_WEIGHTS)
    if metric == "euclidean":
        return _as_unit_srgb(rgb)
    raise ValueError(f"Unknown color metric: {metric!r}. Expected oklab, cielab, luma or euclidean.")


@lru_cache(maxsize=32)
def _palette_to_metric_space_cached(palette_bytes: bytes, rows: int, metric: str) -> np.ndarray:
    palette = np.frombuffer(palette_bytes, dtype=np.uint8).reshape(rows, 3)
    result = to_metric_space(palette, metric)
    result.flags.writeable = False  # shared cache entry: callers must not mutate
    return result


def palette_to_metric_space(palette: np.ndarray, metric: str) -> np.ndarray:
    """Convert an sRGB uint8 palette (N, 3) to metric space, at most once per
    unique palette content + metric.

    Conversion results are cached, so repeated calls (GUI preview frames,
    multi-pass dithering) reuse the same coordinates instead of re-converting
    the palette every time. The returned array is read-only.
    """
    pal = np.ascontiguousarray(np.asarray(palette), dtype=np.uint8)
    if pal.ndim != 2 or pal.shape[1] != 3:
        raise ValueError(f"Expected palette of shape (N, 3), got {pal.shape}")
    return _palette_to_metric_space_cached(pal.tobytes(), pal.shape[0], str(metric).lower().strip())


def match_palette_nearest(
    image: np.ndarray,
    palette: np.ndarray,
    metric: str = "oklab",
) -> np.ndarray:
    """Map every pixel in image (H, W, 3) to the closest color in palette (N, 3).

    Supported distance metrics:
    - 'oklab': Modern 2020 human perceptual color space (Recommended for rich shadow/color fidelity)
    - 'cielab': Classic CIE L*a*b* Delta-E color distance
    - 'luma': Rec. 709 luminance-weighted Euclidean distance
    - 'euclidean': Standard unweighted RGB Euclidean distance

    Both the image and the palette are converted into the metric space up front;
    the palette conversion is cached per unique palette content + metric.
    Pixel matching is chunked in 65,536-pixel blocks to keep memory bounded on large textures.
    """
    h, w, _ = image.shape
    palette_clean = palette.astype(np.uint8)

    img_space = to_metric_space(image, metric).reshape(-1, 3)
    pal_space = palette_to_metric_space(palette_clean, metric).reshape(1, -1, 3)

    total_pixels = h * w
    chunk_size = 65536
    nearest_idx = np.empty(total_pixels, dtype=np.int32)

    for start in range(0, total_pixels, chunk_size):
        end = min(start + chunk_size, total_pixels)
        diff = img_space[start:end, None, :] - pal_space
        dists = np.sum(diff ** 2, axis=2)
        nearest_idx[start:end] = np.argmin(dists, axis=1)

    return palette_clean[nearest_idx].reshape(h, w, 3)
