"""Color grading and adjustment module for retexture.

Provides fast, vectorized NumPy operations for:
- Hue Shift (-180 deg to +180 deg)
- Saturation (0.0 to 2.0)
- Brightness (0.5 to 1.5)
- Contrast (0.5 to 1.5)
"""

from __future__ import annotations

import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float image (0.0 to 1.0) to HSV float image."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc

    s = np.zeros_like(maxc)
    mask = maxc > 1e-6
    s[mask] = delta[mask] / maxc[mask]

    h = np.zeros_like(maxc)
    nonzero_delta = delta > 1e-6

    r_mask = nonzero_delta & (maxc == r)
    g_mask = nonzero_delta & (maxc == g)
    b_mask = nonzero_delta & (maxc == b)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0

    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV float image (0.0 to 1.0) to RGB float image."""
    h = (hsv[..., 0] % 1.0) * 6.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, 1.0)

    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros_like(hsv)

    m0 = i == 0
    rgb[m0] = np.stack([v[m0], t[m0], p[m0]], axis=-1)
    m1 = i == 1
    rgb[m1] = np.stack([q[m1], v[m1], p[m1]], axis=-1)
    m2 = i == 2
    rgb[m2] = np.stack([p[m2], v[m2], t[m2]], axis=-1)
    m3 = i == 3
    rgb[m3] = np.stack([p[m3], q[m3], v[m3]], axis=-1)
    m4 = i == 4
    rgb[m4] = np.stack([t[m4], p[m4], v[m4]], axis=-1)
    m5 = i == 5
    rgb[m5] = np.stack([v[m5], p[m5], q[m5]], axis=-1)

    return np.clip(rgb, 0.0, 1.0)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab values back to clipped sRGB floats in the 0..1 range."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_rgb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    linear = np.stack([r, g, b_rgb], axis=-1)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def apply_oklab_style(
    image: np.ndarray,
    black_point: float = 0.0,
    white_point: float = 1.0,
    s_curve: float = 0.0,
    shadow_hue_shift: float = 0.0,
    highlight_hue_shift: float = 0.0,
    shadow_saturation: float = 1.0,
    highlight_saturation: float = 1.0,
) -> np.ndarray:
    """Apply palette-friendly tonal and split-tone shaping in Oklab.

    Unlike HSV grading, this keeps perceived lightness separate from chroma.
    It is intended to happen before palette matching so every resulting colour
    still belongs to the preset's authored palette.
    """
    if (
        abs(black_point) <= 0.001
        and abs(white_point - 1.0) <= 0.001
        and abs(s_curve) <= 0.001
        and abs(shadow_hue_shift) <= 0.001
        and abs(highlight_hue_shift) <= 0.001
        and abs(shadow_saturation - 1.0) <= 0.001
        and abs(highlight_saturation - 1.0) <= 0.001
    ):
        return image

    from retexture.core.quantize import srgb_to_oklab

    has_alpha = image.shape[-1] == 4
    rgb = image[..., :3].astype(np.float32) / 255.0
    lab = srgb_to_oklab(rgb).astype(np.float32)

    black = float(np.clip(black_point, 0.0, 0.45))
    white = float(np.clip(white_point, 0.55, 1.0))
    lab[..., 0] = np.clip((lab[..., 0] - black) / max(white - black, 1e-4), 0.0, 1.0)

    curve = float(np.clip(s_curve, -1.0, 1.0))
    if abs(curve) > 0.001:
        L = lab[..., 0]
        lab[..., 0] = np.clip(L + curve * 2.0 * L * (1.0 - L) * (2.0 * L - 1.0), 0.0, 1.0)

    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    chroma_weight = np.clip(chroma / 0.08, 0.0, 1.0)
    shadow_weight = np.clip((0.55 - lab[..., 0]) / 0.55, 0.0, 1.0)
    highlight_weight = np.clip((lab[..., 0] - 0.45) / 0.55, 0.0, 1.0)

    hue += np.deg2rad(shadow_hue_shift) * shadow_weight * chroma_weight
    hue += np.deg2rad(highlight_hue_shift) * highlight_weight * chroma_weight
    sat = 1.0 + (shadow_saturation - 1.0) * shadow_weight
    sat += (highlight_saturation - 1.0) * highlight_weight
    chroma = np.maximum(chroma * sat, 0.0)
    lab[..., 1] = np.cos(hue) * chroma
    lab[..., 2] = np.sin(hue) * chroma

    out_rgb = np.clip(oklab_to_srgb(lab) * 255.0, 0.0, 255.0).astype(np.uint8)
    if has_alpha:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def adjust_colors(
    image: np.ndarray,
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    midtones: float = 0.0,
    highlights: float = 0.0,
    luminance_threshold: float = 0.5,
    invert_luminance: bool = False,
) -> np.ndarray:
    """Apply hue, saturation, brightness, contrast, and 3-way tonal shaping."""
    has_alpha = image.shape[-1] == 4
    rgb = image[..., :3].astype(np.float32) / 255.0

    if invert_luminance:
        rgb = 1.0 - rgb

    if abs(hue_shift) > 0.01 or abs(saturation - 1.0) > 0.01:
        hsv = rgb_to_hsv(rgb)
        if abs(hue_shift) > 0.01:
            hsv[..., 0] = (hsv[..., 0] + (hue_shift / 360.0)) % 1.0
        if abs(saturation - 1.0) > 0.01:
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
        rgb = hsv_to_rgb(hsv)

    if abs(contrast - 1.0) > 0.01:
        rgb = (rgb - 0.5) * contrast + 0.5

    if abs(brightness - 1.0) > 0.01:
        rgb = rgb * brightness

    # Midtones shaping (gamma curve)
    if abs(midtones) > 0.01:
        gamma = float(np.power(2.0, -midtones * 1.5))
        rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)

    # Highlights shaping (quadratic weighting on upper band)
    if abs(highlights) > 0.01:
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        weight = np.clip(lum, 0.0, 1.0) ** 2
        weight = np.expand_dims(weight, axis=-1)
        if highlights > 0:
            rgb = rgb + (highlights * weight * (1.0 - rgb))
        else:
            rgb = rgb + (highlights * weight * rgb)

    # Luminance threshold shift
    if abs(luminance_threshold - 0.5) > 0.01:
        bias = (0.5 - luminance_threshold) * 0.4
        rgb = rgb + bias

    out_rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    if has_alpha:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb
