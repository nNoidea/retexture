"""Retro shader-style effects and neural/algorithmic stylization module for retexture.

Provides:
- Chromatic Aberration (RGB channel shift / CRT lens fringing)
- CRT Scanlines (horizontal phosphor mask)
- Grime & Decay (procedural horror/abandoned surface mottling)
- Edge Crunch (high-contrast boundary sharpening)
- Pixel Art Outline Injection (Sobel luminance edge contours)
- Bilateral Planar Simplification (flattens photographic noise into clean planes)
- PixelOE Stylizer (lightness-aware patch clustering and local contrast shaping)
- Cel-Shading Lighting Bands & Split-Toning Temperature Shifts
"""

from __future__ import annotations

import numpy as np


def apply_chromatic_aberration(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Shift Red and Blue color channels horizontally for authentic CRT lens fringing."""
    if amount <= 0.001:
        return image

    shift_pixels = max(1, int(round(amount * 4.0)))
    h, w = image.shape[:2]
    has_alpha = image.shape[-1] == 4

    rgb = image[..., :3].copy()
    res = rgb.copy()

    # Shift Red channel to the right
    res[:, shift_pixels:, 0] = rgb[:, :-shift_pixels, 0]
    res[:, :shift_pixels, 0] = rgb[:, 0:1, 0]

    # Shift Blue channel to the left
    res[:, :-shift_pixels, 2] = rgb[:, shift_pixels:, 2]
    res[:, -shift_pixels:, 2] = rgb[:, -1:, 2]

    # Blend slightly based on amount
    blend = min(amount, 1.0)
    out_rgb = np.clip(rgb.astype(np.float32) * (1.0 - blend) + res.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    if has_alpha:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def apply_crt_scanlines(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Apply horizontal CRT scanline darkening mask."""
    if amount <= 0.001:
        return image

    h, w = image.shape[:2]
    darkness = 1.0 - (amount * 0.45)

    res = image.astype(np.float32).copy()
    res[1::2, :, :3] *= darkness

    return np.clip(res, 0, 255).astype(np.uint8)


def _periodic_value_noise(
    height: int,
    width: int,
    grid_height: int,
    grid_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bilinearly sample a periodic random grid, with no edge discontinuity."""
    grid = rng.random((max(2, grid_height), max(2, grid_width)), dtype=np.float32)
    # Sample both borders at the same phase so the generated mask is exactly
    # repeatable at the pixel boundary, not merely visually similar.
    gy = np.arange(height, dtype=np.float32) * grid.shape[0] / max(height - 1, 1)
    gx = np.arange(width, dtype=np.float32) * grid.shape[1] / max(width - 1, 1)
    y0 = np.floor(gy).astype(int) % grid.shape[0]
    x0 = np.floor(gx).astype(int) % grid.shape[1]
    y1 = (y0 + 1) % grid.shape[0]
    x1 = (x0 + 1) % grid.shape[1]
    fy = gy - np.floor(gy)
    fx = gx - np.floor(gx)
    top = grid[y0][:, x0] * (1.0 - fx)[None, :] + grid[y0][:, x1] * fx[None, :]
    bottom = grid[y1][:, x0] * (1.0 - fx)[None, :] + grid[y1][:, x1] * fx[None, :]
    return top * (1.0 - fy)[:, None] + bottom * fy[:, None]


def _fbm_noise(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 0.55
    for grid_h, grid_w in ((4, 4), (8, 8), (16, 16)):
        total += _periodic_value_noise(height, width, grid_h, grid_w, rng) * amplitude
        amplitude *= 0.5
    return np.clip(total / 0.9625, 0.0, 1.0)


def apply_grime_decay(
    image: np.ndarray,
    amount: float = 0.0,
    seed: int = 42,
    mode: str = "uniform",
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay deterministic, style-aware surface marks before quantization."""
    if amount <= 0.001:
        return image

    h, w = image.shape[:2]
    rng = np.random.default_rng(seed + 999)

    mode = str(mode).lower().strip()
    noise = _fbm_noise(h, w, rng)
    if mode in ("streak", "streaks", "drip"):
        streaks = _fbm_noise(h, w, rng)
        vertical = np.linspace(0.3, 1.0, h, dtype=np.float32)[:, None]
        noise = np.clip(0.25 * noise + 0.75 * streaks * vertical, 0.0, 1.0)
    elif mode in ("edge_wear", "edgewear", "cavity"):
        rgb = image[..., :3].astype(np.float32) / 255.0
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        padded = np.pad(lum, ((1, 1), (1, 1)), mode="edge")
        neighborhood = (
            padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) * 0.25
        cavity = np.clip((neighborhood - lum) * 4.0, 0.0, 1.0)
        noise = np.clip(noise * (0.25 + 0.75 * cavity), 0.0, 1.0)

    if palette is not None and len(palette):
        pal = np.asarray(palette, dtype=np.float32)
        pal_lum = pal @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        dark = pal[pal_lum <= np.percentile(pal_lum, 40)]
        grime_tint = np.mean(dark if len(dark) else pal, axis=0) / 255.0
    else:
        grime_tint = np.array([0.45, 0.35, 0.22], dtype=np.float32)

    img_f = image[..., :3].astype(np.float32) / 255.0
    grime_mask = (noise[..., np.newaxis] ** 1.45) * float(np.clip(amount, 0.0, 1.0)) * 0.78
    blended = img_f * (1.0 - grime_mask) + grime_tint.reshape(1, 1, 3) * grime_mask
    out_rgb = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def apply_kuwahara_filter(
    image: np.ndarray,
    amount: float = 0.0,
    periodic: bool = False,
) -> np.ndarray:
    """Structure-preserving directional Kuwahara filter.

    Flattens micro-surface noise and photographic grain into clean, smooth planes
    by choosing the mean of the local quadrant with lowest variance. Completely
    avoids blurring across high-contrast structural edges (door frames, window borders).
    """
    if amount <= 0.001:
        return image

    rgb = image[..., :3].astype(np.float32)
    h, w, c = rgb.shape
    radius = 2 if amount > 0.6 else 1
    pad = radius

    padded = np.pad(
        rgb,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="wrap" if periodic else "edge",
    )

    quad_means: list[np.ndarray] = []
    quad_vars: list[np.ndarray] = []
    block_len = radius + 1
    count = float(block_len * block_len)

    # 4 quadrants: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    for dy, dx in [(-pad, -pad), (-pad, 0), (0, -pad), (0, 0)]:
        y0 = pad + dy
        x0 = pad + dx
        block = np.zeros((h, w, c), dtype=np.float32)
        block_sq = np.zeros((h, w), dtype=np.float32)
        for oy in range(block_len):
            for ox in range(block_len):
                sample = padded[y0 + oy : y0 + oy + h, x0 + ox : x0 + ox + w]
                block += sample
                block_sq += np.sum(sample ** 2, axis=-1)
        mean = block / count
        var = (block_sq / count) - np.sum(mean ** 2, axis=-1)
        quad_means.append(mean)
        quad_vars.append(var)

    stacked_vars = np.stack(quad_vars, axis=0)
    best_quad = np.argmin(stacked_vars, axis=0)

    stacked_means = np.stack(quad_means, axis=0)
    h_idx, w_idx = np.indices((h, w))
    kuwahara_rgb = stacked_means[best_quad, h_idx, w_idx]

    blended = rgb * (1.0 - amount) + kuwahara_rgb * amount
    out_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def apply_edge_crunch(
    image: np.ndarray,
    amount: float = 0.0,
    noise_gate: float = 0.5,
    periodic: bool = False,
) -> np.ndarray:
    """Sharpen and emphasize high-contrast pixel boundaries for chunky retro edges.

    Uses noise gating to prevent amplifying subtle grain on flat wall surfaces.
    """
    if amount <= 0.001:
        return image

    img_f = image[..., :3].astype(np.float32)
    padded = np.pad(img_f, ((1, 1), (1, 1), (0, 0)), mode="wrap" if periodic else "edge")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    delta = center * 4.0 - (up + down + left + right)

    # Compute local contrast difference
    local_contrast = np.max(
        np.abs(np.stack([center - up, center - down, center - left, center - right], axis=0)),
        axis=(0, -1),
    )
    # Low contrast variations (<20 RGB levels) are treated as noise and dampened
    contrast_weight = np.clip(local_contrast / 20.0, 0.0, 1.0)[..., np.newaxis]
    gate = float(np.clip(noise_gate, 0.0, 1.0))
    weight = (1.0 - gate) + gate * contrast_weight

    sharpened = center + delta * (amount * weight)
    out_rgb = np.clip(sharpened, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def apply_outline_injection(
    image: np.ndarray,
    amount: float = 0.0,
    noise_gate: float = 0.5,
    periodic: bool = False,
) -> np.ndarray:
    """Detect silhouette contours using noise-gated connected Sobel gradients.

    Uses bilateral luminance pre-filtering and 3x3 connected-contour gating to reject
    isolated 1-pixel noise speckles while keeping continuous lines (doors, windows, trim).
    """
    if amount <= 0.001:
        return image

    rgb = image[..., :3].astype(np.float32)
    h, w, _ = rgb.shape
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0

    # 1. Edge-preserving bilateral pre-smoothing on luminance to eliminate micro-textures
    lum_padded = np.pad(lum, ((1, 1), (1, 1)), mode="wrap" if periodic else "edge")
    c_lum = lum
    neighbors_lum = [
        lum_padded[:-2, 1:-1], lum_padded[2:, 1:-1],
        lum_padded[1:-1, :-2], lum_padded[1:-1, 2:],
        lum_padded[:-2, :-2], lum_padded[:-2, 2:],
        lum_padded[2:, :-2], lum_padded[2:, 2:],
    ]
    w_sum = np.ones((h, w), dtype=np.float32)
    accum = c_lum.copy()
    sigma = 0.12
    for n in neighbors_lum:
        diff_sq = (c_lum - n) ** 2
        w = np.exp(-diff_sq / (2.0 * sigma * sigma))
        accum += n * w
        w_sum += w
    smooth_lum = accum / w_sum

    # 2. Sobel Gradients on clean luminance map
    padded_s = np.pad(smooth_lum, ((1, 1), (1, 1)), mode="wrap" if periodic else "edge")
    gx = (
        -1.0 * padded_s[:-2, :-2] + 1.0 * padded_s[:-2, 2:] +
        -2.0 * padded_s[1:-1, :-2] + 2.0 * padded_s[1:-1, 2:] +
        -1.0 * padded_s[2:, :-2] + 1.0 * padded_s[2:, 2:]
    )
    gy = (
        -1.0 * padded_s[:-2, :-2] - 2.0 * padded_s[:-2, 1:-1] - 1.0 * padded_s[:-2, 2:] +
         1.0 * padded_s[2:, :-2] + 2.0 * padded_s[2:, 1:-1] + 1.0 * padded_s[2:, 2:]
    )

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    threshold = 0.10 * (1.0 - amount * 0.4)
    raw_mask = np.clip((grad_mag - threshold) / (threshold + 1e-4), 0.0, 1.0)

    # 3. Connected-contour noise gating: real structural edges (doors, sills) have collinear neighbors;
    #    isolated speckles have 0-1 neighbors and are rejected.
    gate = float(np.clip(noise_gate, 0.0, 1.0))
    if gate > 0.001:
        pad_mask = np.pad(raw_mask, ((1, 1), (1, 1)), mode="wrap" if periodic else "edge")
        neighbor_sum = (
            pad_mask[:-2, :-2] + pad_mask[:-2, 1:-1] + pad_mask[:-2, 2:] +
            pad_mask[1:-1, :-2]                      + pad_mask[1:-1, 2:] +
            pad_mask[2:, :-2]  + pad_mask[2:, 1:-1]  + pad_mask[2:, 2:]
        )
        connectivity = np.clip(neighbor_sum / 1.8, 0.0, 1.0)
        edge_mask = raw_mask * ((1.0 - gate) + gate * connectivity)
    else:
        edge_mask = raw_mask

    edge_strength = edge_mask * amount

    # Darken pixels on detected structural edges
    dark_contour = rgb * (1.0 - edge_strength[..., np.newaxis] * 0.85)
    out_rgb = np.clip(dark_contour, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def _neighborhood_views(
    array: np.ndarray,
    radius: int = 1,
    periodic: bool = False,
    include_center: bool = False,
) -> list[np.ndarray]:
    """Return shifted local-neighbour views with either clamped or tiled borders."""
    radius = max(1, int(radius))
    height, width = array.shape[:2]
    views: list[np.ndarray] = []

    if periodic:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if not include_center and dy == 0 and dx == 0:
                    continue
                views.append(np.roll(array, shift=(dy, dx), axis=(0, 1)))
        return views

    padded = np.pad(
        array,
        ((radius, radius), (radius, radius)) + ((0, 0),) * (array.ndim - 2),
        mode="edge",
    )
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not include_center and dy == 0 and dx == 0:
                continue
            y0 = radius + dy
            x0 = radius + dx
            views.append(padded[y0:y0 + height, x0:x0 + width])
    return views


def _local_min_max(
    channel: np.ndarray,
    radius: int,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local extrema without introducing a new edge value."""
    views = _neighborhood_views(channel, radius=radius, periodic=periodic, include_center=True)
    return np.minimum.reduce(views), np.maximum.reduce(views)


def apply_bilateral_simplify(
    image: np.ndarray,
    amount: float = 0.0,
    periodic: bool = False,
) -> np.ndarray:
    """Edge-preserving bilateral smoothing to flatten photographic noise into clean illustrative planes."""
    if amount <= 0.001:
        return image

    rgb = image[..., :3].astype(np.float32)
    h, w, _ = rgb.shape

    # Fast 3x3 local cross-bilateral approximation
    center = rgb
    neighbors = _neighborhood_views(rgb, radius=1, periodic=periodic)

    sigma_color = 30.0 * (1.0 + amount * 2.0)
    total_weight = np.ones((h, w, 1), dtype=np.float32)
    accum = center.copy()

    for n in neighbors:
        diff_sq = np.sum((center - n) ** 2, axis=-1, keepdims=True)
        w_color = np.exp(-diff_sq / (2.0 * (sigma_color ** 2)))
        accum += n * w_color
        total_weight += w_color

    simplified = accum / total_weight
    blended = center * (1.0 - amount) + simplified * amount
    out_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def _pixeloe_radius(height: int, width: int, amount: float) -> int:
    """Choose a patch radius that respects output resolution and stylization strength."""
    short_side = min(height, width)
    if short_side <= 96:
        return 1
    if short_side <= 256:
        return 2 if amount >= 0.4 else 1
    if short_side <= 512:
        return 3 if amount >= 0.7 else 2
    return 3


def apply_pixeloe_stylizer(
    image: np.ndarray,
    amount: float = 0.0,
    periodic: bool = False,
) -> np.ndarray:
    """Shape readable pixel-art clusters while preserving the source colour identity.

    This is a lightweight, deterministic PixelOE-inspired pass rather than a
    learned model. It smooths a small receptive field, shapes local Oklab
    lightness contrast, and keeps most chroma from the source so the later
    palette pass—not PixelOE—remains responsible for the final colour choice.
    """
    if amount <= 0.001:
        return image

    amount = float(np.clip(amount, 0.0, 1.0))
    rgb = image[..., :3].astype(np.float32)
    height, width = rgb.shape[:2]

    # 1. Bilateral planar smoothing. This is intentionally internal to PixelOE;
    #    the pipeline skips the standalone bilateral pass when PixelOE is on.
    simplified = apply_bilateral_simplify(
        image,
        amount=amount * 0.7,
        periodic=periodic,
    )[..., :3].astype(np.float32)

    # 2. Shape local lightness, not independent RGB channels. That gives us
    #    stronger planes and edges without turning a neutral surface magenta,
    #    green, or orange just because its channels had different ranges.
    from retexture.core.color_adjust import oklab_to_srgb
    from retexture.core.quantize import srgb_to_oklab

    source_lab = srgb_to_oklab(np.clip(rgb / 255.0, 0.0, 1.0)).astype(np.float32)
    smooth_lab = srgb_to_oklab(np.clip(simplified / 255.0, 0.0, 1.0)).astype(np.float32)
    radius = _pixeloe_radius(height, width, amount)
    local_min, local_max = _local_min_max(smooth_lab[..., 0], radius=radius, periodic=periodic)
    local_range = local_max - local_min
    safe_range = np.maximum(local_range, 1e-5)
    normalized = np.clip((smooth_lab[..., 0] - local_min) / safe_range, 0.0, 1.0)

    # A sigmoid creates intentional cluster separation. The gate leaves broad,
    # flat areas alone instead of inventing bands inside a uniform plane.
    sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (normalized - 0.5)))
    contrasted = local_min + sigmoid * local_range
    gate = np.clip(local_range / 0.08, 0.0, 1.0)
    contrasted = smooth_lab[..., 0] + gate * (contrasted - smooth_lab[..., 0])

    base_lightness = source_lab[..., 0] * (1.0 - 0.35 * amount) + smooth_lab[..., 0] * (0.35 * amount)
    output_lab = source_lab.copy()
    output_lab[..., 0] = base_lightness * (1.0 - amount) + contrasted * amount

    # Preserve hue and material colour. Only a restrained amount of smoothed
    # chroma is mixed in to remove isolated colour noise.
    chroma_mix = amount * 0.2
    output_lab[..., 1:] = source_lab[..., 1:] * (1.0 - chroma_mix) + smooth_lab[..., 1:] * chroma_mix
    out_rgb = np.clip(oklab_to_srgb(output_lab) * 255.0, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb


def apply_cel_shading_bands(image: np.ndarray, steps: int = 0) -> np.ndarray:
    """Posterize luminance into discrete, stylized retro lighting bands (cel-shading)."""
    if steps < 2:
        return image

    rgb = image[..., :3].astype(np.float32) / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    lum = np.clip(lum, 1e-5, 1.0)

    quant_lum = np.round(lum * (steps - 1)) / (steps - 1)
    ratio = (quant_lum / lum)[..., np.newaxis]
    cel_rgb = np.clip(rgb * ratio * 255.0, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([cel_rgb, image[..., 3:]], axis=-1)
    return cel_rgb


def apply_color_temperature_shift(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Apply professional color temperature split-toning (warm highlights & cool shadows)."""
    if abs(amount) <= 0.001:
        return image

    rgb = image[..., :3].astype(np.float32)
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0

    highlight_weight = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)[..., np.newaxis]
    shadow_weight = np.clip((0.5 - lum) * 2.0, 0.0, 1.0)[..., np.newaxis]

    warm_tint = np.array([25.0 * amount, 15.0 * amount, -10.0 * amount], dtype=np.float32)
    cool_tint = np.array([-15.0 * amount, 5.0 * amount, 25.0 * amount], dtype=np.float32)

    shifted = rgb + (warm_tint * highlight_weight) + (cool_tint * shadow_weight)
    out_rgb = np.clip(shifted, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb
