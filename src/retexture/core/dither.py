"""Dithering engine for retexture.

Implements:
- Ordered Bayer Dithering (2x2, 4x4, 8x8) with continuous strength multiplier
- Authentic Sega Saturn / PS1 Interlaced Checkerboard Mesh
- Diagonal Cross-Hatching (PC-98 / 1-Bit Manga Shading)
- Yliluoma's 2-Color Optical Blending Algorithm
- Riemersma Dithering (Fractal Hilbert Space-Filling Curve)
- Floyd-Steinberg & Atkinson Error Diffusion
- Blue Noise Dithering
"""

from __future__ import annotations

import numpy as np

# Standard normalized Bayer matrices centered around 0 (-0.5 to +0.5)
BAYER_2X2 = (np.array([
    [0, 2],
    [3, 1]
], dtype=np.float32) / 4.0) - 0.375

BAYER_4X4 = (np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32) / 16.0) - 0.46875

BAYER_8X8 = (np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
], dtype=np.float32) / 64.0) - 0.4921875

# Halftone concentric circular dot matrix (stylized comic / retro-CG dot ramp)
HALFTONE_DOT_4X4 = (np.array([
    [12,  5,  6, 13],
    [ 4,  0,  1,  7],
    [11,  3,  2,  8],
    [15, 10,  9, 14]
], dtype=np.float32) / 16.0) - 0.46875

# Sega Saturn / PS1 Interlaced Checkerboard (50% alternating shadow mesh)
INTERLACED_CHECKER_2X2 = np.array([
    [ 0.5, -0.5],
    [-0.5,  0.5]
], dtype=np.float32)

# PC-98 & 1-Bit Diagonal Cross-Hatching Matrix (45° angled line hatching)
CROSSHATCH_4X4 = (np.array([
    [ 0,  8,  4, 12],
    [14,  2, 10,  6],
    [ 3, 11,  1,  9],
    [13,  5, 15,  7]
], dtype=np.float32) / 16.0) - 0.46875


def _clustered_dot_matrix(n: int) -> np.ndarray:
    """Build a clustered-dot (print-style) threshold matrix of size n x n.

    Dots grow outward from periodic dot centers (toroidal distance), producing
    the classic newspaper halftone look instead of Bayer's dispersed speckle.
    """
    if n == 4:
        centers = np.array([[1.5, 1.5]], dtype=np.float32)
    else:
        centers = np.array([[2.0, 2.0], [6.0, 6.0]], dtype=np.float32)

    cells = np.array([(y, x) for y in range(n) for x in range(n)], dtype=np.float32)
    keys = []
    for (y, x) in cells:
        # Toroidal distance to nearest dot center
        dists = []
        for (cy, cx) in centers:
            dy = np.abs(y - cy)
            dx = np.abs(x - cx)
            dy = np.minimum(dy, n - dy)
            dx = np.minimum(dx, n - dx)
            dists.append(dx * dx + dy * dy)
        d = min(dists)
        # Spiral tie-break around nearest center so dots grow in concentric rings
        nearest = centers[int(np.argmin(dists))]
        angle = np.arctan2(y - nearest[0], x - nearest[1])
        keys.append((d, angle, y, x))

    order = sorted(range(len(keys)), key=lambda i: keys[i])
    rank = np.zeros(n * n, dtype=np.float32)
    for r, idx in enumerate(order):
        rank[idx] = r
    return (rank.reshape(n, n) / (n * n)) - 0.46875


CLUSTER_DOT_4X4 = _clustered_dot_matrix(4)
CLUSTER_DOT_8X8 = _clustered_dot_matrix(8)

# Horizontal Line Halftone (each tile row gets a constant threshold => scanlines)
LINE_HALFTONE_4X4 = (np.array([
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=np.float32) / 4.0) - 0.375

# Knuth-style dot diffusion class matrix. Unlike the old ``(i + j) mod 8``
# placeholder, this is a full 8x8 class pattern; diffusion below only writes
# to pixels in a later class, so already-quantized dots are never revisited.
KNUTH_CLASS_8X8 = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7],
    [7, 0, 1, 2, 3, 4, 5, 6],
    [6, 7, 0, 1, 2, 3, 4, 5],
    [5, 6, 7, 0, 1, 2, 3, 4],
    [4, 5, 6, 7, 0, 1, 2, 3],
    [3, 4, 5, 6, 7, 0, 1, 2],
    [2, 3, 4, 5, 6, 7, 0, 1],
    [1, 2, 3, 4, 5, 6, 7, 0],
], dtype=np.uint8)


from functools import lru_cache


def srgb_to_linear(x: np.ndarray | float) -> np.ndarray | float:
    """Convert sRGB (0.0 to 1.0) to linear photometric space."""
    arr = np.asarray(x, dtype=np.float32)
    linear = np.where(arr <= 0.04045, arr / 12.92, np.power((arr + 0.055) / 1.055, 2.4))
    return linear.item() if np.ndim(x) == 0 else linear


def linear_to_srgb(x: np.ndarray | float) -> np.ndarray | float:
    """Convert linear photometric space (0.0 to 1.0) to sRGB (0.0 to 1.0)."""
    arr = np.asarray(x, dtype=np.float32)
    srgb = np.where(arr <= 0.0031308, arr * 12.92, 1.055 * np.power(np.maximum(arr, 0.0), 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)
    return srgb.item() if np.ndim(x) == 0 else srgb


@lru_cache(maxsize=4)
def get_blue_noise_matrix(size: int = 64) -> np.ndarray:
    """Generate or retrieve a high-frequency toroidal void-and-cluster blue noise matrix."""
    N = size
    sigma = 1.9
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky)
    H = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (KX ** 2 + KY ** 2))

    rng = np.random.default_rng(1337)
    pattern = np.zeros((N, N), dtype=np.float32)
    n_initial = int(N * N * 0.1)
    indices = rng.choice(N * N, n_initial, replace=False)
    pattern.flat[indices] = 1.0

    # Relaxation
    for _ in range(25):
        blurred = np.real(np.fft.ifft2(np.fft.fft2(pattern) * H))
        c_idx = np.argmax(blurred * pattern)
        pattern.flat[c_idx] = 0
        v_idx = np.argmin(blurred + pattern * 1e6)
        pattern.flat[v_idx] = 1

    rank_matrix = np.zeros((N, N), dtype=np.int32)
    p_work = pattern.copy()
    for r in range(n_initial - 1, -1, -1):
        blurred = np.real(np.fft.ifft2(np.fft.fft2(p_work) * H))
        c_idx = np.argmax(blurred * p_work)
        p_work.flat[c_idx] = 0
        rank_matrix.flat[c_idx] = r

    p_work = pattern.copy()
    for r in range(n_initial, N * N):
        blurred = np.real(np.fft.ifft2(np.fft.fft2(p_work) * H))
        v_idx = np.argmin(blurred + p_work * 1e6)
        p_work.flat[v_idx] = 1
        rank_matrix.flat[v_idx] = r

    return (rank_matrix.astype(np.float32) / (N * N)) - 0.5


def get_bayer_matrix(matrix_type: str) -> np.ndarray:
    """Return the 2D threshold matrix for the given algorithm name."""
    clean = matrix_type.lower().replace("_", "").replace("-", "")
    if "blue" in clean:
        return get_blue_noise_matrix(64)
    if "clusterdot" in clean or "clustered" in clean:
        if "4x4" in clean:
            return CLUSTER_DOT_4X4
        if "8x8" in clean:
            return CLUSTER_DOT_8X8
        return CLUSTER_DOT_8X8
    if "linehalftone" in clean or "halftoneline" in clean:
        return LINE_HALFTONE_4X4
    if "halftonecross" in clean:
        return CROSSHATCH_4X4
    if "interlace" in clean or "checker" in clean or "saturn" in clean:
        return INTERLACED_CHECKER_2X2
    if "crosshatch" in clean or "hatch" in clean or "pc98" in clean:
        return CROSSHATCH_4X4
    if "halftonedot" in clean or "dot" in clean:
        return HALFTONE_DOT_4X4
    if "2x2" in clean or "2" in clean:
        return BAYER_2X2
    if "8x8" in clean or "8" in clean:
        return BAYER_8X8
    return BAYER_4X4


# Error diffusion kernels: list of (dx, dy, weight) offsets relative to the
# current pixel. Weights are normalized (they sum to 1.0). Reference weights
# taken from classic literature & libdither (robertkist).
ERROR_DIFFUSION_KERNELS: dict[str, list[tuple[int, int, float]]] = {
    "floyd_steinberg": [
        (1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16),
    ],
    "false_floyd_steinberg": [
        (1, 0, 3 / 8), (0, 1, 3 / 8), (1, 1, 2 / 8),
    ],
    "atkinson": [
        (1, 0, 1 / 8), (2, 0, 1 / 8),
        (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ],
    "jarvis_judice_ninke": [
        (1, 0, 7 / 48), (2, 0, 5 / 48),
        (-2, 1, 3 / 48), (-1, 1, 5 / 48), (0, 1, 7 / 48), (1, 1, 5 / 48), (2, 1, 3 / 48),
        (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48),
    ],
    "stucki": [
        (1, 0, 8 / 42), (2, 0, 4 / 42),
        (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42), (1, 1, 4 / 42), (2, 1, 2 / 42),
        (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42), (1, 2, 2 / 42), (2, 2, 1 / 42),
    ],
    "burkes": [
        (1, 0, 8 / 32), (2, 0, 4 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 8 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
    ],
    "sierra": [
        (1, 0, 5 / 32), (2, 0, 3 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 5 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
        (-1, 2, 2 / 32), (0, 2, 3 / 32), (1, 2, 2 / 32),
    ],
    "sierra_two_row": [
        (1, 0, 4 / 16), (2, 0, 3 / 16),
        (-2, 1, 1 / 16), (-1, 1, 2 / 16), (0, 1, 3 / 16), (1, 1, 2 / 16), (2, 1, 1 / 16),
    ],
    "sierra_lite": [
        (1, 0, 2 / 4), (-1, 1, 1 / 4), (0, 1, 1 / 4),
    ],
    "shiau_fan_1": [
        (1, 0, 8 / 16),
        (-2, 1, 1 / 16), (-1, 1, 1 / 16), (0, 1, 2 / 16), (1, 1, 4 / 16),
    ],
    "shiau_fan_2": [
        (1, 0, 7 / 16),
        (-1, 1, 1 / 16), (0, 1, 3 / 16), (1, 1, 5 / 16),
    ],
    "shiau_fan_3": [
        (1, 0, 4 / 8),
        (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 2 / 8),
    ],
}

# Canonical name lookup for user-facing aliases
KERNEL_ALIASES: dict[str, str] = {
    "false_floyd": "false_floyd_steinberg",
    "fake_floyd": "false_floyd_steinberg",
    "fake_floyd_steinberg": "false_floyd_steinberg",
    "jarvis": "jarvis_judice_ninke",
    "jjn": "jarvis_judice_ninke",
    "sierra3": "sierra",
    "sierra_3": "sierra",
    "sierra2row": "sierra_two_row",
    "sierra_2row": "sierra_two_row",
    "two_row_sierra": "sierra_two_row",
    "tworowsierra": "sierra_two_row",
    "shiaufan1": "shiau_fan_1",
    "shiau_fan": "shiau_fan_1",
    "shiaufan2": "shiau_fan_2",
    "shiaufan3": "shiau_fan_3",
}


def resolve_error_diffusion_kernel(algo: str) -> str | None:
    """Map an algorithm name to its canonical error diffusion kernel key."""
    clean = algo.lower().replace("-", "_").strip()
    if clean in ERROR_DIFFUSION_KERNELS:
        return clean
    return KERNEL_ALIASES.get(clean)


def dither_palette_bayer(
    image: np.ndarray,
    palette: np.ndarray,
    matrix_type: str = "bayer4x4",
    strength: float = 0.8,
    metric: str = "oklab",
    dither_scale: int = 1,
) -> np.ndarray:
    """Apply Bayer or Blue Noise ordered dithering and map strictly to the given palette."""
    from retexture.core.quantize import match_palette_nearest

    if strength <= 0.001 or matrix_type.lower() == "none":
        return match_palette_nearest(image, palette, metric=metric)

    bayer = get_bayer_matrix(matrix_type)
    if dither_scale > 1:
        bayer = np.repeat(np.repeat(bayer, dither_scale, axis=0), dither_scale, axis=1)

    bh, bw = bayer.shape
    h, w, _ = image.shape

    if len(palette) > 1:
        pal_f = palette.astype(np.float32)
        diffs = pal_f[:, np.newaxis, :] - pal_f[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        np.fill_diagonal(dists, 1e6)
        min_dists = np.min(dists, axis=1)
        avg_step = float(np.mean(min_dists))
    else:
        avg_step = 64.0

    tiles_y = (h + bh - 1) // bh
    tiles_x = (w + bw - 1) // bw
    tiled_bayer = np.tile(bayer, (tiles_y, tiles_x))[:h, :w, np.newaxis]

    perturbation = tiled_bayer * (avg_step * strength * 0.7)
    perturbed_img = np.clip(image.astype(np.float32) + perturbation, 0, 255).astype(np.uint8)

    return match_palette_nearest(perturbed_img, palette, metric=metric)


def dither_palette_blue_noise(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 0.85,
    dither_scale: int = 1,
    metric: str = "oklab",
) -> np.ndarray:
    """Apply organic, isotropic Blue Noise void-and-cluster matrix dithering."""
    return dither_palette_bayer(
        image=image,
        palette=palette,
        matrix_type="blue_noise",
        strength=strength,
        metric=metric,
        dither_scale=dither_scale,
    )


def dither_palette_yliluoma(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 0.85,
    metric: str = "oklab",
) -> np.ndarray:
    """Yliluoma-style pairwise optical dithering in the configured metric.

    For every pixel, all palette pairs are projected against the target to
    find the pair and ideal mixing ratio with the smallest error. The Bayer
    cell then chooses the first or second endpoint according to that ratio.
    Work is chunked to keep memory bounded.
    """
    from retexture.core.quantize import match_palette_nearest, palette_to_metric_space, to_metric_space

    if len(palette) <= 2 or strength <= 0.01:
        return match_palette_nearest(image, palette, metric=metric)

    h, w, _ = image.shape
    target = to_metric_space(image, metric).reshape(-1, 3).astype(np.float32)
    pal_space = palette_to_metric_space(palette, metric).astype(np.float32)
    pairs = np.array([(a, b) for a in range(len(palette)) for b in range(a + 1, len(palette))], dtype=np.int32)
    a_space = pal_space[pairs[:, 0]]
    delta = pal_space[pairs[:, 1]] - a_space
    denom = np.maximum(np.sum(delta * delta, axis=1), 1e-8)
    best_a = np.zeros(len(target), dtype=np.int32)
    best_b = np.ones(len(target), dtype=np.int32)
    best_ratio = np.zeros(len(target), dtype=np.float32)

    for start in range(0, len(target), 4096):
        stop = min(start + 4096, len(target))
        block = target[start:stop]
        projection = np.sum((block[:, None, :] - a_space[None, :, :]) * delta[None, :, :], axis=2) / denom[None, :]
        ratio = np.clip(projection, 0.0, 1.0)
        mix = a_space[None, :, :] + ratio[..., None] * delta[None, :, :]
        errors = np.sum((block[:, None, :] - mix) ** 2, axis=2)
        choice = np.argmin(errors, axis=1)
        rows = np.arange(stop - start)
        best_a[start:stop] = pairs[choice, 0]
        best_b[start:stop] = pairs[choice, 1]
        best_ratio[start:stop] = ratio[rows, choice]

    matrix = get_bayer_matrix("bayer4x4")
    tiled = np.tile(matrix, ((h + 3) // 4, (w + 3) // 4))[:h, :w].reshape(-1)
    threshold = 0.5 + tiled
    mix_ratio = 0.5 + (best_ratio - 0.5) * float(np.clip(strength, 0.0, 1.0))
    result_indices = np.where(threshold < mix_ratio, best_b, best_a)
    return palette[result_indices].reshape(h, w, 3).astype(np.uint8)


def dither_1bit_weighted(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 1.0,
    bias: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Reduce to two palette colours using a weighted neighbourhood shader."""
    from retexture.core.quantize import to_metric_space

    if len(palette) < 2:
        return np.repeat(palette[:1], image.shape[0] * image.shape[1], axis=0).reshape(image.shape)

    palette_space = to_metric_space(palette, "oklab")
    order = np.argsort(palette_space[:, 0])[:2]
    dark, light = int(order[0]), int(order[-1])
    rgb = image[..., :3].astype(np.float32) / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    padded = np.pad(lum, ((1, 1), (1, 1)), mode="edge")
    weighted = (
        padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:]
        + 2 * padded[1:-1, :-2] + 4 * padded[1:-1, 1:-1] + 2 * padded[1:-1, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    ) / 16.0
    weighted = (weighted - 0.5) * (1.0 + float(contrast)) + 0.5 + float(bias)
    matrix = get_bayer_matrix("bayer4x4")
    threshold = 0.5 + np.tile(matrix, ((image.shape[0] + 3) // 4, (image.shape[1] + 3) // 4))[:image.shape[0], :image.shape[1]] * 0.35 * np.clip(strength, 0.0, 1.0)
    return np.where((weighted >= threshold)[..., np.newaxis], palette[light], palette[dark]).astype(np.uint8)


def _generate_hilbert_curve(order: int) -> list[tuple[int, int]]:
    """Generate 2D coordinates along a Hilbert space-filling curve of given order."""
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            return y, x
        return x, y

    n = 1 << order
    points = []
    for i in range(n * n):
        t = i
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        points.append((x, y))
    return points


def dither_palette_riemersma(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 1.0,
    metric: str = "oklab",
) -> np.ndarray:
    """Riemersma Dithering using a fractal Hilbert space-filling curve.

    Eliminates linear sweep artifacts and directional worm trails by diffusing
    errors along a multi-scale fractal curve with an exponential decay buffer.

    Nearest-color matching and the error history operate strictly in the
    configured metric space (Oklab by default); output colors are exact sRGB
    palette entries.
    """
    from retexture.core.quantize import palette_to_metric_space, to_metric_space

    h, w, _ = image.shape

    # Determine Hilbert curve order to cover the image
    max_dim = max(h, w)
    order = int(np.ceil(np.log2(max_dim)))
    order = max(order, 1)

    points = _generate_hilbert_curve(order)
    buf = to_metric_space(image, metric).astype(np.float32)
    pal_space = palette_to_metric_space(palette, metric)
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # 16-element exponential decay error history weights
    history_size = 16
    decay_ratio = np.exp(-np.log(history_size) / (history_size - 1))
    weights = np.array([decay_ratio ** i for i in range(history_size)], dtype=np.float32)
    weights /= weights.sum()

    error_history = np.zeros((history_size, 3), dtype=np.float32)

    # In error diffusion / feedback loops, gain > 1.0 creates positive feedback
    # and exponential divergence. Clamp to [0.0, 1.0] for unconditional stability.
    strength = float(np.clip(strength, 0.0, 1.0))

    for x, y in points:
        if x >= w or y >= h:
            continue

        accumulated_error = np.sum(error_history * weights[:, np.newaxis], axis=0) * strength
        target_color = buf[y, x] + accumulated_error

        dists = np.sum((pal_space - target_color) ** 2, axis=1)
        best_idx = int(np.argmin(dists))
        output[y, x] = palette[best_idx]

        error_history = np.roll(error_history, 1, axis=0)
        error_history[0] = target_color - pal_space[best_idx]

    return output


def dither_palette_kernel(
    image: np.ndarray,
    palette: np.ndarray,
    kernel: str = "floyd_steinberg",
    strength: float = 1.0,
    metric: str = "oklab",
) -> np.ndarray:
    """Generic error diffusion dithering with a named diffusion kernel.

    Supports Floyd-Steinberg, Atkinson, Stucki, Burkes, Sierra (3-row),
    Two-Row Sierra, Sierra Lite, Jarvis-Judice-Ninke, False Floyd-Steinberg
    and the Shiau-Fan family.

    Nearest-color matching and error diffusion run strictly in the configured
    metric space (Oklab by default): the image and palette are converted once
    up front, the buffer accumulates diffused error in that space, and only
    the chosen sRGB palette entries are written to the output.
    """
    from retexture.core.quantize import palette_to_metric_space, to_metric_space

    canonical = resolve_error_diffusion_kernel(kernel)
    if canonical is None:
        raise ValueError(f"Unknown error diffusion kernel: {kernel}")
    offsets = ERROR_DIFFUSION_KERNELS[canonical]

    h, w, _ = image.shape
    buf = to_metric_space(image, metric).astype(np.float32)
    pal_space = palette_to_metric_space(palette, metric)
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # In error diffusion / feedback loops, gain > 1.0 creates positive feedback
    # and exponential divergence. Clamp to [0.0, 1.0] for unconditional stability.
    strength = float(np.clip(strength, 0.0, 1.0))

    for y in range(h):
        for x in range(w):
            old_val = buf[y, x]
            dists = np.sum((pal_space - old_val) ** 2, axis=1)
            best_idx = int(np.argmin(dists))
            output[y, x] = palette[best_idx]

            err = (old_val - pal_space[best_idx]) * strength

            for dx, dy, weight in offsets:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    buf[ny, nx] += err * weight

    return output


def dither_palette_floyd_steinberg(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 1.0,
    metric: str = "oklab",
) -> np.ndarray:
    """Apply Floyd-Steinberg error diffusion dithering strictly mapped to palette."""
    return dither_palette_kernel(image, palette, kernel="floyd_steinberg", strength=strength, metric=metric)


def dither_palette_atkinson(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 1.0,
    metric: str = "oklab",
) -> np.ndarray:
    """Apply Atkinson error diffusion dithering strictly mapped to palette."""
    return dither_palette_kernel(image, palette, kernel="atkinson", strength=strength, metric=metric)


def dither_palette_dot_diffusion(
    image: np.ndarray,
    palette: np.ndarray,
    strength: float = 1.0,
    metric: str = "oklab",
) -> np.ndarray:
    """Knuth-style dot diffusion dithering.

    Processes pixels class-by-class (diagonal 8x8 class matrix) in a
    serpentine raster and diffuses quantization error to the four orthogonal
    neighbors, which always belong to the next class.

    Nearest-color matching and error diffusion run strictly in the configured
    metric space (Oklab by default); output colors are exact sRGB palette
    entries.
    """
    from retexture.core.quantize import palette_to_metric_space, to_metric_space

    h, w, _ = image.shape
    buf = to_metric_space(image, metric).astype(np.float32)
    pal_space = palette_to_metric_space(palette, metric)
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # In error diffusion / feedback loops, gain > 1.0 creates positive feedback
    # and exponential divergence. Clamp to [0.0, 1.0] for unconditional stability.
    strength = float(np.clip(strength, 0.0, 1.0))

    for cls in range(8):
        for y in range(h):
            x_iter = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
            for x in x_iter:
                if KNUTH_CLASS_8X8[y % 8, x % 8] != cls:
                    continue

                old_val = buf[y, x]
                dists = np.sum((pal_space - old_val) ** 2, axis=1)
                best_idx = int(np.argmin(dists))
                output[y, x] = palette[best_idx]

                err = (old_val - pal_space[best_idx]) * strength * 0.25
                # Only later classes are still unquantized. This avoids the
                # old implementation's feedback into already-written dots.
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        if KNUTH_CLASS_8X8[ny % 8, nx % 8] > cls:
                            buf[ny, nx] += err

    return output
