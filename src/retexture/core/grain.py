"""Film grain and analogue noise module for retexture.

Generates reproducible or randomized texture noise:
- Configurable grain intensity / amount
- Configurable random seed for deterministic batch outputs
- Monochrome vs. chromatic noise toggle
"""

from __future__ import annotations

import numpy as np


def apply_grain(
    image: np.ndarray,
    amount: float = 0.0,
    seed: int | None = None,
    monochrome: bool = True,
    periodic: bool = False,
) -> np.ndarray:
    """Apply film grain noise to an RGB uint8 image.

    When ``periodic`` is enabled, the outer row/column share their samples so
    the grain itself cannot reintroduce a seamless-tile seam.
    """
    if amount <= 0.001:
        return image

    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    sigma = amount * 128.0

    if monochrome:
        noise = rng.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
        noise = np.repeat(noise, 3, axis=-1)
    else:
        noise = rng.normal(0.0, sigma, (h, w, 3)).astype(np.float32)

    if periodic and h > 1 and w > 1:
        noise[-1, :, :] = noise[0, :, :]
        noise[:, -1, :] = noise[:, 0, :]

    has_alpha = image.shape[-1] == 4
    rgb = image[..., :3].astype(np.float32) + noise
    out_rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if has_alpha:
        return np.concatenate([out_rgb, image[..., 3:]], axis=-1)
    return out_rgb
