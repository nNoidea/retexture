"""Single Source of Truth transformation pipeline for retexture.

All image operations for both the interactive GUI live preview and the batch
CLI engine pass through this exact module to guarantee 100% bit-for-bit parity.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

from retexture.config import RetextureConfig
from retexture.core.color_adjust import adjust_colors, apply_oklab_style
from retexture.core.dither import (
    dither_1bit_weighted,
    dither_palette_bayer,
    dither_palette_blue_noise,
    dither_palette_dot_diffusion,
    dither_palette_kernel,
    dither_palette_riemersma,
    dither_palette_yliluoma,
    get_bayer_matrix,
    resolve_error_diffusion_kernel,
    srgb_to_linear,
)
from retexture.core.grain import apply_grain
from retexture.core.palettes import (
    load_palette_preset,
)
from retexture.core.quantize import (
    match_palette_nearest,
)
from retexture.core.shader_fx import (
    apply_bilateral_simplify,
    apply_cel_shading_bands,
    apply_chromatic_aberration,
    apply_color_temperature_shift,
    apply_crt_scanlines,
    apply_edge_crunch,
    apply_grime_decay,
    apply_kuwahara_filter,
    apply_outline_injection,
    apply_pixeloe_stylizer,
)

RESIZE_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def load_input_image(source: Image.Image | np.ndarray | bytes | str | Path) -> Image.Image:
    """Standardize source input into a PIL RGB/RGBA Image."""
    if isinstance(source, Image.Image):
        return source.copy()
    if isinstance(source, np.ndarray):
        return Image.fromarray(source)
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source))
    if isinstance(source, (str, Path)):
        return Image.open(source)
    raise ValueError(f"Unsupported image input source type: {type(source)}")


def k_centroids_downsample(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Edge-preserving K-Centroids downscaling.

    Subdivides the input image into grid blocks and computes the dominant centroid
    for each block rather than a blurry average, preserving 1px lines and high contrast edges.
    """
    has_alpha = img.mode == "RGBA"
    arr = np.array(img).astype(np.float32)
    h, w, c = arr.shape

    if h < target_h or w < target_w:
        return img.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

    block_h = h / target_h
    block_w = w / target_w

    out = np.zeros((target_h, target_w, c), dtype=np.uint8)

    for ty in range(target_h):
        sy_start = int(ty * block_h)
        sy_end = max(sy_start + 1, int((ty + 1) * block_h))
        for tx in range(target_w):
            sx_start = int(tx * block_w)
            sx_end = max(sx_start + 1, int((tx + 1) * block_w))

            block = arr[sy_start:sy_end, sx_start:sx_end].reshape(-1, c)
            n_pixels = len(block)
            if n_pixels <= 1:
                out[ty, tx] = np.clip(block[0], 0, 255).astype(np.uint8)
                continue

            # Compute block luminance from RGB channels to detect contrast/edges
            lum = 0.299 * block[:, 0] + 0.587 * block[:, 1] + 0.114 * block[:, 2]
            lum_min, lum_max = np.min(lum), np.max(lum)

            if lum_max - lum_min > 40.0:
                # High contrast edge detected in block: split into 2 clusters (dark & bright)
                mid = (lum_min + lum_max) * 0.5
                dark_mask = lum < mid
                bright_mask = ~dark_mask

                dark_count = np.count_nonzero(dark_mask)
                bright_count = n_pixels - dark_count

                # If dark pixels represent an edge line (at least 15% of block), preserve dark edge
                if 0.15 <= (dark_count / n_pixels) <= 0.65:
                    centroid = np.mean(block[dark_mask], axis=0)
                elif 0.15 <= (bright_count / n_pixels) <= 0.65:
                    centroid = np.mean(block[bright_mask], axis=0)
                else:
                    dominant_mask = dark_mask if dark_count >= bright_count else bright_mask
                    centroid = np.mean(block[dominant_mask], axis=0)
            else:
                centroid = np.median(block, axis=0)

            out[ty, tx] = np.clip(centroid, 0, 255).astype(np.uint8)

    mode = "RGBA" if has_alpha else "RGB"
    return Image.fromarray(out, mode=mode)


def process_image(
    source: Image.Image | np.ndarray | bytes | str | Path,
    config: RetextureConfig | None = None,
) -> Image.Image:
    """Execute the complete retro texture pipeline on an image."""
    if config is None:
        config = RetextureConfig()

    img = load_input_image(source)

    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    # 1. Downscale / Resample (Keep original native resolution if size <= 0)
    target_w = int(config.size[0]) if config.size and config.size[0] > 0 else 0
    target_h = int(config.size[1]) if config.size and len(config.size) > 1 and config.size[1] > 0 else 0

    if target_w > 0 and target_h > 0:
        # Aspect-preserving scale: shortest image side maps to min(target_w, target_h)
        scale = min(target_w, target_h) / min(img.width, img.height)
        target_w = max(1, round(img.width * scale))
        target_h = max(1, round(img.height * scale))
    else:
        target_w = target_w if target_w > 0 else img.width
        target_h = target_h if target_h > 0 else img.height

    filter_mode = config.resize_filter.lower()

    if (img.width, img.height) != (target_w, target_h):
        if "centroid" in filter_mode or "k_centroid" in filter_mode:
            img = k_centroids_downsample(img, target_w, target_h)
        else:
            resample_filter = RESIZE_FILTERS.get(filter_mode, Image.Resampling.NEAREST)
            img = img.resize((target_w, target_h), resample=resample_filter)

    img_arr = np.array(img)
    has_alpha_channel = img_arr.ndim == 3 and img_arr.shape[-1] == 4
    alpha_channel = img_arr[..., 3] if has_alpha_channel else None
    rgb_arr = img_arr[..., :3]

    # 2. Seamless Tiling
    if config.seamless_tiling:
        from retexture.core.tiling import make_seamless
        rgb_arr = make_seamless(
            rgb_arr,
            seam_size=config.seam_size,
            method=getattr(config, "seam_method", "offset_wrap"),
        )

    # 2.5 Pre-Blur (Anti-aliasing and noise suppression)
    if getattr(config, "pre_blur", 0.0) > 0.001:
        blur_pimg = Image.fromarray(rgb_arr).filter(ImageFilter.GaussianBlur(radius=config.pre_blur))
        rgb_arr = np.array(blur_pimg)

    # 2.8 Clean Surface: Kuwahara structure filter (flattens wall grain into clean painterly planes without blurring edges)
    if getattr(config, "kuwahara_filter", 0.0) > 0.001:
        rgb_arr = apply_kuwahara_filter(
            rgb_arr,
            amount=config.kuwahara_filter,
            periodic=config.seamless_tiling,
        )

    # 3. Bilateral smoothing pass (flattens photographic speckle into clean planes)
    if getattr(config, "bilateral_simplify", 0.0) > 0.001:
        rgb_arr = apply_bilateral_simplify(
            rgb_arr,
            amount=config.bilateral_simplify,
            periodic=config.seamless_tiling,
        )

    # 4. PixelOE structural stylizer (patch contrast enhancement & pixel sharpening)
    if getattr(config, "pixeloe_strength", 0.0) > 0.001:
        rgb_arr = apply_pixeloe_stylizer(
            rgb_arr,
            amount=config.pixeloe_strength,
            periodic=config.seamless_tiling,
        )

    # 5. Smart Pixel Art Outline Injection (noise-gated connected contour detector)
    if getattr(config, "outline_strength", 0.0) > 0.001:
        rgb_arr = apply_outline_injection(
            rgb_arr,
            amount=config.outline_strength,
            noise_gate=getattr(config, "edge_noise_gate", 0.5),
            periodic=config.seamless_tiling,
        )

    # 6. Smart Edge Crunch (noise-gated boundary sharpness boost)
    if config.edge_crunch > 0.001:
        rgb_arr = apply_edge_crunch(
            rgb_arr,
            amount=config.edge_crunch,
            noise_gate=getattr(config, "edge_noise_gate", 0.5),
            periodic=config.seamless_tiling,
        )

    # 7. Color Grading & 3-Way Tonal Shaping
    rgb_arr = adjust_colors(
        rgb_arr,
        hue_shift=config.hue,
        saturation=config.saturation,
        brightness=config.brightness,
        contrast=config.contrast,
        midtones=getattr(config, "midtones", 0.0),
        highlights=getattr(config, "highlights", 0.0),
        luminance_threshold=getattr(config, "luminance_threshold", 0.5),
        invert_luminance=getattr(config, "invert_luminance", False),
    )

    # 8. Color Temperature Split-Toning (Warm Highlights / Cool Shadows)
    if abs(config.color_temperature) > 0.001:
        rgb_arr = apply_color_temperature_shift(rgb_arr, amount=config.color_temperature)

    rgb_arr = apply_oklab_style(
        rgb_arr,
        black_point=getattr(config, "black_point", 0.0),
        white_point=getattr(config, "white_point", 1.0),
        s_curve=getattr(config, "s_curve", 0.0),
        shadow_hue_shift=getattr(config, "shadow_hue_shift", 0.0),
        highlight_hue_shift=getattr(config, "highlight_hue_shift", 0.0),
        shadow_saturation=getattr(config, "shadow_saturation", 1.0),
        highlight_saturation=getattr(config, "highlight_saturation", 1.0),
    )

    # 9. Cel-Shading Tone Stepping (Discrete Value Ramping)
    if config.cel_shading_steps >= 2:
        rgb_arr = apply_cel_shading_bands(rgb_arr, steps=config.cel_shading_steps)

    # Resolve the authored palette before surface marks. Palette decisions
    # should be made from the clean image rather than from grime or grain that
    # will be quantized later.
    try:
        palette = load_palette_preset(config.palette_preset or "ps1_classic_16")
    except FileNotFoundError:
        palette = load_palette_preset("ps1_classic_16")

    # Optical effects are pre-quantization by default, so a fixed palette stays
    # fixed. bake_optical_fx explicitly opts into post-quantization artifacts.
    if not config.bake_optical_fx and config.chromatic_aberration > 0.001:
        rgb_arr = apply_chromatic_aberration(rgb_arr, amount=config.chromatic_aberration)

    # 11. Grime & Decay (surface wear and mold)
    if config.grime_decay > 0.001:
        seed = config.grain_seed or 42
        rgb_arr = apply_grime_decay(
            rgb_arr,
            amount=config.grime_decay,
            seed=seed,
            mode=getattr(config, "grime_mode", "uniform"),
            palette=palette,
        )

    # 12. Film Grain / Analogue Noise
    if config.grain > 0.001:
        rgb_arr = apply_grain(
            rgb_arr,
            amount=config.grain,
            seed=config.grain_seed,
            monochrome=config.grain_monochrome,
            periodic=config.seamless_tiling,
        )

    if not config.bake_optical_fx and config.crt_scanlines > 0.001:
        rgb_arr = apply_crt_scanlines(rgb_arr, amount=config.crt_scanlines)

    # Blur, bilateral filtering, and local edge operators are intentionally
    # normal-image operations. Reconcile their border neighbourhood one last
    # time before quantization so the exported palette decisions agree at the
    # repeat boundary.
    if config.seamless_tiling:
        from retexture.core.tiling import make_seamless
        rgb_arr = make_seamless(
            rgb_arr,
            seam_size=config.seam_size,
            method=getattr(config, "seam_method", "offset_wrap"),
        )

    # 13. Quantization & Dithering in Perceptual / Linear Space
    dither_algo = config.dither_algorithm.lower()
    strength = config.dither_strength
    dither_scale = getattr(config, "dither_scale", 1)
    use_linear = getattr(config, "linear_color_space", False)
    metric = getattr(config, "color_metric", "oklab")

    # Perceptual metrics (Oklab/CIELAB) linearize internally, in float, as part
    # of their sRGB conversion — pre-linearizing here would apply the sRGB EOTF
    # a second time. The linear input only feeds RGB-space metrics.
    perceptual_metric = metric in ("oklab", "cielab")

    # Optional Linear Photometric transformation for gamma-accurate dither energy
    linear_input: np.ndarray | None = None
    if use_linear:
        linear_input = np.clip(srgb_to_linear(rgb_arr.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)

    dither_input = rgb_arr
    if use_linear and not perceptual_metric:
        dither_input = linear_input

    dither_palette = palette
    if use_linear and not perceptual_metric:
        dither_palette = np.clip(srgb_to_linear(palette.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)

    if dither_algo in ("dither_1bit_weighted", "one_bit", "1bit"):
        matched = dither_1bit_weighted(
            rgb_arr,
            palette,
            strength=strength,
            bias=getattr(config, "one_bit_bias", 0.0),
            contrast=getattr(config, "one_bit_contrast", 1.0),
        )
    elif dither_algo == "none" or strength <= 0.001:
        matched = match_palette_nearest(dither_input, dither_palette, metric=metric)
    elif dither_algo in ("blue_noise", "bluenoise", "blue"):
        matched = dither_palette_blue_noise(
            dither_input,
            dither_palette,
            strength=strength,
            metric=metric,
            dither_scale=dither_scale,
        )
    elif dither_algo in ("yliluoma", "yliluoma2", "optical_blend", "blend"):
        matched = dither_palette_yliluoma(
            dither_input,
            dither_palette,
            strength=strength,
            metric=metric,
        )
    elif dither_algo in ("riemersma", "hilbert", "fractal"):
        matched = dither_palette_riemersma(
            dither_input,
            dither_palette,
            strength=strength,
            metric=metric,
        )
    elif dither_algo in ("dot_diffusion", "dotdiffusion", "knuth", "dot_diff"):
        matched = dither_palette_dot_diffusion(
            dither_input,
            dither_palette,
            strength=strength,
            metric=metric,
        )
    elif dither_algo in (
        "bayer2x2", "bayer4x4", "bayer8x8", "bayer",
        "halftone_dot", "halftone_cross", "halftone",
        "interlaced", "checkerboard", "saturn",
        "crosshatch", "hatching", "pc98",
        "cluster_dot_4x4", "clusterdot4x4",
        "cluster_dot_8x8", "clusterdot8x8", "cluster_dot", "clusterdot",
        "line_halftone", "linehalftone", "halftone_line",
    ):
        matched = dither_palette_bayer(
            dither_input,
            dither_palette,
            matrix_type=dither_algo,
            strength=strength,
            metric=metric,
            dither_scale=dither_scale,
        )
    elif resolve_error_diffusion_kernel(dither_algo) is not None:
        matched = dither_palette_kernel(
            dither_input,
            dither_palette,
            kernel=dither_algo,
            strength=strength,
            metric=metric,
        )
    else:
        matched = match_palette_nearest(dither_input, dither_palette, metric=metric)

    if use_linear and not perceptual_metric:
        # Map linear matched palette colors back to exact sRGB palette colors
        retrofied_rgb = match_palette_nearest(matched, palette, metric="euclidean")
    else:
        retrofied_rgb = matched

    # Optional intentionally non-palette-pure optical bake.
    if config.bake_optical_fx and config.crt_scanlines > 0.001:
        retrofied_rgb = apply_crt_scanlines(retrofied_rgb, amount=config.crt_scanlines)
    if config.bake_optical_fx and config.chromatic_aberration > 0.001:
        retrofied_rgb = apply_chromatic_aberration(retrofied_rgb, amount=config.chromatic_aberration)
    if config.seamless_tiling:
        from retexture.core.tiling import enforce_tile_edges
        retrofied_rgb = enforce_tile_edges(retrofied_rgb)

    # 15. Recombine Alpha Channel if applicable with optional 1-bit screen-door stippling
    if alpha_channel is not None:
        if getattr(config, "alpha_stipple", False):
            stipple_matrix = get_bayer_matrix(dither_algo)
            if dither_scale > 1:
                stipple_matrix = np.repeat(np.repeat(stipple_matrix, dither_scale, axis=0), dither_scale, axis=1)
            sm_h, sm_w = stipple_matrix.shape
            ah, aw = alpha_channel.shape
            tiled_sm = np.tile(stipple_matrix, ((ah + sm_h - 1) // sm_h, (aw + sm_w - 1) // sm_w))[:ah, :aw]
            stipple_thresh = np.clip(128.0 - tiled_sm * 255.0 * 0.9, 1.0, 254.0)
            alpha_channel = np.where(alpha_channel.astype(np.float32) >= stipple_thresh, 255, 0).astype(np.uint8)

        if config.seamless_tiling:
            from retexture.core.tiling import enforce_tile_edges
            alpha_channel = enforce_tile_edges(alpha_channel)

        final_arr = np.dstack([retrofied_rgb, alpha_channel])
        return Image.fromarray(final_arr, mode="RGBA")

    return Image.fromarray(retrofied_rgb, mode="RGB")


def export_image_bytes(
    image: Image.Image,
    fmt: str = "png",
    quality: int = 90,
) -> bytes:
    """Serialize PIL Image to byte buffer in requested format."""
    buf = io.BytesIO()
    fmt_clean = fmt.lower().strip()
    if fmt_clean in ("jpg", "jpeg"):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=quality, optimize=True)
    elif fmt_clean == "webp":
        image.save(buf, format="WEBP", quality=quality)
    elif fmt_clean == "bmp":
        image.save(buf, format="BMP")
    elif fmt_clean == "tga":
        image.save(buf, format="TGA")
    else:  # default png
        image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
