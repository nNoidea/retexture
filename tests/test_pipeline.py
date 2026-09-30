import numpy as np
from PIL import Image
from retexture.config import RetextureConfig
from retexture.core.pipeline import (
    export_image_bytes,
    k_centroids_downsample,
    process_image,
)


def test_pipeline_standard_sizes():
    img = Image.new("RGB", (300, 300), color=(120, 80, 200))
    for size in (64, 128, 256, 512):
        cfg = RetextureConfig(size=[size, size], palette_preset="ps1_classic_16")
        res = process_image(img, cfg)
        assert res.size == (size, size)


def test_pipeline_keep_original_resolution():
    img = Image.new("RGB", (345, 217), color=(80, 150, 220))
    cfg = RetextureConfig(size=[0, 0])
    res = process_image(img, cfg)
    assert res.size == (345, 217)


def test_pipeline_preserves_aspect_ratio_downscale():
    img = Image.new("RGB", (300, 200), color=(80, 150, 220))
    cfg = RetextureConfig(size=[512, 512], palette_preset="ps1_classic_16")
    res = process_image(img, cfg)
    assert res.size == (768, 512)


def test_pipeline_preserves_aspect_ratio_upscale():
    img = Image.new("RGB", (150, 100), color=(80, 150, 220))
    cfg = RetextureConfig(size=[512, 512], palette_preset="ps1_classic_16")
    res = process_image(img, cfg)
    assert res.size == (768, 512)


def test_pipeline_preserves_aspect_ratio_rectangular_target():
    img = Image.new("RGB", (300, 200), color=(80, 150, 220))
    cfg = RetextureConfig(size=[128, 64], palette_preset="ps1_classic_16")
    res = process_image(img, cfg)
    assert res.size == (96, 64)


def test_pipeline_preserves_aspect_ratio_tall_image():
    img = Image.new("RGB", (200, 400), color=(80, 150, 220))
    cfg = RetextureConfig(size=[256, 256], palette_preset="ps1_classic_16")
    res = process_image(img, cfg)
    assert res.size == (256, 512)


def test_mouthwashing_preset():
    img = Image.new("RGB", (200, 200), color=(100, 140, 80))
    cfg = RetextureConfig.load("mouthwashing_psx_horror")
    res = process_image(img, cfg)
    assert res.size == (128, 128)
    assert res.mode == "RGB"


def test_grain_seed_determinism():
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    cfg1 = RetextureConfig(size=[64, 64], grain=0.1, grain_seed=42)
    cfg2 = RetextureConfig(size=[64, 64], grain=0.1, grain_seed=42)
    res1 = np.array(process_image(img, cfg1))
    res2 = np.array(process_image(img, cfg2))
    assert np.array_equal(res1, res2)


def test_export_formats():
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    png_bytes = export_image_bytes(img, fmt="png")
    jpg_bytes = export_image_bytes(img, fmt="jpg", quality=85)
    webp_bytes = export_image_bytes(img, fmt="webp", quality=85)
    assert png_bytes.startswith(b"\x89PNG")
    assert jpg_bytes.startswith(b"\xff\xd8")
    assert webp_bytes.startswith(b"RIFF")


def test_k_centroids_downsample():
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    # Draw a black 1-pixel cross
    arr = np.array(img)
    arr[64, :] = [0, 0, 0]
    arr[:, 64] = [0, 0, 0]
    img = Image.fromarray(arr)

    downscaled = k_centroids_downsample(img, 32, 32)
    assert downscaled.size == (32, 32)
    # Verify dark pixels are retained
    down_arr = np.array(downscaled)
    assert np.min(down_arr) < 50


def test_config_json_serialization():
    cfg = RetextureConfig(
        size=[256, 256],
        chromatic_aberration=0.3,
        grime_decay=0.25,
        color_metric="oklab",
        outline_strength=0.5,
        pixeloe_strength=0.7,
    )
    json_str = cfg.to_json()
    reconstructed = RetextureConfig.from_json(json_str)
    assert reconstructed.size == [256, 256]
    assert reconstructed.chromatic_aberration == 0.3
    assert reconstructed.grime_decay == 0.25
    assert reconstructed.color_metric == "oklab"
    assert reconstructed.outline_strength == 0.5
    assert reconstructed.pixeloe_strength == 0.7
    assert RetextureConfig.from_dict({"export_format": "webp"}).export_format == "webp"


def test_all_builtin_presets_execute():
    from pathlib import Path
    img = Image.new("RGB", (128, 128), color=(140, 90, 70))
    preset_files = list((Path.cwd() / "presets" / "configs").glob("*.json"))
    assert len(preset_files) >= 20
    for pf in preset_files:
        cfg = RetextureConfig.load(pf.stem)
        res = process_image(img, cfg)
        assert res is not None
        assert res.size == (cfg.size[0], cfg.size[1])
        assert res.mode in ("RGB", "RGBA")


def test_pipeline_rgba_k_centroids():
    rgba_img = Image.new("RGBA", (128, 128), color=(255, 100, 50, 180))
    cfg = RetextureConfig(size=[64, 64], resize_filter="k_centroids")
    res = process_image(rgba_img, cfg)
    assert res.size == (64, 64)
    res_arr = np.array(res)
    assert res_arr[..., 3].shape == (64, 64)


def test_pipeline_alpha_stipple():
    # Gradient alpha from 0 to 255
    alpha_ramp = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape((64, 64))
    rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha_ramp])
    img = Image.fromarray(rgba, mode="RGBA")

    cfg = RetextureConfig(
        size=[64, 64],
        alpha_stipple=True,
        dither_algorithm="blue_noise",
    )
    res = process_image(img, cfg)
    assert res.mode == "RGBA"
    res_arr = np.array(res)
    alpha = res_arr[..., 3]
    # Screen-door 1-bit alpha must strictly contain only 0 or 255
    unique_vals = set(np.unique(alpha))
    assert unique_vals.issubset({0, 255})
    # Both transparent and opaque pixels should exist across the ramp
    assert 0 in unique_vals
    assert 255 in unique_vals


def test_pipeline_pre_blur_and_smooth_diffuse():
    # Image with sharp random speckles
    np.random.seed(42)
    noisy_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(noisy_arr, mode="RGB")

    cfg = RetextureConfig(
        size=[64, 64],
        pre_blur=1.5,
        bilateral_simplify=0.8,
        dither_algorithm="none",
    )
    res = process_image(img, cfg)
    res_arr = np.array(res)
    # Variance of smoothed image should be significantly lower than noisy original
    assert np.var(res_arr) < np.var(noisy_arr)


def test_linear_color_space_no_double_transform():
    """Audit point 2: with linear_color_space=True and a perceptual metric, matching
    must equal direct sRGB Oklab matching — the sRGB EOTF must not be applied twice."""
    from retexture.core.color_adjust import adjust_colors
    from retexture.core.palettes import load_palette_preset
    from retexture.core.quantize import match_palette_nearest

    rng = np.random.default_rng(9)
    img_arr = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)

    cfg = RetextureConfig(
        size=[0, 0],
        saturation=1.0,
        brightness=1.0,
        contrast=1.0,
        grain=0.0,
        alpha_stipple=False,
        dither_algorithm="none",
        linear_color_space=True,
        color_metric="oklab",
    )
    result = np.array(process_image(img, cfg))

    # Ground truth: the same neutrally-graded sRGB pixels matched directly in Oklab
    neutral = adjust_colors(
        img_arr,
        hue_shift=0.0,
        saturation=1.0,
        brightness=1.0,
        contrast=1.0,
        midtones=0.0,
        highlights=0.0,
        luminance_threshold=0.5,
        invert_luminance=False,
    )
    palette = load_palette_preset("ps1_classic_16")
    expected = match_palette_nearest(neutral, palette, metric="oklab")

    assert np.array_equal(result, expected)


def test_seamless_pipeline_keeps_palette_and_edges_equal():
    rng = np.random.default_rng(46)
    image = Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))
    cfg = RetextureConfig(
        size=[40, 40],
        palette_preset="ps1_classic_16",
        seamless_tiling=True,
        seam_method="offset_wrap",
        grime_decay=0.15,
        grain=0.05,
        grain_seed=9,
        dither_algorithm="bayer4x4",
    )
    output = np.array(process_image(image, cfg))
    assert np.array_equal(output[:, 0], output[:, -1])
    assert np.array_equal(output[0, :], output[-1, :])


def test_builtin_presets_dir_outside_repo_root(tmp_path, monkeypatch):
    """Ensure get_builtin_presets_dir successfully finds presets even when CWD is outside repo root."""
    from retexture.config import get_builtin_presets_dir, RetextureConfig

    monkeypatch.chdir(tmp_path)
    presets_dir = get_builtin_presets_dir()
    assert presets_dir.is_dir()
    assert (presets_dir / "ps1_classic.json").is_file()

    # Verify loading a preset works from outside repo root
    cfg = RetextureConfig.load("ps1_classic")
    assert cfg.palette_preset == "ps1_classic_16"
