import numpy as np
import retexture.core.quantize as quantize_module
from retexture.core.quantize import (
    match_palette_nearest,
    palette_to_metric_space,
    srgb_to_cielab,
    srgb_to_oklab,
    to_metric_space,
)


def test_match_palette_nearest_strict_containment():
    palette = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)

    rng = np.random.default_rng(42)
    random_img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    for metric in ("oklab", "cielab", "luma", "euclidean"):
        mapped = match_palette_nearest(random_img, palette, metric=metric)
        flat_mapped = mapped.reshape(-1, 3)
        for color in flat_mapped:
            assert any(np.array_equal(color, p) for p in palette)


def test_srgb_to_oklab_and_cielab_transforms():
    rgb = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
    ], dtype=np.uint8)

    oklab = srgb_to_oklab(rgb)
    assert oklab.shape == (2, 3, 3)
    # White has L close to 1.0, black has L close to 0.0
    assert oklab[1, 0, 0] > 0.95
    assert oklab[1, 1, 0] < 0.05

    cielab = srgb_to_cielab(rgb)
    assert cielab.shape == (2, 3, 3)
    # White has L* close to 100, black close to 0
    assert cielab[1, 0, 0] > 95.0
    assert cielab[1, 1, 0] < 5.0


def test_to_metric_space_matches_metric_distances():
    """to_metric_space must embed each metric so plain Euclidean distance in the
    output space equals that metric's distance in sRGB."""
    palette = np.array([
        [0, 0, 0],
        [90, 90, 90],
        [255, 128, 0],
        [10, 200, 30],
        [200, 20, 200],
    ], dtype=np.uint8)
    unit = palette.astype(np.float32) / 255.0

    assert np.allclose(to_metric_space(palette, "oklab"), srgb_to_oklab(unit))
    assert np.allclose(to_metric_space(palette, "cielab"), srgb_to_cielab(unit))
    assert np.allclose(to_metric_space(palette, "euclidean"), unit)

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = to_metric_space(palette, "luma")
    assert np.allclose(luma, unit * np.sqrt(weights))

    # Plain Euclidean distance inside luma space == Rec. 709 luma-weighted RGB distance
    a, b = unit[1], unit[3]
    d_space = float(np.linalg.norm(luma[1] - luma[3]))
    d_luma = float(np.sqrt(np.sum(weights * (a - b) ** 2)))
    assert np.isclose(d_space, d_luma)


def test_palette_metric_space_cache_consistency():
    """palette_to_metric_space must agree with direct conversion and be content-keyed."""
    from retexture.core.quantize import _palette_to_metric_space_cached

    palette = np.array([[10, 200, 30], [200, 20, 200], [90, 90, 90]], dtype=np.uint8)

    for metric in ("oklab", "cielab", "luma", "euclidean"):
        direct = to_metric_space(palette, metric)
        cached = palette_to_metric_space(palette, metric)
        assert np.allclose(cached, direct), metric

    info_before = _palette_to_metric_space_cached.cache_info()
    palette_to_metric_space(palette, "oklab")          # already converted above -> hit
    palette_to_metric_space(palette.copy(), "oklab")   # equal content, new object -> hit
    palette_to_metric_space(palette[:], "oklab")       # same content via view -> hit
    palette_to_metric_space(palette, "euclidean")      # already converted above -> hit
    info_after = _palette_to_metric_space_cached.cache_info()

    assert info_after.hits == info_before.hits + 4
    assert info_after.misses == info_before.misses


def test_palette_oklab_converted_once_across_calls(monkeypatch):
    """The pairwise Yliluoma pass converts the image once and reuses palette space."""
    from retexture.core.dither import dither_palette_yliluoma

    conversion_dims: list[int] = []
    original_srgb_to_oklab = quantize_module.srgb_to_oklab

    def recording_srgb_to_oklab(arr):
        conversion_dims.append(int(np.asarray(arr).ndim))
        return original_srgb_to_oklab(arr)

    monkeypatch.setattr(quantize_module, "srgb_to_oklab", recording_srgb_to_oklab)

    # Distinctive palette content so no other test's cache entry interferes
    palette = np.array([[13, 244, 9], [77, 7, 250], [251, 97, 5]], dtype=np.uint8)
    image = np.full((8, 8, 3), 77, dtype=np.uint8)

    match_palette_nearest(image, palette, metric="oklab")
    match_palette_nearest(image, palette, metric="oklab")
    dither_palette_yliluoma(image, palette, strength=0.9, metric="oklab")

    # ndim==3 calls are whole-image conversions (one per public pass);
    # ndim==2 calls are palette conversions, which must happen exactly once.
    image_conversions = [d for d in conversion_dims if d == 3]
    palette_conversions = [d for d in conversion_dims if d == 2]

    assert len(image_conversions) == 3  # 2 direct matches + 1 pairwise Yliluoma pass
    assert len(palette_conversions) == 1


def test_match_palette_nearest_chunking_large_image():
    """Verify that match_palette_nearest processes large images (>65,536 pixels) accurately across chunks."""
    rng = np.random.default_rng(42)
    # 300x300 = 90,000 pixels, exceeding the 65,536-pixel chunk size
    large_image = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [128, 128, 128],
        [64, 32, 16],
    ], dtype=np.uint8)

    matched = match_palette_nearest(large_image, palette, metric="oklab")
    assert matched.shape == (300, 300, 3)

    # Every output pixel must be an exact palette color
    flat = matched.reshape(-1, 3)
    diffs = flat[:, None, :] - palette[None, :, :]
    exact_match = np.any(np.all(diffs == 0, axis=-1), axis=-1)
    assert np.all(exact_match)
