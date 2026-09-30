import numpy as np
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


def test_chromatic_aberration():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    img[:, :16, 0] = 255
    res = apply_chromatic_aberration(img, amount=0.5)
    assert res.shape == (32, 32, 3)
    assert res.dtype == np.uint8
    assert not np.array_equal(res, img)


def test_crt_scanlines():
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    res = apply_crt_scanlines(img, amount=0.5)
    assert res.shape == (32, 32, 3)
    # Odd rows should be darker than even rows
    assert np.all(res[1::2, :, 0] < res[0::2, :, 0])


def test_grime_decay():
    img = np.full((32, 32, 3), 180, dtype=np.uint8)
    res1 = apply_grime_decay(img, amount=0.5, seed=42)
    res2 = apply_grime_decay(img, amount=0.5, seed=42)
    assert res1.shape == (32, 32, 3)
    assert np.array_equal(res1, res2)


def test_edge_crunch():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 200
    res = apply_edge_crunch(img, amount=0.5)
    assert res.shape == (32, 32, 3)
    assert res.dtype == np.uint8


def test_outline_injection():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 255  # White box in black background
    res = apply_outline_injection(img, amount=0.8)
    assert res.shape == (32, 32, 3)
    assert res.dtype == np.uint8


def test_bilateral_simplify():
    rng = np.random.default_rng(42)
    noisy_img = rng.integers(100, 150, (32, 32, 3), dtype=np.uint8)
    res = apply_bilateral_simplify(noisy_img, amount=0.8)
    assert res.shape == (32, 32, 3)
    assert res.dtype == np.uint8
    # Noise variance should be reduced
    assert np.std(res.astype(np.float32)) < np.std(noisy_img.astype(np.float32))


def test_pixeloe_stylizer():
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (32, 32, 3), dtype=np.uint8)
    res = apply_pixeloe_stylizer(img, amount=0.7)
    assert res.shape == (32, 32, 3)
    assert res.dtype == np.uint8
    assert np.array_equal(res, apply_pixeloe_stylizer(img, amount=0.7))

    rgba = np.dstack([img, np.full(img.shape[:2], 173, dtype=np.uint8)])
    rgba_res = apply_pixeloe_stylizer(rgba, amount=0.7, periodic=True)
    assert np.array_equal(rgba_res[..., 3], rgba[..., 3])

    flat = np.full((16, 16, 3), (128, 82, 64), dtype=np.uint8)
    flat_res = apply_pixeloe_stylizer(flat, amount=1.0)
    assert np.max(np.abs(flat_res.astype(np.int16) - flat.astype(np.int16))) <= 1


def test_cel_shading_bands():
    img = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=-1)
    res = apply_cel_shading_bands(img, steps=4)
    assert res.shape == (16, 16, 3)
    assert res.dtype == np.uint8


def test_color_temperature_shift():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[0, 0] = [200, 200, 200]
    img[1, 1] = [50, 50, 50]
    res_warm = apply_color_temperature_shift(img, amount=0.8)
    assert res_warm[0, 0, 0] > 200
    assert res_warm[1, 1, 2] > 50


def test_kuwahara_filter():
    rng = np.random.default_rng(42)
    # Surface with high-frequency noise
    noisy = rng.integers(120, 140, (32, 32, 3), dtype=np.uint8)
    # Add a sharp door boundary: right half is dark
    noisy[:, 16:] = 30
    
    clean = apply_kuwahara_filter(noisy, amount=0.8)
    assert clean.shape == (32, 32, 3)
    assert clean.dtype == np.uint8
    # Noise on the left side should be reduced
    assert np.std(clean[:, :14].astype(np.float32)) < np.std(noisy[:, :14].astype(np.float32))
    # Boundary sharpness should be preserved (door step at x=15 to x=16)
    contrast = np.mean(clean[:, 14, 0].astype(np.float32)) - np.mean(clean[:, 17, 0].astype(np.float32))
    assert contrast > 70.0


def test_smart_outline_noise_gate():
    img = np.full((32, 32, 3), 160, dtype=np.uint8)
    # Isolated speckles on the wall
    img[5, 5] = 240
    img[22, 10] = 60
    # Connected door edge at column 16
    img[:, 16:] = 40

    # With noise gating enabled
    res = apply_outline_injection(img, amount=0.8, noise_gate=0.8)
    assert res.shape == (32, 32, 3)
    # Door line at column 15 should have dark contour
    assert res[10, 15, 0] < 120
    # Isolated speckle should NOT receive dark outline
    assert res[5, 5, 0] > 200

