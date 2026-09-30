import numpy as np
from retexture.core.dither import (
    dither_1bit_weighted,
    dither_palette_atkinson,
    dither_palette_bayer,
    dither_palette_floyd_steinberg,
    dither_palette_riemersma,
    dither_palette_yliluoma,
)


def test_dither_palette_bayer_strict_colors():
    palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
    ], dtype=np.uint8)
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    res = dither_palette_bayer(img, palette, matrix_type="bayer4x4", strength=1.0)
    for color in res.reshape(-1, 3):
        assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255])


def test_dither_palette_floyd_steinberg_strict_colors():
    palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
    ], dtype=np.uint8)
    img = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=-1)
    res = dither_palette_floyd_steinberg(img, palette, strength=1.0)
    for color in res.reshape(-1, 3):
        assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255])


def test_dither_palette_atkinson_strict_colors():
    palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
    ], dtype=np.uint8)
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    res = dither_palette_atkinson(img, palette, strength=1.0)
    for color in res.reshape(-1, 3):
        assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255])


def test_dither_palette_yliluoma_strict_colors():
    palette = np.array([
        [0, 0, 0],
        [128, 128, 128],
        [255, 255, 255],
    ], dtype=np.uint8)
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    res = dither_palette_yliluoma(img, palette, strength=0.9, metric="oklab")
    assert res.shape == (16, 16, 3)
    for color in res.reshape(-1, 3):
        assert any(np.array_equal(color, p) for p in palette)


def test_weighted_1bit_is_binary_and_tonal():
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    ramp = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8, 1)
    img = np.repeat(ramp, 3, axis=-1)
    res = dither_1bit_weighted(img, palette, contrast=1.2)
    assert res.shape == img.shape
    assert set(map(tuple, np.unique(res.reshape(-1, 3), axis=0))).issubset({(0, 0, 0), (255, 255, 255)})
    assert np.mean(np.all(res[:2] == [0, 0, 0], axis=-1)) > np.mean(np.all(res[-2:] == [0, 0, 0], axis=-1))


def test_dither_palette_riemersma_strict_colors():
    palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
    ], dtype=np.uint8)
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    res = dither_palette_riemersma(img, palette, strength=1.0, metric="oklab")
    assert res.shape == (16, 16, 3)
    for color in res.reshape(-1, 3):
        assert any(np.array_equal(color, p) for p in palette)


def test_dither_palette_interlaced_and_crosshatch():
    palette = np.array([
        [0, 0, 0],
        [128, 128, 128],
        [255, 255, 255],
    ], dtype=np.uint8)
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    res_inter = dither_palette_bayer(img, palette, matrix_type="interlaced", strength=0.9)
    res_hatch = dither_palette_bayer(img, palette, matrix_type="crosshatch", strength=0.9)
    assert res_inter.shape == (16, 16, 3)
    assert res_hatch.shape == (16, 16, 3)


def test_dither_palette_blue_noise():
    from retexture.core.dither import dither_palette_blue_noise
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    res = dither_palette_blue_noise(img, palette, strength=1.0)
    assert res.shape == (64, 64, 3)
    # Output must be binary palette colors
    for color in res.reshape(-1, 3):
        assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255])


def test_dither_scale_bayer_and_blue_noise():
    from retexture.core.dither import dither_palette_bayer, dither_palette_blue_noise
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    
    res_scale2 = dither_palette_bayer(img, palette, matrix_type="bayer4x4", dither_scale=2)
    assert res_scale2.shape == (32, 32, 3)
    # In 2x scale, adjacent 2x2 blocks have identical dither decisions
    assert np.array_equal(res_scale2[0, 0], res_scale2[0, 1])
    assert np.array_equal(res_scale2[0, 0], res_scale2[1, 0])
    assert np.array_equal(res_scale2[0, 0], res_scale2[1, 1])

    res_bn_scale2 = dither_palette_blue_noise(img, palette, dither_scale=2)
    assert res_bn_scale2.shape == (32, 32, 3)
    assert np.array_equal(res_bn_scale2[0, 0], res_bn_scale2[0, 1])


def test_linear_color_space_transforms():
    from retexture.core.dither import srgb_to_linear, linear_to_srgb
    # 0 -> 0, 1 -> 1
    assert np.isclose(srgb_to_linear(0.0), 0.0, atol=1e-4)
    assert np.isclose(srgb_to_linear(1.0), 1.0, atol=1e-4)
    # Mid-gray 0.5 sRGB is roughly ~0.214 in linear space
    assert np.isclose(srgb_to_linear(0.5), 0.214, atol=0.01)
    # Round-trip preserves values
    test_vals = np.linspace(0.0, 1.0, 50)
    round_trip = linear_to_srgb(srgb_to_linear(test_vals))
    assert np.allclose(test_vals, round_trip, atol=1e-4)


def test_error_diffusion_kernel_weights_sum_to_one():
    from retexture.core.dither import ERROR_DIFFUSION_KERNELS
    for name, offsets in ERROR_DIFFUSION_KERNELS.items():
        total = sum(w for _, _, w in offsets)
        if name == "atkinson":
            # Atkinson deliberately diffuses only 6/8 of the error for contrast
            assert abs(total - 0.75) < 1e-6
        else:
            assert abs(total - 1.0) < 1e-6, f"{name} weights sum to {total}"


def test_new_error_diffusion_kernels_strict_colors():
    from retexture.core.dither import dither_palette_kernel
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    img = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=-1)
    for kernel in (
        "stucki", "burkes", "sierra", "sierra_two_row", "sierra_lite",
        "jarvis_judice_ninke", "false_floyd_steinberg",
        "shiau_fan_1", "shiau_fan_2", "shiau_fan_3",
    ):
        res = dither_palette_kernel(img, palette, kernel=kernel, strength=1.0)
        assert res.shape == (16, 16, 3), kernel
        for color in res.reshape(-1, 3):
            assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255]), kernel


def test_error_diffusion_kernel_determinism():
    from retexture.core.dither import dither_palette_kernel
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    rng = np.random.default_rng(11)
    img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    for kernel in ("stucki", "sierra", "jarvis_judice_ninke", "shiau_fan_2"):
        a = dither_palette_kernel(img, palette, kernel=kernel, strength=0.9)
        b = dither_palette_kernel(img, palette, kernel=kernel, strength=0.9)
        assert np.array_equal(a, b), kernel


def test_kernel_alias_resolution():
    from retexture.core.dither import resolve_error_diffusion_kernel
    assert resolve_error_diffusion_kernel("jjn") == "jarvis_judice_ninke"
    assert resolve_error_diffusion_kernel("sierra3") == "sierra"
    assert resolve_error_diffusion_kernel("fake_floyd") == "false_floyd_steinberg"
    assert resolve_error_diffusion_kernel("shiau_fan") == "shiau_fan_1"
    assert resolve_error_diffusion_kernel("floyd_steinberg") == "floyd_steinberg"
    assert resolve_error_diffusion_kernel("bayer4x4") is None


def test_clustered_dot_and_line_halftone():
    from retexture.core.dither import (
        CLUSTER_DOT_4X4,
        CLUSTER_DOT_8X8,
        LINE_HALFTONE_4X4,
        dither_palette_bayer,
    )
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    img = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=-1)

    assert CLUSTER_DOT_4X4.shape == (4, 4)
    assert CLUSTER_DOT_8X8.shape == (8, 8)
    assert LINE_HALFTONE_4X4.shape == (4, 4)
    # Every matrix cell must be unique (proper threshold matrix)
    assert len(set(CLUSTER_DOT_4X4.flatten().tolist())) == 16
    assert len(set(CLUSTER_DOT_8X8.flatten().tolist())) == 64

    for mtype in ("cluster_dot_4x4", "cluster_dot_8x8", "line_halftone"):
        res = dither_palette_bayer(img, palette, matrix_type=mtype, strength=1.0)
        assert res.shape == (16, 16, 3), mtype
        for color in res.reshape(-1, 3):
            assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255]), mtype


def test_clustered_dot_grows_from_center():
    from retexture.core.dither import CLUSTER_DOT_4X4
    # Lowest threshold (darkest first) must be nearest the dot center (1.5, 1.5)
    idx = np.unravel_index(int(np.argmin(CLUSTER_DOT_4X4)), CLUSTER_DOT_4X4.shape)
    dist = min(
        ((idx[0] - 1) ** 2 + (idx[1] - 1) ** 2),
        ((idx[0] - 2) ** 2 + (idx[1] - 1) ** 2),
        ((idx[0] - 1) ** 2 + (idx[1] - 2) ** 2),
        ((idx[0] - 2) ** 2 + (idx[1] - 2) ** 2),
    )
    assert dist <= 1.0


def test_dot_diffusion_strict_colors_and_determinism():
    from retexture.core.dither import dither_palette_dot_diffusion
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    img = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=-1)
    res = dither_palette_dot_diffusion(img, palette, strength=1.0)
    assert res.shape == (16, 16, 3)
    for color in res.reshape(-1, 3):
        assert np.array_equal(color, [0, 0, 0]) or np.array_equal(color, [255, 255, 255])
    res2 = dither_palette_dot_diffusion(img, palette, strength=1.0)
    assert np.array_equal(res, res2)


def test_pipeline_new_algorithms_end_to_end():
    from PIL import Image
    from retexture.config import RetextureConfig
    from retexture.core.pipeline import process_image

    rng = np.random.default_rng(21)
    arr = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    algos = (
        "stucki", "burkes", "sierra", "sierra_two_row", "sierra_lite",
        "jarvis_judice_ninke", "false_floyd_steinberg",
        "shiau_fan_1", "shiau_fan_2", "shiau_fan_3",
        "cluster_dot_4x4", "cluster_dot_8x8", "line_halftone", "dot_diffusion",
    )
    for algo in algos:
        cfg = RetextureConfig(size=[0, 0], palette_preset="ps1_classic_16", dither_algorithm=algo, grain=0.0)
        res = process_image(img, cfg)
        assert res.size == (48, 32), algo
        assert res.mode == "RGB", algo


def test_error_diffusion_honors_color_metric():
    """Audit point 2: nearest-color matching inside the error-diffusion loops must
    follow the configured metric (Oklab), not raw RGB Euclidean."""
    from retexture.core.dither import (
        dither_palette_dot_diffusion,
        dither_palette_kernel,
        dither_palette_riemersma,
    )
    from retexture.core.quantize import match_palette_nearest

    rng = np.random.default_rng(5)
    palette = rng.integers(0, 256, (8, 3), dtype=np.uint8)

    # Search for a pixel whose nearest palette entry differs between the two metrics
    discriminating = None
    for _ in range(500):
        color = rng.integers(0, 256, 3, dtype=np.uint8)
        img = np.tile(color, (4, 4, 1))
        ok_choice = match_palette_nearest(img, palette, metric="oklab")[0, 0]
        euc_choice = match_palette_nearest(img, palette, metric="euclidean")[0, 0]
        if not np.array_equal(ok_choice, euc_choice):
            discriminating = (img, ok_choice, euc_choice)
            break

    assert discriminating is not None, "no color discriminates oklab from euclidean"
    img, ok_choice, euc_choice = discriminating

    for name, dither_fn in (
        (
            "floyd_steinberg",
            lambda im: dither_palette_kernel(im, palette, kernel="floyd_steinberg", strength=1.0, metric="oklab"),
        ),
        (
            "riemersma",
            lambda im: dither_palette_riemersma(im, palette, strength=1.0, metric="oklab"),
        ),
        (
            "dot_diffusion",
            lambda im: dither_palette_dot_diffusion(im, palette, strength=1.0, metric="oklab"),
        ),
    ):
        res = dither_fn(img)
        # Pixel (0, 0) receives no diffused error, so it is a pure nearest match
        assert np.array_equal(res[0, 0], ok_choice), name
        for px in res.reshape(-1, 3):
            assert any(np.array_equal(px, p) for p in palette), name

    # The metric must actually flow through: euclidean picks the euclidean entry
    res_euc = dither_palette_kernel(img, palette, kernel="floyd_steinberg", strength=1.0, metric="euclidean")
    assert np.array_equal(res_euc[0, 0], euc_choice)


def test_error_diffusion_high_strength_no_overflow():
    """Verify that strength > 1.0 (e.g. 1.5 from the GUI slider) does not cause
    exponential divergence or RuntimeWarning overflow in error-diffusion loops."""
    import warnings
    from retexture.core.dither import (
        ERROR_DIFFUSION_KERNELS,
        dither_palette_dot_diffusion,
        dither_palette_kernel,
        dither_palette_riemersma,
    )

    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Test Riemersma with strength=1.5
        res_r = dither_palette_riemersma(img, palette, strength=1.5, metric="oklab")
        assert res_r.shape == img.shape

        # Test Dot Diffusion with strength=1.5
        res_dd = dither_palette_dot_diffusion(img, palette, strength=1.5, metric="oklab")
        assert res_dd.shape == img.shape

        # Test all kernel error diffusion algorithms with strength=1.5
        for kernel_name in ERROR_DIFFUSION_KERNELS:
            res_k = dither_palette_kernel(img, palette, kernel=kernel_name, strength=1.5, metric="oklab")
            assert res_k.shape == img.shape

