import numpy as np
import pytest
from retexture.core.color_adjust import adjust_colors, apply_oklab_style


def test_adjust_colors_midtones():
    # Gradient from 0 to 255
    arr = np.linspace(0, 255, 256, dtype=np.uint8)
    img = np.stack([arr, arr, arr], axis=-1).reshape((16, 16, 3))
    
    # Mid-gray is around index 128
    mid_idx = 128
    original_val = arr[mid_idx]

    # Lift midtones
    lifted = adjust_colors(img, midtones=0.5)
    lifted_mid = lifted.reshape(-1, 3)[mid_idx, 0]
    assert lifted_mid > original_val
    # 0 and 255 should remain roughly pinned
    assert lifted.reshape(-1, 3)[0, 0] == 0
    assert lifted.reshape(-1, 3)[255, 0] == 255

    # Lower midtones
    lowered = adjust_colors(img, midtones=-0.5)
    lowered_mid = lowered.reshape(-1, 3)[mid_idx, 0]
    assert lowered_mid < original_val


def test_adjust_colors_highlights():
    arr = np.linspace(0, 255, 256, dtype=np.uint8)
    img = np.stack([arr, arr, arr], axis=-1).reshape((16, 16, 3))

    # Highlights (e.g. 200) should be affected more than shadows (e.g. 50)
    boosted = adjust_colors(img, highlights=0.4)
    flat_orig = img.reshape(-1, 3)[:, 0]
    flat_boosted = boosted.reshape(-1, 3)[:, 0]

    high_diff = int(flat_boosted[220]) - int(flat_orig[220])
    shadow_diff = int(flat_boosted[30]) - int(flat_orig[30])
    assert high_diff >= shadow_diff


def test_adjust_colors_invert():
    img = np.array([[[0, 100, 255]]], dtype=np.uint8)
    inverted = adjust_colors(img, invert_luminance=True)
    assert np.allclose(inverted[0, 0], [255, 155, 0], atol=1)


def test_adjust_colors_luminance_threshold():
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    # Threshold < 0.5 pushes values brighter, > 0.5 pushes darker
    bright_bias = adjust_colors(arr, luminance_threshold=0.3)
    dark_bias = adjust_colors(arr, luminance_threshold=0.7)
    assert bright_bias[0, 0, 0] > dark_bias[0, 0, 0]


def test_oklab_style_preserves_shape_and_changes_split_tone():
    img = np.array([[[30, 40, 50], [210, 190, 170]]], dtype=np.uint8)
    styled = apply_oklab_style(
        img,
        black_point=0.04,
        white_point=0.96,
        s_curve=0.4,
        shadow_hue_shift=-18.0,
        highlight_hue_shift=14.0,
        shadow_saturation=1.15,
        highlight_saturation=1.08,
    )
    assert styled.shape == img.shape
    assert styled.dtype == np.uint8
    assert not np.array_equal(styled, img)
