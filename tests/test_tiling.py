import numpy as np
from retexture.core.tiling import make_seamless


def test_make_seamless_continuity():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    seamless = make_seamless(img, seam_size=0.2)
    assert seamless.shape == (64, 64, 3)
    assert seamless.dtype == np.uint8


def test_offset_wrap_converges_at_all_borders():
    img = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    seamless = make_seamless(img, seam_size=0.2, method="offset_wrap")
    assert np.array_equal(seamless[:, 0], seamless[:, -1])
    assert np.array_equal(seamless[0, :], seamless[-1, :])
