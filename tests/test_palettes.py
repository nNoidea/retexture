import numpy as np
import pytest
from PIL import Image

from retexture.config import RetextureConfig
from retexture.core.palettes import (
    list_palette_presets,
    load_palette_preset,
    parse_lospec_png_palette,
)


def test_parse_lospec_1x_preserves_order_and_deduplicates():
    image = Image.new("RGB", (4, 1))
    image.putdata([(10, 20, 30), (40, 50, 60), (10, 20, 30), (70, 80, 90)])

    palette = parse_lospec_png_palette(image)

    assert palette.tolist() == [[10, 20, 30], [40, 50, 60], [70, 80, 90]]


def test_parse_lospec_32x_strip():
    colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    image_array = np.zeros((32, len(colors) * 32, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        image_array[:, index * 32:(index + 1) * 32, :] = color
    image = Image.fromarray(image_array, mode="RGB")

    palette = parse_lospec_png_palette(image)

    assert palette.tolist() == [list(color) for color in colors]


def test_parse_lospec_rejects_non_uniform_32x_swatch():
    image = Image.new("RGB", (64, 32), color=(10, 20, 30))
    image.putpixel((0, 0), (255, 255, 255))

    with pytest.raises(ValueError, match="solid-color"):
        parse_lospec_png_palette(image)


def test_parse_lospec_rejects_arbitrary_image():
    with pytest.raises(ValueError, match="Expected a Lospec"):
        parse_lospec_png_palette(Image.new("RGB", (16, 16), color=(10, 20, 30)))


def test_custom_png_palette_is_discovered_and_loaded(tmp_path, monkeypatch):
    import retexture.core.palettes as palettes_module

    presets_dir = tmp_path / "presets"
    (presets_dir / "palettes").mkdir(parents=True)
    custom_dir = tmp_path / "custom_palettes"
    custom_dir.mkdir(parents=True)
    monkeypatch.setattr(palettes_module, "get_presets_dir", lambda: presets_dir)
    monkeypatch.setattr(palettes_module, "get_custom_palettes_dir", lambda: custom_dir)

    image = Image.new("RGB", (2, 1))
    image.putdata([(1, 2, 3), (200, 201, 202)])
    image.save(custom_dir / "test_palette.png", format="PNG")

    image_hyphen = Image.new("RGB", (2, 1))
    image_hyphen.putdata([(10, 20, 30), (40, 50, 60)])
    image_hyphen.save(custom_dir / "custodian-8-32x.png", format="PNG")

    entries = list_palette_presets()
    entry = next(item for item in entries if item["id"] == "custom/test_palette")

    assert entry["is_custom"] is True
    assert entry["color_count"] == 2
    assert load_palette_preset("custom/test_palette").tolist() == [[1, 2, 3], [200, 201, 202]]
    # Hyphenated ID matching check
    assert load_palette_preset("custom/custodian-8-32x").tolist() == [[10, 20, 30], [40, 50, 60]]
    assert load_palette_preset("custodian-8-32x").tolist() == [[10, 20, 30], [40, 50, 60]]
    assert load_palette_preset("custodian_8_32x").tolist() == [[10, 20, 30], [40, 50, 60]]


def test_legacy_palette_mode_fields_are_ignored():
    config = RetextureConfig.from_dict({
        "palette_mode": "ps1_rgb555",
        "palette_preset": None,
        "max_colors": 16,
        "custom_palette_colors": None,
    })

    assert config.palette_preset == "ps1_classic_16"
    serialized = config.to_dict()
    assert "palette_mode" not in serialized
    assert "max_colors" not in serialized
    assert "custom_palette_colors" not in serialized


def test_palette_cache_and_clear():
    from retexture.core.palettes import _LOADED_PALETTE_CACHE, clear_palette_cache

    clear_palette_cache()
    assert len(_LOADED_PALETTE_CACHE) == 0

    p1 = load_palette_preset("ps1_classic_16")
    assert len(_LOADED_PALETTE_CACHE) > 0
    assert "ps1_classic_16" in _LOADED_PALETTE_CACHE

    # Verify mutating the returned array does not corrupt the cached array
    orig_val = int(p1[0, 0])
    p1[0, 0] = (orig_val + 10) % 256
    p2 = load_palette_preset("ps1_classic_16")
    assert p2[0, 0] == orig_val

    clear_palette_cache()
    assert len(_LOADED_PALETTE_CACHE) == 0
