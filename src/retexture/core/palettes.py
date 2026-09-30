"""Palette discovery, loading, and Lospec PNG importing helpers.

Built-in palettes are JSON files in ``presets/palettes/``. User-imported
Lospec palettes are kept as their original PNG files in ``custom_palettes/``
(repo root, gitignored) so the import remains lossless, fully local, and
never committed to the public repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


MAX_PALETTE_COLORS = 256


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert a hex color string like '#ff00aa' or 'ff00aa' to (R, G, B) tuple."""
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    if len(hex_clean) != 6:
        raise ValueError(f"Invalid hex color string: {hex_str}")
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    return (r, g, b)


def rgb_to_hex(rgb: tuple[int, int, int] | np.ndarray) -> str:
    """Convert RGB tuple or array to hex color string like '#ff00aa'."""
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def get_presets_dir() -> Path:
    """Locate the presets directory (checks working directory, package root, then repo root)."""
    cwd_presets = Path.cwd() / "presets" / "palettes"
    if cwd_presets.is_dir():
        return Path.cwd() / "presets"

    pkg_presets = Path(__file__).resolve().parent.parent / "presets"
    if (pkg_presets / "palettes").is_dir():
        return pkg_presets

    module_root = Path(__file__).resolve().parent.parent.parent.parent / "presets"
    if (module_root / "palettes").is_dir():
        return module_root

    return Path.cwd() / "presets"


def get_palettes_dir() -> Path:
    """Return the directory containing built-in and custom palettes."""
    return get_presets_dir() / "palettes"


def get_custom_palettes_dir() -> Path:
    """Return the writable, gitignored directory for imported palette PNGs.

    Mirrors ``get_custom_presets_dir``: ``<cwd>/custom_palettes``.
    Callers that write must ``mkdir(parents=True, exist_ok=True)``;
    readers treat a missing dir as empty.
    """
    return Path.cwd() / "custom_palettes"


def _palette_id_for_file(file_path: Path, palettes_dir: Path) -> str:
    """Build a stable dropdown/config ID for a palette file."""
    # PNG imports are always custom, regardless of location (new + legacy).
    if file_path.suffix.lower() == ".png":
        return f"custom/{file_path.stem}"
    try:
        resolved = file_path.resolve()
        if resolved.is_relative_to(get_custom_palettes_dir().resolve()):
            return f"custom/{file_path.stem}"
    except Exception:
        pass
    try:
        if file_path.resolve().is_relative_to((palettes_dir / "custom").resolve()):
            return f"custom/{file_path.stem}"
    except Exception:
        pass
    relative = file_path.relative_to(palettes_dir).with_suffix("")
    return "/".join(relative.parts)


def _is_custom_palette_file(file_path: Path, palettes_dir: Path) -> bool:
    """Return whether a palette file lives in the custom palette directory."""
    if file_path.suffix.lower() == ".png":
        return True
    try:
        if file_path.resolve().is_relative_to(get_custom_palettes_dir().resolve()):
            return True
    except Exception:
        pass
    try:
        return file_path.resolve().is_relative_to((palettes_dir / "custom").resolve())
    except Exception:
        return False


def _ordered_unique_colors(colors: np.ndarray) -> np.ndarray:
    """Remove duplicate RGB colors while preserving palette order."""
    ordered: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for color in np.asarray(colors, dtype=np.uint8).reshape(-1, 3):
        key = (int(color[0]), int(color[1]), int(color[2]))
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    if not ordered:
        return np.empty((0, 3), dtype=np.uint8)
    return np.asarray(ordered, dtype=np.uint8)


def parse_lospec_png_palette(source: Image.Image | str | Path) -> np.ndarray:
    """Parse a Lospec 1x or 32x PNG palette, preserving color order.

    Accepted layouts are a one-pixel strip or a strip of 32x32 solid-color
    swatches. Both horizontal and vertical strips are accepted for resilience,
    although Lospec's downloads are normally horizontal.
    """
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        image = Image.open(source)

    try:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    finally:
        if not isinstance(source, Image.Image):
            image.close()

    if np.any(rgba[..., 3] != 255):
        raise ValueError("Palette PNG must be fully opaque")

    rgb = rgba[..., :3]
    height, width, _ = rgb.shape
    colors: np.ndarray

    if height == 1:
        colors = rgb[0, :, :]
    elif width == 1:
        colors = rgb[:, 0, :]
    elif height == 32 and width % 32 == 0:
        block_count = width // 32
        blocks = rgb.reshape(32, block_count, 32, 3).transpose(1, 0, 2, 3)
        samples = blocks[:, 0, 0, :]
        if not np.all(blocks == samples[:, np.newaxis, np.newaxis, :]):
            raise ValueError("32x palette PNG must contain solid-color 32x32 swatches")
        colors = samples
    elif width == 32 and height % 32 == 0:
        block_count = height // 32
        blocks = rgb.reshape(block_count, 32, 32, 3)
        samples = blocks[:, 0, 0, :]
        if not np.all(blocks == samples[:, np.newaxis, np.newaxis, :]):
            raise ValueError("32x palette PNG must contain solid-color 32x32 swatches")
        colors = samples
    else:
        raise ValueError("Expected a Lospec 1x strip or 32x32 swatch strip")

    palette = _ordered_unique_colors(colors)
    if len(palette) < 2:
        raise ValueError("Palette PNG must contain at least two colors")
    if len(palette) > MAX_PALETTE_COLORS:
        raise ValueError(f"Palette PNG may contain at most {MAX_PALETTE_COLORS} colors")
    return palette


def list_palette_presets() -> list[dict[str, Any]]:
    """Scan built-in JSON and imported custom PNG palette files."""
    palettes_dir = get_palettes_dir()
    custom_dir = get_custom_palettes_dir()
    results = []

    if not palettes_dir.exists():
        palettes_dir = None

    palette_files: list[Path] = []
    if palettes_dir is not None:
        palette_files.extend(palettes_dir.glob("*.json"))
        # Legacy fallback (pre-public): presets/palettes/custom/*.png
        legacy_custom = palettes_dir / "custom"
        if legacy_custom.is_dir():
            palette_files.extend(legacy_custom.glob("*.png"))
    if custom_dir.is_dir():
        palette_files.extend(custom_dir.glob("*.png"))
    for file_path in sorted(set(palette_files), key=lambda p: str(p).lower()):
        try:
            # _helpers need a palettes_dir; fall back to parent for custom-only setups
            ref_dir = palettes_dir if palettes_dir is not None else custom_dir.parent / "presets" / "palettes"
            is_custom = file_path.suffix.lower() == ".png" or _is_custom_palette_file(file_path, ref_dir)
            preset_id = f"custom/{file_path.stem}" if is_custom else _palette_id_for_file(file_path, ref_dir)
            if file_path.suffix.lower() == ".png":
                palette = parse_lospec_png_palette(file_path)
                data = {
                    "name": file_path.stem,
                    "description": "Imported Lospec PNG palette",
                    "author": "",
                    "style_tags": ["custom"],
                    "colors": [rgb_to_hex(color) for color in palette],
                }
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            results.append({
                "id": preset_id,
                "name": data.get("name", preset_id),
                "description": data.get("description", ""),
                "author": data.get("author", ""),
                "style_tags": data.get("style_tags", []),
                "color_count": len(data.get("colors", [])),
                "colors": data.get("colors", []),
                "path": str(file_path),
                "filename": file_path.name,
                "is_custom": is_custom,
            })
        except Exception:
            continue

    return results


_LOADED_PALETTE_CACHE: dict[str, np.ndarray] = {}


def clear_palette_cache() -> None:
    """Clear the in-memory cache of loaded palette arrays."""
    _LOADED_PALETTE_CACHE.clear()


def load_palette_preset(preset_name: str) -> np.ndarray:
    """Load a palette preset by ID or name and return uint8 numpy array of shape (N, 3)."""
    raw_query = str(preset_name).strip().lower()
    clean_query = raw_query.replace("-", "_").replace(" ", "_")

    if clean_query in _LOADED_PALETTE_CACHE:
        return _LOADED_PALETTE_CACHE[clean_query].copy()

    clean_stem = clean_query.split("/")[-1]
    palettes_dir = get_palettes_dir()

    for palette in list_palette_presets():
        p_id_raw = str(palette["id"]).lower()
        p_id_clean = p_id_raw.replace("-", "_").replace(" ", "_")
        p_name_raw = str(palette["name"]).lower()
        p_name_clean = p_name_raw.replace("-", "_").replace(" ", "_")

        if (
            p_id_raw == raw_query
            or p_id_clean == clean_query
            or p_name_raw == raw_query
            or p_name_clean == clean_query
            or p_id_clean.split("/")[-1] == clean_stem
            or p_name_clean == clean_stem
        ):
            if palette["filename"].lower().endswith(".png"):
                arr = parse_lospec_png_palette(palette["path"])
            else:
                arr = np.array([hex_to_rgb(c) for c in palette["colors"]], dtype=np.uint8)

            _LOADED_PALETTE_CACHE[clean_query] = arr
            _LOADED_PALETTE_CACHE[p_id_clean] = arr
            _LOADED_PALETTE_CACHE[p_name_clean] = arr
            return arr.copy()

    raise FileNotFoundError(f"Palette preset '{preset_name}' not found in {palettes_dir}")
