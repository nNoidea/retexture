"""Core retro texture processing engine."""

from retexture.core.color_adjust import adjust_colors, apply_oklab_style
from retexture.core.dither import (
    dither_1bit_weighted,
    dither_palette_atkinson,
    dither_palette_bayer,
    dither_palette_floyd_steinberg,
)
from retexture.core.grain import apply_grain
from retexture.core.palettes import (
    list_palette_presets,
    load_palette_preset,
    parse_lospec_png_palette,
)
from retexture.core.pipeline import export_image_bytes, process_image
from retexture.core.quantize import (
    match_palette_nearest,
)
from retexture.core.shader_fx import (
    apply_chromatic_aberration,
    apply_crt_scanlines,
    apply_edge_crunch,
    apply_grime_decay,
)
from retexture.core.tiling import enforce_tile_edges, make_seamless

__all__ = [
    "adjust_colors",
    "apply_oklab_style",
    "apply_grain",
    "make_seamless",
    "enforce_tile_edges",
    "apply_chromatic_aberration",
    "apply_crt_scanlines",
    "apply_grime_decay",
    "apply_edge_crunch",
    "load_palette_preset",
    "list_palette_presets",
    "parse_lospec_png_palette",
    "match_palette_nearest",
    "dither_palette_bayer",
    "dither_palette_floyd_steinberg",
    "dither_palette_atkinson",
    "dither_1bit_weighted",
    "process_image",
    "export_image_bytes",
]
