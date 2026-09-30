"""Configuration management and preset serialization for retexture."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Strict allowlist for user-supplied preset names / ids.
# Mirrors the palette-import sanitizer: lowercase slug, no path separators.
_PRESET_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_AUTOSAVE_FILENAME_RE = re.compile(r"^autosave_[A-Za-z0-9_-]{1,64}\.json$")
_RESERVED_PRESET_IDS = {"latest_saved"}


def sanitize_preset_id(raw: str | None) -> str:
    """Normalize a user-supplied preset name to a safe filesystem slug.

    Raises ValueError on empty / reserved / invalid input so callers can
    return HTTP 400 instead of writing outside ``custom_presets/``.
    """
    if raw is None:
        raise ValueError("Preset name is required")
    text = raw.strip()
    if not text:
        raise ValueError("Preset name is required")
    # Reject path tricks explicitly instead of silently collapsing them into
    # a colliding slug (e.g. '../../evil' -> 'evil').
    if "\x00" in text or "/" in text or "\\" in text or ".." in text:
        raise ValueError(f"Invalid preset name '{raw}': path separators are not allowed")
    if Path(text).is_absolute():
        raise ValueError(f"Invalid preset name '{raw}'")
    slug = text.lower().replace(" ", "_").replace("-", "_")
    # Collapse runs like the palette importer does, then validate strictly.
    slug = re.sub(r"[^a-z0-9_-]+", "_", slug).strip("._-")
    if not slug:
        raise ValueError("Preset name must contain letters or numbers")
    if slug in _RESERVED_PRESET_IDS:
        raise ValueError(f"Preset name '{raw}' is reserved")
    if not _PRESET_ID_RE.fullmatch(slug):
        raise ValueError(f"Invalid preset name '{raw}'")
    if slug.startswith("autosave_"):
        raise ValueError("Preset name may not use the 'autosave_' prefix")
    return slug


def get_custom_presets_dir() -> Path:
    """Get or create custom_presets directory."""
    d = Path.cwd() / "custom_presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_autosaves_dir() -> Path:
    """Get or create custom_presets/autosaves directory."""
    d = get_custom_presets_dir() / "autosaves"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_builtin_presets_dir() -> Path:
    """Get builtin configs directory."""
    cwd_presets = Path.cwd() / "presets" / "configs"
    if cwd_presets.is_dir():
        return cwd_presets

    pkg_root = Path(__file__).resolve().parent
    if (pkg_root / "presets" / "configs").is_dir():
        return pkg_root / "presets" / "configs"

    repo_root = Path(__file__).resolve().parent.parent.parent / "presets" / "configs"
    if repo_root.is_dir():
        return repo_root

    return Path.cwd() / "presets" / "configs"


@dataclass
class RetextureConfig:
    """Complete configuration for retro texture transformation pipeline."""

    name: str = "Custom Preset"
    description: str = ""
    base_preset: str = ""
    size: list[int] = field(default_factory=lambda: [128, 128])
    resize_filter: str = "nearest"
    export_format: Literal["png", "jpg", "jpeg", "webp", "bmp", "tga"] = "png"
    jpg_quality: int = 90

    # Palette & Quantization
    palette_preset: str | None = "ps1_classic_16"
    color_metric: Literal["oklab", "cielab", "luma", "euclidean"] = "oklab"

    # Dithering
    dither_algorithm: str = "bayer4x4"
    dither_strength: float = 0.8
    dither_scale: int = 1
    linear_color_space: bool = True
    alpha_stipple: bool = True
    one_bit_bias: float = 0.0
    one_bit_contrast: float = 1.0

    # Film Grain & Noise
    grain: float = 0.06
    grain_seed: int | None = 42
    grain_monochrome: bool = True

    # Color & Tonal Adjustments
    hue: float = 0.0
    saturation: float = 1.15
    brightness: float = 1.05
    contrast: float = 1.12
    color_temperature: float = 0.0
    midtones: float = 0.0
    highlights: float = 0.0
    luminance_threshold: float = 0.5
    invert_luminance: bool = False
    black_point: float = 0.0
    white_point: float = 1.0
    s_curve: float = 0.0
    shadow_hue_shift: float = 0.0
    highlight_hue_shift: float = 0.0
    shadow_saturation: float = 1.0
    highlight_saturation: float = 1.0

    # Surface Prep & Shader-Style Retro FX
    pre_blur: float = 0.0
    chromatic_aberration: float = 0.0
    crt_scanlines: float = 0.0
    grime_decay: float = 0.0
    edge_crunch: float = 0.0
    outline_strength: float = 0.0
    edge_noise_gate: float = 0.5
    kuwahara_filter: float = 0.0
    bilateral_simplify: float = 0.0
    pixeloe_strength: float = 0.0
    cel_shading_steps: int = 0
    grime_mode: Literal["uniform", "streak", "edge_wear"] = "uniform"
    bake_optical_fx: bool = False

    # Seamless Tiling
    seamless_tiling: bool = False
    seam_size: float = 0.2
    seam_method: Literal["offset_wrap", "mirror"] = "offset_wrap"
    # Kept as a read-compatible field for older custom presets. The new
    # implementation uses seam_method; it is intentionally not emitted by
    # newly-created configs.
    seamless_wrap: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # seamless_wrap was an abandoned boolean switch. Read it for
        # backwards compatibility, but keep it out of new preset files so the
        # real seam choice is always explicit.
        data.pop("seamless_wrap", None)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetextureConfig:
        """Instantiate RetextureConfig from a dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        if "size" in filtered_data:
            s = filtered_data["size"]
            if isinstance(s, int):
                filtered_data["size"] = [s, s]
            elif isinstance(s, (list, tuple)) and len(s) >= 2:
                filtered_data["size"] = [int(s[0]), int(s[1])]

        if "export_format" in filtered_data and isinstance(filtered_data["export_format"], str):
            fmt = filtered_data["export_format"].lower().replace(".", "")
            if fmt in ("jpg", "jpeg"):
                filtered_data["export_format"] = "jpg"
            elif fmt in ("png", "webp", "bmp", "tga"):
                filtered_data["export_format"] = fmt
            else:
                filtered_data["export_format"] = "png"

        # Palette modes were removed from the product surface. Older preset
        # files may still contain those fields, so ignore them and normalize
        # missing/null palette selections to the current fixed-palette default.
        if not filtered_data.get("palette_preset"):
            filtered_data["palette_preset"] = "ps1_classic_16"

        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_str: str) -> RetextureConfig:
        """Parse configuration from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, file_path: str | Path) -> RetextureConfig:
        """Load configuration from file or preset name."""
        path = Path(file_path)

        # 1. Direct file path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_json(f.read())

        # 2. Check custom_presets/autosaves/
        autosave_file = get_autosaves_dir() / f"{path.name}"
        if not autosave_file.suffix:
            autosave_file = get_autosaves_dir() / f"{path.stem}.json"
        if autosave_file.exists():
            with open(autosave_file, "r", encoding="utf-8") as f:
                return cls.from_json(f.read())

        # 3. Check custom_presets/
        custom_file = get_custom_presets_dir() / f"{path.stem}.json"
        if custom_file.exists():
            with open(custom_file, "r", encoding="utf-8") as f:
                return cls.from_json(f.read())

        # 4. Check presets/configs/
        builtin_file = get_builtin_presets_dir() / f"{path.stem}.json"
        if builtin_file.exists():
            with open(builtin_file, "r", encoding="utf-8") as f:
                return cls.from_json(f.read())

        raise FileNotFoundError(f"Configuration not found: {file_path}")

    def save(self, file_path: str | Path) -> Path:
        """Save configuration to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    def autosave(self, create_snapshot: bool = False, max_history: int = 100) -> Path:
        """Auto-save current working configuration.

        If create_snapshot is True, also saves a timestamped snapshot into
        custom_presets/autosaves/ (capped at max_history files via FIFO).
        """
        target = get_custom_presets_dir() / "latest_saved.json"
        self.save(target)

        if create_snapshot:
            autosaves_dir = get_autosaves_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            snapshot_path = autosaves_dir / f"autosave_{ts}.json"
            self.save(snapshot_path)

            # Enforce 100 limit FIFO
            existing = sorted(autosaves_dir.glob("autosave_*.json"), key=lambda p: p.stat().st_mtime)
            if len(existing) > max_history:
                for old_file in existing[:-max_history]:
                    try:
                        old_file.unlink()
                    except Exception:
                        pass

        return target

    def save_as_custom(self, name: str | None = None) -> tuple[Path, str]:
        """Save as named custom preset in custom_presets/{name}.json."""
        if not name or not name.strip():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = (self.name or "preset").lower().replace(" ", "_")
            name = f"{base}_{ts}"

        clean_name = sanitize_preset_id(name)
        self.name = name.strip()
        custom_dir = get_custom_presets_dir().resolve()
        target = (custom_dir / f"{clean_name}.json").resolve()
        # Defense in depth: sanitized slug must stay inside custom_presets/.
        if target.parent != custom_dir:
            raise ValueError(f"Invalid preset name '{name}'")
        self.save(target)
        return (target, clean_name)


def delete_custom_preset(preset_id: str) -> bool:
    """Delete a custom preset from custom_presets/{preset_id}.json."""
    try:
        clean_id = sanitize_preset_id(preset_id.replace(".json", ""))
    except ValueError:
        return False
    custom_dir = get_custom_presets_dir().resolve()
    target = (custom_dir / f"{clean_id}.json").resolve()
    if target.parent != custom_dir:
        return False
    if target.name == "latest_saved.json":
        return False
    if target.exists() and target.is_file():
        target.unlink()
        return True
    return False


def delete_autosave(filename: str) -> bool:
    """Delete a specific autosave file from custom_presets/autosaves/."""
    if not filename or "/" in filename or "\\" in filename or "\x00" in filename:
        return False
    if ".." in filename:
        return False
    if not _AUTOSAVE_FILENAME_RE.fullmatch(filename):
        return False
    autosaves_dir = get_autosaves_dir().resolve()
    target = (autosaves_dir / filename).resolve()
    if target.parent != autosaves_dir:
        return False
    if target.exists() and target.is_file():
        target.unlink()
        return True
    return False


def clear_all_autosaves() -> int:
    """Clear all history autosaves except the single latest one in custom_presets/autosaves/."""
    autosaves_dir = get_autosaves_dir()
    files = sorted(autosaves_dir.glob("autosave_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    count = 0
    # Keep the latest one (files[0]), remove all older ones
    for f in files[1:]:
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def list_config_presets() -> dict[str, Any]:
    """Scan built-in configs, custom presets, and autosaves history."""
    builtin_dir = get_builtin_presets_dir()
    custom_dir = get_custom_presets_dir()
    autosaves_dir = get_autosaves_dir()

    builtin_list = []
    if builtin_dir.exists():
        for fp in sorted(builtin_dir.glob("*.json")):
            try:
                cfg = RetextureConfig.load(fp)
                builtin_list.append({
                    "id": fp.stem,
                    "name": cfg.name or fp.stem,
                    "is_custom": False,
                    "config": cfg.to_dict(),
                })
            except Exception:
                continue

    custom_list = []
    latest_saved = None
    if custom_dir.exists():
        for fp in sorted(custom_dir.glob("*.json")):
            if fp.name == "latest_saved.json":
                try:
                    cfg = RetextureConfig.load(fp)
                    latest_saved = {
                        "id": "latest_saved",
                        "name": "● Working Config (Auto-saved)",
                        "is_custom": True,
                        "config": cfg.to_dict(),
                    }
                except Exception:
                    pass
                continue

            try:
                cfg = RetextureConfig.load(fp)
                custom_list.append({
                    "id": fp.stem,
                    "name": cfg.name or fp.stem,
                    "is_custom": True,
                    "created_time": fp.stat().st_mtime,
                    "config": cfg.to_dict(),
                })
            except Exception:
                continue

    autosaves_list = []
    if autosaves_dir.exists():
        # Newest first
        files = sorted(autosaves_dir.glob("autosave_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for fp in files[:100]:
            try:
                cfg = RetextureConfig.load(fp)
                mtime = fp.stat().st_mtime
                dt = datetime.fromtimestamp(mtime).strftime("%H:%M:%S (%b %d)")
                based_on = cfg.base_preset or cfg.name or "Custom"
                autosaves_list.append({
                    "id": fp.stem,
                    "filename": fp.name,
                    "timestamp_label": dt,
                    "based_on": based_on,
                    "palette": cfg.palette_preset or "unknown",
                    "size": f"{cfg.size[0]}×{cfg.size[1]}",
                    "config": cfg.to_dict(),
                })
            except Exception:
                continue

    return {
        "builtin": builtin_list,
        "custom": custom_list,
        "latest_saved": latest_saved,
        "autosaves": autosaves_list,
        "autosaves_count": len(autosaves_list),
        "limit": 100,
    }
