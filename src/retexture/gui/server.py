"""FastAPI local studio server for retexture.

Provides REST endpoints for live preview generation, auto-save presets,
custom preset management, and in-GUI mass batch conversion.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from retexture.config import RetextureConfig, list_config_presets
from retexture.core.palettes import (
    clear_palette_cache,
    get_custom_palettes_dir,
    list_palette_presets,
    load_palette_preset,
    parse_lospec_png_palette,
    rgb_to_hex,
)
from retexture.core.pipeline import export_image_bytes, process_image

import secrets

# ---------------------------------------------------------------------------
# Resource limits (DoS hardening).
# Textures rarely exceed 4K (16.7 MP); 32 MP headroom still blocks
# decompression bombs while keeping legitimate 8K work usable.
# ---------------------------------------------------------------------------
MAX_IMAGE_PIXELS = 33_554_432  # ~32 MP
MAX_IMAGE_DIM = 8192  # max width/height in px
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # decoded image bytes per request
MAX_BASE64_CHARS = 28 * 1024 * 1024  # ~20 MB decoded
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # per uploaded file
MAX_BATCH_FILES = 100
MAX_BROWSE_ENTRIES = 5000
MAX_FILENAME_LEN = 128

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="Retexture", version="0.2.0")

SESSION_TOKEN: str | None = os.environ.get("RETEXTURE_SESSION_TOKEN")

# Whitelist of public paths that do not require cryptographic token authentication
PUBLIC_STATIC_PATHS = {"/", "/favicon.ico", "/api/sample-image"}


def _security_headers(path: str) -> dict[str, str]:
    """Headers applied to every response (including early 401/403)."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-origin",
        # Local single-origin app: no CDNs, no inline scripts.
        # 'unsafe-inline' kept for style-src only (dynamic JS styles + one
        # <strong style=> in preset cards); scripts stay strict 'self'.
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        ),
    }
    if path.startswith("/api/"):
        headers["Cache-Control"] = "no-store"
    return headers


def is_public_path(path: str) -> bool:
    """Check if the requested path is a public static asset."""
    if path in PUBLIC_STATIC_PATHS:
        return True
    if path.startswith("/static/"):
        return True
    return False


def get_session_token() -> str | None:
    """Get active ephemeral session token."""
    return SESSION_TOKEN


def set_session_token(token: str | None) -> None:
    """Set active ephemeral session token."""
    global SESSION_TOKEN
    SESSION_TOKEN = token


# ============================================================================
# Global Security Gateway Middleware Layer
# Every incoming request (current and future) MUST pass through this security gate
# before any route or endpoint handler can execute.
# ============================================================================
@app.middleware("http")
async def security_gateway_middleware(request: Request, call_next):
    path = request.url.path
    # Layer 0: Cross-Site Request Blocker
    # Drop any cross-site request originating from external websites in another browser tab
    sec_fetch_site = request.headers.get("sec-fetch-site")
    if sec_fetch_site == "cross-site":
        return Response(
            content="Access Denied: Cross-site requests from external websites are strictly blocked.",
            status_code=403,
            media_type="text/plain",
            headers=_security_headers(path),
        )

    # Layer 1: Network Interface Firewall
    # Immediately drop any connection not originating strictly from the local loopback interface (127.0.0.1)
    client_ip = request.client.host if request.client else None
    if client_ip not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return Response(
            content="Access Denied: Retexture Studio only accepts connections from the local machine (127.0.0.1).",
            status_code=403,
            media_type="text/plain",
            headers=_security_headers(path),
        )

    # Layer 2: Default-Deny Cryptographic Session Authentication
    # All non-static endpoints (both existing and any future endpoints) are locked by default.
    # NOTE: the session token is accepted via header (X-Retexture-Token,
    # optionally `Bearer`) or HttpOnly cookie ONLY. Query-string tokens were
    # removed to avoid leakage in logs/history/caches. <img> thumbnails rely
    # on the cookie set by GET / (browsers send it automatically).
    if not is_public_path(path) and SESSION_TOKEN:
        token_header = request.headers.get("x-retexture-token") or request.headers.get("authorization")
        if token_header and token_header.lower().startswith("bearer "):
            token_header = token_header[7:].strip()
        token_cookie = request.cookies.get("retexture_token")
        req_token = token_header or token_cookie

        if not req_token or not secrets.compare_digest(req_token, SESSION_TOKEN):
            return Response(
                content='{"detail": "Unauthorized: Invalid or missing session token."}',
                status_code=401,
                media_type="application/json",
                headers=_security_headers(path),
            )

    response = await call_next(request)
    for k, v in _security_headers(path).items():
        response.headers.setdefault(k, v)
    return response


# Security: Restrict CORS origins exclusively to local loopback ports (prevent cross-origin website snooping)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Shared upload / filesystem validation helpers
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp"}
ALLOWED_EXPORT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga"}

# Sensitive locations never served, listed, or written by the explorer.
# (Explorer is a local-user file picker, not a sandbox escape hatch.)
_SENSITIVE_DIR_NAMES = {".ssh", ".gnupg", ".aws", ".pki", ".gnome-keyring"}
_SENSITIVE_ROOTS = ("proc", "sys", "dev", "run/secrets", "etc/ssh", "etc/ssl/private")


def _is_blocked_path(p: Path) -> bool:
    """True if a resolved path points at credentials / kernel pseudo-FS."""
    try:
        parts = [c.lower() for c in p.parts]
    except Exception:
        return True
    if any(name in _SENSITIVE_DIR_NAMES for name in parts):
        return True
    posix = p.as_posix().lower()
    for root in _SENSITIVE_ROOTS:
        if posix == f"/{root}" or posix.startswith(f"/{root}/"):
            return True
    return False


def _check_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Invalid image dimensions")
    if max(width, height) > MAX_IMAGE_DIM:
        raise HTTPException(
            status_code=413,
            detail=f"Image dimensions {width}x{height} exceed {MAX_IMAGE_DIM}px limit",
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image ({width}x{height}) exceeds {MAX_IMAGE_PIXELS} pixel limit",
        )


def _decode_image_bytes(raw_b64: str, *, field: str = "image") -> bytes:
    if len(raw_b64) > MAX_BASE64_CHARS:
        raise HTTPException(status_code=413, detail=f"{field} exceeds size limit")
    try:
        data = base64.b64decode(raw_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid base64 {field}")
    if not data:
        raise HTTPException(status_code=400, detail=f"{field} is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field} exceeds size limit")
    return data


def _open_validated_image(data: bytes) -> Image.Image:
    """Open bytes with Pillow, enforcing pixel/dimension caps. Returns copy."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.draft(probe.mode, probe.size)  # cheap header probe
            w, h = probe.size
            _check_image_dimensions(w, h)
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode inside try (catches bomb/truncation)
        _check_image_dimensions(img.width, img.height)
        return img
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail=f"Image too large: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def _safe_export_filename(filename: str | None) -> str:
    """Strict basename allowlist for single-file exports (no traversal)."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename is required")
    raw = filename.strip()
    if len(raw) > MAX_FILENAME_LEN:
        raise HTTPException(status_code=400, detail="Filename is too long")
    if "\x00" in raw or "/" in raw or "\\" in raw:
        raise HTTPException(status_code=400, detail="Filename must be a plain basename without path separators")
    if Path(raw).is_absolute():
        raise HTTPException(status_code=400, detail="Absolute paths are not allowed")
    name = Path(raw).name
    if name in ("", ".", "..") or ".." in Path(name).parts:
        raise HTTPException(status_code=400, detail="Invalid filename")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXPORT_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported export extension '{suffix}'")
    stem = Path(name).stem
    if not stem.strip():
        raise HTTPException(status_code=400, detail="Filename must contain letters or numbers")
    clean_stem = re.sub(r"[^A-Za-z0-9 _-]+", "_", stem).strip("._- ")
    if not clean_stem:
        raise HTTPException(status_code=400, detail="Filename must contain letters or numbers")
    return f"{clean_stem}{suffix}"


def _resolve_output_inside(out_dir: Path, filename: str) -> Path:
    """Join + resolve, then prove containment (defense in depth)."""
    out_dir = out_dir.resolve()
    target = (out_dir / filename).resolve()
    try:
        target.relative_to(out_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Resolved path escapes output folder") from None
    return target


class ProcessJsonRequest(BaseModel):
    image_base64: str
    config: dict[str, Any]


class AutosaveRequest(BaseModel):
    config: dict[str, Any]
    create_snapshot: bool = False


class SavePresetRequest(BaseModel):
    name: str
    config: dict[str, Any]


class BatchConvertRequest(BaseModel):
    input_paths: list[str] | None = None
    input_folder: str | None = None
    output_folder: str
    config: dict[str, Any]


@app.get("/")
async def root():
    """Serve index.html and set session cookie if active."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        response = FileResponse(index_file)
        if SESSION_TOKEN:
            # HttpOnly: JS uses sessionStorage for fetch headers; <img>
            # thumbnails authenticate via this cookie automatically, so no
            # token-in-URL is needed anywhere.
            response.set_cookie(
                "retexture_token",
                SESSION_TOKEN,
                httponly=True,
                samesite="lax",
            )
        return response
    return {"status": "Retexture API active"}


@app.get("/api/presets")
async def get_presets():
    """Return all palette presets, built-in configs, custom presets, and working config."""
    return {
        "palettes": list_palette_presets(),
        "configs": list_config_presets(),
    }


def _safe_custom_palette_filename(filename: str | None) -> str:
    """Normalize an uploaded palette filename to a safe PNG basename."""
    raw_name = Path(filename or "").name
    if Path(raw_name).suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Only PNG palette files are supported")

    stem = Path(raw_name).stem.lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem).strip("._-")
    if not stem:
        raise HTTPException(status_code=400, detail="Palette filename must contain letters or numbers")
    return f"{stem}.png"


@app.post("/api/palettes/import")
async def import_palette_endpoint(
    palette_file: UploadFile = File(...),
    overwrite: bool = Form(False),
):
    """Import and persist a Lospec 1x or 32x PNG palette."""
    try:
        filename = _safe_custom_palette_filename(palette_file.filename)
        contents = await palette_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Palette file is empty")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Palette file is too large")

        try:
            with Image.open(io.BytesIO(contents)) as image:
                if image.format != "PNG":
                    raise ValueError("File is not a PNG image")
                palette = parse_lospec_png_palette(image)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid Lospec palette PNG: {exc}") from exc

        custom_dir = get_custom_palettes_dir()
        custom_dir.mkdir(parents=True, exist_ok=True)
        target = custom_dir / filename
        if target.exists() and not overwrite:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": f"A custom palette named '{filename}' already exists",
                    "filename": filename,
                    "palette_id": f"custom/{Path(filename).stem}",
                },
            )

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{Path(filename).stem}.",
                suffix=".tmp",
                dir=custom_dir,
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(contents)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, target)
            temp_path = None
            clear_palette_cache()
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink()
                except FileNotFoundError:
                    pass

        palette_id = f"custom/{Path(filename).stem}"
        palette_metadata = {
            "id": palette_id,
            "name": Path(filename).stem,
            "description": "Imported Lospec PNG palette",
            "author": "",
            "style_tags": ["custom"],
            "color_count": len(palette),
            "colors": [rgb_to_hex(color) for color in palette],
            "path": str(target),
            "filename": filename,
            "is_custom": True,
        }
        return {
            "status": "ok",
            "palette": palette_metadata,
            "palettes": list_palette_presets(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/presets/autosave")
async def autosave_preset(req: AutosaveRequest):
    """Auto-save the current working configuration into custom_presets/latest_saved.json."""
    try:
        cfg = RetextureConfig.from_dict(req.config)
        cfg.autosave(create_snapshot=req.create_snapshot, max_history=100)
        return {
            "status": "ok",
            "saved_to": "custom_presets/latest_saved.json",
            "snapshot_created": req.create_snapshot,
            "configs": list_config_presets(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/presets/save")
async def save_custom_preset_endpoint(req: SavePresetRequest):
    """Save a named custom preset into custom_presets/{name}.json."""
    try:
        cfg = RetextureConfig.from_dict(req.config)
        path, clean_name = cfg.save_as_custom(req.name)
        return {
            "status": "ok",
            "name": clean_name,
            "path": str(path),
            "configs": list_config_presets(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/presets/custom/{preset_id}")
async def delete_custom_preset_endpoint(preset_id: str):
    """Delete a user-saved custom preset."""
    from retexture.config import delete_custom_preset
    ok = delete_custom_preset(preset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"status": "ok", "configs": list_config_presets()}


@app.delete("/api/presets/autosaves/{filename}")
async def delete_autosave_endpoint(filename: str):
    """Delete an autosave file."""
    from retexture.config import delete_autosave
    ok = delete_autosave(filename)
    if not ok:
        raise HTTPException(status_code=404, detail="Autosave not found")
    return {"status": "ok", "configs": list_config_presets()}


@app.post("/api/presets/autosaves/clear")
async def clear_autosaves_endpoint():
    """Clear all history autosaves."""
    from retexture.config import clear_all_autosaves
    count = clear_all_autosaves()
    return {"status": "ok", "cleared_count": count, "configs": list_config_presets()}


@app.post("/api/process")
def process_texture_endpoint(request: ProcessJsonRequest):
    """Process an image using the Single Source of Truth Python pipeline in threadpool."""
    try:
        header_split = request.image_base64.split(",")
        raw_b64 = header_split[-1]
        img_bytes = _decode_image_bytes(raw_b64)
        input_img = _open_validated_image(img_bytes)

        cfg = RetextureConfig.from_dict(request.config)
        processed_img = process_image(input_img, cfg)

        fmt = cfg.export_format.lower()
        out_bytes = export_image_bytes(processed_img, fmt=fmt, quality=cfg.jpg_quality)
        if fmt in ("jpg", "jpeg"):
            media_type = "image/jpeg"
        elif fmt == "webp":
            media_type = "image/webp"
        elif fmt == "bmp":
            media_type = "image/bmp"
        else:
            media_type = "image/png"

        return Response(content=out_bytes, media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





class DirectBatchConvertRequest(BaseModel):
    file_paths: list[str]
    output_folder: str
    config: dict[str, Any]
    filename_pattern: str | None = None


class CheckConflictsRequest(BaseModel):
    file_paths: list[str]
    output_folder: str
    config: dict[str, Any]
    filename_pattern: str | None = None


class ExportSingleRequest(BaseModel):
    filename: str
    output_folder: str
    image_base64: str


def resolve_output_filename(
    original_stem: str,
    pattern: str | None,
    cfg: RetextureConfig,
    ext: str,
) -> str:
    """Format output filename using customizable tokens."""
    if not pattern or not pattern.strip():
        pattern = "{name}"

    preset_name = cfg.name or cfg.base_preset or "custom"
    clean_preset = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in preset_name.lower())
    size_str = f"{cfg.size[0]}x{cfg.size[1]}" if cfg.size and cfg.size[0] > 0 else "original"
    try:
        palette_count = len(load_palette_preset(cfg.palette_preset or "ps1_classic_16"))
        colors_str = f"{palette_count}c"
    except Exception:
        colors_str = "palette"
    dither_str = cfg.dither_algorithm

    resolved = pattern
    resolved = resolved.replace("{name}", original_stem)
    resolved = resolved.replace("{preset}", clean_preset)
    resolved = resolved.replace("{stylename}", clean_preset)
    resolved = resolved.replace("{style}", clean_preset)
    resolved = resolved.replace("{size}", size_str)
    resolved = resolved.replace("{resolution}", size_str)
    resolved = resolved.replace("{pixel}", size_str)
    resolved = resolved.replace("{colors}", colors_str)
    resolved = resolved.replace("{dither}", dither_str)

    clean_filename = "".join(c if c.isalnum() or c in ("-", "_", ".", " ") else "_" for c in resolved).strip()
    if not clean_filename:
        # original_stem is already a Path.stem (no separators); still strip.
        clean_filename = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in original_stem) or "texture"
    # Cap length (keep extension) so patterns cannot create absurd names.
    if len(clean_filename) > MAX_FILENAME_LEN:
        clean_filename = clean_filename[:MAX_FILENAME_LEN].rstrip("._- ")
        if not clean_filename:
            clean_filename = "texture"
    if not clean_filename.endswith(f".{ext}"):
        clean_filename = f"{clean_filename}.{ext}"
    return clean_filename


@app.get("/api/sample-image")
async def get_default_sample():
    """Serve default startup sample image."""
    sample_path = Path.cwd() / "presets" / "samples" / "default_sample.png"
    if sample_path.exists():
        return FileResponse(sample_path, media_type="image/png")
    img = Image.new("RGB", (256, 256), color=(75, 80, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/explorer/quick-locations")
def get_quick_locations():
    """Return common quick-jump locations on the system."""
    home = Path.home()
    cwd = Path.cwd()
    locations = [
        {"name": "Current Workspace", "path": str(cwd)},
    ]
    downloads = home / "Downloads"
    if downloads.exists():
        locations.append({"name": "Downloads", "path": str(downloads)})
    pictures = home / "Pictures"
    if pictures.exists():
        locations.append({"name": "Pictures", "path": str(pictures)})

    return {"locations": locations}


@app.get("/api/explorer/browse")
async def explore_directory(path: str | None = None, all_files: bool = False):
    """Browse a local directory, listing subfolders and either image files or all files."""
    try:
        if path and path.strip():
            base_dir = Path(path.strip()).expanduser().resolve()
            if not base_dir.exists() or not base_dir.is_dir():
                raise HTTPException(status_code=404, detail=f"Directory not found: '{path}'")
        else:
            base_dir = Path.cwd().resolve()

        if _is_blocked_path(base_dir):
            raise HTTPException(status_code=403, detail="Access to this system location is forbidden")

        items_dirs = []
        items_files = []
        parent_dir = str(base_dir.parent) if base_dir.parent != base_dir else None

        try:
            entries = sorted(base_dir.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        if len(entries) > MAX_BROWSE_ENTRIES:
            raise HTTPException(
                status_code=400,
                detail=f"Directory lists more than {MAX_BROWSE_ENTRIES} entries",
            )

        for item in entries:
            if item.name.startswith("."):
                continue
            try:
                if item.is_symlink():
                    # Resolve links but never follow them outside the tree or
                    # into blocked locations (e.g. ~/.ssh).
                    target = item.resolve()
                    if _is_blocked_path(target):
                        continue
                if item.is_dir():
                    items_dirs.append({
                        "name": item.name,
                        "path": str(item),
                        "is_dir": True,
                    })
                elif item.is_file():
                    if all_files:
                        try:
                            size_kb = round(item.stat().st_size / 1024, 1)
                        except OSError:
                            continue
                        items_files.append({
                            "name": item.name,
                            "path": str(item),
                            "size_kb": size_kb,
                            "dimensions": "",
                            "is_dir": False,
                        })
                    elif item.suffix.lower() in IMAGE_EXTS:
                        try:
                            st_size = item.stat().st_size
                        except OSError:
                            continue
                        if st_size > MAX_UPLOAD_BYTES:
                            dim_str = "too large"
                        else:
                            dim_str = ""
                            try:
                                with Image.open(item) as im:
                                    _check_image_dimensions(im.width, im.height)
                                    dim_str = f"{im.width}×{im.height}"
                            except HTTPException:
                                dim_str = "too large"
                            except Exception:
                                pass
                        size_kb = round(st_size / 1024, 1)
                        items_files.append({
                            "name": item.name,
                            "path": str(item),
                            "size_kb": size_kb,
                            "dimensions": dim_str,
                            "is_dir": False,
                        })
            except OSError:
                continue

        return {
            "current_path": str(base_dir),
            "parent_path": parent_dir,
            "directories": items_dirs,
            "files": items_files,
            "total_images": len(items_files),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CreateFolderRequest(BaseModel):
    folder_path: str


@app.post("/api/explorer/create-folder")
async def create_folder(req: CreateFolderRequest):
    """Create a new folder safely (only final folder if parent exists)."""
    try:
        raw = req.folder_path.strip()
        if not raw or len(raw) > 1024 or "\x00" in raw:
            raise HTTPException(status_code=400, detail="Invalid folder path")
        p = Path(raw).expanduser().resolve()
        if _is_blocked_path(p) or _is_blocked_path(p.parent):
            raise HTTPException(status_code=403, detail="Creating folders in this system location is forbidden")
        if p.exists():
            return {"status": "ok", "path": str(p), "created": False}
        if not p.parent.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Parent folder '{p.parent}' does not exist. Cannot create multi-level directory tree."
            )
        p.mkdir(parents=False, exist_ok=True)
        return {"status": "ok", "path": str(p), "created": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/explorer/file")
async def get_explorer_file(path: str):
    """Serve a local image file for viewport loading (strictly restricted to image files)."""
    if not path or len(path) > 4096 or "\x00" in path:
        raise HTTPException(status_code=400, detail="Invalid path")
    p = Path(path).expanduser().resolve()
    if _is_blocked_path(p):
        raise HTTPException(status_code=403, detail="Access to this system location is forbidden")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if p.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=403, detail="Forbidden: file is not a supported image texture")
    try:
        if p.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image file exceeds size limit")
        # Verify actual image magic bytes (not just extension) so a symlinked
        # /etc/shadow renamed to .png cannot be exfiltrated.
        with Image.open(p) as im:
            _check_image_dimensions(im.width, im.height)
            im.load()
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail=f"Image too large: {exc}") from exc
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden: file is not a valid image texture")
    media_type = "image/png"
    if p.suffix.lower() in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif p.suffix.lower() == ".webp":
        media_type = "image/webp"
    return FileResponse(p, media_type=media_type, headers={"Cache-Control": "no-store"})


@app.post("/api/explorer/batch-convert")
async def direct_batch_convert(req: DirectBatchConvertRequest):
    """Convert selected files and write directly to the target output directory on disk."""
    try:
        if not req.file_paths or len(req.file_paths) > MAX_BATCH_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Provide 1..{MAX_BATCH_FILES} input files",
            )
        out_dir = Path(req.output_folder).expanduser().resolve()
        if _is_blocked_path(out_dir):
            raise HTTPException(status_code=403, detail="Writing to this system location is forbidden")
        if not out_dir.exists():
            parent_dir = out_dir.parent
            if not parent_dir.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Parent folder '{parent_dir}' does not exist on disk. Retexture will only create the final folder '{out_dir.name}'."
                )
            # Create ONLY the latest folder
            out_dir.mkdir(parents=False, exist_ok=True)
        if not out_dir.is_dir():
            raise HTTPException(status_code=400, detail="Output folder is not a directory")

        cfg = RetextureConfig.from_dict(req.config)
        ext = cfg.export_format.lower()

        results = []
        for fp_str in req.file_paths:
            if not fp_str or len(fp_str) > 4096 or "\x00" in fp_str:
                continue
            fp = Path(fp_str).expanduser().resolve()
            if _is_blocked_path(fp):
                results.append({"filename": Path(fp_str).name, "error": "Blocked system location", "status": "failed"})
                continue
            if not fp.exists() or not fp.is_file():
                continue
            if fp.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                if fp.stat().st_size > MAX_UPLOAD_BYTES:
                    raise ValueError("Input file exceeds size limit")
                with Image.open(fp) as probe:
                    _check_image_dimensions(probe.width, probe.height)
                input_img = Image.open(fp)
                input_img.load()
                processed = process_image(input_img, cfg)
                out_name = resolve_output_filename(fp.stem, req.filename_pattern, cfg, ext)
                out_file = _resolve_output_inside(out_dir, out_name)
                out_bytes = export_image_bytes(processed, fmt=ext, quality=cfg.jpg_quality)
                with open(out_file, "wb") as f:
                    f.write(out_bytes)
                results.append({
                    "filename": fp.name,
                    "output_name": out_name,
                    "output_path": str(out_file),
                    "status": "success",
                })
            except HTTPException as ex:
                results.append({
                    "filename": fp.name,
                    "error": str(ex.detail),
                    "status": "failed",
                })
            except Exception as ex:
                results.append({
                    "filename": fp.name,
                    "error": str(ex),
                    "status": "failed",
                })

        return {
            "status": "ok",
            "count": len(results),
            "output_folder": str(out_dir),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explorer/check-conflicts")
async def check_conflicts_endpoint(req: CheckConflictsRequest):
    """Check if any resolved output filenames already exist in the target output directory."""
    try:
        if req.file_paths and len(req.file_paths) > MAX_BATCH_FILES:
            raise HTTPException(status_code=400, detail=f"At most {MAX_BATCH_FILES} files per request")
        out_dir = Path(req.output_folder).expanduser().resolve()
        if _is_blocked_path(out_dir):
            raise HTTPException(status_code=403, detail="Access to this system location is forbidden")
        cfg = RetextureConfig.from_dict(req.config)
        ext = cfg.export_format.lower()

        conflicts: list[str] = []
        if out_dir.exists() and out_dir.is_dir():
            for fp_str in req.file_paths:
                stem = Path(fp_str).stem[:128]
                out_name = resolve_output_filename(stem, req.filename_pattern, cfg, ext)
                if (out_dir / out_name).exists():
                    conflicts.append(out_name)

        return {
            "status": "ok",
            "total": len(req.file_paths),
            "conflicts_count": len(conflicts),
            "conflicting_files": conflicts,
            "output_folder": str(out_dir),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explorer/export-single")
async def export_single_endpoint(req: ExportSingleRequest):
    """Save a single exported texture to disk in output_folder."""
    try:
        safe_name = _safe_export_filename(req.filename)
        out_dir = Path(req.output_folder).expanduser().resolve()
        if _is_blocked_path(out_dir):
            raise HTTPException(status_code=403, detail="Writing to this system location is forbidden")
        if not out_dir.exists():
            parent_dir = out_dir.parent
            if not parent_dir.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Parent folder '{parent_dir}' does not exist. Retexture will only create the final folder.",
                )
            out_dir.mkdir(parents=False, exist_ok=True)
        if not out_dir.is_dir():
            raise HTTPException(status_code=400, detail="Output folder is not a directory")
        out_file = _resolve_output_inside(out_dir, safe_name)
        raw_b64 = req.image_base64.split(",")[-1]
        img_bytes = _decode_image_bytes(raw_b64)
        # Prove the payload is a real image within pixel caps before writing
        # anything to disk (prevents arbitrary-byte writes).
        _open_validated_image(img_bytes)
        with open(out_file, "wb") as f:
            f.write(img_bytes)
        return {"status": "ok", "saved_path": str(out_file), "filename": safe_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-convert")
async def batch_convert_endpoint(
    images: List[UploadFile] = File(...),
    config_json: str = Form(...),
):
    """Mass convert multiple uploaded files and return results summary."""
    try:
        if len(images) > MAX_BATCH_FILES:
            raise HTTPException(status_code=400, detail=f"At most {MAX_BATCH_FILES} files per request")
        if len(config_json) > 1_000_000:
            raise HTTPException(status_code=413, detail="Config payload too large")
        cfg_dict = json.loads(config_json)
        cfg = RetextureConfig.from_dict(cfg_dict)
        results = []

        for upload in images:
            contents = await upload.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File '{upload.filename}' exceeds size limit")
            input_img = _open_validated_image(contents)
            processed_img = process_image(input_img, cfg)

            fmt = cfg.export_format.lower()
            out_bytes = export_image_bytes(processed_img, fmt=fmt, quality=cfg.jpg_quality)
            b64_out = base64.b64encode(out_bytes).decode("utf-8")
            media_type = "image/jpeg" if fmt in ("jpg", "jpeg") else "image/png"

            stem = Path(upload.filename or "texture").stem[:64]
            clean_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("._-") or "texture"
            filename = f"{clean_stem}.{fmt}"

            results.append({
                "filename": filename,
                "data_url": f"data:{media_type};base64,{b64_out}",
                "size": [processed_img.width, processed_img.height],
            })

        return {
            "status": "ok",
            "count": len(results),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
