"""Command-line interface for retexture."""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
from pathlib import Path
import click
from PIL import Image

from retexture.config import RetextureConfig, list_config_presets
from retexture.core.palettes import list_palette_presets
from retexture.core.pipeline import export_image_bytes, process_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp", ".tiff", ".tif"}


def _process_single_file_worker(args: tuple[str, str, dict]) -> tuple[str, bool, str]:
    input_path_str, output_path_str, config_dict = args
    try:
        cfg = RetextureConfig.from_dict(config_dict)
        with Image.open(input_path_str) as img:
            res = process_image(img, cfg)
            out_path = Path(output_path_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_bytes = export_image_bytes(res, fmt=cfg.export_format, quality=cfg.jpg_quality)
            out_path.write_bytes(out_bytes)

        return (input_path_str, True, "")
    except Exception as e:
        return (input_path_str, False, str(e))


@click.group()
@click.version_option(version="0.2.0", prog_name="retexture")
def main() -> None:
    """Retexture: High-speed retro texture processing pipeline and gamedev studio."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(), help="Output directory path.")
@click.option("-c", "--config", "config_file", type=click.Path(), help="Path to JSON configuration preset.")
@click.option("--preset", type=str, help="Name of config preset.")
@click.option("--palette", type=str, help="Palette preset name.")
@click.option("--size", type=str, help="Target texture size, e.g. '128' or '128x128'.")
@click.option("--dither", type=click.Choice(["bayer2x2", "bayer4x4", "bayer8x8", "blue_noise", "yliluoma", "interlaced", "crosshatch", "halftone_dot", "halftone_cross", "floyd_steinberg", "atkinson", "stucki", "burkes", "sierra", "sierra_two_row", "sierra_lite", "jarvis_judice_ninke", "false_floyd_steinberg", "shiau_fan_1", "shiau_fan_2", "shiau_fan_3", "cluster_dot_4x4", "cluster_dot_8x8", "line_halftone", "dot_diffusion", "dither_1bit_weighted", "none"], case_sensitive=False), help="Dithering algorithm.")
@click.option("--dither-strength", type=float, help="Dither strength multiplier.")
@click.option("--grain", type=float, help="Film grain intensity.")
@click.option("--seed", type=int, help="Random seed for grain/grime.")
@click.option("--grime-mode", type=click.Choice(["uniform", "streak", "edge_wear"], case_sensitive=False), help="Surface-mark pattern.")
@click.option("--format", "export_format", type=click.Choice(["png", "jpg", "jpeg", "webp", "bmp", "tga"], case_sensitive=False), help="Export file format.")
@click.option("-r", "--recursive", is_flag=True, default=True, help="Scan directories recursively.")
@click.option("-w", "--workers", type=int, default=None, help="Number of parallel worker processes.")
def batch(
    input_path: str,
    output_dir: str,
    config_file: str | None,
    preset: str | None,
    palette: str | None,
    size: str | None,
    dither: str | None,
    dither_strength: float | None,
    grain: float | None,
    seed: int | None,
    grime_mode: str | None,
    export_format: str | None,
    recursive: bool,
    workers: int | None,
) -> None:
    """Batch convert textures in parallel across all CPU cores."""
    if config_file:
        cfg = RetextureConfig.load(config_file)
    elif preset:
        cfg = RetextureConfig.load(preset)
    else:
        cfg = RetextureConfig()

    if palette:
        cfg.palette_preset = palette

    if size:
        if "x" in size.lower():
            w_str, h_str = size.lower().split("x", 1)
            cfg.size = [int(w_str), int(h_str)]
        else:
            dim = int(size)
            cfg.size = [dim, dim]

    if dither:
        cfg.dither_algorithm = dither.lower()  # type: ignore
    if dither_strength is not None:
        cfg.dither_strength = float(dither_strength)
    if grain is not None:
        cfg.grain = float(grain)
    if seed is not None:
        cfg.grain_seed = int(seed)
    if grime_mode:
        cfg.grime_mode = grime_mode.lower()  # type: ignore
    if export_format:
        fmt = export_format.lower()
        cfg.export_format = "jpg" if fmt in ("jpg", "jpeg") else fmt  # type: ignore

    in_p = Path(input_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    file_pairs: list[tuple[Path, Path]] = []

    if in_p.is_file():
        ext = f".{cfg.export_format}"
        out_file = out_p / f"{in_p.stem}{ext}" if out_p.is_dir() else out_p
        file_pairs.append((in_p, out_file))
    else:
        pattern = "**/*" if recursive else "*"
        for f in in_p.glob(pattern):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                rel_path = f.relative_to(in_p)
                ext = f".{cfg.export_format}"
                target_file = out_p / rel_path.with_suffix(ext)
                file_pairs.append((f, target_file))

    if not file_pairs:
        click.echo(click.style(f"No image files found in {input_path}", fg="yellow"))
        return

    num_workers = workers or max(1, os.cpu_count() or 1)
    click.echo(click.style(f"Retexture Batch Conversion", bold=True, fg="cyan"))
    click.echo(f"  Input:         {in_p}")
    click.echo(f"  Output:        {out_p}")
    click.echo(f"  Images Found:  {len(file_pairs)}")
    click.echo(f"  Target Size:   {cfg.size[0]}x{cfg.size[1]}")
    click.echo(f"  Preset:        {cfg.name or cfg.palette_preset}")
    click.echo(f"  Workers:       {num_workers} processes\n")

    tasks = [(str(inp), str(outp), cfg.to_dict()) for inp, outp in file_pairs]

    start_time = time.perf_counter()
    success_count = 0
    fail_count = 0

    def run_with(executor_type: type[concurrent.futures.Executor]) -> tuple[int, int]:
        succeeded = 0
        failed = 0
        with executor_type(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_single_file_worker, t) for t in tasks]
            with click.progressbar(concurrent.futures.as_completed(futures), length=len(futures), label="Processing") as bar:
                for fut in bar:
                    _, ok, err = fut.result()
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
        return succeeded, failed

    try:
        success_count, fail_count = run_with(concurrent.futures.ProcessPoolExecutor)
    except PermissionError:
        # Some sandboxed/containerized environments deny forkserver sockets.
        # Threads still parallelize Pillow/NumPy work and keep the CLI usable.
        click.echo(click.style("Process workers unavailable; using thread workers.", fg="yellow"))
        success_count, fail_count = run_with(concurrent.futures.ThreadPoolExecutor)

    elapsed = time.perf_counter() - start_time
    click.echo()
    if fail_count == 0:
        click.echo(click.style(f"✓ Successfully processed {success_count} textures in {elapsed:.2f}s!", fg="green", bold=True))
    else:
        click.echo(click.style(f"Completed: {success_count} succeeded, {fail_count} failed in {elapsed:.2f}s", fg="yellow", bold=True))


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", required=True, type=click.Path(), help="Output file path.")
@click.option("-c", "--config", "config_file", type=click.Path(), help="Path to JSON configuration preset.")
@click.option("--preset", type=str, help="Config preset name.")
@click.option("--palette", type=str, help="Palette preset name.")
@click.option("--size", type=str, help="Target texture size, e.g. '128'.")
@click.option("--format", "export_format", type=click.Choice(["png", "jpg", "jpeg", "webp", "bmp", "tga"], case_sensitive=False), help="Export file format.")
def process(
    input_file: str,
    output_file: str,
    config_file: str | None,
    preset: str | None,
    palette: str | None,
    size: str | None,
    export_format: str | None,
) -> None:
    """Convert a single image file."""
    if config_file:
        cfg = RetextureConfig.load(config_file)
    elif preset:
        cfg = RetextureConfig.load(preset)
    else:
        cfg = RetextureConfig()

    if palette:
        cfg.palette_preset = palette
    if size:
        dim = int(size.split("x")[0])
        cfg.size = [dim, dim]
    if export_format:
        fmt = export_format.lower()
        cfg.export_format = "jpg" if fmt in ("jpg", "jpeg") else fmt  # type: ignore

    with Image.open(input_file) as img:
        res = process_image(img, cfg)
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_bytes = export_image_bytes(res, fmt=cfg.export_format, quality=cfg.jpg_quality)
        out_p.write_bytes(out_bytes)

    click.echo(click.style(f"✓ Saved texture to {output_file}", fg="green"))


@main.command()
@click.option("-p", "--port", type=int, default=8080, help="Local server port.")
@click.option("-h", "--host", type=str, default="127.0.0.1", help="Local server host (strictly loopback).")
@click.option("--no-browser", is_flag=True, help="Do not open browser automatically.")
def gui(port: int, host: str, no_browser: bool) -> None:
    """Launch the interactive preview studio (strictly local loopback)."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        click.echo(
            click.style(
                f"Error: Host '{host}' is rejected for security. Retexture allows browsing local files and must strictly bind to loopback (127.0.0.1).",
                fg="red",
                bold=True,
            )
        )
        sys.exit(1)

    import secrets
    import uvicorn
    from retexture.gui.server import app, set_session_token

    # Generate an ephemeral session token for this studio run
    token = secrets.token_hex(16)
    set_session_token(token)

    url = f"http://{host}:{port}/?token={token}"
    display_url = f"http://{host}:{port}"
    click.echo(click.style(f"\n Retexture Studio (100% Offline & Authenticated)", bold=True, fg="cyan"))
    click.echo(f" Server running at: {click.style(display_url, underline=True, fg='blue')}")
    click.echo(f" {click.style('✓', fg='green')} Session token generated (local-only)")
    click.echo(" Press Ctrl+C to quit.\n")

    if not no_browser:
        import threading
        import webbrowser

        def open_tab():
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=open_tab, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


@main.group()
def presets() -> None:
    """Manage and inspect presets."""
    pass


@presets.command("list")
def list_presets_cmd() -> None:
    """List all available palettes and configuration presets."""
    palettes = list_palette_presets()
    configs = list_config_presets()

    click.echo(click.style("\n=== Palette Presets ===", bold=True, fg="cyan"))
    for p in palettes:
        click.echo(f"  • {click.style(p['id'], bold=True)}: {p['name']} ({p['color_count']} colors)")

    click.echo(click.style("\n=== Built-in Presets ===", bold=True, fg="green"))
    for c in configs["builtin"]:
        click.echo(f"  • {click.style(c['id'], bold=True)}: {c['name']}")

    if configs["custom"]:
        click.echo(click.style("\n=== Custom Presets (custom_presets/) ===", bold=True, fg="yellow"))
        for c in configs["custom"]:
            click.echo(f"  • {click.style(c['id'], bold=True)}: {c['name']}")
    click.echo()


@main.group()
def config() -> None:
    """Configuration utilities."""
    pass


@config.command("export")
@click.option("-o", "--output", "output_file", type=click.Path(), default="retexture_preset.json", help="Output JSON path.")
@click.option("--preset", type=str, default="ps1_classic", help="Base preset name.")
def export_config(output_file: str, preset: str) -> None:
    """Export configuration preset JSON file."""
    try:
        cfg = RetextureConfig.load(preset)
    except Exception:
        cfg = RetextureConfig()

    cfg.save(output_file)
    click.echo(click.style(f"✓ Exported config preset to {output_file}", fg="green"))


if __name__ == "__main__":
    main()
