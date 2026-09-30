from click.testing import CliRunner
from PIL import Image
from retexture.cli import main


def test_cli_presets_list():
    runner = CliRunner()
    result = runner.invoke(main, ["presets", "list"])
    assert result.exit_code == 0
    assert "Palette Presets" in result.output
    assert "ps1_classic_16" in result.output


def test_cli_single_process(tmp_path):
    in_file = tmp_path / "input.png"
    out_file = tmp_path / "output.png"

    img = Image.new("RGB", (64, 64), color=(255, 128, 0))
    img.save(in_file)

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(in_file), "-o", str(out_file), "--palette", "ps1_classic_16", "--size", "64"])
    assert result.exit_code == 0
    assert out_file.exists()


def test_cli_batch(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    for i in range(3):
        img = Image.new("RGB", (32, 32), color=(i * 50, 100, 150))
        img.save(in_dir / f"tex_{i}.png")

    runner = CliRunner()
    result = runner.invoke(main, ["batch", str(in_dir), "-o", str(out_dir), "--palette", "ps1_classic_16", "--size", "32"])
    assert result.exit_code == 0
    assert (out_dir / "tex_0.png").exists()
    assert (out_dir / "tex_1.png").exists()
    assert (out_dir / "tex_2.png").exists()


def test_cli_config_export(tmp_path):
    out_file = tmp_path / "exported.json"
    runner = CliRunner()
    result = runner.invoke(main, ["config", "export", "-o", str(out_file), "--preset", "ps1_classic"])
    assert result.exit_code == 0
    assert out_file.exists()


def test_cli_export_formats(tmp_path):
    in_file = tmp_path / "input.png"
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(in_file)

    runner = CliRunner()
    for fmt, expected_pil_format in [("webp", "WEBP"), ("bmp", "BMP"), ("tga", "TGA"), ("jpg", "JPEG")]:
        out_file = tmp_path / f"output.{fmt}"
        result = runner.invoke(main, ["process", str(in_file), "-o", str(out_file), "--format", fmt, "--size", "32"])
        assert result.exit_code == 0
        assert out_file.exists()
        with Image.open(out_file) as loaded:
            assert loaded.format == expected_pil_format


def test_cli_batch_formats(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    img = Image.new("RGB", (32, 32), color=(50, 60, 70))
    img.save(in_dir / "sample.png")

    runner = CliRunner()
    result = runner.invoke(main, ["batch", str(in_dir), "-o", str(out_dir), "--format", "webp", "--size", "32"])
    assert result.exit_code == 0
    out_file = out_dir / "sample.webp"
    assert out_file.exists()
    with Image.open(out_file) as loaded:
        assert loaded.format == "WEBP"
