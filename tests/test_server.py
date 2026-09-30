import base64
import io
import json
from fastapi.testclient import TestClient
from PIL import Image
from retexture.gui.server import app, set_session_token

TEST_TOKEN = "test_secret_token_123"
set_session_token(TEST_TOKEN)

client = TestClient(app, headers={"X-Retexture-Token": TEST_TOKEN})


def test_server_session_token_authentication():
    # 1. Request with wrong token -> 401 Unauthorized
    unauth_client = TestClient(app, headers={"X-Retexture-Token": "wrong_token"})
    res = unauth_client.get("/api/presets")
    assert res.status_code == 401
    assert "Unauthorized" in res.text

    # 2. Request with no token -> 401 Unauthorized
    no_token_client = TestClient(app)
    res_no = no_token_client.get("/api/presets")
    assert res_no.status_code == 401

    # 3. Query-string tokens are no longer accepted (log/history leak hardening).
    # Must be 401 even with the correct token in ?token=.
    res_query = no_token_client.get(f"/api/presets?token={TEST_TOKEN}")
    assert res_query.status_code == 401

    # 4. Request with valid header token -> 200 OK
    valid_client = TestClient(app, headers={"X-Retexture-Token": TEST_TOKEN})
    res_valid = valid_client.get("/api/presets")
    assert res_valid.status_code == 200

    # 5. Request with HttpOnly session cookie (used by <img> thumbnails) -> 200 OK
    cookie_client = TestClient(app, cookies={"retexture_token": TEST_TOKEN})
    res_cookie = cookie_client.get("/api/presets")
    assert res_cookie.status_code == 200


def test_server_presets_endpoint():
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "palettes" in data
    assert "configs" in data
    assert len(data["palettes"]) > 0
    assert len(data["configs"]["builtin"]) > 0


def test_server_process_endpoint():
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "image_base64": b64,
        "config": {
            "size": [64, 64],
            "palette_preset": "ps1_classic_16",
            "export_format": "png",
        },
    }

    res = client.post("/api/process", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content.startswith(b"\x89PNG")


def test_server_autosave_and_save_preset():
    # 1. Autosave with snapshot
    cfg_data = {
        "name": "Test Working",
        "size": [128, 128],
        "chromatic_aberration": 0.35,
    }
    res = client.post("/api/presets/autosave", json={"config": cfg_data, "create_snapshot": True})
    assert res.status_code == 200
    data = res.json()
    assert data["snapshot_created"] is True
    assert len(data["configs"]["autosaves"]) > 0

    # 2. Save Custom Preset
    res = client.post("/api/presets/save", json={"name": "test_saved_v1", "config": cfg_data})
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "test_saved_v1"

    # 3. Delete Custom Preset
    res_del = client.delete("/api/presets/custom/test_saved_v1")
    assert res_del.status_code == 200

    # 4. Clear Autosaves
    res_clear = client.post("/api/presets/autosaves/clear")
    assert res_clear.status_code == 200


def test_server_sample_image_endpoint():
    res = client.get("/api/sample-image")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 0


def test_server_import_lospec_palette(tmp_path, monkeypatch):
    import retexture.core.palettes as palettes_module
    import retexture.gui.server as server_module

    presets_dir = tmp_path / "presets"
    (presets_dir / "palettes").mkdir(parents=True)
    custom_dir = tmp_path / "custom_palettes"
    custom_dir.mkdir(parents=True)
    monkeypatch.setattr(palettes_module, "get_presets_dir", lambda: presets_dir)
    monkeypatch.setattr(palettes_module, "get_custom_palettes_dir", lambda: custom_dir)
    monkeypatch.setattr(server_module, "get_custom_palettes_dir", lambda: custom_dir)

    image = Image.new("RGB", (3, 1))
    image.putdata([(12, 34, 56), (78, 90, 123), (200, 210, 220)])
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    res = client.post(
        "/api/palettes/import",
        files={"palette_file": ("My Palette.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["palette"]["id"] == "custom/my_palette"
    assert data["palette"]["color_count"] == 3
    target = custom_dir / "my_palette.png"
    assert target.exists()

    conflict = client.post(
        "/api/palettes/import",
        files={"palette_file": ("My Palette.png", buf.getvalue(), "image/png")},
    )
    assert conflict.status_code == 409
    assert "already exists" in conflict.json()["detail"]["message"]

    replacement = Image.new("RGB", (2, 1))
    replacement.putdata([(1, 2, 3), (4, 5, 6)])
    replacement_buf = io.BytesIO()
    replacement.save(replacement_buf, format="PNG")
    overwritten = client.post(
        "/api/palettes/import",
        data={"overwrite": "true"},
        files={"palette_file": ("My Palette.png", replacement_buf.getvalue(), "image/png")},
    )
    assert overwritten.status_code == 200
    assert overwritten.json()["palette"]["color_count"] == 2


def test_server_explorer_endpoints(tmp_path):
    # Create test directory structure with images
    img1 = tmp_path / "tex_a.png"
    img2 = tmp_path / "tex_b.jpg"
    sub = tmp_path / "subfolder"
    sub.mkdir()

    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img1)
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(img2)

    # 1. Browse
    res = client.get(f"/api/explorer/browse?path={tmp_path}")
    assert res.status_code == 200
    data = res.json()
    assert data["total_images"] == 2
    assert len(data["directories"]) == 1
    assert data["directories"][0]["name"] == "subfolder"

    # 2. Get file
    res_file = client.get(f"/api/explorer/file?path={img1}")
    assert res_file.status_code == 200
    assert res_file.headers["content-type"] == "image/png"

    # 3. Direct disk batch convert (creates only the latest folder when parent exists)
    out_dir = tmp_path / "retrofied_out"
    payload = {
        "file_paths": [str(img1), str(img2)],
        "output_folder": str(out_dir),
        "config": {
            "size": [32, 32],
            "palette_preset": "signalis_replika_16",
            "export_format": "png",
        },
    }
    res_batch = client.post("/api/explorer/batch-convert", json=payload)
    assert res_batch.status_code == 200
    batch_data = res_batch.json()
    assert batch_data["count"] == 2
    assert (out_dir / "tex_a.png").exists()
    assert (out_dir / "tex_b.png").exists()

    # 4. Fail if parent of destination does not exist
    deep_nested_dir = tmp_path / "nonexistent_parent" / "child_folder"
    payload["output_folder"] = str(deep_nested_dir)
    res_fail = client.post("/api/explorer/batch-convert", json=payload)
    assert res_fail.status_code == 400
    assert "Parent folder" in res_fail.json()["detail"]


def test_server_explorer_nonexistent_browse():
    res = client.get("/api/explorer/browse?path=/this/path/absolutely/does/not/exist_12345")
    assert res.status_code == 404
    assert "Directory not found" in res.json()["detail"]


def test_server_rejects_non_local_lan_clients():
    # Simulate a request coming from another device on the LAN
    lan_client = TestClient(app, client=("192.168.1.120", 54321))
    res = lan_client.get("/api/presets")
    assert res.status_code == 403
    assert "Access Denied" in res.text


def test_server_rejects_cross_site_requests():
    # Simulate an external website in another tab attempting a cross-site request
    cross_client = TestClient(app, headers={"Sec-Fetch-Site": "cross-site", "X-Retexture-Token": TEST_TOKEN})
    res = cross_client.get("/api/presets")
    assert res.status_code == 403
    assert "cross-site" in res.text.lower()


def test_server_check_conflicts(tmp_path):
    out_dir = tmp_path / "output_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crate.png").write_bytes(b"existing_file")

    payload = {
        "file_paths": [str(tmp_path / "crate.png"), str(tmp_path / "stone.png")],
        "output_folder": str(out_dir),
        "filename_pattern": "{name}",
        "config": {"export_format": "png"},
    }

    res = client.post("/api/explorer/check-conflicts", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert data["conflicts_count"] == 1
    assert data["conflicting_files"] == ["crate.png"]


def test_server_quick_locations():
    res = client.get("/api/explorer/quick-locations")
    assert res.status_code == 200
    data = res.json()
    assert "locations" in data
    names = [loc["name"] for loc in data["locations"]]
    assert "Current Workspace" in names
    for name in names:
        assert name in ("Current Workspace", "Downloads", "Pictures")
