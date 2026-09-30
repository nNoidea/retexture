/**
 * Retexture — Interactive Studio Frontend
 * Single Source of Truth architecture with auto-save, side-by-side mode, and mass batch converter.
 */

// ============================================================================
// Ephemeral Session Token Authentication
// ============================================================================
function getCookie(name) {
  const match = document.cookie.match(new RegExp("(^|; )" + name + "=([^;]*)"));
  return match ? decodeURIComponent(match[2]) : null;
}

const urlParams = new URLSearchParams(window.location.search);
const tokenFromUrl = urlParams.get("token");
if (tokenFromUrl) {
  sessionStorage.setItem("retexture_session_token", tokenFromUrl);
  // Clean token from browser URL address bar for privacy
  window.history.replaceState({}, document.title, window.location.pathname);
}
const sessionToken = tokenFromUrl || sessionStorage.getItem("retexture_session_token") || getCookie("retexture_token") || "";

// ============================================================================
// HTML escaping (stored-XSS hardening).
// Disk filenames / preset names / server errors are attacker-controlled and
// must never be interpolated raw into innerHTML or attributes.
// ============================================================================
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"'`=\/]/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#x27;",
    "`": "&#x60;",
    "=": "&#x3D;",
    "/": "&#x2F;",
  })[c]);
}

function apiFetch(url, options = {}) {
  const opts = { ...options };
  opts.headers = { ...(opts.headers || {}) };
  if (sessionToken) {
    opts.headers["X-Retexture-Token"] = sessionToken;
  }
  return fetch(url, opts);
}

function getAuthenticatedFileUrl(path) {
  // Auth relies on the HttpOnly session cookie set by GET / (browsers send
  // it automatically for <img>). No token-in-URL: avoids history/log leaks.
  return `/api/explorer/file?path=${encodeURIComponent(path)}`;
}

function getAuthenticatedSampleUrl() {
  return "/api/sample-image";
}

const state = {
  originalImage: null,
  processedImage: null,
  processedBlob: null,
  viewMode: "sidebyside", // "sidebyside", "split", "tiled", "3d"
  splitRatio: 0.5,
  zoom: 1.0,
  pan: { x: 0, y: 0 },
  isPanning: false,
  panStart: { x: 0, y: 0 },
  isDraggingSplit: false,
  crispNearest: true, // 100% crisp pixel-art nearest-neighbor view
  activePresetId: "mouthwashing_psx_horror",
  basePresetName: "mouthwashing_psx_horror",
  presetsData: { palettes: [], configs: { builtin: [], custom: [], latest_saved: null } },
  
  // Asset Explorer & Bulk Queue State
  explorerPath: ".",
  explorerParentPath: null,
  explorerFiles: [],
  explorerDirectories: [],
  selectedFiles: new Set(),
  activeFilePath: null,
  activeFileName: "default_sample",
  outputFolder: localStorage.getItem("retexture_last_save_path") || "./retrofied_textures",
  namingPattern: localStorage.getItem("retexture_filename_pattern") || "{name}",
  lockResolution: localStorage.getItem("retexture_lock_resolution") === "true",

  config: {
    name: "Custom Preset",
    description: "",
    base_preset: "",
    size: [128, 128],
    resize_filter: "nearest",
    export_format: "png",
    jpg_quality: 90,
    palette_preset: "mouthwashing_decay_16",
    color_metric: "oklab",
    dither_algorithm: "bayer4x4",
    dither_strength: 0.95,
    dither_scale: 1,
    linear_color_space: true,
    alpha_stipple: true,
    one_bit_bias: 0.0,
    one_bit_contrast: 1.0,
    grain: 0.08,
    grain_seed: 42,
    grain_monochrome: true,
    hue: 0.0,
    saturation: 0.85,
    brightness: 0.95,
    contrast: 1.25,
    color_temperature: 0.0,
    midtones: 0.0,
    highlights: 0.0,
    luminance_threshold: 0.5,
    invert_luminance: false,
    black_point: 0.0,
    white_point: 1.0,
    s_curve: 0.0,
    shadow_hue_shift: 0.0,
    highlight_hue_shift: 0.0,
    shadow_saturation: 1.0,
    highlight_saturation: 1.0,
    pre_blur: 0.0,
    chromatic_aberration: 0.4,
    crt_scanlines: 0.25,
    grime_decay: 0.35,
    edge_crunch: 0.4,
    outline_strength: 0.0,
    bilateral_simplify: 0.0,
    pixeloe_strength: 0.0,
    cel_shading_steps: 0,
    grime_mode: "uniform",
    bake_optical_fx: false,
    seamless_tiling: false,
    seam_size: 0.2,
    seam_method: "offset_wrap",
  },
};

let processDebounceTimer = null;
let autosaveDebounceTimer = null;
let activeProcessAbortController = null;
let latestProcessRequestId = 0;

// ============================================================================
// Mode & Context Awareness Helpers
// ============================================================================

const ORDERED_DITHERS = new Set([
  "bayer2x2", "bayer4x4", "bayer8x8", "bayer",
  "halftone_dot", "halftone_cross", "halftone",
  "interlaced", "checkerboard", "saturn",
  "crosshatch", "hatching", "pc98",
  "cluster_dot_4x4", "cluster_dot_8x8", "cluster_dot", "clusterdot",
  "line_halftone", "linehalftone", "halftone_line",
  "blue_noise", "bluenoise", "blue"
]);

function setUIMode(mode) {
  const isSimple = mode !== "pro";
  state.uiMode = isSimple ? "simple" : "pro";
  const sidebar = document.getElementById("controlsSidebar");
  const btnSimple = document.getElementById("btnModeSimple");
  const btnPro = document.getElementById("btnModePro");

  if (sidebar) {
    sidebar.classList.toggle("mode-simple", isSimple);
    sidebar.classList.toggle("mode-pro", !isSimple);
  }
  if (btnSimple) {
    btnSimple.classList.toggle("active", isSimple);
    btnSimple.setAttribute("aria-selected", isSimple ? "true" : "false");
  }
  if (btnPro) {
    btnPro.classList.toggle("active", !isSimple);
    btnPro.setAttribute("aria-selected", !isSimple ? "true" : "false");
  }
  localStorage.setItem("retexture_ui_mode", state.uiMode);
}

function updateDitherScaleSupport(algo) {
  const isOrdered = ORDERED_DITHERS.has((algo || "").toLowerCase());
  const scaleWrap = document.getElementById("ditherScaleWrap");
  const badge = document.getElementById("scaleSupportBadge");
  if (scaleWrap) {
    scaleWrap.style.opacity = isOrdered ? "1.0" : "0.45";
    scaleWrap.title = isOrdered ? "Pattern Scale (1×-4× for ordered matrices)" : "Pattern Scale only applies to ordered matrices (Bayer, Halftone, Crosshatch, Blue Noise)";
  }
  if (badge) {
    badge.textContent = isOrdered ? "Ordered" : "Not applicable";
    badge.style.opacity = isOrdered ? "1.0" : "0.5";
  }
}

function checkImageHasAlpha(img) {
  try {
    if (!img || !img.width || !img.height) return false;
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(img.width, 64);
    canvas.height = Math.min(img.height, 64);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] < 250) return true;
    }
    return false;
  } catch (e) {
    return false;
  }
}

function updateAlphaBadge(hasAlpha) {
  const badge = document.getElementById("alphaStippleBadge");
  if (badge) {
    badge.textContent = hasAlpha ? "RGBA Active" : "RGB (No Alpha)";
    badge.style.color = hasAlpha ? "var(--accent-green)" : "var(--text-muted)";
    badge.style.borderColor = hasAlpha ? "rgba(80, 250, 123, 0.4)" : "var(--border-dim)";
  }
}

function onOriginalImageLoaded(img) {
  state.originalImage = img;
  updateAlphaBadge(checkImageHasAlpha(img));
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  initUIListeners();
  initPanelToggles();
  initSidebarResizers();
  initCanvas();
  initSplitSlider();
  initSavePresetModal();
  initPresetsManagerModal();
  initNamingPatternModal();
  initSaveFolderBrowserModal();
  initOverwriteModal();
  initAssetExplorer();
  initPaletteImporter();

  await fetchPresets();
  loadDefaultStartupTexture();
});

// ============================================================================
// Presets & Autosave System
// ============================================================================

async function fetchPresets() {
  try {
    const res = await apiFetch("/api/presets");
    if (!res.ok) throw new Error("Failed to load presets");
    state.presetsData = await res.json();
    populatePresetDropdowns();
  } catch (err) {
    showToast(`Error loading presets: ${err.message}`);
  }
}

function populatePresetDropdowns() {
  // Palettes
  const palSelect = document.getElementById("palettePresetSelect");
  palSelect.innerHTML = "";
  const palettes = state.presetsData.palettes || [];
  [
    { label: "Built-in Palettes", items: palettes.filter((p) => !p.is_custom) },
    { label: "Custom Palettes", items: palettes.filter((p) => p.is_custom) },
  ].forEach((group) => {
    if (!group.items.length) return;
    const optGroup = document.createElement("optgroup");
    optGroup.label = group.label;
    group.items.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.color_count})`;
      optGroup.appendChild(opt);
    });
    palSelect.appendChild(optGroup);
  });
  if (state.config.palette_preset) {
    palSelect.value = state.config.palette_preset;
  }
  renderPaletteSwatches();

  // Config Presets (Organized: Working Config, Custom, Built-in)
  const cfgSelect = document.getElementById("configPresetSelect");
  cfgSelect.innerHTML = "";

  if (state.presetsData.configs.latest_saved) {
    const opt = document.createElement("option");
    opt.value = "latest_saved";
    opt.textContent = "● Working Config (Auto-saved)";
    cfgSelect.appendChild(opt);
  }

  if (state.presetsData.configs.custom && state.presetsData.configs.custom.length > 0) {
    const optGroup = document.createElement("optgroup");
    optGroup.label = "Custom Presets";
    state.presetsData.configs.custom.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      optGroup.appendChild(opt);
    });
    cfgSelect.appendChild(optGroup);
  }

  if (state.presetsData.configs.builtin && state.presetsData.configs.builtin.length > 0) {
    const optGroup = document.createElement("optgroup");
    optGroup.label = "Built-in Presets";
    state.presetsData.configs.builtin.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      optGroup.appendChild(opt);
    });
    cfgSelect.appendChild(optGroup);
  }

  if (state.activePresetId) {
    cfgSelect.value = state.activePresetId;
  }
}

function initPaletteImporter() {
  const button = document.getElementById("btnImportPalette");
  const input = document.getElementById("paletteFileInput");
  if (!button || !input) return;

  button.addEventListener("click", () => input.click());
  input.addEventListener("change", async () => {
    const file = input.files && input.files[0];
    input.value = "";
    if (!file) return;

    const importPalette = async (overwrite = false) => {
      const formData = new FormData();
      formData.append("palette_file", file);
      formData.append("overwrite", String(overwrite));

      const res = await apiFetch("/api/palettes/import", {
        method: "POST",
        body: formData,
      });

      let data = {};
      try {
        data = await res.json();
      } catch (_) {
        // Keep the generic HTTP error below if the server did not return JSON.
      }

      if (res.status === 409 && !overwrite) {
        const detail = data.detail || {};
        const message = typeof detail === "string"
          ? detail
          : (detail.message || `A palette named "${file.name}" already exists`);
        if (confirm(`${message}. Overwrite it with this file?`)) {
          return importPalette(true);
        }
        showToast("Palette import cancelled.");
        return;
      }

      if (!res.ok) {
        const detail = data.detail || "Palette import failed";
        throw new Error(typeof detail === "string" ? detail : detail.message || "Palette import failed");
      }

      state.presetsData.palettes = data.palettes || state.presetsData.palettes || [];
      if (data.palette && data.palette.id) {
        state.config.palette_preset = data.palette.id;
      }
      populatePresetDropdowns();
      syncUIFromConfig();
      triggerProcess(true, true);
      triggerAutosave(true);
      showToast(`Imported palette "${data.palette?.name || file.name}".`);
    };

    try {
      await importPalette();
    } catch (err) {
      showToast(`Palette import error: ${err.message}`);
    }
  });
}

function renderPaletteSwatches() {
  const container = document.getElementById("paletteSwatches");
  container.innerHTML = "";
  const pal = state.presetsData.palettes.find((p) => p.id === state.config.palette_preset);
  if (!pal || !pal.colors) return;

  pal.colors.forEach((hex) => {
    const swatch = document.createElement("div");
    swatch.className = "swatch";
    swatch.style.backgroundColor = hex;
    swatch.title = hex;
    container.appendChild(swatch);
  });
}

function markConfigModified() {
  if (state.activePresetId !== "latest_saved") {
    state.activePresetId = "latest_saved";
    const select = document.getElementById("configPresetSelect");
    if (select) {
      select.value = "latest_saved";
    }
  }
}

function triggerAutosave(createSnapshot = false) {
  const tag = document.getElementById("autoSaveIndicator");
  if (tag) tag.textContent = createSnapshot ? "Saved" : "Saving...";

  clearTimeout(autosaveDebounceTimer);
  autosaveDebounceTimer = setTimeout(async () => {
    try {
      const res = await apiFetch("/api/presets/autosave", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config: state.config,
          create_snapshot: createSnapshot,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.configs) {
          state.presetsData.configs = data.configs;
        }
      }
      if (tag) tag.textContent = "Saved";
    } catch (e) {
      if (tag) tag.textContent = "Error";
    }
  }, createSnapshot ? 10 : 350);
}

function setProcessingIndicator(show, customText = null) {
  const badge = document.getElementById("processingIndicator");
  const textEl = document.getElementById("processingStatusText");
  if (!badge) return;

  if (show) {
    if (textEl) {
      if (customText) {
        textEl.textContent = customText;
      } else if ((state.config.pixeloe_strength || 0) > 0.001) {
        textEl.textContent = "PixelOE Stylizing...";
      } else if ((state.config.outline_strength || 0) > 0.001) {
        textEl.textContent = "Injecting Outlines...";
      } else if (state.config.dither_algorithm === "riemersma") {
        textEl.textContent = "Riemersma Fractal Dither...";
      } else {
        textEl.textContent = "Processing Texture...";
      }
    }
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }
}

function triggerProcess(isUserModification = true, createSnapshot = false, immediate = false) {
  if (isUserModification) {
    markConfigModified();
  }
  triggerAutosave(createSnapshot);

  // Instantly cancel any ongoing in-flight HTTP request
  if (activeProcessAbortController) {
    try {
      activeProcessAbortController.abort();
    } catch (_) {}
    activeProcessAbortController = null;
  }
  activeProcessAbortController = new AbortController();

  // Increment request ID so any pending/stale responses are immediately discarded
  latestProcessRequestId++;
  const requestId = latestProcessRequestId;

  clearTimeout(processDebounceTimer);
  setProcessingIndicator(true);

  // If discrete action (preset select, randomizer click, format switch), execute immediately with 0ms delay.
  // If continuous slider dragging, debounce slightly (45ms) to prevent TCP socket floods.
  const delay = (immediate || createSnapshot) ? 0 : 45;

  processDebounceTimer = setTimeout(async () => {
    // Drop if a newer request arrived while in debounce delay
    if (requestId !== latestProcessRequestId) return;
    if (!state.originalImage) {
      setProcessingIndicator(false);
      return;
    }

    const signal = activeProcessAbortController ? activeProcessAbortController.signal : undefined;

    try {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = state.originalImage.naturalWidth || state.originalImage.width;
      tempCanvas.height = state.originalImage.naturalHeight || state.originalImage.height;
      const ctx = tempCanvas.getContext("2d");
      ctx.drawImage(state.originalImage, 0, 0);
      const b64 = tempCanvas.toDataURL("image/png");

      if (requestId !== latestProcessRequestId || signal?.aborted) return;

      const response = await apiFetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: signal,
        body: JSON.stringify({
          image_base64: b64,
          config: state.config,
        }),
      });

      // If newer request was dispatched while waiting for network, drop this stale response immediately
      if (requestId !== latestProcessRequestId || signal?.aborted) return;
      if (!response.ok) throw new Error("Processing failed");

      const blob = await response.blob();
      if (requestId !== latestProcessRequestId || signal?.aborted) return;

      state.processedBlob = blob;

      const img = new Image();
      img.onload = () => {
        // Critical: only update viewport if this is STILL the latest active request
        if (requestId !== latestProcessRequestId) {
          URL.revokeObjectURL(img.src);
          return;
        }
        state.processedImage = img;
        if (state.viewMode === "3d") {
          if (window.update3DTexture) window.update3DTexture(img);
        } else {
          render2D();
        }
        setProcessingIndicator(false);
      };
      img.onerror = () => {
        if (requestId === latestProcessRequestId) {
          setProcessingIndicator(false);
        }
      };
      img.src = URL.createObjectURL(blob);
    } catch (err) {
      if (err.name === "AbortError") {
        // Deliberately aborted by newer user interaction - do nothing
        return;
      }
      console.error(err);
      if (requestId === latestProcessRequestId) {
        setProcessingIndicator(false);
      }
    }
  }, delay);
}

// ============================================================================
// 2D Canvas Rendering (Side-by-Side, Split Slider, 3x3 Tiled)
// ============================================================================

function initCanvas() {
  const container = document.getElementById("viewportArea");
  const wrapper = document.getElementById("view2dWrapper");

  // Mouse wheel zoom across the entire viewport, anchored at the cursor position
  container.addEventListener("wheel", (e) => {
    if (state.viewMode === "3d") return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    zoomAt(e.clientX, e.clientY, factor);
  }, { passive: false });

  // Pan via dragging on wrapper
  wrapper.addEventListener("mousedown", (e) => {
    if (state.isDraggingSplit) return;
    state.isPanning = true;
    state.panStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
  });

  window.addEventListener("mousemove", (e) => {
    if (state.isPanning && !state.isDraggingSplit) {
      state.pan.x = e.clientX - state.panStart.x;
      state.pan.y = e.clientY - state.panStart.y;
      updateCanvasTransform();
    }
  });

  window.addEventListener("mouseup", () => {
    state.isPanning = false;
  });
}

function setZoom(newZoom) {
  state.zoom = Math.min(Math.max(newZoom, 0.2), 32.0);
  updateCanvasTransform();
}

// Zoom anchored at a screen position: the content point under the cursor stays
// under it while the stage scales. The stage scales about its own center
// (transform-origin: center center), so the pan must be shifted to compensate.
function zoomAt(clientX, clientY, factor) {
  const container = document.getElementById("viewportArea");
  const stage = document.getElementById("canvasStage");
  if (!container || !stage) {
    setZoom(state.zoom * factor);
    return;
  }

  const newZoom = Math.min(Math.max(state.zoom * factor, 0.2), 32.0);
  const ratio = newZoom / state.zoom;
  if (ratio === 1) return; // clamped at a zoom limit, nothing to anchor

  const containerRect = container.getBoundingClientRect();
  const stageRect = stage.getBoundingClientRect();
  const originX = stageRect.left + stageRect.width / 2 - containerRect.left;
  const originY = stageRect.top + stageRect.height / 2 - containerRect.top;
  const mouseX = clientX - containerRect.left;
  const mouseY = clientY - containerRect.top;

  state.pan.x += (mouseX - originX) * (1 - ratio);
  state.pan.y += (mouseY - originY) * (1 - ratio);
  state.zoom = newZoom;
  updateCanvasTransform();
}

function updateCanvasTransform() {
  const stage = document.getElementById("canvasStage");
  if (stage) {
    stage.style.transform = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
  }
  const resetBtn = document.getElementById("btnZoomReset");
  if (resetBtn) {
    resetBtn.textContent = `${Math.round(state.zoom * 100)}%`;
  }
  updateSplitDividerPosition();
}

window.state = state;

function render2D() {
  const canvas = document.getElementById("mainCanvas");
  const ctx = canvas.getContext("2d");
  if (!state.originalImage || !state.processedImage) return;

  canvas.classList.toggle("pixelated", state.crispNearest);

  const targetW = state.processedImage.naturalWidth || state.processedImage.width || 256;
  const targetH = state.processedImage.naturalHeight || state.processedImage.height || 256;

  // Clean integer texel scaling so every pixel in the retro texture is an exact integer block
  let scale = 1;
  if (targetW <= 64) scale = 6;       // 64 * 6 = 384px (each retro texel is 6x6 solid pixels)
  else if (targetW <= 128) scale = 3; // 128 * 3 = 384px (each retro texel is 3x3 solid pixels)
  else if (targetW <= 256) scale = 2; // 256 * 2 = 512px (each retro texel is 2x2 solid pixels)
  else scale = 1;

  const boxW = targetW * scale;
  const boxH = targetH * scale;

  if (state.viewMode === "sidebyside") {
    // ------------------------------------------------------------------------
    // Side-by-Side View: [ ORIGINAL ]      [ RETROFIED (Crisp Pixel Art) ]
    // ------------------------------------------------------------------------
    const gap = 24;
    const totalW = boxW * 2 + gap;
    const totalH = boxH;

    canvas.width = totalW;
    canvas.height = totalH;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    // Clear canvas so the space between textures is completely transparent
    ctx.clearRect(0, 0, totalW, totalH);

    // 1. Draw Original on the left
    ctx.imageSmoothingEnabled = !state.crispNearest;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(state.originalImage, 0, 0, boxW, boxH);

    // 2. Draw Retrofied on the right (pure nearest-neighbor, crisp pixel grid)
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(state.processedImage, boxW + gap, 0, boxW, boxH);

  } else if (state.viewMode === "split") {
    // ------------------------------------------------------------------------
    // Split Slider View
    // ------------------------------------------------------------------------
    canvas.width = boxW;
    canvas.height = boxH;
    canvas.style.width = `${boxW}px`;
    canvas.style.height = `${boxH}px`;

    ctx.clearRect(0, 0, boxW, boxH);

    // 1. Draw Retrofied underlayer (crisp pixel grid)
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(state.processedImage, 0, 0, boxW, boxH);

    // 2. Clip and draw original overlayer
    const splitX = Math.floor(boxW * state.splitRatio);
    if (splitX > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, boxH);
      ctx.clip();
      ctx.imageSmoothingEnabled = !state.crispNearest;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(state.originalImage, 0, 0, boxW, boxH);
      ctx.restore();
    }
  } else if (state.viewMode === "tiled") {
    // ------------------------------------------------------------------------
    // 3x3 Tiled View
    // ------------------------------------------------------------------------
    canvas.width = boxW * 3;
    canvas.height = boxH * 3;
    canvas.style.width = `${boxW * 3}px`;
    canvas.style.height = `${boxH * 3}px`;

    ctx.clearRect(0, 0, boxW * 3, boxH * 3);
    ctx.imageSmoothingEnabled = false;

    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        ctx.drawImage(state.processedImage, col * boxW, row * boxH, boxW, boxH);
      }
    }
  }

  updateCanvasTransform();
}

// ============================================================================
// Split Slider Controller
// ============================================================================

function initSplitSlider() {
  const divider = document.getElementById("splitDivider");
  const wrapper = document.getElementById("view2dWrapper");

  divider.addEventListener("mousedown", (e) => {
    state.isDraggingSplit = true;
    if (wrapper) wrapper.classList.add("is-dragging-split");
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.isDraggingSplit) return;
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return;

    const x = e.clientX - rect.left;
    const ratio = Math.min(Math.max(x / rect.width, 0.0), 1.0);
    state.splitRatio = ratio;
    render2D();
  });

  window.addEventListener("mouseup", () => {
    state.isDraggingSplit = false;
    if (wrapper) wrapper.classList.remove("is-dragging-split");
  });
}

function updateSplitDividerPosition() {
  const divider = document.getElementById("splitDivider");
  const wrapper = document.getElementById("view2dWrapper");
  if (!divider || !wrapper) return;
  const isSplit = state.viewMode === "split";
  wrapper.classList.toggle("mode-split", isSplit);
  divider.style.left = `${state.splitRatio * 100}%`;
  divider.style.height = "100%";
}

// ============================================================================
// UI Listeners & Controls
// ============================================================================

function updateLockResolutionUI() {
  const btn = document.getElementById("btnLockResolution");
  const icon = document.getElementById("lockResIcon");
  const text = document.getElementById("lockResText");
  if (!btn) return;
  btn.classList.toggle("active", state.lockResolution);
  if (text) text.textContent = state.lockResolution ? "Locked" : "Unlock";
  if (icon) {
    icon.innerHTML = state.lockResolution
      ? `<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>`
      : `<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/>`;
  }
}

function initUIListeners() {
  // Texture Interpolation (Blender-style Closest vs Linear) Toggle
  const btnNearest = document.getElementById("btnToggleNearest");
  const nearestLabel = document.getElementById("nearestLabel");
  const canvas = document.getElementById("mainCanvas");

  const savedCrisp = localStorage.getItem("retexture_crisp_nearest");
  if (savedCrisp !== null) {
    state.crispNearest = savedCrisp === "true";
  }

  if (btnNearest) {
    btnNearest.classList.toggle("active", state.crispNearest);
    if (nearestLabel) {
      nearestLabel.textContent = state.crispNearest ? "Interpolation: Closest" : "Interpolation: Linear";
    }
    if (canvas) canvas.classList.toggle("pixelated", state.crispNearest);

    btnNearest.addEventListener("click", () => {
      state.crispNearest = !state.crispNearest;
      btnNearest.classList.toggle("active", state.crispNearest);
      if (nearestLabel) {
        nearestLabel.textContent = state.crispNearest ? "Interpolation: Closest" : "Interpolation: Linear";
      }
      canvas.classList.toggle("pixelated", state.crispNearest);
      localStorage.setItem("retexture_crisp_nearest", state.crispNearest);
      if (window.set3DTextureFiltering) {
        window.set3DTextureFiltering(state.crispNearest);
      }
      render2D();
      showToast(state.crispNearest ? "Interpolation: Closest (Pixel Art)" : "Interpolation: Linear (Smooth Bilinear)");
    });
  }

  // View Mode Tabs
  const savedViewMode = localStorage.getItem("retexture_last_view_mode");
  if (savedViewMode) {
    state.viewMode = savedViewMode;
    document.querySelectorAll(".v-tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.view === savedViewMode);
    });
    const v2d = document.getElementById("view2dWrapper");
    const v3d = document.getElementById("view3dWrapper");
    const isSplit = savedViewMode === "split";
    if (v2d) v2d.classList.toggle("mode-split", isSplit);
    if (savedViewMode === "3d") {
      if (v2d) v2d.classList.add("hidden");
      if (v3d) v3d.classList.remove("hidden");
      if (window.init3DViewport) window.init3DViewport();
    }
  }

  // Mode Selector (Simple vs Pro)
  const savedUIMode = localStorage.getItem("retexture_ui_mode") || "simple";
  setUIMode(savedUIMode);

  const btnSimple = document.getElementById("btnModeSimple");
  const btnPro = document.getElementById("btnModePro");
  if (btnSimple) {
    btnSimple.addEventListener("click", () => setUIMode("simple"));
  }
  if (btnPro) {
    btnPro.addEventListener("click", () => setUIMode("pro"));
  }

  document.querySelectorAll(".v-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".v-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.dataset.view;
      state.viewMode = mode;
      localStorage.setItem("retexture_last_view_mode", mode);

      const v2d = document.getElementById("view2dWrapper");
      const v3d = document.getElementById("view3dWrapper");
      const divider = document.getElementById("splitDivider");

      const isSplit = mode === "split";
      v2d.classList.toggle("mode-split", isSplit);
      if (divider) divider.style.height = "100%";

      if (mode === "3d") {
        v2d.classList.add("hidden");
        v3d.classList.remove("hidden");
        if (window.init3DViewport) window.init3DViewport();
        if (state.processedImage && window.update3DTexture) window.update3DTexture(state.processedImage);
      } else {
        v3d.classList.add("hidden");
        v2d.classList.remove("hidden");
        render2D();
      }
    });
  });

  // Config Presets Dropdown
  document.getElementById("configPresetSelect").addEventListener("change", (e) => {
    const val = e.target.value;
    let target = null;
    if (val === "latest_saved") {
      target = state.presetsData.configs.latest_saved?.config;
      state.activePresetId = "latest_saved";
    } else {
      const all = [...state.presetsData.configs.custom, ...state.presetsData.configs.builtin];
      const match = all.find((c) => c.id === val);
      if (match) {
        target = match.config;
        state.activePresetId = match.id;
        state.basePresetName = match.name || match.id;
      }
    }

    if (target) {
      const preservedSize = state.lockResolution ? [...state.config.size] : null;
      state.config = { ...state.config, ...target, base_preset: state.basePresetName };
      if (preservedSize) {
        state.config.size = preservedSize;
      }
      syncUIFromConfig();
      triggerProcess(false, false); // Do not switch to working config because user explicitly selected this preset
    }
  });

  // Lock Resolution Toggle
  const lockResBtn = document.getElementById("btnLockResolution");
  if (lockResBtn) {
    updateLockResolutionUI();
    lockResBtn.addEventListener("click", () => {
      state.lockResolution = !state.lockResolution;
      localStorage.setItem("retexture_lock_resolution", state.lockResolution);
      updateLockResolutionUI();
      showToast(state.lockResolution ? "Resolution Locked (Presets will preserve size)" : "Resolution Unlocked");
    });
  }

  // Resolution Scale Slider (32px to 512px in steps of 32)
  const sliderRes = document.getElementById("sliderResolution");
  if (sliderRes) {
    sliderRes.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      state.config.size = [val, val];
      document.getElementById("valResolution").textContent = `${val}px`;
      document.getElementById("resolutionSliderWrap").classList.remove("dimmed");
      document.getElementById("btnOriginalSize").classList.remove("active");
      triggerProcess(false, false);
    });
    sliderRes.addEventListener("change", (e) => {
      const val = parseInt(e.target.value, 10);
      state.config.size = [val, val];
      document.getElementById("valResolution").textContent = `${val}px`;
      document.getElementById("resolutionSliderWrap").classList.remove("dimmed");
      document.getElementById("btnOriginalSize").classList.remove("active");
      triggerProcess(true, true);
    });
  }

  // Original Size Button (1:1 Native Resolution)
  const btnOrig = document.getElementById("btnOriginalSize");
  if (btnOrig) {
    btnOrig.addEventListener("click", () => {
      state.config.size = [0, 0];
      document.getElementById("resolutionSliderWrap").classList.add("dimmed");
      btnOrig.classList.add("active");
      document.getElementById("valResolution").textContent = "Native 1:1";
      triggerProcess(true, true);
    });
  }

  // Filters & Formats
  document.getElementById("resizeFilterSelect").addEventListener("change", (e) => {
    state.config.resize_filter = e.target.value;
    triggerProcess(true, true);
  });

  document.getElementById("exportFormatSelect").addEventListener("change", (e) => {
    state.config.export_format = e.target.value;
    triggerProcess(true, true);
  });

  document.getElementById("palettePresetSelect").addEventListener("change", (e) => {
    state.config.palette_preset = e.target.value;
    renderPaletteSwatches();
    triggerProcess(true, true);
  });

  const cmSelect = document.getElementById("colorMetricSelect");
  if (cmSelect) {
    cmSelect.addEventListener("change", (e) => {
      state.config.color_metric = e.target.value;
      triggerProcess(true, true);
    });
  }

  // Dither & Noise
  document.getElementById("ditherAlgoSelect").addEventListener("change", (e) => {
    state.config.dither_algorithm = e.target.value;
    updateDitherScaleSupport(e.target.value);
    triggerProcess(true, true);
  });

  document.querySelectorAll("#ditherScaleGroup .scale-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const s = parseInt(btn.dataset.scale, 10) || 1;
      state.config.dither_scale = s;
      document.querySelectorAll("#ditherScaleGroup .scale-chip").forEach(b => b.classList.toggle("active", b === btn));
      triggerProcess(true, true);
      triggerAutosave(true);
    });
  });

  const chkLinear = document.getElementById("chkLinearColorSpace");
  if (chkLinear) {
    chkLinear.addEventListener("change", (e) => {
      state.config.linear_color_space = e.target.checked;
      triggerProcess(true, true);
      triggerAutosave(true);
    });
  }

  const chkAlpha = document.getElementById("chkAlphaStipple");
  if (chkAlpha) {
    chkAlpha.addEventListener("change", (e) => {
      state.config.alpha_stipple = e.target.checked;
      triggerProcess(true, true);
      triggerAutosave(true);
    });
  }

  bindSlider("sliderDitherStrength", "valDitherStrength", (val) => {
    state.config.dither_strength = parseFloat(val);
  });

  bindSlider("sliderGrain", "valGrain", (val) => {
    state.config.grain = parseFloat(val);
  });

  document.getElementById("grainSeedInput").addEventListener("input", (e) => {
    state.config.grain_seed = parseInt(e.target.value, 10) || 0;
    triggerProcess(true, false);
  });
  document.getElementById("grainSeedInput").addEventListener("change", () => {
    triggerAutosave(true);
  });

  document.getElementById("btnRandomSeed").addEventListener("click", () => {
    const seed = Math.floor(Math.random() * 100000);
    document.getElementById("grainSeedInput").value = seed;
    state.config.grain_seed = seed;
    triggerProcess(true, true);
  });

  document.getElementById("chkMonochromeGrain").addEventListener("change", (e) => {
    state.config.grain_monochrome = e.target.checked;
    triggerProcess(true, true);
  });

  // Retro Shader FX & AI Stylization
  bindSlider("sliderKuwahara", "valKuwahara", (val) => {
    state.config.kuwahara_filter = parseFloat(val);
  });

  bindSlider("sliderOutline", "valOutline", (val) => {
    state.config.outline_strength = parseFloat(val);
  });

  bindSlider("sliderEdgeNoiseGate", "valEdgeNoiseGate", (val) => {
    state.config.edge_noise_gate = parseFloat(val);
  });

  bindSlider("sliderPixelOE", "valPixelOE", (val) => {
    state.config.pixeloe_strength = parseFloat(val);
  });

  bindSlider("sliderCelSteps", "valCelSteps", (val) => {
    const v = parseInt(val, 10);
    state.config.cel_shading_steps = v;
    document.getElementById("valCelSteps").textContent = v === 0 ? "Off" : `${v} steps`;
  });

  bindSlider("sliderChromatic", "valChromatic", (val) => {
    state.config.chromatic_aberration = parseFloat(val);
  });

  bindSlider("sliderScanlines", "valScanlines", (val) => {
    state.config.crt_scanlines = parseFloat(val);
  });

  bindSlider("sliderGrime", "valGrime", (val) => {
    state.config.grime_decay = parseFloat(val);
  });

  const grimeMode = document.getElementById("grimeModeSelect");
  if (grimeMode) {
    grimeMode.addEventListener("change", (e) => {
      state.config.grime_mode = e.target.value;
      triggerProcess(true, true);
    });
  }
  const bakeOptical = document.getElementById("chkBakeOpticalFx");
  if (bakeOptical) {
    bakeOptical.addEventListener("change", (e) => {
      state.config.bake_optical_fx = e.target.checked;
      triggerProcess(true, true);
    });
  }

  bindSlider("sliderEdgeCrunch", "valEdgeCrunch", (val) => {
    state.config.edge_crunch = parseFloat(val);
  });

  // Surface Prep
  bindSlider("sliderSmoothDiffuse", "valSmoothDiffuse", (val) => {
    state.config.bilateral_simplify = parseFloat(val);
  });

  bindSlider("sliderPreBlur", "valPreBlur", (val) => {
    state.config.pre_blur = parseFloat(val);
  });

  bindSlider("sliderMidtones", "valMidtones", (val) => {
    state.config.midtones = parseFloat(val);
  });

  bindSlider("sliderHighlights", "valHighlights", (val) => {
    state.config.highlights = parseFloat(val);
  });

  bindSlider("sliderLuminanceThreshold", "valLuminanceThreshold", (val) => {
    state.config.luminance_threshold = parseFloat(val);
  });

  const chkInvert = document.getElementById("chkInvertLuminance");
  if (chkInvert) {
    chkInvert.addEventListener("change", (e) => {
      state.config.invert_luminance = e.target.checked;
      triggerProcess(true, true);
      triggerAutosave(true);
    });
  }

  // Color Grading
  bindSlider("sliderColorTemp", "valColorTemp", (val) => {
    state.config.color_temperature = parseFloat(val);
  });

  bindSlider("sliderHue", "valHue", (val) => {
    state.config.hue = parseFloat(val);
  }, "°");

  bindSlider("sliderSaturation", "valSaturation", (val) => {
    state.config.saturation = parseFloat(val);
  });

  bindSlider("sliderBrightness", "valBrightness", (val) => {
    state.config.brightness = parseFloat(val);
  });

  bindSlider("sliderContrast", "valContrast", (val) => {
    state.config.contrast = parseFloat(val);
  });

  document.getElementById("btnResetColors").addEventListener("click", () => {
    state.config.color_temperature = 0.0;
    state.config.hue = 0.0;
    state.config.saturation = 1.0;
    state.config.brightness = 1.0;
    state.config.contrast = 1.0;
    state.config.midtones = 0.0;
    state.config.highlights = 0.0;
    state.config.luminance_threshold = 0.5;
    state.config.invert_luminance = false;
    state.config.pre_blur = 0.0;
    state.config.bilateral_simplify = 0.0;
    syncUIFromConfig();
    triggerProcess(true, true);
  });

  // ============================================================================
  // Section Randomizers
  // ============================================================================

  // 1. Random Preset
  const btnRandPreset = document.getElementById("btnRandomPreset");
  if (btnRandPreset) {
    btnRandPreset.addEventListener("click", () => {
      const allPresets = [...state.presetsData.configs.custom, ...state.presetsData.configs.builtin];
      if (!allPresets.length) return;
      const pick = allPresets[Math.floor(Math.random() * allPresets.length)];
      const preservedSize = state.lockResolution ? [...state.config.size] : null;
      state.config = { ...state.config, ...pick.config, base_preset: pick.name || pick.id };
      if (preservedSize) {
        state.config.size = preservedSize;
      }
      state.activePresetId = pick.id;
      state.basePresetName = pick.name || pick.id;
      const select = document.getElementById("configPresetSelect");
      if (select) select.value = pick.id;
      syncUIFromConfig();
      triggerProcess(false, false);
      showToast(`🎲 Preset: ${pick.name || pick.id}`);
    });
  }

  // 2. Random Resolution & Format
  const btnRandRes = document.getElementById("btnRandomResolution");
  if (btnRandRes) {
    btnRandRes.addEventListener("click", () => {
      const filters = ["nearest", "k_centroids", "bilinear", "bicubic", "box"];
      if (!state.lockResolution) {
        if (Math.random() < 0.12) {
          state.config.size = [0, 0];
        } else {
          const steps = [];
          for (let s = 32; s <= 512; s += 32) steps.push(s);
          const randSize = steps[Math.floor(Math.random() * steps.length)];
          state.config.size = [randSize, randSize];
        }
      }
      const randFilter = filters[Math.floor(Math.random() * filters.length)];
      state.config.resize_filter = randFilter;
      syncUIFromConfig();
      triggerProcess(true, true);
      const isOrig = !state.config.size || state.config.size[0] <= 0;
      showToast(`🎲 Resolution: ${isOrig ? 'Original 1:1' : state.config.size[0] + 'px'} (${randFilter})`);
    });
  }

  // 3. Random Color (Palette + Grading)
  const btnRandPal = document.getElementById("btnRandomPalette");
  if (btnRandPal) {
    btnRandPal.addEventListener("click", () => {
      const metrics = ["oklab", "cielab", "luma", "euclidean"];
      state.config.color_metric = metrics[Math.floor(Math.random() * metrics.length)];

      const palList = state.presetsData.palettes || [];
      if (palList.length) {
        const pick = palList[Math.floor(Math.random() * palList.length)];
        state.config.palette_preset = pick.id;
        syncUIFromConfig();
        triggerProcess(true, true);
        showToast(`🎲 Palette: ${pick.name}`);
      }

      if (Math.random() < 0.5) {
        state.config.hue = Math.round(Math.random() * 90 - 45);
        state.config.saturation = Math.round((0.85 + Math.random() * 0.7) * 20) / 20;
        state.config.brightness = Math.round((0.9 + Math.random() * 0.25) * 50) / 50;
        state.config.contrast = Math.round((0.95 + Math.random() * 0.35) * 50) / 50;
        state.config.color_temperature = Math.round((Math.random() * 0.8 - 0.4) * 20) / 20;
        state.config.midtones = Math.round((Math.random() * 0.8 - 0.4) * 20) / 20;
        state.config.highlights = Math.round((Math.random() * 0.6 - 0.3) * 20) / 20;
        state.config.luminance_threshold = Math.round((0.35 + Math.random() * 0.3) * 50) / 50;
        syncUIFromConfig();
        triggerProcess(true, true);
        showToast("🎲 Randomized Color Grading");
      }
    });
  }

  // 4. Random Dithering & Noise
  const btnRandDither = document.getElementById("btnRandomDither");
  if (btnRandDither) {
    btnRandDither.addEventListener("click", () => {
      const algos = [
        "blue_noise", "bayer4x4", "yliluoma", "interlaced", "crosshatch",
        "riemersma", "atkinson", "floyd_steinberg", "bayer2x2", "bayer8x8",
        "halftone_dot", "cluster_dot_4x4", "cluster_dot_8x8", "line_halftone",
        "dot_diffusion", "stucki", "burkes", "sierra", "sierra_two_row",
        "sierra_lite", "jarvis_judice_ninke", "false_floyd_steinberg",
        "shiau_fan_1", "shiau_fan_2", "shiau_fan_3", "none",
      ];
      state.config.dither_algorithm = algos[Math.floor(Math.random() * algos.length)];
      state.config.dither_strength = Math.round((0.3 + Math.random() * 0.9) * 20) / 20;
      state.config.dither_scale = Math.random() < 0.25 ? 2 : 1;
      state.config.grain = Math.random() < 0.4 ? 0 : Math.round(Math.random() * 0.12 * 100) / 100;
      state.config.grain_seed = Math.floor(Math.random() * 100000);
      syncUIFromConfig();
      triggerProcess(true, true);
      showToast(`🎲 Dither: ${state.config.dither_algorithm} (${state.config.dither_strength})`);
    });
  }

  // 5. Random Effects
  const btnRandShaders = document.getElementById("btnRandomShaders");
  if (btnRandShaders) {
    btnRandShaders.addEventListener("click", () => {
      state.config.kuwahara_filter = Math.random() < 0.5 ? 0 : Math.round(Math.random() * 0.7 * 20) / 20;
      state.config.pixeloe_strength = Math.random() < 0.45 ? 0 : Math.round(Math.random() * 0.85 * 20) / 20;
      state.config.outline_strength = Math.random() < 0.5 ? 0 : Math.round(Math.random() * 0.7 * 20) / 20;
      state.config.cel_shading_steps = Math.random() < 0.5 ? 0 : Math.floor(Math.random() * 6) + 2;
      state.config.chromatic_aberration = Math.random() < 0.6 ? 0 : Math.round(Math.random() * 0.35 * 20) / 20;
      state.config.crt_scanlines = Math.random() < 0.6 ? 0 : Math.round(Math.random() * 0.4 * 20) / 20;
      state.config.grime_decay = Math.random() < 0.6 ? 0 : Math.round(Math.random() * 0.4 * 20) / 20;
      state.config.edge_crunch = Math.random() < 0.55 ? 0 : Math.round(Math.random() * 0.6 * 20) / 20;
      syncUIFromConfig();
      triggerProcess(true, true);
      showToast("🎲 Randomized Effects");
    });
  }

  // Seamless Tiling
  document.getElementById("chkSeamlessTiling").addEventListener("change", (e) => {
    state.config.seamless_tiling = e.target.checked;
    document.getElementById("seamlessTilingGroup").classList.toggle("hidden", !e.target.checked);
    triggerProcess(true, true);
  });

  bindSlider("sliderSeamSize", "valSeamSize", (val) => {
    state.config.seam_size = parseFloat(val);
  });

  const seamMethod = document.getElementById("seamMethodSelect");
  if (seamMethod) {
    seamMethod.addEventListener("change", (e) => {
      state.config.seam_method = e.target.value;
      triggerProcess(true, true);
    });
  }

  // Zoom Toolbar
  document.getElementById("btnZoomIn").addEventListener("click", () => setZoom(state.zoom * 1.25));
  document.getElementById("btnZoomOut").addEventListener("click", () => setZoom(state.zoom / 1.25));
  document.getElementById("btnZoomReset").addEventListener("click", () => {
    state.zoom = 1.0;
    state.pan = { x: 0, y: 0 };
    updateCanvasTransform();
  });

  // File Input
  const fileInput = document.getElementById("fileInput");
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        loadImageFromFile(e.target.files[0]);
      }
    });
  }

  // Copy JSON & Export
  const copyBtn = document.getElementById("btnCopyJson");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(JSON.stringify(state.config, null, 2));
      showToast("Configuration JSON copied to clipboard!");
    });
  }

  const exportBtn = document.getElementById("btnExportImage");
  if (exportBtn) {
    exportBtn.addEventListener("click", async () => {
      if (!state.processedBlob) return;
      const ext = state.config.export_format === "jpg" ? "jpg" : "png";
      const stem = state.activeFileName ? state.activeFileName.replace(/\.[^/.]+$/, "") : "texture";
      const filename = resolveFilename(stem, ext);
      const outFolder = (document.getElementById("explorerOutFolderInput")?.value.trim()) || state.outputFolder || "./retrofied_textures";

      const doDownload = () => {
        const url = URL.createObjectURL(state.processedBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        showToast(`Exported ${filename}`);
      };

      try {
        const res = await apiFetch("/api/explorer/check-conflicts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_paths: [stem],
            output_folder: outFolder,
            filename_pattern: state.namingPattern || "{name}",
            config: state.config,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.conflicts_count > 0) {
            promptOverwriteConfirmation({
              title: "Export Overwrite Warning",
              subtitle: "A file with this name already exists in your destination folder.",
              total: 1,
              conflicts: data.conflicting_files,
              outputFolder: data.output_folder,
              onConfirm: () => {
                doDownload();
              },
            });
            return;
          }
        }
      } catch (_) {}

      doDownload();
    });
  }
}

function resolveFilename(originalStem, ext = "png") {
  const pattern = state.namingPattern || "{name}";
  const presetName = (state.config.name || state.config.base_preset || "custom")
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "_");
  const sizeStr = (state.config.size && state.config.size[0] > 0)
    ? `${state.config.size[0]}x${state.config.size[1]}`
    : "original";
  const palette = (state.presetsData.palettes || []).find((p) => p.id === state.config.palette_preset);
  const colorsStr = palette ? `${palette.color_count}c` : "palette";
  const ditherStr = state.config.dither_algorithm || "dither";

  let out = pattern
    .replace(/\{name\}/g, originalStem)
    .replace(/\{preset\}/g, presetName)
    .replace(/\{stylename\}/g, presetName)
    .replace(/\{style\}/g, presetName)
    .replace(/\{size\}/g, sizeStr)
    .replace(/\{resolution\}/g, sizeStr)
    .replace(/\{pixel\}/g, sizeStr)
    .replace(/\{colors\}/g, colorsStr)
    .replace(/\{dither\}/g, ditherStr);

  out = out.replace(/[^a-zA-Z0-9_\-\. ]/g, "_").trim() || originalStem;
  if (!out.endsWith(`.${ext}`)) {
    out = `${out}.${ext}`;
  }
  return out;
}

function bindSlider(sliderId, valId, onUpdate, suffix = "", defaultVal = null) {
  const slider = document.getElementById(sliderId);
  const tag = document.getElementById(valId);
  if (!slider || !tag) return;

  const initialVal = defaultVal !== null ? defaultVal : slider.defaultValue || slider.value;

  tag.title = "Click to reset to default";
  tag.addEventListener("click", () => {
    slider.value = initialVal;
    tag.textContent = `${initialVal}${suffix}`;
    onUpdate(initialVal);
    triggerProcess(true, true);
    showToast(`↺ Reset ${sliderId.replace("slider", "")} to ${initialVal}${suffix}`);
  });
  
  // Real-time live preview while dragging (no history snapshot)
  slider.addEventListener("input", (e) => {
    tag.textContent = `${e.target.value}${suffix}`;
    onUpdate(e.target.value);
    triggerProcess(true, false);
  });

  // Commit history snapshot ONLY when user releases slider thumb
  slider.addEventListener("change", () => {
    triggerAutosave(true);
  });
}

function syncUIFromConfig() {
  document.getElementById("resizeFilterSelect").value = state.config.resize_filter;
  document.getElementById("exportFormatSelect").value = state.config.export_format;
  if (state.config.palette_preset) {
    document.getElementById("palettePresetSelect").value = state.config.palette_preset;
    renderPaletteSwatches();
  }

  if (state.config.color_metric) {
    const cm = document.getElementById("colorMetricSelect");
    if (cm) cm.value = state.config.color_metric;
  }

  document.getElementById("ditherAlgoSelect").value = state.config.dither_algorithm;
  updateDitherScaleSupport(state.config.dither_algorithm);
  document.getElementById("sliderDitherStrength").value = state.config.dither_strength;
  document.getElementById("valDitherStrength").textContent = state.config.dither_strength.toFixed(2);

  document.getElementById("sliderGrain").value = state.config.grain;
  document.getElementById("valGrain").textContent = state.config.grain.toFixed(2);
  document.getElementById("grainSeedInput").value = state.config.grain_seed || 0;
  document.getElementById("chkMonochromeGrain").checked = state.config.grain_monochrome;

  const kuwahara = state.config.kuwahara_filter || 0.0;
  const elKuwahara = document.getElementById("sliderKuwahara");
  if (elKuwahara) {
    elKuwahara.value = kuwahara;
    document.getElementById("valKuwahara").textContent = kuwahara.toFixed(2);
  }

  const noiseGate = state.config.edge_noise_gate !== undefined ? state.config.edge_noise_gate : 0.5;
  const elNoiseGate = document.getElementById("sliderEdgeNoiseGate");
  if (elNoiseGate) {
    elNoiseGate.value = noiseGate;
    document.getElementById("valEdgeNoiseGate").textContent = noiseGate.toFixed(2);
  }

  const pixeloe = state.config.pixeloe_strength || 0.0;
  const elPixelOE = document.getElementById("sliderPixelOE");
  if (elPixelOE) {
    elPixelOE.value = pixeloe;
    document.getElementById("valPixelOE").textContent = pixeloe.toFixed(2);
  }

  const outline = state.config.outline_strength || 0.0;
  const elOutline = document.getElementById("sliderOutline");
  if (elOutline) {
    elOutline.value = outline;
    document.getElementById("valOutline").textContent = outline.toFixed(2);
  }

  const bilateral = state.config.bilateral_simplify || 0.0;
  const elSmooth = document.getElementById("sliderSmoothDiffuse");
  if (elSmooth) {
    elSmooth.value = bilateral;
    document.getElementById("valSmoothDiffuse").textContent = bilateral.toFixed(2);
  }

  const preBlur = state.config.pre_blur || 0.0;
  const elPreBlur = document.getElementById("sliderPreBlur");
  if (elPreBlur) {
    elPreBlur.value = preBlur;
    document.getElementById("valPreBlur").textContent = preBlur.toFixed(2);
  }

  const midtones = state.config.midtones || 0.0;
  const elMidtones = document.getElementById("sliderMidtones");
  if (elMidtones) {
    elMidtones.value = midtones;
    document.getElementById("valMidtones").textContent = midtones.toFixed(2);
  }

  const highlights = state.config.highlights || 0.0;
  const elHighlights = document.getElementById("sliderHighlights");
  if (elHighlights) {
    elHighlights.value = highlights;
    document.getElementById("valHighlights").textContent = highlights.toFixed(2);
  }

  const lumThresh = state.config.luminance_threshold ?? 0.5;
  const elLumThresh = document.getElementById("sliderLuminanceThreshold");
  if (elLumThresh) {
    elLumThresh.value = lumThresh;
    document.getElementById("valLuminanceThreshold").textContent = lumThresh.toFixed(2);
  }

  const elInvert = document.getElementById("chkInvertLuminance");
  if (elInvert) elInvert.checked = !!state.config.invert_luminance;

  const elLinear = document.getElementById("chkLinearColorSpace");
  if (elLinear) elLinear.checked = state.config.linear_color_space !== false;

  const elAlphaStipple = document.getElementById("chkAlphaStipple");
  if (elAlphaStipple) elAlphaStipple.checked = state.config.alpha_stipple !== false;

  const currentScale = state.config.dither_scale || 1;
  document.querySelectorAll("#ditherScaleGroup .scale-chip").forEach((btn) => {
    btn.classList.toggle("active", parseInt(btn.dataset.scale, 10) === currentScale);
  });

  const celSteps = state.config.cel_shading_steps || 0;
  document.getElementById("sliderCelSteps").value = celSteps;
  document.getElementById("valCelSteps").textContent = celSteps === 0 ? "Off" : `${celSteps} steps`;

  document.getElementById("sliderChromatic").value = state.config.chromatic_aberration;
  document.getElementById("valChromatic").textContent = state.config.chromatic_aberration.toFixed(2);

  document.getElementById("sliderScanlines").value = state.config.crt_scanlines;
  document.getElementById("valScanlines").textContent = state.config.crt_scanlines.toFixed(2);

  document.getElementById("sliderGrime").value = state.config.grime_decay;
  document.getElementById("valGrime").textContent = state.config.grime_decay.toFixed(2);

  const elGrimeMode = document.getElementById("grimeModeSelect");
  if (elGrimeMode) elGrimeMode.value = state.config.grime_mode || "uniform";
  const elBakeOptical = document.getElementById("chkBakeOpticalFx");
  if (elBakeOptical) elBakeOptical.checked = !!state.config.bake_optical_fx;

  document.getElementById("sliderEdgeCrunch").value = state.config.edge_crunch;
  document.getElementById("valEdgeCrunch").textContent = state.config.edge_crunch.toFixed(2);

  const colorTemp = state.config.color_temperature || 0.0;
  document.getElementById("sliderColorTemp").value = colorTemp;
  document.getElementById("valColorTemp").textContent = colorTemp.toFixed(2);

  document.getElementById("sliderHue").value = state.config.hue;
  document.getElementById("valHue").textContent = `${Math.round(state.config.hue)}°`;

  document.getElementById("sliderSaturation").value = state.config.saturation;
  document.getElementById("valSaturation").textContent = state.config.saturation.toFixed(2);

  document.getElementById("sliderBrightness").value = state.config.brightness;
  document.getElementById("valBrightness").textContent = state.config.brightness.toFixed(2);

  document.getElementById("sliderContrast").value = state.config.contrast;
  document.getElementById("valContrast").textContent = state.config.contrast.toFixed(2);

  document.getElementById("chkSeamlessTiling").checked = state.config.seamless_tiling;
  document.getElementById("seamlessTilingGroup").classList.toggle("hidden", !state.config.seamless_tiling);
  document.getElementById("sliderSeamSize").value = state.config.seam_size;
  document.getElementById("valSeamSize").textContent = state.config.seam_size.toFixed(2);
  const elSeamMethod = document.getElementById("seamMethodSelect");
  if (elSeamMethod) elSeamMethod.value = state.config.seam_method || "offset_wrap";

  const isOriginal = !state.config.size || state.config.size[0] <= 0;
  const sliderWrap = document.getElementById("resolutionSliderWrap");
  const sliderResolution = document.getElementById("sliderResolution");
  const valResolution = document.getElementById("valResolution");
  const btnOriginal = document.getElementById("btnOriginalSize");

  if (isOriginal) {
    if (sliderWrap) sliderWrap.classList.add("dimmed");
    if (btnOriginal) btnOriginal.classList.add("active");
    if (valResolution) valResolution.textContent = "Native 1:1";
  } else {
    if (sliderWrap) sliderWrap.classList.remove("dimmed");
    if (btnOriginal) btnOriginal.classList.remove("active");
    const currentSize = state.config.size[0];
    if (sliderResolution) {
      const snapped = Math.max(32, Math.min(512, Math.round(currentSize / 32) * 32));
      sliderResolution.value = snapped;
      if (valResolution) valResolution.textContent = `${snapped}px`;
    }
  }
}

// ============================================================================
// Save Preset Modal
// ============================================================================

function initSavePresetModal() {
  const modal = document.getElementById("savePresetModal");
  const openBtn = document.getElementById("btnSavePresetModal");
  const closeBtn = document.getElementById("btnCloseSavePresetModal");
  const cancelBtn = document.getElementById("btnCancelSavePreset");
  const confirmBtn = document.getElementById("btnConfirmSavePreset");
  const nameInput = document.getElementById("presetNameInput");

  openBtn.addEventListener("click", () => {
    const rawBase = (state.basePresetName || state.config.palette_preset || "preset")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .toLowerCase();

    // Check custom presets to find next version number (e.g. ps1_classic_2, etc.)
    const customList = state.presetsData.configs.custom || [];
    let nextName = `${rawBase}_2`;
    let counter = 2;
    while (customList.some(c => c.id.toLowerCase() === nextName.toLowerCase() || (c.name && c.name.toLowerCase() === nextName.toLowerCase()))) {
      counter++;
      nextName = `${rawBase}_${counter}`;
    }

    nameInput.value = nextName;
    modal.classList.remove("hidden");
    nameInput.focus();
    nameInput.select();
  });

  const closeModal = () => modal.classList.add("hidden");
  closeBtn.addEventListener("click", closeModal);
  cancelBtn.addEventListener("click", closeModal);

  confirmBtn.addEventListener("click", async () => {
    const name = nameInput.value.trim();
    if (!name) return;

    try {
      const res = await apiFetch("/api/presets/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, config: state.config }),
      });
      if (!res.ok) throw new Error("Failed to save preset");
      const data = await res.json();
      state.presetsData.configs = data.configs;
      state.activePresetId = data.name;
      state.basePresetName = data.name;
      populatePresetDropdowns();
      document.getElementById("configPresetSelect").value = data.name;
      closeModal();
      showToast(`✓ Preset saved as custom_presets/${data.name}.json`);
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  });
}

// ============================================================================
// Mass Batch Converter Modal
// ============================================================================

function initBatchModal() {
  const modal = document.getElementById("batchModal");
  const openBtn = document.getElementById("btnOpenBatchModal");
  const closeBtn = document.getElementById("btnCloseBatchModal");
  const cancelBtn = document.getElementById("btnCancelBatch");
  const runBtn = document.getElementById("btnRunBatch");
  const fileInput = document.getElementById("batchFilesInput");
  const statusArea = document.getElementById("batchStatusArea");
  const progBar = document.getElementById("batchProgressBar");
  const logBox = document.getElementById("batchLog");

  if (!modal || !openBtn) return;

  openBtn.addEventListener("click", () => {
    statusArea.classList.add("hidden");
    progBar.style.width = "0%";
    logBox.innerHTML = "";
    modal.classList.remove("hidden");
  });

  const closeModal = () => modal.classList.add("hidden");
  closeBtn.addEventListener("click", closeModal);
  cancelBtn.addEventListener("click", closeModal);

  runBtn.addEventListener("click", async () => {
    const files = fileInput.files;
    if (!files || files.length === 0) {
      alert("Please select one or more image files first.");
      return;
    }

    statusArea.classList.remove("hidden");
    progBar.style.width = "20%";
    logBox.innerHTML = `<div>Sending ${files.length} textures to converter...</div>`;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("images", files[i]);
    }
    formData.append("config_json", JSON.stringify(state.config));

    try {
      const res = await apiFetch("/api/batch-convert", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Batch conversion request failed");
      const data = await res.json();

      progBar.style.width = "100%";
      logBox.innerHTML += `<div class="state-success">✓ Successfully converted ${Number(data.count) || 0} textures!</div>`;

      // Trigger automatic downloads for converted textures
      data.results.forEach((item) => {
        const a = document.createElement("a");
        a.href = item.data_url;
        a.download = item.filename;
        a.click();
      });

      showToast(`Mass conversion completed for ${data.count} textures!`);
    } catch (err) {
      logBox.innerHTML += `<div class="state-error">Error: ${escapeHtml(err.message)}</div>`;
    }
  });
}

// ============================================================================
// Image & Texture Loaders
// ============================================================================

function loadImageFromFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      onOriginalImageLoaded(img);
      triggerProcess();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function loadDefaultStartupTexture() {
  const lastFilePath = localStorage.getItem("retexture_last_active_file");
  const img = new Image();

  img.onload = () => {
    onOriginalImageLoaded(img);
    if (lastFilePath) {
      state.activeFilePath = lastFilePath;
      state.activeFileName = lastFilePath.split("/").pop() || "";
    }
    triggerProcess(false, false);
  };

  img.onerror = () => {
    // If loading last file fails (e.g. file moved or deleted), fallback to builtin sample
    const fallbackImg = new Image();
    fallbackImg.onload = () => {
      onOriginalImageLoaded(fallbackImg);
      state.activeFilePath = null;
      state.activeFileName = "default_sample.png";
      triggerProcess(false, false);
    };
    fallbackImg.onerror = () => {
      // Emergency procedural canvas
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#4a4e5a";
      ctx.fillRect(0, 0, 256, 256);
      const emergencyImg = new Image();
      emergencyImg.onload = () => {
        onOriginalImageLoaded(emergencyImg);
        triggerProcess(false, false);
      };
      emergencyImg.src = canvas.toDataURL("image/png");
    };
    fallbackImg.src = getAuthenticatedSampleUrl();
  };

  if (lastFilePath) {
    img.src = getAuthenticatedFileUrl(lastFilePath);
  } else {
    img.src = getAuthenticatedSampleUrl();
  }
}

// ============================================================================
// Auto-Rename Naming Pattern Modal
// ============================================================================

function initNamingPatternModal() {
  const modal = document.getElementById("namingPatternModal");
  const openBtn = document.getElementById("btnOpenNamingModal");
  const closeBtn = document.getElementById("btnCloseNamingModal");
  const resetBtn = document.getElementById("btnResetNamingPattern");
  const saveBtn = document.getElementById("btnSaveNamingPattern");
  const patternInput = document.getElementById("namingPatternInput");
  const livePreview = document.getElementById("namingLivePreview");
  const badge = document.getElementById("currentPatternBadge");

  if (!modal) return;

  const updatePreview = (pat) => {
    const template = pat || "{name}";
    const presetName = (state.config.name || "psx_horror").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const sizeStr = `${state.config.size[0]}x${state.config.size[1]}`;
    const palette = (state.presetsData.palettes || []).find((p) => p.id === state.config.palette_preset);
    const colorsStr = palette ? `${palette.color_count}c` : "palette";
    const ditherStr = state.config.dither_algorithm || "bayer4x4";

    let res = template
      .replace(/\{name\}/g, "wall_stone")
      .replace(/\{preset\}/g, presetName)
      .replace(/\{stylename\}/g, presetName)
      .replace(/\{style\}/g, presetName)
      .replace(/\{size\}/g, sizeStr)
      .replace(/\{resolution\}/g, sizeStr)
      .replace(/\{pixel\}/g, sizeStr)
      .replace(/\{colors\}/g, colorsStr)
      .replace(/\{dither\}/g, ditherStr);
    
    res = res.replace(/[^a-zA-Z0-9_\-\. ]/g, "_").trim() || "wall_stone";
    if (!res.endsWith(".png")) res += ".png";
    if (livePreview) livePreview.textContent = res;
  };

  if (openBtn) {
    openBtn.addEventListener("click", () => {
      patternInput.value = state.namingPattern || "{name}";
      updatePreview(patternInput.value);
      modal.classList.remove("hidden");
    });
  }

  const closeModal = () => modal.classList.add("hidden");
  if (closeBtn) closeBtn.addEventListener("click", closeModal);

  if (patternInput) {
    patternInput.addEventListener("input", (e) => {
      updatePreview(e.target.value);
    });
  }

  // Token chip click: appends token to input
  document.querySelectorAll(".token-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const token = chip.dataset.token;
      patternInput.value = (patternInput.value ? `${patternInput.value}_` : "") + token;
      updatePreview(patternInput.value);
    });
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      patternInput.value = "{name}";
      updatePreview("{name}");
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", () => {
      const val = patternInput.value.trim() || "{name}";
      state.namingPattern = val;
      localStorage.setItem("retexture_filename_pattern", val);
      if (badge) badge.textContent = val;
      closeModal();
      showToast(`Naming Pattern set to: ${val}`);
    });
  }

  if (badge) {
    badge.textContent = state.namingPattern || "{name}";
  }
}

// ============================================================================
// Destination Save Folder Browser Modal
// ============================================================================

let currentSaveBrowserPath = "";
let saveBrowserParentPath = null;

function initSaveFolderBrowserModal() {
  const modal = document.getElementById("saveFolderModal");
  const btnOpen = document.getElementById("btnBrowseOutFolder");
  const btnClose = document.getElementById("btnCloseSaveFolderModal");
  const btnCancel = document.getElementById("btnCancelSaveFolder");
  const btnConfirm = document.getElementById("btnConfirmSaveFolder");
  const pathInput = document.getElementById("saveBrowserPathInput");
  const btnGo = document.getElementById("btnSaveBrowserGo");
  const btnUp = document.getElementById("btnSaveBrowserUp");
  const btnRefresh = document.getElementById("btnSaveBrowserRefresh");
  const btnNewFolder = document.getElementById("btnSaveBrowserNewFolder");
  const quickLocationsWrap = document.getElementById("saveBrowserQuickLocations");
  const outFolderInput = document.getElementById("explorerOutFolderInput");

  if (!modal || !btnOpen) return;

  btnOpen.addEventListener("click", () => {
    modal.classList.remove("hidden");
    loadQuickLocations();
    const currentPath = outFolderInput.value.trim() || ".";
    loadSaveBrowserDirectory(currentPath);
  });

  if (btnClose) btnClose.addEventListener("click", () => modal.classList.add("hidden"));
  if (btnCancel) btnCancel.addEventListener("click", () => modal.classList.add("hidden"));

  if (btnGo) {
    btnGo.addEventListener("click", () => {
      loadSaveBrowserDirectory(pathInput.value.trim());
    });
  }

  if (pathInput) {
    pathInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        loadSaveBrowserDirectory(pathInput.value.trim());
      }
    });
  }

  if (btnUp) {
    btnUp.addEventListener("click", () => {
      if (saveBrowserParentPath) {
        loadSaveBrowserDirectory(saveBrowserParentPath);
      }
    });
  }

  if (btnRefresh) {
    btnRefresh.addEventListener("click", () => {
      loadSaveBrowserDirectory(currentSaveBrowserPath || pathInput.value.trim() || ".");
    });
  }

  if (btnNewFolder) {
    btnNewFolder.addEventListener("click", async () => {
      const folderName = prompt("Enter name for the new folder:");
      if (!folderName || !folderName.trim()) return;
      const cleanName = folderName.trim().replace(/[\\/]/g, "_");
      const newFolderPath = `${currentSaveBrowserPath}/${cleanName}`;
      try {
        const res = await apiFetch("/api/explorer/create-folder", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_path: newFolderPath }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Failed to create folder");
        }
        showToast(`Created folder "${cleanName}"`);
        loadSaveBrowserDirectory(newFolderPath);
      } catch (err) {
        showToast(`Error: ${err.message}`);
      }
    });
  }

  if (btnConfirm) {
    btnConfirm.addEventListener("click", () => {
      if (currentSaveBrowserPath) {
        state.outputFolder = currentSaveBrowserPath;
        if (outFolderInput) outFolderInput.value = currentSaveBrowserPath;
        localStorage.setItem("retexture_last_save_path", currentSaveBrowserPath);
        modal.classList.add("hidden");
        showToast(`Save destination set to: ${currentSaveBrowserPath}`);
      }
    });
  }

  async function loadQuickLocations() {
    try {
      const res = await apiFetch("/api/explorer/quick-locations");
      if (!res.ok) return;
      const data = await res.json();
      if (!quickLocationsWrap) return;
      quickLocationsWrap.innerHTML = "";
      data.locations.forEach((loc) => {
        const chip = document.createElement("button");
        chip.className = "quick-loc-chip";
        chip.textContent = loc.name;
        chip.title = loc.path;
        chip.addEventListener("click", () => {
          loadSaveBrowserDirectory(loc.path);
        });
        quickLocationsWrap.appendChild(chip);
      });
    } catch (_) {}
  }
}

async function loadSaveBrowserDirectory(path) {
  const fileList = document.getElementById("saveBrowserFileList");
  const pathInput = document.getElementById("saveBrowserPathInput");
  const currentBadge = document.getElementById("saveBrowserCurrentPath");
  if (!fileList) return;

  fileList.innerHTML = `<div class="state-msg">Loading directory...</div>`;

  try {
    const res = await apiFetch(`/api/explorer/browse?path=${encodeURIComponent(path || ".")}&all_files=true`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to load directory");
    }
    const data = await res.json();
    currentSaveBrowserPath = data.current_path;
    saveBrowserParentPath = data.parent_path;

    if (pathInput) pathInput.value = data.current_path;
    if (currentBadge) currentBadge.textContent = data.current_path;

    fileList.innerHTML = "";

    if (data.directories.length === 0 && data.files.length === 0) {
      fileList.innerHTML = `<div class="state-empty">(Empty folder)</div>`;
      return;
    }

    // 1. Render Subdirectories
    data.directories.forEach((dir) => {
      const item = document.createElement("div");
      item.className = "save-browser-item";
      item.innerHTML = `
        <span class="save-browser-item-icon save-browser-item-dir">${getIcon("folder", "icon-sm")}</span>
        <span class="save-browser-item-name">${escapeHtml(dir.name)}</span>
        <span class="save-browser-item-meta">folder</span>
      `;
      item.addEventListener("click", () => {
        loadSaveBrowserDirectory(dir.path);
      });
      fileList.appendChild(item);
    });

    // 2. Render Files (Read-only orientation display without loading payloads)
    data.files.forEach((file) => {
      const item = document.createElement("div");
      item.className = "save-browser-item is-file";
      item.innerHTML = `
        <span class="save-browser-item-icon save-browser-item-file">${getIcon("file", "icon-sm")}</span>
        <span class="save-browser-item-name">${escapeHtml(file.name)}</span>
        <span class="save-browser-item-meta">${escapeHtml(String(file.size_kb))} KB</span>
      `;
      fileList.appendChild(item);
    });

  } catch (err) {
    fileList.innerHTML = `<div class="state-error">${escapeHtml(err.message)}</div>`;
    showToast(`Error: ${err.message}`);
  }
}

// ============================================================================
// Overwrite Conflict Warning Modal
// ============================================================================

let pendingOverwriteCallback = null;

function initOverwriteModal() {
  const modal = document.getElementById("overwriteModal");
  const closeBtn = document.getElementById("btnCloseOverwriteModal");
  const cancelBtn = document.getElementById("btnCancelOverwrite");
  const confirmBtn = document.getElementById("btnConfirmOverwrite");
  const changePatternBtn = document.getElementById("btnOverwriteChangePattern");
  const changeFolderBtn = document.getElementById("btnOverwriteChangeFolder");

  if (!modal) return;

  const closeModal = () => {
    modal.classList.add("hidden");
    pendingOverwriteCallback = null;
  };

  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeModal);

  if (confirmBtn) {
    confirmBtn.addEventListener("click", () => {
      const cb = pendingOverwriteCallback;
      closeModal();
      if (cb) cb();
    });
  }

  if (changePatternBtn) {
    changePatternBtn.addEventListener("click", () => {
      closeModal();
      const openNamingBtn = document.getElementById("btnOpenNamingModal");
      if (openNamingBtn) openNamingBtn.click();
    });
  }

  if (changeFolderBtn) {
    changeFolderBtn.addEventListener("click", () => {
      closeModal();
      const openFolderBtn = document.getElementById("btnBrowseOutFolder");
      if (openFolderBtn) openFolderBtn.click();
    });
  }
}

function promptOverwriteConfirmation({ title, subtitle, total, conflicts, outputFolder, onConfirm }) {
  const modal = document.getElementById("overwriteModal");
  if (!modal) {
    if (confirm(`${conflicts.length} of ${total} files already exist in ${outputFolder}. Overwrite?`)) {
      onConfirm();
    }
    return;
  }

  document.getElementById("overwriteModalTitle").textContent = title || "Existing Files Detected";
  document.getElementById("overwriteModalSubtitle").textContent = subtitle || "Files with the same name already exist.";
  document.getElementById("overwriteCountSummary").textContent = `${conflicts.length} of ${total} files already exist`;
  document.getElementById("overwriteTargetFolder").textContent = outputFolder;
  document.getElementById("btnConfirmOverwriteText").textContent = `Overwrite ${conflicts.length} File${conflicts.length > 1 ? "s" : ""}`;

  const list = document.getElementById("overwriteFilesList");
  list.innerHTML = "";
  conflicts.forEach((name) => {
    const item = document.createElement("div");
    item.className = "overwrite-file-item";
    item.innerHTML = `<svg class="icon icon-xs" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg><span>${escapeHtml(name)}</span>`;
    list.appendChild(item);
  });

  pendingOverwriteCallback = onConfirm;
  modal.classList.remove("hidden");
}

// ============================================================================
// Collapsible Panels & Resizable Sidebars Controller
// ============================================================================

function initPanelToggles() {
  const expBtn = document.getElementById("btnToggleExplorer");
  const expSidebar = document.getElementById("explorerSidebar");
  const resizerLeft = document.getElementById("resizerLeft");

  // Restore saved Left Sidebar collapsed state
  const leftCollapsed = localStorage.getItem("retexture_sidebar_left_collapsed") === "true";
  if (leftCollapsed && expSidebar) {
    expSidebar.classList.add("collapsed");
    if (expBtn) expBtn.classList.remove("active");
    if (resizerLeft) resizerLeft.classList.add("hidden");
  }

  if (expBtn && expSidebar) {
    expBtn.addEventListener("click", () => {
      const isCollapsed = expSidebar.classList.toggle("collapsed");
      expBtn.classList.toggle("active", !isCollapsed);
      if (resizerLeft) resizerLeft.classList.toggle("hidden", isCollapsed);
      localStorage.setItem("retexture_sidebar_left_collapsed", isCollapsed);
    });
  }

  const setBtn = document.getElementById("btnToggleSettings");
  const setSidebar = document.getElementById("controlsSidebar");
  const resizerRight = document.getElementById("resizerRight");

  // Restore saved Right Sidebar collapsed state
  const rightCollapsed = localStorage.getItem("retexture_sidebar_right_collapsed") === "true";
  if (rightCollapsed && setSidebar) {
    setSidebar.classList.add("collapsed");
    if (setBtn) setBtn.classList.remove("active");
    if (resizerRight) resizerRight.classList.add("hidden");
  }

  if (setBtn && setSidebar) {
    setBtn.addEventListener("click", () => {
      const isCollapsed = setSidebar.classList.toggle("collapsed");
      setBtn.classList.toggle("active", !isCollapsed);
      if (resizerRight) resizerRight.classList.toggle("hidden", isCollapsed);
      localStorage.setItem("retexture_sidebar_right_collapsed", isCollapsed);
    });
  }
}

function initSidebarResizers() {
  const resizerLeft = document.getElementById("resizerLeft");
  const resizerRight = document.getElementById("resizerRight");
  const expSidebar = document.getElementById("explorerSidebar");
  const setSidebar = document.getElementById("controlsSidebar");
  const mainBody = document.querySelector(".main-body");

  // Restore saved widths from localStorage
  const savedLeft = localStorage.getItem("retexture_sidebar_left_width");
  if (savedLeft && expSidebar) {
    expSidebar.style.width = `${savedLeft}px`;
  }
  const savedRight = localStorage.getItem("retexture_sidebar_right_width");
  if (savedRight && setSidebar) {
    setSidebar.style.width = `${savedRight}px`;
  }

  // Left Sidebar Resizer
  if (resizerLeft && expSidebar && mainBody) {
    let isDraggingLeft = false;

    resizerLeft.addEventListener("mousedown", (e) => {
      isDraggingLeft = true;
      resizerLeft.classList.add("active");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDraggingLeft) return;
      const rect = mainBody.getBoundingClientRect();
      let newW = e.clientX - rect.left;
      newW = Math.max(180, Math.min(650, newW));
      expSidebar.style.width = `${newW}px`;
    });

    window.addEventListener("mouseup", () => {
      if (isDraggingLeft) {
        isDraggingLeft = false;
        resizerLeft.classList.remove("active");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        const w = parseInt(expSidebar.style.width, 10);
        if (w) localStorage.setItem("retexture_sidebar_left_width", w);
      }
    });
  }

  // Right Sidebar Resizer
  if (resizerRight && setSidebar && mainBody) {
    let isDraggingRight = false;

    resizerRight.addEventListener("mousedown", (e) => {
      isDraggingRight = true;
      resizerRight.classList.add("active");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDraggingRight) return;
      const rect = mainBody.getBoundingClientRect();
      let newW = rect.right - e.clientX;
      newW = Math.max(220, Math.min(650, newW));
      setSidebar.style.width = `${newW}px`;
    });

    window.addEventListener("mouseup", () => {
      if (isDraggingRight) {
        isDraggingRight = false;
        resizerRight.classList.remove("active");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        const w = parseInt(setSidebar.style.width, 10);
        if (w) localStorage.setItem("retexture_sidebar_right_width", w);
      }
    });
  }
}

// ============================================================================
// Left Asset Explorer & Direct Disk Batch Converter
// ============================================================================

function initAssetExplorer() {
  const pathInput = document.getElementById("explorerPathInput");
  const goBtn = document.getElementById("btnGoPath");
  const searchInput = document.getElementById("explorerSearchInput");
  const refreshBtn = document.getElementById("btnRefreshExplorer");
  const upBtn = document.getElementById("btnUpFolder");
  const selectAllChk = document.getElementById("chkSelectAllFiles");
  const convertBtn = document.getElementById("btnExplorerConvertBatch");
  const outFolderInput = document.getElementById("explorerOutFolderInput");

  // Load saved output folder
  if (outFolderInput) {
    outFolderInput.value = state.outputFolder || "./retrofied_textures";
    outFolderInput.addEventListener("input", (e) => {
      state.outputFolder = e.target.value.trim();
      localStorage.setItem("retexture_last_save_path", state.outputFolder);
    });
  }

  // Direct Path Input & Go Button
  if (pathInput) {
    pathInput.value = state.explorerPath || ".";
    pathInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadExplorerDirectory(pathInput.value.trim());
      }
    });
  }

  // Load Quick Common Locations into Sidebar
  async function loadExplorerQuickLocations() {
    const wrap = document.getElementById("explorerQuickLocations");
    if (!wrap) return;
    try {
      const res = await apiFetch("/api/explorer/quick-locations");
      if (!res.ok) return;
      const data = await res.json();
      wrap.innerHTML = "";
      data.locations.forEach((loc) => {
        const chip = document.createElement("button");
        chip.className = "quick-loc-chip";
        chip.textContent = loc.name;
        chip.title = loc.path;
        chip.addEventListener("click", () => {
          loadExplorerDirectory(loc.path);
        });
        wrap.appendChild(chip);
      });
    } catch (_) {}
  }
  loadExplorerQuickLocations();

  if (goBtn) {
    goBtn.addEventListener("click", () => {
      if (pathInput) {
        loadExplorerDirectory(pathInput.value.trim());
      }
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      renderExplorerFiles();
    });
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      loadExplorerDirectory(state.explorerPath);
    });
  }

  if (upBtn) {
    upBtn.addEventListener("click", () => {
      if (state.explorerParentPath) {
        loadExplorerDirectory(state.explorerParentPath);
      }
    });
  }

  if (selectAllChk) {
    selectAllChk.addEventListener("change", (e) => {
      const checked = e.target.checked;
      const query = (searchInput ? searchInput.value : "").trim().toLowerCase();
      const files = state.explorerFiles.filter((f) => !query || f.name.toLowerCase().includes(query));

      if (checked) {
        files.forEach((f) => state.selectedFiles.add(f.path));
      } else {
        files.forEach((f) => state.selectedFiles.delete(f.path));
      }
      updateBatchControls();
      renderExplorerFiles();
    });
  }

  if (convertBtn) {
    convertBtn.addEventListener("click", async () => {
      const outFolder = (outFolderInput ? outFolderInput.value.trim() : "") || "./retrofied_textures";
      const filesToConvert = Array.from(state.selectedFiles);
      if (filesToConvert.length === 0) return;

      // 1. First check if any resolved output filenames already exist on disk
      try {
        const checkRes = await apiFetch("/api/explorer/check-conflicts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_paths: filesToConvert,
            output_folder: outFolder,
            filename_pattern: state.namingPattern || "{name}",
            config: state.config,
          }),
        });

        if (checkRes.ok) {
          const checkData = await checkRes.json();
          if (checkData.conflicts_count > 0) {
            promptOverwriteConfirmation({
              title: "Batch Export Overwrite Warning",
              subtitle: "Files with identical output names already exist in your destination folder.",
              total: checkData.total,
              conflicts: checkData.conflicting_files,
              outputFolder: checkData.output_folder,
              onConfirm: () => {
                executeBatchConvert(filesToConvert, outFolder);
              },
            });
            return;
          }
        }
      } catch (_) {}

      // If no conflicts or check skipped, proceed immediately
      executeBatchConvert(filesToConvert, outFolder);
    });
  }

  // Initial browse: restore last explorer directory if saved
  const savedExpPath = localStorage.getItem("retexture_last_explorer_path") || ".";
  loadExplorerDirectory(savedExpPath);
}

async function executeBatchConvert(filesToConvert, outFolder) {
  const convertBtn = document.getElementById("btnExplorerConvertBatch");
  const progressArea = document.getElementById("explorerBatchProgress");
  const progressBar = document.getElementById("explorerProgressBar");
  const statusText = document.getElementById("explorerBatchStatusText");

  progressArea.classList.remove("hidden");
  progressBar.style.width = "25%";
  statusText.textContent = `Converting ${filesToConvert.length} textures...`;
  if (convertBtn) convertBtn.disabled = true;

  try {
    const res = await apiFetch("/api/explorer/batch-convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_paths: filesToConvert,
        output_folder: outFolder,
        config: state.config,
        filename_pattern: state.namingPattern || "{name}",
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Batch conversion failed");
    }
    const data = await res.json();

    progressBar.style.width = "100%";
    statusText.textContent = `✓ Done! Converted ${data.count} textures`;
    showToast(`✓ Converted ${data.count} textures to ${outFolder}`);
    setTimeout(() => {
      progressArea.classList.add("hidden");
    }, 3500);
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
    showToast(`Batch Error: ${err.message}`);
  } finally {
    if (convertBtn) convertBtn.disabled = false;
  }
}

async function loadExplorerDirectory(path) {
  try {
    const res = await apiFetch(`/api/explorer/browse?path=${encodeURIComponent(path || ".")}`);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Failed to explore directory");
    }
    const data = await res.json();

    state.explorerPath = data.current_path;
    state.explorerParentPath = data.parent_path;
    state.explorerDirectories = data.directories || [];
    state.explorerFiles = data.files || [];

    localStorage.setItem("retexture_last_explorer_path", data.current_path);

    // Update path input box with current path
    const pathInput = document.getElementById("explorerPathInput");
    if (pathInput) {
      pathInput.value = data.current_path;
      pathInput.title = data.current_path;
    }

    const upBtn = document.getElementById("btnUpFolder");
    if (upBtn) {
      upBtn.disabled = !data.parent_path;
    }

    renderExplorerFiles();
  } catch (err) {
    showToast(`Explorer: ${err.message}`);
  }
}

function renderExplorerFiles() {
  const listEl = document.getElementById("explorerFileList");
  const searchInput = document.getElementById("explorerSearchInput");
  if (!listEl) return;

  listEl.innerHTML = "";
  const query = (searchInput ? searchInput.value : "").trim().toLowerCase();

  // 1. Render Subdirectories
  state.explorerDirectories.forEach((dir) => {
    if (query && !dir.name.toLowerCase().includes(query)) return;
    const item = document.createElement("div");
    item.className = "explorer-item";
    item.innerHTML = `
      <span class="explorer-item-dir">${getIcon("folder", "icon-sm")}</span>
      <span class="explorer-item-name explorer-dir-label">${escapeHtml(dir.name)}</span>
    `;
    item.addEventListener("click", () => {
      loadExplorerDirectory(dir.path);
    });
    listEl.appendChild(item);
  });

  // 2. Render Texture Files
  const matchedFiles = state.explorerFiles.filter((f) => !query || f.name.toLowerCase().includes(query));

  if (matchedFiles.length === 0 && state.explorerDirectories.length === 0) {
    listEl.innerHTML = `<div class="state-msg">No image textures found in this folder.</div>`;
    updateBatchControls();
    return;
  }

  matchedFiles.forEach((file) => {
    const isSelected = state.selectedFiles.has(file.path);
    const isActive = state.activeFilePath === file.path;

    const item = document.createElement("div");
    item.className = `explorer-item ${isActive ? "active" : ""}`;
    item.innerHTML = `
      <input type="checkbox" class="explorer-chk" ${isSelected ? "checked" : ""} />
      <img src="${escapeHtml(getAuthenticatedFileUrl(file.path))}" class="explorer-item-thumb" alt="${escapeHtml(file.name)}" loading="lazy" />
      <div class="explorer-item-info">
        <span class="explorer-item-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
        <span class="explorer-item-meta">${escapeHtml(file.dimensions || "")} • ${escapeHtml(String(file.size_kb))} KB</span>
      </div>
    `;

    // Checkbox toggle
    const chk = item.querySelector(".explorer-chk");
    chk.addEventListener("click", (e) => {
      e.stopPropagation();
      if (chk.checked) {
        state.selectedFiles.add(file.path);
      } else {
        state.selectedFiles.delete(file.path);
      }
      updateBatchControls();
    });

    // Item click -> Load as active texture
    item.addEventListener("click", () => {
      state.activeFilePath = file.path;
      state.activeFileName = file.name;
      localStorage.setItem("retexture_last_active_file", file.path);
      const img = new Image();
      img.onload = () => {
        onOriginalImageLoaded(img);
        renderExplorerFiles();
        triggerProcess(false, false);
      };
      img.src = getAuthenticatedFileUrl(file.path);
    });

    listEl.appendChild(item);
  });

  updateBatchControls();
}

function updateBatchControls() {
  const count = state.selectedFiles.size;
  const countBadge = document.getElementById("selectedCountText");
  const convertBtn = document.getElementById("btnExplorerConvertBatch");
  const selectAllChk = document.getElementById("chkSelectAllFiles");

  if (countBadge) countBadge.textContent = `${count} selected`;
  if (convertBtn) {
    const btnText = document.getElementById("btnExplorerConvertBatchText");
    if (btnText) btnText.textContent = `Convert Selected (${count})`;
    convertBtn.disabled = count === 0;
  }

  if (selectAllChk) {
    const total = state.explorerFiles.length;
    selectAllChk.checked = total > 0 && count === total;
  }
}

function showToast(msg) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}

// ============================================================================
// Presets & Autosave History Manager Modal
// ============================================================================

function initPresetsManagerModal() {
  const modal = document.getElementById("presetsManagerModal");
  const openBtn = document.getElementById("btnOpenPresetsModal");
  const closeBtn = document.getElementById("btnClosePresetsModal");
  const closeBtn2 = document.getElementById("btnClosePresetsManager");
  const clearBtn = document.getElementById("btnClearAutosaves");
  const saveNewBtn = document.getElementById("btnModalSaveNew");

  if (openBtn) {
    openBtn.addEventListener("click", () => {
      renderPresetsManager();
      modal.classList.remove("hidden");
    });
  }

  const closeModal = () => modal.classList.add("hidden");
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (closeBtn2) closeBtn2.addEventListener("click", closeModal);

  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      if (!confirm("Clear all autosave history?")) return;
      try {
        const res = await apiFetch("/api/presets/autosaves/clear", { method: "POST" });
        if (res.ok) {
          const data = await res.json();
          state.presetsData.configs = data.configs;
          renderPresetsManager();
          showToast("Autosave history cleared.");
        }
      } catch (e) {
        showToast(`Error: ${e.message}`);
      }
    });
  }

  if (saveNewBtn) {
    saveNewBtn.addEventListener("click", () => {
      closeModal();
      document.getElementById("btnSavePresetModal").click();
    });
  }
}

function renderPresetsManager() {
  const autosaveList = document.getElementById("autosaveHistoryList");
  const customList = document.getElementById("customPresetsList");
  const countEl = document.getElementById("autosaveCount");

  const autosaves = state.presetsData.configs.autosaves || [];
  const customs = state.presetsData.configs.custom || [];

  if (countEl) countEl.textContent = autosaves.length;

  // 1. Render Autosave History (Left Column)
  autosaveList.innerHTML = "";
  if (autosaves.length === 0) {
    autosaveList.innerHTML = `<div class="state-msg">No autosaves recorded yet. Snapshots are created when you release a slider.</div>`;
  } else {
    autosaves.forEach((item) => {
      const card = document.createElement("div");
      card.className = "preset-card-item";
      card.innerHTML = `
        <div class="preset-info">
          <span class="preset-item-name">${escapeHtml(item.timestamp_label)}</span>
          <span class="preset-item-meta">Based on: <strong style="color: var(--accent-cyan); font-weight: 600;">${escapeHtml(item.based_on || 'Custom')}</strong> • ${escapeHtml(item.size)}</span>
        </div>
        <div class="preset-item-actions">
          <button class="btn btn-secondary btn-load-autosave btn-xs">Load</button>
          <button class="btn-icon-danger btn-del-autosave" title="Delete this autosave">${getIcon("x", "icon-xs")}</button>
        </div>
      `;

      card.querySelector(".btn-load-autosave").addEventListener("click", (e) => {
        e.stopPropagation();
        const preservedSize = state.lockResolution ? [...state.config.size] : null;
        state.config = { ...state.config, ...item.config };
        if (preservedSize) {
          state.config.size = preservedSize;
        }
        state.activePresetId = "latest_saved";
        syncUIFromConfig();
        triggerProcess(false, false);
        document.getElementById("presetsManagerModal").classList.add("hidden");
        showToast(`Loaded autosave from ${item.timestamp_label}`);
      });

      card.querySelector(".btn-del-autosave").addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          const res = await apiFetch(`/api/presets/autosaves/${item.filename}`, { method: "DELETE" });
          if (res.ok) {
            const data = await res.json();
            state.presetsData.configs = data.configs;
            renderPresetsManager();
          }
        } catch (err) {
          showToast(`Error: ${err.message}`);
        }
      });

      autosaveList.appendChild(card);
    });
  }

  // 2. Render Custom Presets (Right Column)
  customList.innerHTML = "";
  if (customs.length === 0) {
    customList.innerHTML = `<div class="state-msg">No custom presets saved yet. Click "+ Save Current" to create one.</div>`;
  } else {
    customs.forEach((item) => {
      const card = document.createElement("div");
      card.className = "preset-card-item";
      card.innerHTML = `
        <div class="preset-info">
          <span class="preset-item-name">${escapeHtml(item.name)}</span>
          <span class="preset-item-meta">${escapeHtml(item.id)}</span>
        </div>
        <div class="preset-item-actions">
          <button class="btn btn-secondary btn-load-custom btn-xs">Load</button>
          <button class="btn-icon-danger btn-del-custom" title="Delete custom preset">${getIcon("trash", "icon-xs")}</button>
        </div>
      `;

      card.querySelector(".btn-load-custom").addEventListener("click", (e) => {
        e.stopPropagation();
        const preservedSize = state.lockResolution ? [...state.config.size] : null;
        state.config = { ...state.config, ...item.config };
        if (preservedSize) {
          state.config.size = preservedSize;
        }
        state.activePresetId = item.id;
        state.basePresetName = item.id;
        syncUIFromConfig();
        populatePresetDropdowns();
        triggerProcess(false, false);
        document.getElementById("presetsManagerModal").classList.add("hidden");
        showToast(`Loaded preset "${item.name}"`);
      });

      card.querySelector(".btn-del-custom").addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete custom preset "${item.name}"?`)) return;
        try {
          const res = await apiFetch(`/api/presets/custom/${item.id}`, { method: "DELETE" });
          if (res.ok) {
            const data = await res.json();
            state.presetsData.configs = data.configs;
            populatePresetDropdowns();
            renderPresetsManager();
            showToast(`Deleted preset "${item.name}"`);
          }
        } catch (err) {
          showToast(`Error: ${err.message}`);
        }
      });

      customList.appendChild(card);
    });
  }
}
