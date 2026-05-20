// CleanDIFT interactive correspondence frontend
const $ = (id) => document.getElementById(id);

const state = {
  src: null, // {id, w, h, img}
  tgt: null,
  busy: false,
  topMatches: [],          // array of {src_x, src_y, tgt_x, tgt_y, score}
  clickPoints: [],         // array of {srcPt:{x,y}, tgtPt:{x,y}, score, heatmap}
};

const palette = [
  "#ff6b6b","#ffd166","#06d6a0","#4cc9f0","#b08cff","#ff9f1c",
  "#ef476f","#118ab2","#a3e635","#f72585","#22d3ee","#facc15",
];

function log(msg, cls = "") {
  const el = $("log");
  const d = document.createElement("div");
  if (cls) d.className = cls;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.prepend(d);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${path} -> ${r.status}: ${t}`);
  }
  return r.json();
}

async function loadHealth() {
  try {
    const h = await api("/api/health");
    $("modelStatus").textContent =
      `${h.device} · ${h.sd_version} · ${h.image_size}px · ${h.model_loaded ? "loaded" : "lazy"}`;
    $("status").className = "status ok";
  } catch (e) {
    $("modelStatus").textContent = "offline";
    $("status").className = "status err";
  }
}

function drawImage(canvas, img) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext("2d").drawImage(img, 0, 0);
}

function dispScale(rec) {
  const r = rec.canvas.getBoundingClientRect();
  return r.width / rec.img.naturalWidth;
}

function relayoutCanvases() {
  // Shared scale: both canvases render at the same px-per-natural-px factor.
  // Fill the larger of the two panes; cap height at most of the viewport.
  // Use the viewer's overall width (independent of pane content) so we don't loop.
  const viewer = $("viewer");
  const viewerW = viewer.clientWidth - 32; // padding 16+16
  const paneW = Math.max(120, Math.floor(viewerW / 2) - 8);
  const maxH = Math.max(240, window.innerHeight * 0.82);
  let maxNW = 0, maxNH = 0;
  for (const k of ["src", "tgt"]) if (state[k]) {
    maxNW = Math.max(maxNW, state[k].img.naturalWidth);
    maxNH = Math.max(maxNH, state[k].img.naturalHeight);
  }
  if (!maxNW) return;
  // Allow upscaling up to 3x so small thumbnails are visible.
  const scale = Math.min(paneW / maxNW, maxH / maxNH, 3);
  for (const k of ["src", "tgt"]) if (state[k]) {
    const w = Math.round(state[k].img.naturalWidth * scale);
    const h = Math.round(state[k].img.naturalHeight * scale);
    for (const c of [state[k].canvas, state[k].overlay]) {
      c.style.width = w + "px";
      c.style.height = h + "px";
    }
  }
  redrawDots();
  renderLines();
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await api("/api/upload", { method: "POST", body: fd });
  const img = await loadImg(URL.createObjectURL(file));
  return { id: r.image_id, w: r.width, h: r.height, img };
}

function loadImg(url) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = url;
  });
}

async function onFile(which, file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) { log(`not an image: ${file.name}`, "err"); return; }
  log(`uploading ${which}: ${file.name}`);
  try {
    const rec = await uploadFile(file);
    state[which] = rec;
    const canvas = $(which === "src" ? "srcCanvas" : "tgtCanvas");
    const overlay = $(which === "src" ? "srcOverlay" : "tgtOverlay");
    drawImage(canvas, rec.img);
    overlay.width = canvas.width;
    overlay.height = canvas.height;
    rec.canvas = canvas;
    rec.overlay = overlay;
    $(which === "src" ? "srcWrap" : "tgtWrap").classList.remove("empty");
    clearOverlays();
    requestAnimationFrame(relayoutCanvases);
    log(`${which} ready ${rec.w}x${rec.h}`, "ok");
  } catch (e) {
    log(`upload ${which} failed: ${e.message}`, "err");
  }
}

function clearOverlays() {
  for (const k of ["src", "tgt"]) {
    if (state[k] && state[k].overlay) {
      const ctx = state[k].overlay.getContext("2d");
      ctx.clearRect(0, 0, state[k].overlay.width, state[k].overlay.height);
    }
  }
  $("lines").innerHTML = "";
  state.topMatches = [];
  state.clickPoints = [];
}

function imgToCanvas(rec, x, y) {
  // Overlay bitmap == natural-image coords, no scaling needed.
  return { x, y };
}

function drawDot(rec, x, y, color, label) {
  const ctx = rec.overlay.getContext("2d");
  const p = imgToCanvas(rec, x, y);
  // Scale dot/text relative to display size so they look consistent across images.
  const s = 1 / Math.max(0.01, dispScale(rec));
  const r = 7 * s;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.lineWidth = 2 * s;
  ctx.strokeStyle = "white";
  ctx.fillStyle = color;
  ctx.fill(); ctx.stroke();
  if (label !== undefined) {
    ctx.font = `bold ${Math.round(11 * s)}px ui-monospace, Menlo, monospace`;
    ctx.fillStyle = "white";
    ctx.strokeStyle = "black";
    ctx.lineWidth = 3 * s;
    const t = String(label);
    ctx.strokeText(t, p.x + 9 * s, p.y - 9 * s);
    ctx.fillText(t, p.x + 9 * s, p.y - 9 * s);
  }
}

function drawHeatmapOnTarget(dataUrl) {
  return new Promise((res) => {
    const ctx = state.tgt.overlay.getContext("2d");
    const img = new Image();
    img.onload = () => {
      ctx.globalAlpha = 0.55;
      ctx.drawImage(img, 0, 0, state.tgt.overlay.width, state.tgt.overlay.height);
      ctx.globalAlpha = 1;
      res();
    };
    img.src = dataUrl;
  });
}

function renderLines() {
  const svg = $("lines");
  svg.innerHTML = "";
  const svgR = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${svgR.width} ${svgR.height}`);
  svg.setAttribute("preserveAspectRatio", "none");

  function paneCoord(rec, ix, iy) {
    const r = rec.canvas.getBoundingClientRect();
    const s = r.width / rec.img.naturalWidth;
    return { x: r.left - svgR.left + ix * s, y: r.top - svgR.top + iy * s };
  }

  const drawPair = (a, b, color) => {
    const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
    l.setAttribute("x1", a.x); l.setAttribute("y1", a.y);
    l.setAttribute("x2", b.x); l.setAttribute("y2", b.y);
    l.setAttribute("stroke", color);
    svg.appendChild(l);
  };

  state.clickPoints.forEach((p, i) => {
    if (!p.tgtPt) return;
    const a = paneCoord(state.src, p.srcPt.x, p.srcPt.y);
    const b = paneCoord(state.tgt, p.tgtPt.x, p.tgtPt.y);
    drawPair(a, b, palette[i % palette.length]);
  });
  state.topMatches.forEach((m, i) => {
    const a = paneCoord(state.src, m.src_x, m.src_y);
    const b = paneCoord(state.tgt, m.tgt_x, m.tgt_y);
    drawPair(a, b, palette[i % palette.length]);
  });
}

async function handleCanvasClick(which, ev) {
  const rec = state[which];
  const other = which === "src" ? state.tgt : state.src;
  if (!rec || !other) { log("need both images first", "err"); return; }
  if (state.busy) return;

  const rect = rec.canvas.getBoundingClientRect();
  const s = rect.width / rec.img.naturalWidth;
  const ix = (ev.clientX - rect.left) / s;
  const iy = (ev.clientY - rect.top) / s;

  state.busy = true;
  try {
    log(`matching click on ${which} (${ix.toFixed(1)}, ${iy.toFixed(1)})`);
    const t0 = performance.now();
    const r = await api("/api/match", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        src_id: rec.id, tgt_id: other.id,
        x: ix, y: iy, src_w: rec.w, src_h: rec.h,
        caption: $("caption").value, feat_key: $("featKey").value,
        return_heatmap: $("showHeatmap").checked,
        heatmap_size: 256,
      }),
    });
    log(`match in ${(performance.now() - t0).toFixed(0)}ms · sim=${r.similarity.toFixed(3)}`, "ok");

    const idx = state.clickPoints.length;
    const color = palette[idx % palette.length];
    state.clickPoints.push({
      srcPt: which === "src" ? { x: ix, y: iy } : { x: r.target_x, y: r.target_y },
      tgtPt: which === "src" ? { x: r.target_x, y: r.target_y } : { x: ix, y: iy },
      score: r.similarity,
    });

    // Redraw all overlays
    clearOverlayCanvases();
    if (r.heatmap_png && $("showHeatmap").checked) {
      // draw heatmap onto the *other* pane
      const tgtPane = which === "src" ? state.tgt : state.src;
      await drawHeatmapImg(tgtPane, r.heatmap_png);
    }
    redrawDots();
    renderLines();
  } catch (e) {
    log(`match failed: ${e.message}`, "err");
  } finally {
    state.busy = false;
  }
}

function drawHeatmapImg(rec, dataUrl) {
  return new Promise((res) => {
    const ctx = rec.overlay.getContext("2d");
    const img = new Image();
    img.onload = () => {
      ctx.globalAlpha = 0.6;
      ctx.drawImage(img, 0, 0, rec.overlay.width, rec.overlay.height);
      ctx.globalAlpha = 1;
      res();
    };
    img.src = dataUrl;
  });
}

function clearOverlayCanvases() {
  for (const k of ["src", "tgt"]) {
    if (state[k] && state[k].overlay) {
      const ctx = state[k].overlay.getContext("2d");
      ctx.clearRect(0, 0, state[k].overlay.width, state[k].overlay.height);
    }
  }
}

function redrawDots() {
  state.clickPoints.forEach((p, i) => {
    const color = palette[i % palette.length];
    drawDot(state.src, p.srcPt.x, p.srcPt.y, color, i + 1);
    if (p.tgtPt) drawDot(state.tgt, p.tgtPt.x, p.tgtPt.y, color, i + 1);
  });
  state.topMatches.forEach((m, i) => {
    const color = palette[i % palette.length];
    drawDot(state.src, m.src_x, m.src_y, color, i + 1);
    drawDot(state.tgt, m.tgt_x, m.tgt_y, color, i + 1);
  });
}

async function runTopN() {
  if (!state.src || !state.tgt) { log("need both images first", "err"); return; }
  state.busy = true; $("topBtn").disabled = true;
  try {
    const n = Math.max(1, parseInt($("topN").value || "12", 10));
    log(`computing top ${n} mutual matches…`);
    const t0 = performance.now();
    const r = await api("/api/top_matches", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        src_id: state.src.id, tgt_id: state.tgt.id,
        n, caption: $("caption").value, feat_key: $("featKey").value,
        min_similarity: 0.0, nms_radius_frac: 0.06,
      }),
    });
    log(`top-N done in ${(performance.now()-t0).toFixed(0)}ms · ${r.matches.length} matches`, "ok");
    state.topMatches = r.matches;
    state.clickPoints = [];
    clearOverlayCanvases();
    redrawDots();
    renderLines();
  } catch (e) {
    log(`top-N failed: ${e.message}`, "err");
  } finally {
    state.busy = false; $("topBtn").disabled = false;
  }
}

function setupCanvasClicks() {
  $("srcCanvas").addEventListener("click", (e) => handleCanvasClick("src", e));
  $("tgtCanvas").addEventListener("click", (e) => handleCanvasClick("tgt", e));
}

$("srcFile").addEventListener("change", (e) => onFile("src", e.target.files[0]));
$("tgtFile").addEventListener("change", (e) => onFile("tgt", e.target.files[0]));
$("topBtn").addEventListener("click", () => runTopN());
$("clearBtn").addEventListener("click", () => {
  clearOverlays();
  renderLines();
  log("cleared overlays");
});
$("swapBtn").addEventListener("click", () => {
  const a = state.src, b = state.tgt;
  state.src = b; state.tgt = a;
  for (const k of ["src", "tgt"]) {
    if (state[k]) {
      const canvas = $(k === "src" ? "srcCanvas" : "tgtCanvas");
      const overlay = $(k === "src" ? "srcOverlay" : "tgtOverlay");
      state[k].canvas = canvas;
      state[k].overlay = overlay;
      drawImage(canvas, state[k].img);
      overlay.width = canvas.width; overlay.height = canvas.height;
    }
  }
  clearOverlays();
  relayoutCanvases();
});
window.addEventListener("resize", () => requestAnimationFrame(relayoutCanvases));

function setupDnD() {
  for (const which of ["src", "tgt"]) {
    const el = $(which === "src" ? "srcWrap" : "tgtWrap");
    // show placeholder text via ::before pseudo? Easier: set innerText on empty.
    // Inject a small label element when empty.
    const hint = document.createElement("div");
    hint.className = "dropHint";
    hint.textContent = el.dataset.emptyText || "drop image here";
    el.prepend(hint);

    el.addEventListener("dragenter", (e) => { e.preventDefault(); el.classList.add("dragover"); });
    el.addEventListener("dragover", (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
    el.addEventListener("dragleave", (e) => {
      if (e.target === el) el.classList.remove("dragover");
    });
    el.addEventListener("drop", (e) => {
      e.preventDefault();
      el.classList.remove("dragover");
      const f = e.dataTransfer.files[0];
      if (f) onFile(which, f);
    });
  }
  // Prevent the browser from navigating if user drops outside a pane.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
}

setupCanvasClicks();
setupDnD();
loadHealth();
setInterval(loadHealth, 5000);
