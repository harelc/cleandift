// CleanDIFT interactive correspondence frontend
const $ = (id) => document.getElementById(id);

const state = {
  src: null, // {id, w, h, img}
  tgt: null,
  busy: false,
  topMatches: [],          // array of {src_x, src_y, tgt_x, tgt_y, score}
  clickPoints: [],         // array of {srcPt:{x,y}, tgtPt:{x,y}, score, heatmap}
  keypoint: null,          // {scheme, matches:[{idx,name,src_x,src_y,tgt_x,tgt_y}], edges:[[a,b],...]}
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

function dispScale(rec) {
  const r = rec.imgEl.getBoundingClientRect();
  return r.width / rec.img.naturalWidth;
}

function relayoutCanvases() {
  // Shared scale: both canvases render at the same px-per-natural-px factor.
  // Fill the larger of the two panes; cap height at most of the viewport.
  // Use the viewer's overall width (independent of pane content) so we don't loop.
  const viewer = $("viewer");
  // Available width: viewer minus its 32px padding and the 16px flex gap between panes.
  const viewerW = viewer.clientWidth - 32 - 16;
  const paneW = Math.max(120, Math.floor(viewerW / 2));
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
    // Only set width on the main canvas; CSS `height: auto` keeps aspect correct.
    state[k].imgEl.style.width = w + "px";
    // Overlay is position:absolute width:100% height:100% — follows the img automatically.
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
    const imgEl = $(which === "src" ? "srcImg" : "tgtImg");
    const overlay = $(which === "src" ? "srcOverlay" : "tgtOverlay");
    imgEl.src = rec.img.src;
    overlay.width = rec.img.naturalWidth;
    overlay.height = rec.img.naturalHeight;
    rec.imgEl = imgEl;
    rec.overlay = overlay;
    $(which === "src" ? "srcWrap" : "tgtWrap").classList.remove("empty");
    clearOverlays();
    relayoutCanvases();
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
  state.keypoint = null;
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
    const r = rec.imgEl.getBoundingClientRect();
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

  const rect = rec.imgEl.getBoundingClientRect();
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
        restrict_to_salient: $("restrictSalient").checked,
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
  if (state.keypoint) drawKeypointOverlay();
}

function drawKeypointOverlay() {
  const kp = state.keypoint;
  if (!kp || !state.src || !state.tgt) return;
  // Skeleton/contour lines first, then dots on top.
  const byIdx = new Map(kp.matches.map((m) => [m.idx, m]));
  for (const side of ["src", "tgt"]) {
    const rec = state[side];
    if (!rec) continue;
    const ctx = rec.overlay.getContext("2d");
    const s = 1 / Math.max(0.01, dispScale(rec));
    ctx.lineWidth = 2 * s;
    ctx.strokeStyle = side === "src" ? "rgba(255,255,255,0.85)" : "rgba(255,210,120,0.9)";
    for (const [a, b] of kp.edges) {
      const ma = byIdx.get(a), mb = byIdx.get(b);
      if (!ma || !mb) continue;
      const xa = side === "src" ? ma.src_x : ma.tgt_x;
      const ya = side === "src" ? ma.src_y : ma.tgt_y;
      const xb = side === "src" ? mb.src_x : mb.tgt_x;
      const yb = side === "src" ? mb.src_y : mb.tgt_y;
      ctx.beginPath();
      ctx.moveTo(xa, ya);
      ctx.lineTo(xb, yb);
      ctx.stroke();
    }
  }
  // Smaller dots than top-N — too many to use big numbered circles.
  for (const side of ["src", "tgt"]) {
    const rec = state[side];
    if (!rec) continue;
    const ctx = rec.overlay.getContext("2d");
    const s = 1 / Math.max(0.01, dispScale(rec));
    for (const m of kp.matches) {
      const x = side === "src" ? m.src_x : m.tgt_x;
      const y = side === "src" ? m.src_y : m.tgt_y;
      ctx.beginPath();
      ctx.arc(x, y, 3 * s, 0, Math.PI * 2);
      ctx.fillStyle = side === "src" ? "#5acdff" : "#ffd166";
      ctx.lineWidth = 1 * s;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.fill();
      ctx.stroke();
    }
  }
}

async function runTopN() {
  if (!state.src || !state.tgt) { log("need both images first", "err"); return; }
  state.busy = true; $("topBtn").disabled = true;
  try {
    const n = Math.max(1, parseInt($("topN").value || "12", 10));
    const restrict = $("restrictSalient").checked;
    const nmsR = parseFloat($("nmsRadius").value);
    log(`computing top ${n} mutual matches${restrict ? " (salient only)" : ""} · nms=${nmsR.toFixed(3)}…`);
    const t0 = performance.now();
    const r = await api("/api/top_matches", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        src_id: state.src.id, tgt_id: state.tgt.id,
        n, caption: $("caption").value, feat_key: $("featKey").value,
        min_similarity: 0.0, nms_radius_frac: nmsR,
        restrict_to_salient: restrict,
      }),
    });
    log(`top-N done in ${(performance.now()-t0).toFixed(0)}ms · ${r.matches.length} matches`, "ok");
    state.topMatches = r.matches;
    state.clickPoints = [];
    // updateMaskOverlay clears overlays, redraws mask (if enabled), then dots.
    await updateMaskOverlay();
  } catch (e) {
    log(`top-N failed: ${e.message}`, "err");
  } finally {
    state.busy = false; $("topBtn").disabled = false;
  }
}

async function updateMaskOverlay() {
  const show = $("showMask").checked;
  for (const k of ["src", "tgt"]) {
    const rec = state[k];
    if (!rec) continue;
    const ctx = rec.overlay.getContext("2d");
    // Repaint dots first (clears any prior mask).
    ctx.clearRect(0, 0, rec.overlay.width, rec.overlay.height);
    if (show) {
      try {
        if (!rec.maskImg) {
          log(`fetching salient mask for ${k}…`);
          const img = new Image();
          img.src = `/api/saliency/${rec.id}?ts=${Date.now()}`;
          await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
          rec.maskImg = img;
        }
        // Draw mask in red with the mask itself as the alpha channel.
        const off = document.createElement("canvas");
        off.width = rec.maskImg.naturalWidth;
        off.height = rec.maskImg.naturalHeight;
        const octx = off.getContext("2d");
        octx.drawImage(rec.maskImg, 0, 0);
        const data = octx.getImageData(0, 0, off.width, off.height);
        // The mask is grayscale; use luminance as alpha and tint blue-green.
        for (let i = 0; i < data.data.length; i += 4) {
          const v = data.data[i];
          data.data[i] = 32;       // R
          data.data[i + 1] = 220;  // G
          data.data[i + 2] = 180;  // B
          data.data[i + 3] = Math.round(v * 0.45); // A (semi-transparent)
        }
        octx.putImageData(data, 0, 0);
        ctx.drawImage(off, 0, 0, rec.overlay.width, rec.overlay.height);
      } catch (e) {
        log(`saliency failed for ${k}: ${e.message}`, "err");
      }
    }
  }
  redrawDots();
  renderLines();
}

function setupCanvasClicks() {
  $("srcImg").addEventListener("click", (e) => handleCanvasClick("src", e));
  $("tgtImg").addEventListener("click", (e) => handleCanvasClick("tgt", e));
}

$("srcFile").addEventListener("change", (e) => onFile("src", e.target.files[0]));
$("tgtFile").addEventListener("change", (e) => onFile("tgt", e.target.files[0]));
$("topBtn").addEventListener("click", () => runTopN());

async function runKeypoints() {
  if (!state.src || !state.tgt) { log("need both images first", "err"); return; }
  state.busy = true; $("kpBtn").disabled = true;
  try {
    const scheme = $("kpScheme").value;
    log(`detecting ${scheme} keypoints on source…`);
    const t0 = performance.now();
    const r = await api("/api/keypoint_correspondences", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        src_id: state.src.id, tgt_id: state.tgt.id,
        scheme,
        caption: $("caption").value,
        feat_key: $("featKey").value,
        restrict_to_salient: $("restrictSalient").checked,
      }),
    });
    log(`keypoints done in ${(performance.now()-t0).toFixed(0)}ms · ${r.n_keypoints} detected · ${r.matches.length} matched`, "ok");
    if (!r.matches.length) { log("no keypoints found", "err"); return; }
    state.keypoint = { scheme, matches: r.matches, edges: r.edges || [] };
    state.topMatches = [];
    state.clickPoints = [];
    await updateMaskOverlay();
  } catch (e) {
    log(`keypoints failed: ${e.message}`, "err");
  } finally {
    state.busy = false; $("kpBtn").disabled = false;
  }
}
$("kpBtn").addEventListener("click", () => runKeypoints());
$("nmsRadius").addEventListener("input", () => {
  $("nmsRadiusVal").textContent = parseFloat($("nmsRadius").value).toFixed(3);
});

async function flipImage(which) {
  const rec = state[which];
  if (!rec) { log(`no ${which} image to flip`, "err"); return; }
  state.busy = true;
  try {
    await api(`/api/flip/${rec.id}`, { method: "POST" });
    // Reload the <img> from server with cache-busting; also rebuild the in-memory
    // Image() so naturalWidth still matches and dispScale stays correct.
    const url = `/api/image/${rec.id}?ts=${Date.now()}`;
    rec.imgEl.src = url;
    const fresh = await loadImg(url);
    rec.img = fresh;
    rec.maskImg = null; // invalidate cached saliency overlay
    clearOverlays();
    relayoutCanvases();
    await updateMaskOverlay();
    log(`${which} flipped`, "ok");
  } catch (e) {
    log(`flip ${which} failed: ${e.message}`, "err");
  } finally {
    state.busy = false;
  }
}

document.querySelectorAll(".flipBtn").forEach((btn) => {
  btn.addEventListener("click", () => flipImage(btn.dataset.which));
});

function openHow() { $("howModal").classList.add("open"); $("howModal").setAttribute("aria-hidden", "false"); }
function closeHow() { $("howModal").classList.remove("open"); $("howModal").setAttribute("aria-hidden", "true"); }
$("howBtn").addEventListener("click", openHow);
$("howClose").addEventListener("click", closeHow);
$("howModal").querySelector(".modalBackdrop").addEventListener("click", closeHow);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeHow(); });
$("clearBtn").addEventListener("click", () => {
  clearOverlays();
  renderLines();
  log("cleared overlays");
});
$("showMask").addEventListener("change", () => updateMaskOverlay());
$("restrictSalient").addEventListener("change", () => {
  if ($("restrictSalient").checked) $("showMask").checked = true;
  updateMaskOverlay();
});
$("swapBtn").addEventListener("click", () => {
  const a = state.src, b = state.tgt;
  state.src = b; state.tgt = a;
  for (const k of ["src", "tgt"]) {
    if (state[k]) {
      const imgEl = $(k === "src" ? "srcImg" : "tgtImg");
      const overlay = $(k === "src" ? "srcOverlay" : "tgtOverlay");
      imgEl.src = state[k].img.src;
      overlay.width = state[k].img.naturalWidth;
      overlay.height = state[k].img.naturalHeight;
      state[k].imgEl = imgEl;
      state[k].overlay = overlay;
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
