"""Interactive CleanDIFT correspondence server.

Run from repo root:
    uv run uvicorn app.server:app --host 0.0.0.0 --port 8000

Exposes a small REST API plus static frontend at /.
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from safetensors.torch import load_file
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("cleandift.server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SD_VERSION = os.environ.get("CLEANDIFT_SD", "sd21")  # sd21 or sd15
IMAGE_SIZE = int(os.environ.get("CLEANDIFT_IMG_SIZE", "768" if SD_VERSION == "sd21" else "512"))
FEAT_KEY_DEFAULT = os.environ.get("CLEANDIFT_FEAT_KEY", "us6")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

HF_REPO = "CompVis/cleandift"
WEIGHT_FILE = {
    "sd21": "cleandift_sd21_full.safetensors",
    "sd15": "cleandift_sd15_full.safetensors",
}[SD_VERSION]
CFG_FILE = REPO_ROOT / "configs" / f"{SD_VERSION}_feature_extractor.yaml"

# ---------------------------------------------------------------------------
# Global model + image cache
# ---------------------------------------------------------------------------
MODEL = None
MODEL_LOCK = Lock()


@dataclass
class ImageRecord:
    image_id: str
    pil: Image.Image
    width: int
    height: int
    # cache features keyed by (caption, feat_key) -> tensor [1, D, h, w] on GPU bf16
    features: dict = field(default_factory=dict)
    # cached saliency mask (PIL 'L' at natural size), None until computed
    saliency: Image.Image | None = None


IMAGES: dict[str, ImageRecord] = {}
IMAGES_LOCK = Lock()

# Lazy rembg session for salient-object/foreground detection.
_REMBG_SESSION = None
_REMBG_LOCK = Lock()


def _rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    with _REMBG_LOCK:
        if _REMBG_SESSION is None:
            from rembg import new_session  # local import: heavy
            model = os.environ.get("CLEANDIFT_SALIENCY", "isnet-general-use")
            log.info("loading rembg session: %s", model)
            _REMBG_SESSION = new_session(model)
    return _REMBG_SESSION


def compute_saliency(rec: ImageRecord) -> Image.Image:
    if rec.saliency is not None:
        return rec.saliency
    from rembg import remove

    sess = _rembg_session()
    t0 = time.time()
    # Downsample 2x on each axis for speed; the salient mask is coarse anyway.
    small = rec.pil.resize((max(1, rec.pil.width // 2), max(1, rec.pil.height // 2)), Image.BILINEAR)
    mask_small = remove(small, session=sess, only_mask=True, post_process_mask=True)
    if mask_small.mode != "L":
        mask_small = mask_small.convert("L")
    # Upsample back to natural size for the cached mask (so frontend / feat-grid
    # downsampling work as before).
    mask = mask_small.resize((rec.pil.width, rec.pil.height), Image.BILINEAR)
    rec.saliency = mask
    log.info("saliency %s in %.2fs (input %dx%d)", rec.image_id, time.time() - t0,
             small.width, small.height)
    return mask


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        log.info("Loading CleanDIFT model (%s) from %s", SD_VERSION, CFG_FILE)
        cfg = OmegaConf.load(CFG_FILE)["model"]
        m = hydra.utils.instantiate(cfg)
        if DEVICE == "cuda":
            m = m.cuda().to(DTYPE)
        else:
            m = m.to(DTYPE)
        log.info("Downloading weights %s/%s", HF_REPO, WEIGHT_FILE)
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(repo_id=HF_REPO, filename=WEIGHT_FILE)
        sd = load_file(ckpt)
        m.load_state_dict(sd, strict=True)
        m.eval()
        MODEL = m
        log.info("Model ready on %s", DEVICE)
        return MODEL


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _preprocess(pil: Image.Image) -> torch.Tensor:
    img = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    t = to_tensor(img)[None].to(DEVICE) * 2 - 1
    return t.to(DTYPE)


def compute_features(rec: ImageRecord, caption: str, feat_key: str) -> torch.Tensor:
    key = (caption, feat_key)
    if key in rec.features:
        return rec.features[key]
    model = load_model()
    x = _preprocess(rec.pil)
    with torch.no_grad():
        feats = model.get_features(x, [caption], t=None, feat_key=feat_key)
    # feats: [1, D, h, w] bf16 on GPU. Keep on GPU; small enough.
    rec.features[key] = feats
    return feats


def _to_full_res(feats: torch.Tensor, w: int, h: int) -> torch.Tensor:
    return F.interpolate(feats.float(), size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class MatchRequest(BaseModel):
    src_id: str
    tgt_id: str
    x: float  # in source image coords (display coords)
    y: float
    src_w: int  # original width client used for click
    src_h: int
    caption: str = ""
    feat_key: str = FEAT_KEY_DEFAULT
    return_heatmap: bool = True
    heatmap_size: int = 256


class TopMatchesRequest(BaseModel):
    src_id: str
    tgt_id: str
    n: int = 12
    caption: str = ""
    feat_key: str = FEAT_KEY_DEFAULT
    min_similarity: float = 0.0
    nms_radius_frac: float = 0.06  # fraction of min(src_w, src_h)
    restrict_to_salient: bool = False
    salient_threshold: float = 0.5  # 0-1, applied to normalized mask


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="CleanDIFT correspondences")

STATIC_DIR = REPO_ROOT / "app" / "static"


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "sd_version": SD_VERSION,
        "image_size": IMAGE_SIZE,
        "model_loaded": MODEL is not None,
        "n_images": len(IMAGES),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"bad image: {e}")
    image_id = uuid.uuid4().hex
    rec = ImageRecord(image_id=image_id, pil=pil, width=pil.width, height=pil.height)
    with IMAGES_LOCK:
        IMAGES[image_id] = rec
    log.info("uploaded %s (%dx%d)", image_id, pil.width, pil.height)
    return {"image_id": image_id, "width": pil.width, "height": pil.height}


@app.post("/api/precompute")
async def precompute(
    image_id: str = Form(...),
    caption: str = Form(""),
    feat_key: str = Form(FEAT_KEY_DEFAULT),
):
    rec = IMAGES.get(image_id)
    if not rec:
        raise HTTPException(404, "unknown image_id")
    t0 = time.time()
    feats = compute_features(rec, caption, feat_key)
    return {
        "feat_shape": list(feats.shape),
        "elapsed_s": round(time.time() - t0, 3),
    }


def _cos_sim_matrix(src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
    """Return cos-sim of every src spatial location to every tgt spatial location.

    src_feats: [1, D, hs, ws]; tgt_feats: [1, D, ht, wt]
    returns: [hs*ws, ht*wt] float32
    """
    s = src_feats.float()
    t = tgt_feats.float()
    s = s / (s.norm(dim=1, keepdim=True) + 1e-8)
    t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
    s_flat = einops.rearrange(s, "1 c h w -> (h w) c")
    t_flat = einops.rearrange(t, "1 c h w -> c (h w)")
    return s_flat @ t_flat


def _heatmap_png(sim: torch.Tensor, size: int) -> str:
    """sim: [h, w] float; encode as colored png base64."""
    s = sim.detach().cpu().float().numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    h, w = s.shape
    # cheap colormap: viridis-like
    r = np.clip(1.5 * s - 0.4, 0, 1)
    g = np.clip(1.5 * s * (1 - s) * 4, 0, 1)
    b = np.clip(1.2 * (1 - s), 0, 1)
    a = (s * 220).astype(np.uint8)
    rgba = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
        a,
    ], axis=-1)
    img = Image.fromarray(rgba, "RGBA")
    if max(h, w) != size:
        img = img.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/match")
async def match(req: MatchRequest):
    src = IMAGES.get(req.src_id)
    tgt = IMAGES.get(req.tgt_id)
    if not src or not tgt:
        raise HTTPException(404, "unknown image_id")

    src_feats = compute_features(src, req.caption, req.feat_key)
    tgt_feats = compute_features(tgt, req.caption, req.feat_key)

    # Map click (in client's src image coords) to feature-grid coords
    _, _, fhs, fws = src_feats.shape
    _, _, fht, fwt = tgt_feats.shape

    u = req.x / max(1.0, req.src_w)  # [0, 1]
    v = req.y / max(1.0, req.src_h)
    fx = int(np.clip(u * fws, 0, fws - 1))
    fy = int(np.clip(v * fhs, 0, fhs - 1))

    s = src_feats.float()
    s = s / (s.norm(dim=1, keepdim=True) + 1e-8)
    t = tgt_feats.float()
    t = t / (t.norm(dim=1, keepdim=True) + 1e-8)

    q = s[0, :, fy, fx]  # [D]
    sims = einops.rearrange(t[0], "c h w -> (h w) c") @ q  # [ht*wt]
    sims = einops.rearrange(sims, "(h w) -> h w", h=fht)

    idx = int(sims.argmax().item())
    ty, tx = idx // fwt, idx % fwt
    tx_img = (tx + 0.5) / fwt * tgt.width
    ty_img = (ty + 0.5) / fht * tgt.height
    best = float(sims.max().item())

    out = {
        "target_x": tx_img,
        "target_y": ty_img,
        "similarity": best,
        "tgt_w": tgt.width,
        "tgt_h": tgt.height,
        "src_feat_grid": [fws, fhs],
        "tgt_feat_grid": [fwt, fht],
    }
    if req.return_heatmap:
        out["heatmap_png"] = _heatmap_png(sims, req.heatmap_size)
    return out


def _mask_to_feat_grid(mask_pil: Image.Image, h: int, w: int, threshold: float) -> torch.Tensor:
    """Resize a PIL saliency mask to (h,w) and threshold to bool."""
    m = mask_pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    return torch.from_numpy(arr >= threshold).to("cuda" if torch.cuda.is_available() else "cpu")


def _top_mutual_matches(src_feats, tgt_feats, n, min_sim, nms_radius_frac,
                        src_mask: torch.Tensor | None = None,
                        tgt_mask: torch.Tensor | None = None):
    """Compute top-N mutual nearest-neighbor matches with NMS in source grid.

    Optional boolean masks (feature-grid sized) restrict matches: a candidate
    is kept only if its source location is in src_mask AND its best target
    location is in tgt_mask.
    """
    _, _, hs, ws = src_feats.shape
    _, _, ht, wt = tgt_feats.shape

    sims = _cos_sim_matrix(src_feats, tgt_feats)  # [Ns, Nt]
    Ns, Nt = sims.shape

    # Mask out non-salient cells on BOTH sides before any argmax, so mutual-NN
    # is computed within the salient region of each image.
    if src_mask is not None:
        flat_src = src_mask.view(-1)
        sims = sims.masked_fill(~flat_src[:, None], -1.0)
    if tgt_mask is not None:
        flat_tgt = tgt_mask.view(-1)
        sims = sims.masked_fill(~flat_tgt[None, :], -1.0)

    src_best = sims.argmax(dim=1)  # [Ns] -> tgt idx
    src_best_val = sims.gather(1, src_best[:, None]).squeeze(1)
    tgt_best = sims.argmax(dim=0)  # [Nt] -> src idx

    src_idx = torch.arange(Ns, device=sims.device)
    mutual_mask = tgt_best[src_best] == src_idx
    mutual_mask &= src_best_val >= min_sim
    if src_mask is not None:
        mutual_mask &= src_mask.view(-1)

    # Primary candidates: mutual NN within masks, ranked by similarity.
    cand_src = src_idx[mutual_mask]
    cand_tgt = src_best[mutual_mask]
    cand_score = src_best_val[mutual_mask]

    # Top-up candidates: one-way matches (src->tgt argmax) inside the source
    # mask. We'll only use these after exhausting mutual matches via NMS, so
    # the result list always reaches `n` if there are enough salient cells.
    extra_valid = src_best_val >= min_sim
    if src_mask is not None:
        extra_valid &= src_mask.view(-1)
    extra_valid &= ~mutual_mask
    extra_src = src_idx[extra_valid]
    extra_tgt = src_best[extra_valid]
    extra_score = src_best_val[extra_valid]
    cand_src = torch.cat([cand_src, extra_src])
    cand_tgt = torch.cat([cand_tgt, extra_tgt])
    cand_score = torch.cat([cand_score, extra_score])

    order = torch.argsort(cand_score, descending=True)
    cand_src = cand_src[order].cpu().numpy()
    cand_tgt = cand_tgt[order].cpu().numpy()
    cand_score = cand_score[order].cpu().numpy()

    # NMS in source feature grid coords
    nms_r = nms_radius_frac * min(hs, ws)
    chosen_xy: list[tuple[float, float]] = []
    picks = []
    for s_idx, t_idx, score in zip(cand_src, cand_tgt, cand_score):
        sy, sx = int(s_idx) // ws, int(s_idx) % ws
        ok = True
        for (cx, cy) in chosen_xy:
            if (sx - cx) ** 2 + (sy - cy) ** 2 < nms_r * nms_r:
                ok = False
                break
        if not ok:
            continue
        ty, tx = int(t_idx) // wt, int(t_idx) % wt
        picks.append({
            "src_fx": sx, "src_fy": sy,
            "tgt_fx": tx, "tgt_fy": ty,
            "score": float(score),
        })
        chosen_xy.append((sx, sy))
        if len(picks) >= n:
            break
    return picks, (ws, hs), (wt, ht)


@app.post("/api/top_matches")
async def top_matches(req: TopMatchesRequest):
    src = IMAGES.get(req.src_id)
    tgt = IMAGES.get(req.tgt_id)
    if not src or not tgt:
        raise HTTPException(404, "unknown image_id")
    src_feats = compute_features(src, req.caption, req.feat_key)
    tgt_feats = compute_features(tgt, req.caption, req.feat_key)

    src_mask = tgt_mask = None
    if req.restrict_to_salient:
        _, _, hs, ws = src_feats.shape
        _, _, ht, wt = tgt_feats.shape
        src_mask = _mask_to_feat_grid(compute_saliency(src), hs, ws, req.salient_threshold)
        tgt_mask = _mask_to_feat_grid(compute_saliency(tgt), ht, wt, req.salient_threshold)

    picks, (ws, hs), (wt, ht) = _top_mutual_matches(
        src_feats, tgt_feats, req.n, req.min_similarity, req.nms_radius_frac,
        src_mask=src_mask, tgt_mask=tgt_mask,
    )
    # convert feat-grid coords to image coords
    out = []
    for p in picks:
        out.append({
            "src_x": (p["src_fx"] + 0.5) / ws * src.width,
            "src_y": (p["src_fy"] + 0.5) / hs * src.height,
            "tgt_x": (p["tgt_fx"] + 0.5) / wt * tgt.width,
            "tgt_y": (p["tgt_fy"] + 0.5) / ht * tgt.height,
            "score": p["score"],
        })
    return {
        "matches": out,
        "src_size": [src.width, src.height],
        "tgt_size": [tgt.width, tgt.height],
    }


@app.post("/api/flip/{image_id}")
async def flip(image_id: str):
    """Horizontally flip an image; invalidate cached features and saliency."""
    rec = IMAGES.get(image_id)
    if not rec:
        raise HTTPException(404, "unknown image_id")
    rec.pil = rec.pil.transpose(Image.FLIP_LEFT_RIGHT)
    rec.features.clear()
    rec.saliency = None
    log.info("flipped %s", image_id)
    return {"image_id": image_id, "width": rec.width, "height": rec.height}


@app.get("/api/saliency/{image_id}")
async def get_saliency(image_id: str):
    """Return the saliency mask as a PNG (grayscale, natural-image size)."""
    from fastapi.responses import Response

    rec = IMAGES.get(image_id)
    if not rec:
        raise HTTPException(404, "unknown image_id")
    mask = compute_saliency(rec)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    from fastapi.responses import Response

    rec = IMAGES.get(image_id)
    if not rec:
        raise HTTPException(404, "unknown image_id")
    buf = io.BytesIO()
    rec.pil.save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


# Static frontend (mounted last to avoid swallowing /api)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.on_event("startup")
async def warm():
    if os.environ.get("CLEANDIFT_LAZY", "0") != "1":
        try:
            load_model()
        except Exception as e:
            log.warning("model preload failed (will retry on first request): %s", e)
