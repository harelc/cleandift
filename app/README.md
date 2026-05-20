# CleanDIFT Interactive Correspondences

Interactive web UI for finding dense semantic correspondences between two images
using CleanDIFT features.

## What it does

- Upload two images of any size/aspect ratio.
- Click anywhere on either image — the server computes the best matching point
  on the other image, plus a similarity heatmap overlay.
- Or switch to **Top-N** mode to automatically pick the N most prominent
  mutually-consistent correspondences (mutual nearest neighbors + NMS).

## Architecture

```
browser ──▶ FastAPI (uvicorn) ──▶ CleanDIFT (SD2.1 UNet) on CUDA
```

- `app/server.py` — feature server. Caches features per (image, caption, feature-key).
- `app/static/` — zero-build vanilla JS frontend served from the same process.

## Running on the H100 coder box

```bash
ssh main.harel-8g.harel.coder
cd /home/user/cleandift
git pull
source .venv/bin/activate

# first run downloads weights into the HF cache (~5GB)
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
```

From your laptop, port-forward and open the UI:

```bash
ssh -N -L 8000:localhost:8000 main.harel-8g.harel.coder &
open http://localhost:8000
```

## Env vars

| var | default | notes |
| --- | --- | --- |
| `CLEANDIFT_SD` | `sd21` | `sd15` also supported |
| `CLEANDIFT_IMG_SIZE` | `768` (sd21) / `512` (sd15) | input resolution |
| `CLEANDIFT_FEAT_KEY` | `us6` | UNet feature to use |
| `CLEANDIFT_LAZY` | `0` | set to `1` to skip warm-up at startup |

## REST API

- `POST /api/upload` (multipart `file`) → `{image_id, width, height}`
- `POST /api/precompute` (form `image_id`, `caption`, `feat_key`) → `{feat_shape, elapsed_s}`
- `POST /api/match` (JSON) → `{target_x, target_y, similarity, heatmap_png?}`
- `POST /api/top_matches` (JSON `n`, …) → `{matches: [{src_x,src_y,tgt_x,tgt_y,score}]}`
- `GET  /api/image/{id}` → JPEG
- `GET  /api/health` → status
