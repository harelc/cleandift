# CleanDIFT Interactive Correspondences

Browser UI for finding dense semantic correspondences between two arbitrary
images using CleanDIFT features (SD 2.1 U-Net), plus an optional
salient-object filter via `rembg`.

![architecture](https://img.shields.io/badge/stack-FastAPI%20%2B%20vanilla%20JS-blue) ![gpu](https://img.shields.io/badge/runs%20on-NVIDIA%20H100-success)

---

## Features

- **Click for correspondence** — click any pixel in either image; the server returns
  the best-matching point in the other image and a similarity heatmap overlay.
- **Top-N correspondences** — automatic mutual-nearest-neighbour search with
  adjustable NMS radius.
- **Salient region only** — restrict matching to foreground pixels in both
  images (rembg / ISNet). Top-up one-way matches if mutual NN can't reach N.
- **Horizontal flip per image** — clears CleanDIFT feature + saliency caches and
  treats the flipped pixels as a new image on the next match.
- **Drag-and-drop image upload** with per-pane drop hints.
- **Built-in "How does it work?" modal** explaining the full pipeline.

---

## One-time setup on the GPU VM

Assumes you can SSH to a Linux box with an NVIDIA GPU (we use
`main.harel-8g.harel.coder`, an H100 80GB) and that `uv` is installed
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
ssh main.harel-8g.harel.coder

# clone the fork (interactive-app branch contains the UI + server)
git clone -b interactive-app https://github.com/harelc/cleandift.git ~/cleandift
cd ~/cleandift

# create a Python 3.11 venv and install everything
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -r requirements.txt fastapi 'uvicorn[standard]' \
               python-multipart huggingface_hub safetensors pillow \
               'rembg>=2.0.50' onnxruntime
```

The first server run downloads:
- the SD 2.1 mirror weights (`sd2-community/stable-diffusion-2-1`, ~5 GB)
- the CleanDIFT checkpoint (`CompVis/cleandift`, ~3.7 GB)
- the rembg `isnet-general-use` ONNX model (~170 MB)

All land in `~/.cache/huggingface/hub` and `~/.u2net/` and are reused across
restarts.

---

## Running the server on the VM

```bash
ssh main.harel-8g.harel.coder
cd ~/cleandift

# foreground (you see logs; Ctrl-C stops it):
.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000

# OR detached so it survives your SSH session ending:
nohup setsid .venv/bin/uvicorn app.server:app \
  --host 0.0.0.0 --port 8000 </dev/null >/tmp/cleandift.log 2>&1 &
disown
```

Tail the log in detached mode:

```bash
tail -f /tmp/cleandift.log
```

Stop the detached server:

```bash
pkill -f "uvicorn app.server"
```

Wait for the line `INFO cleandift.server | Model ready on cuda` before issuing
requests — the warm-up takes ~15–40 s on first start (~3 s once the HF cache is
populated).

---

## Opening the UI from your laptop

The server binds to `0.0.0.0:8000` on the VM. Forward it over SSH:

```bash
# on your laptop
ssh -N -L 8000:localhost:8000 main.harel-8g.harel.coder
```

Then open <http://localhost:8000> in your browser. Keep the SSH process running
in a terminal while you use the UI.

### Sharing with a colleague

- **Same office Wi-Fi:** rerun the tunnel bound to `0.0.0.0` so the laptop
  accepts LAN connections, then give them your laptop's IP:
  `ssh -N -L 0.0.0.0:8000:localhost:8000 main.harel-8g.harel.coder`
- **Anywhere on the internet:** `brew install cloudflared && cloudflared tunnel --url http://localhost:8000`
  prints a public `https://*.trycloudflare.com` URL.

---

## Environment variables

| Var | Default | Effect |
| --- | --- | --- |
| `CLEANDIFT_SD`        | `sd21`             | Backbone: `sd21` or `sd15` |
| `CLEANDIFT_IMG_SIZE`  | `768` (sd21) / `512` (sd15) | Resize input to this square before feature extraction |
| `CLEANDIFT_FEAT_KEY`  | `us6`              | UNet feature layer used by default (`us4`–`us9` are also valid) |
| `CLEANDIFT_SALIENCY`  | `isnet-general-use`| rembg model name; alternatives: `u2net`, `u2netp`, `silueta` |
| `CLEANDIFT_LAZY`      | `0`                | Set to `1` to skip model warm-up at startup |

Example: run with the lighter SD 1.5 backbone:

```bash
CLEANDIFT_SD=sd15 CLEANDIFT_IMG_SIZE=512 \
  .venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000
```

---

## REST API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/api/upload` | multipart `file` | `{image_id, width, height}` |
| POST | `/api/precompute` | form `image_id`, `caption`, `feat_key` | `{feat_shape, elapsed_s}` |
| POST | `/api/match` | JSON `{src_id,tgt_id,x,y,src_w,src_h,caption,feat_key,return_heatmap,heatmap_size}` | `{target_x,target_y,similarity, heatmap_png?}` |
| POST | `/api/top_matches` | JSON `{src_id,tgt_id,n,caption,feat_key,min_similarity,nms_radius_frac,restrict_to_salient,salient_threshold}` | `{matches:[{src_x,src_y,tgt_x,tgt_y,score}], src_size, tgt_size}` |
| POST | `/api/flip/{id}` | — | `{image_id,width,height}` (invalidates feature + saliency caches) |
| GET  | `/api/image/{id}` | — | JPEG |
| GET  | `/api/saliency/{id}` | — | PNG (grayscale mask, natural-image size) |
| GET  | `/api/health` | — | `{ok, device, sd_version, image_size, model_loaded, n_images}` |

---

## Troubleshooting

- **`401 Unauthorized` on `stabilityai/stable-diffusion-2-1`** — that repo is
  gated/deprecated. This fork already points at the
  `sd2-community/stable-diffusion-2-1` mirror; if you've edited
  `src/ae.py` / `src/sd_feature_extraction.py` / `configs/sd21_*.yaml` back to
  the original ID, restore the mirror.
- **First match is slow (~30 s)** — that's the per-image CleanDIFT feature
  extraction. Subsequent requests on the same image hit the GPU feature cache
  and return in tens of ms.
- **Saliency takes ~1 s per image** — first time loads the ONNX model (~170 MB
  download); thereafter it runs on a 2×-downsampled copy and is reused for the
  image's lifetime.
- **`pip not found` in the venv** — uv venvs don't ship pip; install packages
  via `~/.local/bin/uv pip install ...` with `VIRTUAL_ENV=~/cleandift/.venv`,
  or activate the venv first.
- **Static UI changes don't show** — they're served live; just hard-refresh
  (Cmd-Shift-R / Ctrl-Shift-R). Backend changes need a server restart.

---

## Updating after pulling new code

```bash
ssh main.harel-8g.harel.coder
cd ~/cleandift
git pull
pkill -f "uvicorn app.server"
nohup setsid .venv/bin/uvicorn app.server:app \
  --host 0.0.0.0 --port 8000 </dev/null >/tmp/cleandift.log 2>&1 &
disown
```
