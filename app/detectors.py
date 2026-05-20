"""Standard-convention keypoint detectors used by the correspondence demo.

Supported schemes:
- "dwpose_body"     — 17-point COCO body via rtmlib Wholebody
- "dwpose_face"     — 68-point face via rtmlib Wholebody (same model)
- "mediapipe_face"  — 478-point dense face landmarks via mediapipe FaceMesh

All detectors return a list of dicts in *natural-image* pixel coordinates:
    [{"x": float, "y": float, "name": str, "score": float}, ...]
The first person in the frame is used for DWPose; the first face for MediaPipe.
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Literal

import numpy as np
from PIL import Image

log = logging.getLogger("cleandift.detectors")

# COCO whole-body layout: 17 body + 6 feet + 68 face + 21 + 21 hand = 133.
COCO_BODY_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
BODY_RANGE = (0, 17)
FACE_RANGE = (23, 23 + 68)  # 17 body + 6 feet = 23

# Canonical COCO 17-point skeleton edges (indices into COCO_BODY_NAMES).
COCO_BODY_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12),
    (5, 11), (6, 12),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]


def _dlib68_edges() -> list[tuple[int, int]]:
    """Standard dlib/iBUG 68-point face contour edges, indices 0..67."""
    edges: list[tuple[int, int]] = []
    # jaw 0-16 (open chain)
    edges += [(i, i + 1) for i in range(16)]
    # right brow 17-21
    edges += [(i, i + 1) for i in range(17, 21)]
    # left brow 22-26
    edges += [(i, i + 1) for i in range(22, 26)]
    # nose bridge 27-30
    edges += [(i, i + 1) for i in range(27, 30)]
    # nose bottom 31-35
    edges += [(i, i + 1) for i in range(31, 35)]
    # right eye 36-41 closed loop
    edges += [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
    # left eye 42-47 closed loop
    edges += [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
    # outer mouth 48-59 closed loop
    edges += [(i, i + 1) for i in range(48, 59)] + [(59, 48)]
    # inner mouth 60-67 closed loop
    edges += [(i, i + 1) for i in range(60, 67)] + [(67, 60)]
    return edges


DLIB68_FACE_EDGES = _dlib68_edges()


def get_edges(scheme: str) -> list[tuple[int, int]]:
    if scheme == "dwpose_body":
        return list(COCO_BODY_EDGES)
    if scheme == "dwpose_face":
        return list(DLIB68_FACE_EDGES)
    if scheme == "mediapipe_face":
        edges = set()
        for a, b in MEDIAPIPE_FACE_CONTOURS:
            edges.add((min(a, b), max(a, b)))
        return sorted(edges)
    return []

MP_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)

# Canonical MediaPipe FaceMesh contour edges (pasted from mediapipe's
# python/solutions/face_mesh_connections.py, since 0.10.35 ships only the
# Tasks API and the `solutions` namespace is gone).
_MP_LIPS = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (78, 95),
    (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318),
    (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
)
_MP_LEFT_EYE = (
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
)
_MP_LEFT_EYEBROW = (
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
)
_MP_RIGHT_EYE = (
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
)
_MP_RIGHT_EYEBROW = (
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
)
_MP_FACE_OVAL = (
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
)
MEDIAPIPE_FACE_CONTOURS = (
    list(_MP_LIPS) + list(_MP_LEFT_EYE) + list(_MP_LEFT_EYEBROW)
    + list(_MP_RIGHT_EYE) + list(_MP_RIGHT_EYEBROW) + list(_MP_FACE_OVAL)
)

# Lazy globals — each detector is initialized on first use.
_WHOLEBODY = None
_WHOLEBODY_LOCK = Lock()
_MP_FACE_LANDMARKER = None
_MP_LOCK = Lock()


def _get_wholebody():
    global _WHOLEBODY
    if _WHOLEBODY is not None:
        return _WHOLEBODY
    with _WHOLEBODY_LOCK:
        if _WHOLEBODY is None:
            from rtmlib import Wholebody

            mode = os.environ.get("CLEANDIFT_RTMLIB_MODE", "balanced")
            backend = os.environ.get("CLEANDIFT_RTMLIB_BACKEND", "onnxruntime")
            # rtmlib auto-downloads ONNX weights into ~/.cache on first call.
            log.info("loading rtmlib Wholebody (mode=%s, backend=%s)", mode, backend)
            try:
                import torch as _t

                device = "cuda" if _t.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            _WHOLEBODY = Wholebody(backend=backend, device=device, mode=mode)
    return _WHOLEBODY


def _mp_task_file() -> str:
    """Return local path to face_landmarker.task, downloading if missing."""
    cache = os.path.expanduser(os.environ.get(
        "CLEANDIFT_MP_CACHE", "~/.cache/mediapipe_tasks"))
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, "face_landmarker.task")
    if not os.path.exists(dest):
        import urllib.request

        log.info("downloading %s -> %s", MP_TASK_URL, dest)
        urllib.request.urlretrieve(MP_TASK_URL, dest)
    return dest


def _get_facemesh():
    global _MP_FACE_LANDMARKER
    if _MP_FACE_LANDMARKER is not None:
        return _MP_FACE_LANDMARKER
    with _MP_LOCK:
        if _MP_FACE_LANDMARKER is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            log.info("loading mediapipe FaceLandmarker (tasks API)")
            base = mp_python.BaseOptions(model_asset_path=_mp_task_file())
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _MP_FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)
    return _MP_FACE_LANDMARKER


def detect_keypoints(
    pil: Image.Image,
    scheme: Literal["dwpose_body", "dwpose_face", "mediapipe_face"],
    min_score: float = 0.3,
) -> list[dict]:
    img = np.array(pil.convert("RGB"))
    h, w = img.shape[:2]

    if scheme in ("dwpose_body", "dwpose_face"):
        det = _get_wholebody()
        keypoints, scores = det(img)
        if len(keypoints) == 0:
            return []
        kpt = keypoints[0]  # first detected person, shape (133, 2)
        sc = scores[0] if scores is not None else np.ones(len(kpt))
        if scheme == "dwpose_body":
            idx0, idx1 = BODY_RANGE
            names = COCO_BODY_NAMES
        else:
            idx0, idx1 = FACE_RANGE
            names = [f"f{i}" for i in range(idx1 - idx0)]
        out = []
        for local_idx, j in enumerate(range(idx0, idx1)):
            s = float(sc[j])
            if s < min_score:
                continue
            out.append({
                "idx": local_idx,
                "x": float(kpt[j, 0]),
                "y": float(kpt[j, 1]),
                "name": names[local_idx],
                "score": s,
            })
        return out

    if scheme == "mediapipe_face":
        import mediapipe as mp

        fm = _get_facemesh()
        # Tasks API expects a mediapipe.Image wrapper.
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        r = fm.detect(mp_img)
        if not r.face_landmarks:
            return []
        lms = r.face_landmarks[0]  # list of NormalizedLandmark
        return [
            {"idx": i, "x": l.x * w, "y": l.y * h, "name": f"m{i}", "score": 1.0}
            for i, l in enumerate(lms)
        ]

    raise ValueError(f"unknown scheme: {scheme}")
