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
        # Pull contour edges from mediapipe's canonical face mesh topology.
        import mediapipe as mp
        edges = set()
        for a, b in mp.solutions.face_mesh.FACEMESH_CONTOURS:
            edges.add((min(a, b), max(a, b)))
        return sorted(edges)
    return []

# Lazy globals — each detector is initialized on first use.
_WHOLEBODY = None
_WHOLEBODY_LOCK = Lock()
_MP_FACEMESH = None
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


def _get_facemesh():
    global _MP_FACEMESH
    if _MP_FACEMESH is not None:
        return _MP_FACEMESH
    with _MP_LOCK:
        if _MP_FACEMESH is None:
            import mediapipe as mp

            log.info("loading mediapipe FaceMesh")
            _MP_FACEMESH = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
    return _MP_FACEMESH


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
        fm = _get_facemesh()
        # mediapipe expects RGB uint8
        r = fm.process(img)
        if not r.multi_face_landmarks:
            return []
        lms = r.multi_face_landmarks[0].landmark
        return [
            {"idx": i, "x": l.x * w, "y": l.y * h, "name": f"m{i}", "score": 1.0}
            for i, l in enumerate(lms)
        ]

    raise ValueError(f"unknown scheme: {scheme}")
