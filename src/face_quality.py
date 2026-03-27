"""
Face landmark quality analysis using MediaPipe Face Landmarker (Tasks API).

Provides a positive quality boost for frames with well-visible,
properly-oriented faces — especially useful for Kids YouTube thumbnails
where character faces drive click-through.

Frames with no detected faces get face_quality_score = 0.0 but are
NOT rejected; the score only acts as a boost during per-shot selection.
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

from src.config import (
    FACE_EAR_BLINK_THRESHOLD,
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_MIN_SIZE_FRACTION,
    FACE_PITCH_PENALTY_DEG,
    FACE_YAW_PENALTY_DEG,
)

# MediaPipe Face Mesh landmark indices (478 total in Tasks API).
# Eye landmarks for Eye Aspect Ratio (EAR).
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362

# Nose tip and face outline for head pose estimation.
_NOSE_TIP = 1
_CHIN = 152
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454
_FOREHEAD = 10

# Default model path (relative to project root).
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)


@dataclass
class FaceResult:
    """Per-frame face analysis result."""
    n_faces: int = 0
    largest_face_area: float = 0.0       # fraction of frame area
    face_centering: float = 0.0          # 0–1, 1 = perfectly centered
    eye_openness: float = 0.0            # average EAR
    head_pose_score: float = 0.0         # 0–1, 1 = frontal
    face_quality_score: float = 0.0      # composite 0–1


def _landmark_distance_norm(lm_a, lm_b) -> float:
    """Euclidean distance between two NormalizedLandmark objects."""
    dx = lm_a.x - lm_b.x
    dy = lm_a.y - lm_b.y
    dz = lm_a.z - lm_b.z
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _compute_ear(landmarks) -> float:
    """
    Eye Aspect Ratio (EAR) averaged over both eyes.
    EAR = |top - bottom| / |outer - inner|  (simplified 2-point version).
    """
    def _eye_ear(top_idx, bottom_idx, outer_idx, inner_idx):
        vertical = _landmark_distance_norm(landmarks[top_idx], landmarks[bottom_idx])
        horizontal = _landmark_distance_norm(landmarks[outer_idx], landmarks[inner_idx])
        if horizontal < 1e-6:
            return 0.0
        return vertical / horizontal

    left_ear = _eye_ear(_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM, _LEFT_EYE_OUTER, _LEFT_EYE_INNER)
    right_ear = _eye_ear(_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER)
    return (left_ear + right_ear) / 2.0


def _estimate_head_pose(landmarks) -> tuple:
    """
    Estimate approximate yaw and pitch from landmark geometry.
    Returns (yaw_degrees, pitch_degrees).
    """
    nose = landmarks[_NOSE_TIP]
    left = landmarks[_LEFT_CHEEK]
    right = landmarks[_RIGHT_CHEEK]
    top = landmarks[_FOREHEAD]
    bottom = landmarks[_CHIN]

    # Yaw: ratio of nose-to-left vs nose-to-right horizontal distance.
    nose_to_left = abs(nose.x - left.x)
    nose_to_right = abs(nose.x - right.x)
    total_width = nose_to_left + nose_to_right
    if total_width < 1e-6:
        yaw_deg = 0.0
    else:
        ratio = nose_to_left / total_width  # 0.5 = frontal
        yaw_deg = abs(ratio - 0.5) * 2.0 * 90.0

    # Pitch: ratio of nose-to-top vs nose-to-bottom vertical distance.
    nose_to_top = abs(nose.y - top.y)
    nose_to_bottom = abs(nose.y - bottom.y)
    total_height = nose_to_top + nose_to_bottom
    if total_height < 1e-6:
        pitch_deg = 0.0
    else:
        ratio = nose_to_top / total_height
        pitch_deg = abs(ratio - 0.5) * 2.0 * 90.0

    return yaw_deg, pitch_deg


def _face_bounding_box(landmarks) -> tuple:
    """Return (area_fraction, center_x, center_y) for a face from NormalizedLandmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    area_fraction = (x_max - x_min) * (y_max - y_min)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return area_fraction, center_x, center_y


class FaceLandmarkAnalyzer:
    """Wraps MediaPipe Face Landmarker (Tasks API) for per-frame face quality analysis."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is required for face detection. "
                "Install with: pip install mediapipe>=0.10.0"
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Face Landmarker model not found at: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_MIN_DETECTION_CONFIDENCE,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def analyze_frame(self, rgb: np.ndarray) -> FaceResult:
        """
        Analyze a single RGB frame for face quality.

        Args:
            rgb: RGB image (H, W, 3), uint8.

        Returns:
            FaceResult with face metrics and composite score.
        """
        if self._landmarker is None:
            return FaceResult()

        # Create MediaPipe Image from numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FaceResult()

        # Evaluate each face.
        face_data = []
        for face_lms in result.face_landmarks:
            area_frac, cx, cy = _face_bounding_box(face_lms)

            if area_frac < FACE_MIN_SIZE_FRACTION:
                continue

            ear = _compute_ear(face_lms)
            yaw_deg, pitch_deg = _estimate_head_pose(face_lms)

            # Centering: distance from face center to frame center (normalized).
            dist_to_center = float(np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
            max_dist = float(np.sqrt(0.5))  # corner distance
            centering = 1.0 - min(dist_to_center / max_dist, 1.0)

            # Head pose score: penalize extreme angles.
            yaw_penalty = max(0.0, 1.0 - max(0.0, yaw_deg - FACE_YAW_PENALTY_DEG) / 45.0)
            pitch_penalty = max(0.0, 1.0 - max(0.0, pitch_deg - FACE_PITCH_PENALTY_DEG) / 60.0)
            pose_score = yaw_penalty * pitch_penalty

            face_data.append({
                "area_frac": area_frac,
                "centering": centering,
                "ear": ear,
                "pose_score": pose_score,
            })

        if not face_data:
            return FaceResult()

        n_faces = len(face_data)

        # Use the largest face as the primary subject.
        largest = max(face_data, key=lambda f: f["area_frac"])

        # Normalize face size (clamp to reasonable range).
        face_size_norm = min(largest["area_frac"] / 0.25, 1.0)

        # Eye openness: 1.0 if above threshold, ramp down below.
        raw_ear = largest["ear"]
        if raw_ear >= FACE_EAR_BLINK_THRESHOLD:
            eye_openness_norm = 1.0
        else:
            eye_openness_norm = max(0.0, raw_ear / FACE_EAR_BLINK_THRESHOLD)

        # Multi-face bonus (capped).
        multi_face_bonus = min(n_faces / 3.0, 1.0)

        # Composite score.
        face_quality = (
            0.25 * face_size_norm
            + 0.20 * largest["centering"]
            + 0.25 * eye_openness_norm
            + 0.20 * largest["pose_score"]
            + 0.10 * multi_face_bonus
        )

        return FaceResult(
            n_faces=n_faces,
            largest_face_area=largest["area_frac"],
            face_centering=largest["centering"],
            eye_openness=raw_ear,
            head_pose_score=largest["pose_score"],
            face_quality_score=float(np.clip(face_quality, 0.0, 1.0)),
        )


def compute_face_scores(face_results: List[FaceResult]) -> List[float]:
    """Extract composite face quality scores from a list of FaceResults."""
    return [fr.face_quality_score for fr in face_results]
