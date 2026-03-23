"""
Stage 0 - Frame streaming utilities.
"""

from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


@dataclass
class FrameData:
    """Metadata and image payload for one decoded frame."""

    index: int
    timestamp: float
    rgb: np.ndarray
    gray: np.ndarray
    width: int
    height: int


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Extract robust video metadata from an opened VideoCapture."""
    fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps_reported
    fps_fallback_used = False

    if not np.isfinite(fps) or fps <= 0.0:
        fps = 25.0
        fps_fallback_used = True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = (total_frames / fps) if fps > 0 else 0.0

    return {
        "fps": float(fps),
        "fps_reported": float(fps_reported),
        "fps_fallback_used": bool(fps_fallback_used),
        "total_frames": int(total_frames),
        "width": int(width),
        "height": int(height),
        "duration_seconds": float(duration_seconds),
        "duration_minutes": float(duration_seconds / 60.0),
    }


def stream_frames(video_path: str) -> Generator[FrameData, None, None]:
    """Decode frames sequentially, yielding one frame at a time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 25.0

    frame_index = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            timestamp = frame_index / fps
            h, w = bgr.shape[:2]

            yield FrameData(
                index=frame_index,
                timestamp=timestamp,
                rgb=rgb,
                gray=gray,
                width=w,
                height=h,
            )

            frame_index += 1
    finally:
        cap.release()
