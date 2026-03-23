"""
Stage 1 - Scene / shot segmentation utilities.
"""

from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import CUT_THRESHOLD, HIST_BINS, POST_CUT_SKIP_SECONDS


def compute_histogram(bgr_frame: np.ndarray) -> np.ndarray:
    """Compute a normalized concatenated BGR histogram."""
    hist_channels = []
    for ch in range(3):
        h = cv2.calcHist([bgr_frame], [ch], None, [HIST_BINS], [0, 256])
        cv2.normalize(h, h)
        hist_channels.append(h)
    return np.concatenate(hist_channels).flatten()


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between two normalized histograms."""
    h1 = hist1.astype(np.float32).reshape(-1, 1)
    h2 = hist2.astype(np.float32).reshape(-1, 1)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def detect_scenes(
    frames_bgr: List[np.ndarray],
    fps: float,
    timestamps: List[float],
) -> Tuple[List[float], List[int], List[int], List[bool]]:
    """Detect cuts and return cut scores, shot IDs, cut indices, and exclusion mask."""
    del timestamps

    n = len(frames_bgr)
    skip_frames = int(POST_CUT_SKIP_SECONDS * fps)
    histograms = [compute_histogram(f) for f in frames_bgr]

    cut_scores = [0.0]
    for i in range(1, n):
        cut_scores.append(bhattacharyya_distance(histograms[i - 1], histograms[i]))

    cut_indices = [i for i in range(1, n) if cut_scores[i] > CUT_THRESHOLD]
    cut_set = set(cut_indices)

    shot_ids = [0] * n
    is_excluded = [False] * n
    current_shot = 0

    for i in range(n):
        if i in cut_set:
            current_shot += 1
            for j in range(i, min(i + skip_frames, n)):
                is_excluded[j] = True
        shot_ids[i] = current_shot

    return cut_scores, shot_ids, cut_indices, is_excluded


def plot_cut_scores(
    cut_scores: List[float],
    cut_indices: List[int],
    fps: float,
    save_path: str,
    threshold: float = CUT_THRESHOLD,
):
    """Plot cut score timeline with cut markers and threshold."""
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(cut_scores))
    time_axis = x / max(fps, 1e-6)

    ax.plot(time_axis, cut_scores, color="#4A90D9", linewidth=0.6, alpha=0.85)
    ax.axhline(
        y=threshold,
        color="#E74C3C",
        linestyle="--",
        linewidth=1,
        label=f"Threshold = {threshold:.3f}",
    )

    for ci in cut_indices:
        ax.axvline(x=ci / max(fps, 1e-6), color="#E74C3C", linewidth=0.5, alpha=0.55)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cut score (Bhattacharyya)")
    ax.set_title("Stage 1 - Scene cut detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_contact_sheet(
    frames_rgb,
    cut_indices: List[int],
    save_path: str,
    cols: int = 5,
    thumb_width: int = 256,
):
    """Save a contact sheet of cut frames."""
    del thumb_width

    if not cut_indices:
        return

    n = len(cut_indices)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, ci in enumerate(cut_indices):
        if isinstance(frames_rgb, dict):
            img = frames_rgb.get(ci)
        else:
            img = frames_rgb[ci] if ci < len(frames_rgb) else None
        if img is not None:
            axes[i].imshow(img)
        axes[i].set_title(f"Cut @ {ci}", fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Scene cut contact sheet", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
