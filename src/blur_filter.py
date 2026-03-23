"""
Stage 3 - Blur filtering utilities.
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_sharpness_scores(grays: List[np.ndarray]) -> List[float]:
    """Compute Variance of Laplacian sharpness for each frame."""
    scores = []
    for gray in grays:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        scores.append(float(np.var(lap)))
    return scores


def filter_by_sharpness(sharpness_scores: List[float], threshold: float) -> List[bool]:
    """Return pass mask where True means frame is sharp enough."""
    return [s > threshold for s in sharpness_scores]


def plot_sharpness_histogram(sharpness_scores: List[float], threshold: float, save_path: str):
    """Save sharpness histogram with threshold line."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sharpness_scores, bins=100, color="#4A90D9", alpha=0.75, edgecolor="white")
    ax.axvline(x=threshold, color="#E74C3C", linestyle="--", linewidth=2, label=f"Threshold = {threshold}")
    ax.set_xlabel("Sharpness (Variance of Laplacian)")
    ax.set_ylabel("Frame count")
    ax.set_title("Stage 3 - Sharpness distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_best_worst_frames(
    frames_rgb: Dict[int, np.ndarray],
    sharpness_scores: List[float],
    save_path: str,
    n: int = 5,
    best_indices: Optional[List[int]] = None,
    worst_indices: Optional[List[int]] = None,
):
    """Save top/bottom sharpness frames as a 2-row contact sheet."""
    if best_indices is None or worst_indices is None:
        order = np.argsort(sharpness_scores)
        worst_indices = order[:n].tolist()
        best_indices = order[-n:][::-1].tolist()

    best = [idx for idx in best_indices if idx in frames_rgb][:n]
    worst = [idx for idx in worst_indices if idx in frames_rgb][:n]

    cols = max(len(best), len(worst), 1)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 6.0))

    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for c in range(cols):
        ax_top = axes[0, c]
        ax_bot = axes[1, c]

        if c < len(best):
            idx = best[c]
            ax_top.imshow(frames_rgb[idx])
            ax_top.set_title(f"#{idx} {sharpness_scores[idx]:.1f}", fontsize=8)
        ax_top.axis("off")

        if c < len(worst):
            idx = worst[c]
            ax_bot.imshow(frames_rgb[idx])
            ax_bot.set_title(f"#{idx} {sharpness_scores[idx]:.1f}", fontsize=8)
        ax_bot.axis("off")

    axes[0, 0].set_ylabel("Best", fontsize=10)
    axes[1, 0].set_ylabel("Worst", fontsize=10)
    plt.suptitle("Stage 3 - Sharpest vs blurriest", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
