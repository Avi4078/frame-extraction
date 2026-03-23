"""
Stage 5 — Colorfulness Filter

Kids content is highly saturated. Filter out desaturated frames
using the Hasler & Süsstrunk colorfulness metric.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def compute_colorfulness(rgbs: List[np.ndarray]) -> List[float]:
    """
    Compute Hasler & Süsstrunk colorfulness metric for each RGB frame.

    Higher values = more colorful.
    """
    scores = []
    for rgb in rgbs:
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)
        B = rgb[:, :, 2].astype(np.float32)

        rg = R - G
        yb = 0.5 * (R + G) - B

        std_root = np.sqrt(np.var(rg) + np.var(yb))
        mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

        colorfulness = std_root + 0.3 * mean_root
        scores.append(float(colorfulness))
    return scores


def filter_by_colorfulness(
    colorfulness_scores: List[float],
    threshold: float,
) -> List[bool]:
    """Return boolean mask — True means frame is colorful enough."""
    return [c >= threshold for c in colorfulness_scores]


def plot_colorfulness_histogram(
    colorfulness_scores: List[float],
    threshold: float,
    save_path: str,
):
    """Histogram of colorfulness scores."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(colorfulness_scores, bins=80, color="#F39C12", alpha=0.7, edgecolor="white")
    ax.axvline(x=threshold, color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold}")
    ax.set_xlabel("Colorfulness (Hasler & Süsstrunk)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Stage 5 — Colorfulness Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
