"""
Stage 4 — Brightness & Contrast Filter

Reject dark and flat frames. Kids thumbnails need visual pop.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def compute_brightness_contrast(
    grays: List[np.ndarray],
) -> Tuple[List[float], List[float]]:
    """
    Compute brightness (mean) and contrast (std) of grayscale frames.

    Returns:
        (brightness_scores, contrast_scores)
    """
    brightness = [float(np.mean(g)) for g in grays]
    contrast = [float(np.std(g)) for g in grays]
    return brightness, contrast


def filter_by_brightness_contrast(
    brightness: List[float],
    contrast: List[float],
    dark_threshold: float,
    contrast_threshold: float,
) -> List[bool]:
    """Return boolean mask — True means frame passes brightness + contrast check."""
    return [
        (b >= dark_threshold and c >= contrast_threshold)
        for b, c in zip(brightness, contrast)
    ]


def plot_brightness_contrast_scatter(
    brightness: List[float],
    contrast: List[float],
    passes: List[bool],
    dark_threshold: float,
    contrast_threshold: float,
    save_path: str,
):
    """Scatter plot of brightness vs contrast with accepted region highlighted."""
    b = np.array(brightness)
    c = np.array(contrast)
    p = np.array(passes)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(b[~p], c[~p], s=2, c="#E74C3C", alpha=0.3, label="Rejected")
    ax.scatter(b[p], c[p], s=2, c="#2ECC71", alpha=0.3, label="Accepted")

    ax.axvline(x=dark_threshold, color="#E74C3C", linestyle="--", linewidth=1)
    ax.axhline(y=contrast_threshold, color="#E74C3C", linestyle="--", linewidth=1)

    # Shade accepted region
    ax.axvspan(dark_threshold, 255, alpha=0.05, color="#2ECC71")

    ax.set_xlabel("Brightness (mean gray)")
    ax.set_ylabel("Contrast (std gray)")
    ax.set_title("Stage 4 — Brightness vs Contrast")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
