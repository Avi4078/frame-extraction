"""
Stage 6 - Clutter filtering utilities.
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_edge_density(grays: List[np.ndarray]) -> List[float]:
    """Compute Canny edge density for each grayscale frame."""
    scores = []
    for gray in grays:
        edges = cv2.Canny(gray, 50, 150)
        scores.append(float(np.count_nonzero(edges)) / max(edges.size, 1))
    return scores


def filter_by_clutter(edge_densities: List[float], min_density: float, max_density: float) -> List[bool]:
    """Return pass mask where True means frame falls inside allowed edge-density range."""
    return [min_density < d < max_density for d in edge_densities]


def plot_edge_density_histogram(
    edge_densities: List[float],
    min_density: float,
    max_density: float,
    save_path: str,
):
    """Save edge-density histogram with accepted window."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(edge_densities, bins=90, color="#8E44AD", alpha=0.75, edgecolor="white")
    ax.axvline(x=min_density, color="#E74C3C", linestyle="--", linewidth=2, label=f"Min = {min_density}")
    ax.axvline(x=max_density, color="#E74C3C", linestyle="--", linewidth=2, label=f"Max = {max_density}")
    ax.axvspan(min_density, max_density, color="#2ECC71", alpha=0.08)
    ax.set_xlabel("Edge density")
    ax.set_ylabel("Frame count")
    ax.set_title("Stage 6 - Edge-density distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_clean_busy_examples(
    frames_rgb: Dict[int, np.ndarray],
    edge_densities: List[float],
    save_path: str,
    clean_indices: Optional[List[int]] = None,
    busy_indices: Optional[List[int]] = None,
    n: int = 5,
):
    """Save low-edge and high-edge example frames."""
    order = np.argsort(edge_densities)
    if clean_indices is None:
        clean_indices = order[:n].tolist()
    if busy_indices is None:
        busy_indices = order[-n:][::-1].tolist()

    clean = [idx for idx in clean_indices if idx in frames_rgb][:n]
    busy = [idx for idx in busy_indices if idx in frames_rgb][:n]

    cols = max(len(clean), len(busy), 1)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 6.0))

    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for c in range(cols):
        ax_top = axes[0, c]
        ax_bot = axes[1, c]

        if c < len(clean):
            idx = clean[c]
            ax_top.imshow(frames_rgb[idx])
            ax_top.set_title(f"#{idx} {edge_densities[idx]:.3f}", fontsize=8)
        ax_top.axis("off")

        if c < len(busy):
            idx = busy[c]
            ax_bot.imshow(frames_rgb[idx])
            ax_bot.set_title(f"#{idx} {edge_densities[idx]:.3f}", fontsize=8)
        ax_bot.axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=10)
    axes[1, 0].set_ylabel("Busy", fontsize=10)
    plt.suptitle("Stage 6 - Clean vs busy examples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
