"""
Stage 2 — Temporal Stability Filtering

Reject frames during camera or object motion using
Mean Absolute Difference (MAD) between consecutive grayscale frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def compute_motion_scores(grays: List[np.ndarray]) -> List[float]:
    """
    Compute MAD between consecutive grayscale frames.

    Args:
        grays: list of grayscale frames (H×W uint8)

    Returns:
        motion_scores: per-frame motion score (0.0 for first frame)
    """
    scores = [0.0]
    for i in range(1, len(grays)):
        diff = np.abs(grays[i].astype(np.float32) - grays[i - 1].astype(np.float32))
        scores.append(float(np.mean(diff)))
    return scores


def filter_by_motion(
    motion_scores: List[float],
    threshold: float,
) -> List[bool]:
    """Return boolean mask — True means frame is stable (passes filter)."""
    return [score < threshold for score in motion_scores]


def plot_motion_scores(
    motion_scores: List[float],
    is_stable: List[bool],
    fps: float,
    threshold: float,
    save_path: str,
):
    """Plot motion score over time, coloring accepted vs rejected frames."""
    n = len(motion_scores)
    time_axis = np.arange(n) / fps
    scores = np.array(motion_scores)
    stable = np.array(is_stable)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot rejected first (red underneath)
    ax.scatter(time_axis[~stable], scores[~stable],
               s=1, c="#E74C3C", alpha=0.4, label="Rejected (motion)")
    ax.scatter(time_axis[stable], scores[stable],
               s=1, c="#2ECC71", alpha=0.4, label="Accepted (stable)")

    ax.axhline(y=threshold, color="#E74C3C", linestyle="--", linewidth=1,
               label=f"Threshold = {threshold}")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Motion Score (MAD)")
    ax.set_title("Stage 2 — Temporal Stability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
