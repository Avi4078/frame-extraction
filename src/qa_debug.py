"""
QA debug helpers for rejection sampling visualizations.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def sample_stage_rejections(
    stage_rejections: Dict[str, List[int]],
    n_samples: int = 8,
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Sample up to n rejected frame indices per stage deterministically."""
    rng = random.Random(seed)
    out: Dict[str, List[int]] = {}
    for stage, rejected in stage_rejections.items():
        if not rejected:
            out[stage] = []
            continue
        k = min(n_samples, len(rejected))
        out[stage] = sorted(rng.sample(rejected, k))
    return out


def _ensure_axes_grid(rows: int, cols: int):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])
    return fig, axes


def save_all_qa_sheets_from_frames(
    stage_rejections: Dict[str, List[int]],
    sampled_by_stage: Dict[str, List[int]],
    frames: Dict[int, np.ndarray],
    qa_dir: str,
    cols: int = 4,
):
    """
    Save one contact sheet per stage from already-read frames.

    This avoids extra video decode passes in QA mode.
    """
    os.makedirs(qa_dir, exist_ok=True)

    for stage, sampled in sampled_by_stage.items():
        valid = [idx for idx in sampled if idx in frames]
        if not valid:
            continue

        n = len(valid)
        rows = (n + cols - 1) // cols
        fig, axes = _ensure_axes_grid(rows, cols)

        for i, idx in enumerate(valid):
            r, c = divmod(i, cols)
            axes[r, c].imshow(frames[idx])
            axes[r, c].set_title(f"Frame {idx}", fontsize=8)
            axes[r, c].axis("off")

        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")

        total_rejected = len(stage_rejections.get(stage, []))
        safe = stage.replace(" ", "_").lower()
        plt.suptitle(f"QA - Rejected by {stage} ({n}/{total_rejected})", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(qa_dir, f"qa_{safe}.png"), dpi=140)
        plt.close(fig)


def save_qa_sheet(
    stage_name: str,
    rejected_indices: List[int],
    video_path: str,
    qa_dir: str,
    n_samples: int = 8,
    cols: int = 4,
):
    """Compatibility helper retained for manual one-stage QA (not used by main)."""
    del video_path

    sampled = sample_stage_rejections({stage_name: rejected_indices}, n_samples=n_samples)
    # Cannot render without frames dict in this lightweight compatibility wrapper.
    if sampled.get(stage_name):
        os.makedirs(qa_dir, exist_ok=True)


def save_all_qa_sheets(
    stage_rejections: Dict[str, List[int]],
    video_path: str,
    qa_dir: str,
    n_samples: int = 8,
):
    """Compatibility wrapper retained for older callers."""
    del video_path

    sampled = sample_stage_rejections(stage_rejections, n_samples=n_samples)
    if any(sampled.values()):
        os.makedirs(qa_dir, exist_ok=True)
