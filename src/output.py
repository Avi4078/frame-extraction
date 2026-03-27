"""
Output — Save frames, metadata, funnel report, and visualizations.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


def save_selected_frames(
    frames_rgb: Dict[int, np.ndarray],
    selected_indices: List[int],
    timestamps: List[float],
    quality_scores: List[float],
    output_dir: str,
):
    """
    Save selected frames as JPGs.
    Format: still_NNNN_t=XX.XX_score=X.XX.jpg

    frames_rgb: dict mapping frame_index → RGB numpy array
    """
    os.makedirs(output_dir, exist_ok=True)
    for rank, idx in enumerate(selected_indices, start=1):
        if idx not in frames_rgb:
            continue
        t = timestamps[idx]
        score = quality_scores[idx]
        filename = f"still_{rank:04d}_t={t:.2f}_score={score:.2f}.jpg"
        filepath = os.path.join(output_dir, filename)
        bgr = cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def save_metadata_jsonl(
    selected_indices: List[int],
    timestamps: List[float],
    shot_ids: List[int],
    sharpness: List[float],
    motion: List[float],
    brightness: List[float],
    contrast: List[float],
    colorfulness: List[float],
    edge_density: List[float],
    quality_scores: List[float],
    output_dir: str,
    face_scores: Optional[List[float]] = None,
    face_counts: Optional[List[int]] = None,
    face_ear: Optional[List[float]] = None,
):
    """Write JSONL metadata for each selected frame."""
    filepath = os.path.join(output_dir, "metadata.jsonl")
    with open(filepath, "w") as f:
        for idx in selected_indices:
            record = {
                "frame_index": idx,
                "timestamp": round(timestamps[idx], 3),
                "shot_id": shot_ids[idx],
                "sharpness": round(sharpness[idx], 2),
                "motion": round(motion[idx], 2),
                "brightness": round(brightness[idx], 2),
                "contrast": round(contrast[idx], 2),
                "colorfulness": round(colorfulness[idx], 2),
                "edge_density": round(edge_density[idx], 4),
                "quality_score": round(quality_scores[idx], 4),
            }
            if face_scores is not None:
                record["face_quality_score"] = round(face_scores[idx], 4)
            if face_counts is not None:
                record["n_faces"] = face_counts[idx]
            if face_ear is not None:
                record["face_ear"] = round(face_ear[idx], 4)
            f.write(json.dumps(record) + "\n")


def print_funnel_report(funnel_data: List[Dict[str, Any]], total_frames: int):
    """Print a formatted funnel report table."""
    print("\n" + "=" * 55)
    print("  📊 Frame Funnel Report")
    print("=" * 55)
    print(f"  {'Stage':<22} {'Remaining':>10}  {'% of Total':>10}")
    print("-" * 55)

    for row in funnel_data:
        pct = (row["count"] / total_frames * 100) if total_frames > 0 else 0
        print(f"  {row['stage']:<22} {row['count']:>10,}  {pct:>9.1f}%")

    print("=" * 55)
    return funnel_data


def save_funnel_chart(
    funnel_data: List[Dict[str, Any]],
    save_path: str,
):
    """Visual bar chart of frame reduction funnel."""
    stages = [d["stage"] for d in funnel_data]
    counts = [d["count"] for d in funnel_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(stages)))
    bars = ax.barh(range(len(stages)), counts, color=colors)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Frames Remaining")
    ax.set_title("Frame Reduction Funnel")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_final_contact_sheet(
    frames_rgb: Dict[int, np.ndarray],
    selected_indices: List[int],
    quality_scores: List[float],
    save_path: str,
    cols: int = 5,
):
    """Contact sheet of all final selected stills."""
    # Filter to indices that actually have frames
    valid = [idx for idx in selected_indices if idx in frames_rgb]
    n = len(valid)
    if n == 0:
        return

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, idx in enumerate(valid):
        r, c = divmod(i, cols)
        axes[r][c].imshow(frames_rgb[idx])
        axes[r][c].set_title(f"#{idx} q={quality_scores[idx]:.2f}", fontsize=7)
        axes[r][c].axis("off")

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    plt.suptitle("Final Selected Stills", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
