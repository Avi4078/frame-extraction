"""
Stage 8 - Global deduplication and similarity visualizations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import (
    ACROSS_SHOT_PHASH_THRESHOLD,
    TOP_PER_CLUSTER,
    WITHIN_SHOT_PHASH_THRESHOLD,
)


def compute_phashes(
    frames_rgb: Dict[int, np.ndarray],
    indices: Sequence[int],
) -> Dict[int, imagehash.ImageHash]:
    """Compute perceptual hash for provided frame indices."""
    hashes: Dict[int, imagehash.ImageHash] = {}
    for idx in indices:
        img = frames_rgb.get(idx)
        if img is None:
            continue
        hashes[idx] = imagehash.phash(Image.fromarray(img))
    return hashes


def _cluster_by_phash(
    indices: Sequence[int],
    hashes: Dict[int, imagehash.ImageHash],
    threshold: int,
) -> List[List[int]]:
    """Greedy clustering by pHash distance threshold."""
    clusters: List[List[int]] = []
    reps: List[imagehash.ImageHash] = []

    for idx in indices:
        h = hashes.get(idx)
        if h is None:
            continue

        assigned = False
        for ci, rep in enumerate(reps):
            if abs(h - rep) < threshold:
                clusters[ci].append(idx)
                assigned = True
                break

        if not assigned:
            clusters.append([idx])
            reps.append(h)

    return clusters


def cluster_indices_by_phash(
    indices: Sequence[int],
    frames_rgb: Dict[int, np.ndarray],
    threshold: int = WITHIN_SHOT_PHASH_THRESHOLD,
) -> List[List[int]]:
    """Build pHash clusters across selected indices."""
    hashes = compute_phashes(frames_rgb, indices)
    return _cluster_by_phash(indices, hashes, threshold)


def deduplicate(
    selected_indices: List[int],
    frames_rgb: Dict[int, np.ndarray],
    quality_scores: List[float],
    shot_ids: List[int],
) -> List[int]:
    """
    Two-tier dedup:
    1) within shot using WITHIN_SHOT_PHASH_THRESHOLD
    2) across shots using ACROSS_SHOT_PHASH_THRESHOLD
    """
    if not selected_indices:
        return []

    hashes = compute_phashes(frames_rgb, selected_indices)

    # Pass 1: within-shot clustering and top-per-cluster keep.
    by_shot: Dict[int, List[int]] = defaultdict(list)
    for idx in selected_indices:
        if idx in hashes:
            by_shot[shot_ids[idx]].append(idx)

    after_within: List[int] = []
    for _, indices in by_shot.items():
        clusters = _cluster_by_phash(indices, hashes, WITHIN_SHOT_PHASH_THRESHOLD)
        for cluster in clusters:
            cluster.sort(key=lambda i: quality_scores[i], reverse=True)
            after_within.extend(cluster[:TOP_PER_CLUSTER])

    # Pass 2: across-shot strict dedup keeping better quality first.
    after_within.sort(key=lambda i: quality_scores[i], reverse=True)
    keep: List[int] = []
    keep_hashes: List[imagehash.ImageHash] = []
    keep_shots: List[int] = []

    for idx in after_within:
        h = hashes.get(idx)
        if h is None:
            continue

        sid = shot_ids[idx]
        dup = False
        for kh, ks in zip(keep_hashes, keep_shots):
            if ks == sid:
                continue
            if abs(h - kh) < ACROSS_SHOT_PHASH_THRESHOLD:
                dup = True
                break

        if not dup:
            keep.append(idx)
            keep_hashes.append(h)
            keep_shots.append(sid)

    return sorted(keep)


def _phash_distance_matrix(
    indices: Sequence[int],
    hashes: Dict[int, imagehash.ImageHash],
) -> np.ndarray:
    n = len(indices)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        hi = hashes[indices[i]]
        for j in range(i + 1, n):
            hj = hashes[indices[j]]
            # imagehash uses 64-bit default hash; normalize to [0,1]
            dist = float(abs(hi - hj)) / 64.0
            d[i, j] = dist
            d[j, i] = dist
    return d


def _classical_mds(distance_matrix: np.ndarray, dim: int = 2) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if n == 1:
        return np.zeros((1, dim), dtype=np.float32)

    d2 = distance_matrix.astype(np.float64) ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j

    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    vals = np.maximum(vals[:dim], 0.0)
    vecs = vecs[:, :dim]
    coords = vecs * np.sqrt(vals)

    if coords.shape[1] < dim:
        coords = np.hstack([coords, np.zeros((n, dim - coords.shape[1]))])

    return coords.astype(np.float32)


def save_similarity_cluster_sheet(
    clusters: List[List[int]],
    frames_rgb: Dict[int, np.ndarray],
    quality_scores: Sequence[float],
    save_path: str,
    max_clusters: int = 0,
    max_per_cluster: int = 0,
):
    """Save row-wise cluster sheet for stage-8 similarity groups."""
    if not clusters:
        return

    sorted_clusters = sorted(clusters, key=len, reverse=True)
    if max_clusters and max_clusters > 0:
        sorted_clusters = sorted_clusters[:max_clusters]

    prepared: List[List[int]] = []
    for c in sorted_clusters:
        ordered = sorted(c, key=lambda i: quality_scores[i], reverse=True)
        if max_per_cluster and max_per_cluster > 0:
            ordered = ordered[:max_per_cluster]
        prepared.append(ordered)

    max_cols = max((len(c) for c in prepared), default=1)
    rows = len(prepared)

    fig, axes = plt.subplots(rows, max_cols, figsize=(max_cols * 2.2, rows * 2.0))

    if rows == 1 and max_cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif max_cols == 1:
        axes = np.array([[a] for a in axes])

    for r, cluster in enumerate(prepared):
        for c in range(max_cols):
            ax = axes[r, c]
            if c < len(cluster):
                idx = cluster[c]
                img = frames_rgb.get(idx)
                if img is not None:
                    ax.imshow(img)
                ax.set_title(f"#{idx} q={quality_scores[idx]:.2f}", fontsize=7)
            ax.axis("off")

        # Label cluster row on first cell.
        axes[r, 0].text(
            0.02,
            0.98,
            f"C{r} n={len(cluster)}",
            transform=axes[r, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
        )

    plt.suptitle(f"Stage 8 - Similarity clusters (rows={rows})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_similarity_scatter(
    indices: List[int],
    frames_rgb: Dict[int, np.ndarray],
    quality_scores: Sequence[float],
    clusters: List[List[int]],
    shot_ids: Sequence[int],
    save_path: str,
):
    """Save pHash-distance scatter via MDS."""
    if not indices:
        return

    hashes = compute_phashes(frames_rgb, indices)
    use_indices = [i for i in indices if i in hashes]
    if not use_indices:
        return

    dist = _phash_distance_matrix(use_indices, hashes)
    coords = _classical_mds(dist, dim=2)

    cluster_map: Dict[int, int] = {}
    for cid, members in enumerate(clusters):
        for idx in members:
            cluster_map[idx] = cid

    x = coords[:, 0]
    y = coords[:, 1]
    colors = np.array([cluster_map.get(idx, -1) for idx in use_indices], dtype=np.int32)
    sizes = np.array([40.0 + 120.0 * float(quality_scores[idx]) for idx in use_indices])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        x,
        y,
        c=colors,
        cmap="tab20",
        s=sizes,
        alpha=0.85,
        edgecolors="#666666",
        linewidths=0.6,
    )

    # Annotate a small set of high-quality points.
    top = sorted(use_indices, key=lambda i: quality_scores[i], reverse=True)[:8]
    row_map = {idx: r for r, idx in enumerate(use_indices)}
    for idx in top:
        r = row_map[idx]
        ax.text(x[r], y[r], str(idx), fontsize=9)

    n_shots = len({shot_ids[idx] for idx in use_indices}) if shot_ids else 0
    ax.set_title(
        f"Stage 8 - Similarity scatter (MDS)\npoints={len(use_indices)}, clusters={len(clusters)}, shots={n_shots}",
        fontsize=12,
    )
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
