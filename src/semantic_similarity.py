"""
Semantic-style similarity and diversity reporting utilities.

This uses classical CV signals (structure, color, local keypoints) to approximate
semantic redundancy without heavy model dependencies.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    SEMANTIC_CLUSTER_THRESHOLD,
    SEMANTIC_COLOR_WEIGHT,
    SEMANTIC_HEATMAP_MAX_POINTS,
    SEMANTIC_ORB_WEIGHT,
    SEMANTIC_REDUNDANCY_THRESHOLD,
    SEMANTIC_RESIZE,
    SEMANTIC_SSIM_WEIGHT,
)


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM approximation without external dependency."""
    x = a.astype(np.float32)
    y = b.astype(np.float32)

    ux = float(np.mean(x))
    uy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cxy = float(np.mean((x - ux) * (y - uy)))

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    num = (2.0 * ux * uy + c1) * (2.0 * cxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    if abs(den) < 1e-8:
        return 0.0
    val = num / den
    return float(np.clip(val, 0.0, 1.0))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as sk_ssim  # type: ignore

        val = sk_ssim(a, b, data_range=255)
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return _ssim_fallback(a, b)


def _color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB)

    hist_a = cv2.calcHist([lab_a], [1, 2], None, [16, 16], [0, 256, 0, 256])
    hist_b = cv2.calcHist([lab_b], [1, 2], None, [16, 16], [0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    # Map [-1, 1] -> [0, 1]
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def _orb_similarity(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=500)
    k1, d1 = orb.detectAndCompute(a_gray, None)
    k2, d2 = orb.detectAndCompute(b_gray, None)

    if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return 0.0

    # Reward count and quality of matches.
    distances = np.array([m.distance for m in matches], dtype=np.float32)
    quality = 1.0 - float(np.mean(distances) / 256.0)
    coverage = min(len(matches) / max(len(k1), len(k2), 1), 1.0)
    return float(np.clip(0.6 * quality + 0.4 * coverage, 0.0, 1.0))


def compute_semantic_similarity(
    frames_rgb: Dict[int, np.ndarray],
    indices: Sequence[int],
    resize: int = SEMANTIC_RESIZE,
) -> np.ndarray:
    """Compute pairwise semantic-style similarity matrix in [0, 1]."""
    n = len(indices)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    prep_rgb: List[np.ndarray] = []
    prep_gray: List[np.ndarray] = []

    for idx in indices:
        img = frames_rgb[idx]
        small = _resize_rgb(img, resize)
        prep_rgb.append(small)
        prep_gray.append(_to_gray(small))

    sim = np.eye(n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            ssim_v = _ssim(prep_gray[i], prep_gray[j])
            color_v = _color_similarity(prep_rgb[i], prep_rgb[j])
            orb_v = _orb_similarity(prep_gray[i], prep_gray[j])

            score = (
                SEMANTIC_SSIM_WEIGHT * ssim_v
                + SEMANTIC_COLOR_WEIGHT * color_v
                + SEMANTIC_ORB_WEIGHT * orb_v
            )
            score = float(np.clip(score, 0.0, 1.0))
            sim[i, j] = score
            sim[j, i] = score

    return sim


def cluster_by_similarity(similarity: np.ndarray, threshold: float = SEMANTIC_CLUSTER_THRESHOLD) -> List[List[int]]:
    """Greedy clustering from pairwise similarity matrix."""
    n = similarity.shape[0]
    clusters: List[List[int]] = []

    for i in range(n):
        assigned = False
        for c in clusters:
            # Attach to first cluster where similarity to any member passes threshold.
            if max(float(similarity[i, j]) for j in c) >= threshold:
                c.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    return clusters


def extract_submatrix(similarity: np.ndarray, keep_rows: Sequence[int]) -> np.ndarray:
    """Extract square submatrix using selected row/column indices."""
    if similarity.size == 0 or not keep_rows:
        return np.zeros((0, 0), dtype=np.float32)
    ix = np.array(list(keep_rows), dtype=np.int32)
    return similarity[np.ix_(ix, ix)]


def _classical_mds(distance_matrix: np.ndarray, dim: int = 2) -> np.ndarray:
    """Classical MDS embedding from a distance matrix."""
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
        pad = np.zeros((n, dim - coords.shape[1]), dtype=np.float64)
        coords = np.hstack([coords, pad])

    return coords.astype(np.float32)


def save_semantic_scatter(
    indices: Sequence[int],
    similarity: np.ndarray,
    quality_scores: Sequence[float],
    shot_ids: Sequence[int],
    save_path: str,
):
    """Save 2D MDS scatter derived from semantic similarity."""
    if len(indices) == 0 or similarity.size == 0:
        return

    dist = 1.0 - np.clip(similarity, 0.0, 1.0)
    coords = _classical_mds(dist, dim=2)

    clusters = cluster_by_similarity(similarity, threshold=SEMANTIC_CLUSTER_THRESHOLD)
    cluster_id = {}
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_id[m] = cid

    x = coords[:, 0]
    y = coords[:, 1]
    c = np.array([cluster_id.get(i, 0) for i in range(len(indices))])
    sizes = np.array([max(20.0, 40.0 + 120.0 * float(quality_scores[idx])) for idx in indices])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, c=c, s=sizes, cmap="tab20", alpha=0.85, edgecolors="#666666", linewidths=0.6)

    # Annotate a few high-quality points.
    top = sorted(indices, key=lambda i: quality_scores[i], reverse=True)[:8]
    index_to_row = {idx: i for i, idx in enumerate(indices)}
    for idx in top:
        row = index_to_row[idx]
        ax.text(x[row], y[row], str(idx), fontsize=9)

    n_shots = len({shot_ids[idx] for idx in indices}) if shot_ids else 0
    ax.set_title(
        f"Stage 9 - Semantic scatter (MDS)\npoints={len(indices)}, clusters={len(clusters)}, shots={n_shots}",
        fontsize=14,
    )
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_semantic_heatmap(
    similarity: np.ndarray,
    indices: Sequence[int],
    save_path: str,
    max_points: int = SEMANTIC_HEATMAP_MAX_POINTS,
):
    """Save similarity heatmap (subsampled if too many points)."""
    if similarity.size == 0 or len(indices) == 0:
        return

    if len(indices) <= max_points:
        sub = similarity
        labels = [str(i) for i in indices]
    else:
        sample_idx = np.linspace(0, len(indices) - 1, num=max_points, dtype=int)
        sub = similarity[np.ix_(sample_idx, sample_idx)]
        labels = [str(indices[i]) for i in sample_idx]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sub, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Stage 9 - Semantic similarity heatmap")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")

    if len(labels) <= 40:
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def build_diversity_report(
    final_indices: Sequence[int],
    shot_ids: Sequence[int],
    timestamps: Sequence[float],
    semantic_indices: Sequence[int],
    semantic_similarity: np.ndarray,
    save_path: str | None = None,
) -> Dict[str, float]:
    """Compute and optionally save a diversity summary report."""
    if not final_indices:
        report: Dict[str, float] = {
            "n_final": 0,
            "n_unique_shots": 0,
            "shot_coverage_ratio": 0.0,
            "mean_pairwise_similarity": 0.0,
            "max_pairwise_similarity": 0.0,
            "redundant_pairs": 0,
            "time_span_seconds": 0.0,
        }
    else:
        unique_shots = {shot_ids[i] for i in final_indices}
        total_shots = len({shot_ids[i] for i in semantic_indices}) if semantic_indices else 0

        row_map = {idx: i for i, idx in enumerate(semantic_indices)}
        vals = []
        redundant = 0
        for i in range(len(final_indices)):
            for j in range(i + 1, len(final_indices)):
                a = row_map.get(final_indices[i])
                b = row_map.get(final_indices[j])
                if a is None or b is None:
                    continue
                s = float(semantic_similarity[a, b])
                vals.append(s)
                if s >= SEMANTIC_REDUNDANCY_THRESHOLD:
                    redundant += 1

        ts = [timestamps[i] for i in final_indices]
        span = (max(ts) - min(ts)) if ts else 0.0

        report = {
            "n_final": int(len(final_indices)),
            "n_unique_shots": int(len(unique_shots)),
            "shot_coverage_ratio": float(len(unique_shots) / total_shots) if total_shots > 0 else 0.0,
            "mean_pairwise_similarity": float(np.mean(vals)) if vals else 0.0,
            "max_pairwise_similarity": float(np.max(vals)) if vals else 0.0,
            "redundant_pairs": int(redundant),
            "time_span_seconds": float(span),
        }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
