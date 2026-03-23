"""
Stage 7 - Per-shot selection with diversity controls.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    HERO_SIGMA_FRACTION,
    MIN_TEMPORAL_DISTANCE,
    SHOT_K_DIVISOR,
    SHOT_K_MAX,
    SHOT_K_MIN,
    STAGE7_ALLOW_BACKFILL,
    STAGE7_DIVERSITY_BONUS_K_MAX,
    STAGE7_DIVERSITY_ENABLED,
    STAGE7_DIVERSITY_LONG_SHOT_SECONDS,
    STAGE7_DIVERSITY_MAX_K,
    STAGE7_DIVERSITY_STD_STEP,
    STAGE7_DIVERSITY_STD_THRESHOLD,
    STAGE7_MIN_FEATURE_DISTANCE,
    W_COLORFULNESS,
    W_CONTRAST,
    W_EDGE_DENSITY,
    W_HERO_BIAS,
    W_SHARPNESS,
)


def normalize(values: List[float]) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_quality_scores(
    sharpness: List[float],
    contrast: List[float],
    colorfulness: List[float],
    edge_density: List[float],
) -> List[float]:
    """Compute base quality score per frame (without hero bias)."""
    n_sharp = normalize(sharpness)
    n_contrast = normalize(contrast)
    n_color = normalize(colorfulness)
    n_edge = normalize(edge_density)

    base_weight = W_SHARPNESS + W_CONTRAST + W_COLORFULNESS + W_EDGE_DENSITY
    scores = (
        W_SHARPNESS * n_sharp
        + W_CONTRAST * n_contrast
        + W_COLORFULNESS * n_color
        - W_EDGE_DENSITY * n_edge
    ) / max(base_weight, 1e-8)

    return scores.tolist()


def compute_feature_vectors(
    sharpness: List[float],
    contrast: List[float],
    colorfulness: List[float],
    edge_density: List[float],
) -> List[np.ndarray]:
    """Build simple normalized feature vectors for diversity checks."""
    n_sharp = normalize(sharpness)
    n_contrast = normalize(contrast)
    n_color = normalize(colorfulness)
    n_edge = normalize(edge_density)

    out: List[np.ndarray] = []
    for i in range(len(sharpness)):
        out.append(
            np.array([n_sharp[i], n_contrast[i], n_color[i], n_edge[i]], dtype=np.float32)
        )
    return out


def _compute_hero_bias(
    timestamps: List[float],
    indices: List[int],
    shot_start_time: float,
    shot_end_time: float,
) -> Dict[int, float]:
    """
    Compute center bias per candidate within a shot.
    center_weight = exp(-((t - center)^2) / sigma^2)
    """
    shot_duration = shot_end_time - shot_start_time
    if shot_duration <= 0:
        return {idx: 1.0 for idx in indices}

    center = (shot_start_time + shot_end_time) / 2.0
    sigma = max(0.01, HERO_SIGMA_FRACTION * shot_duration)

    bias: Dict[int, float] = {}
    for idx in indices:
        t = timestamps[idx]
        bias[idx] = float(np.exp(-((t - center) ** 2) / (sigma**2)))
    return bias


def _dynamic_k_for_shot(
    shot_duration: float,
    shot_candidate_indices: List[int],
    feature_vectors: Optional[List[np.ndarray]],
) -> int:
    """Compute per-shot K with optional diversity bonus for long, varied shots."""
    base_k = max(SHOT_K_MIN, min(SHOT_K_MAX, math.ceil(shot_duration / SHOT_K_DIVISOR)))

    if (
        not STAGE7_DIVERSITY_ENABLED
        or feature_vectors is None
        or shot_duration < STAGE7_DIVERSITY_LONG_SHOT_SECONDS
        or len(shot_candidate_indices) < 3
    ):
        return base_k

    shot_feats = np.asarray([feature_vectors[i] for i in shot_candidate_indices], dtype=np.float32)
    diversity_std = float(np.std(shot_feats))

    if diversity_std <= STAGE7_DIVERSITY_STD_THRESHOLD:
        return base_k

    bonus_steps = int((diversity_std - STAGE7_DIVERSITY_STD_THRESHOLD) / max(STAGE7_DIVERSITY_STD_STEP, 1e-6)) + 1
    bonus = max(0, min(STAGE7_DIVERSITY_BONUS_K_MAX, bonus_steps))
    return min(STAGE7_DIVERSITY_MAX_K, base_k + bonus)


def _enforce_constraints(
    idx: int,
    selected: List[int],
    timestamps: List[float],
    feature_vectors: Optional[List[np.ndarray]],
    use_feature_distance: bool,
) -> bool:
    """Check temporal and optional feature-distance constraints."""
    t = timestamps[idx]
    for sel_idx in selected:
        if abs(t - timestamps[sel_idx]) < MIN_TEMPORAL_DISTANCE:
            return False

    if use_feature_distance and feature_vectors is not None:
        fv = feature_vectors[idx]
        for sel_idx in selected:
            dist = float(np.linalg.norm(fv - feature_vectors[sel_idx]))
            if dist < STAGE7_MIN_FEATURE_DISTANCE:
                return False

    return True


def select_per_shot(
    quality_scores: List[float],
    shot_ids: List[int],
    candidate_mask: List[bool],
    timestamps: List[float],
    fps: float,
    feature_vectors: Optional[List[np.ndarray]] = None,
) -> List[int]:
    """
    Select up to K frames per shot with hero bias and diversity constraints.

    K = clamp(ceil(shot_duration_sec / SHOT_K_DIVISOR), SHOT_K_MIN, SHOT_K_MAX)
    plus optional diversity bonus for long, varied shots.
    """
    del fps

    n = len(quality_scores)

    shots: Dict[int, List[int]] = {}
    shot_bounds: Dict[int, Tuple[float, float]] = {}

    for i in range(n):
        sid = shot_ids[i]
        t = timestamps[i]
        if sid not in shot_bounds:
            shot_bounds[sid] = (t, t)
        else:
            shot_bounds[sid] = (shot_bounds[sid][0], t)

        if candidate_mask[i]:
            shots.setdefault(sid, []).append(i)

    selected: List[int] = []

    for sid, indices in sorted(shots.items()):
        if not indices:
            continue

        start_t, end_t = shot_bounds[sid]
        shot_duration = max(0.0, end_t - start_t)
        k = _dynamic_k_for_shot(shot_duration, indices, feature_vectors)

        hero_bias = _compute_hero_bias(timestamps, indices, start_t, end_t)

        scored: List[Tuple[int, float]] = []
        for idx in indices:
            base = quality_scores[idx]
            hero = hero_bias.get(idx, 0.5)
            combined = (1.0 - W_HERO_BIAS) * base + W_HERO_BIAS * hero
            scored.append((idx, combined))
        scored.sort(key=lambda x: x[1], reverse=True)

        shot_selected: List[int] = []
        for idx, _ in scored:
            if _enforce_constraints(
                idx,
                shot_selected,
                timestamps,
                feature_vectors,
                use_feature_distance=True,
            ):
                shot_selected.append(idx)
                if len(shot_selected) >= k:
                    break

        # Optional backfill if feature distance was too strict.
        if STAGE7_ALLOW_BACKFILL and len(shot_selected) < k:
            for idx, _ in scored:
                if idx in shot_selected:
                    continue
                if _enforce_constraints(
                    idx,
                    shot_selected,
                    timestamps,
                    feature_vectors,
                    use_feature_distance=False,
                ):
                    shot_selected.append(idx)
                    if len(shot_selected) >= k:
                        break

        selected.extend(shot_selected)

    return sorted(selected)
