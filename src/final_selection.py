"""
Stage 9 - Final selection with optional diversity-aware reranking.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.config import (
    FINAL_K_LONG,
    FINAL_K_SHORT,
    FINAL_MAX_PER_SHOT,
    FINAL_MIN_TIME_GAP_SEC,
    FINAL_RERANK_CANDIDATE_MULTIPLIER,
    FINAL_RERANK_ENABLED,
    FINAL_RERANK_LAMBDA,
    SHORT_VIDEO_MINUTES,
)


def _topk_by_quality(
    indices: List[int],
    quality_scores: List[float],
    k: int,
) -> List[int]:
    scored = [(idx, quality_scores[idx]) for idx in indices]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:k]]


def _passes_global_constraints(
    idx: int,
    selected: List[int],
    shot_ids: Optional[List[int]],
    timestamps: Optional[List[float]],
    shot_counts: Dict[int, int],
) -> bool:
    if shot_ids is not None and FINAL_MAX_PER_SHOT > 0:
        sid = shot_ids[idx]
        if shot_counts[sid] >= FINAL_MAX_PER_SHOT:
            return False

    if timestamps is not None and FINAL_MIN_TIME_GAP_SEC > 0:
        t = timestamps[idx]
        for s in selected:
            if abs(t - timestamps[s]) < FINAL_MIN_TIME_GAP_SEC:
                return False

    return True


def select_final(
    deduped_indices: List[int],
    quality_scores: List[float],
    video_duration_minutes: float,
    shot_ids: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    semantic_indices: Optional[List[int]] = None,
    semantic_similarity: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Select final top-K frames.

    If semantic matrix and indices are provided and rerank is enabled, run MMR-like
    quality-diversity rerank. Otherwise fallback to quality-only top-K.
    """
    k = FINAL_K_SHORT if video_duration_minutes < SHORT_VIDEO_MINUTES else FINAL_K_LONG
    if not deduped_indices:
        return []

    if (
        not FINAL_RERANK_ENABLED
        or semantic_similarity is None
        or semantic_indices is None
        or semantic_similarity.size == 0
    ):
        return sorted(_topk_by_quality(deduped_indices, quality_scores, k))

    ranked_by_quality = _topk_by_quality(
        deduped_indices,
        quality_scores,
        min(len(deduped_indices), max(k, k * FINAL_RERANK_CANDIDATE_MULTIPLIER)),
    )

    sem_map = {idx: i for i, idx in enumerate(semantic_indices)}
    selected: List[int] = []
    shot_counts: Dict[int, int] = defaultdict(int)

    while len(selected) < k:
        best_idx = None
        best_score = -1e9

        for idx in ranked_by_quality:
            if idx in selected:
                continue
            if not _passes_global_constraints(idx, selected, shot_ids, timestamps, shot_counts):
                continue

            quality = float(quality_scores[idx])
            row = sem_map.get(idx)
            redundancy = 0.0

            if row is not None and selected:
                sims = []
                for s in selected:
                    col = sem_map.get(s)
                    if col is not None:
                        sims.append(float(semantic_similarity[row, col]))
                if sims:
                    redundancy = max(sims)

            score = quality - FINAL_RERANK_LAMBDA * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        if shot_ids is not None and FINAL_MAX_PER_SHOT > 0:
            shot_counts[shot_ids[best_idx]] += 1

    # Quality backfill if constraints/rerank left gaps.
    if len(selected) < k:
        for idx in _topk_by_quality(deduped_indices, quality_scores, len(deduped_indices)):
            if idx in selected:
                continue
            if not _passes_global_constraints(idx, selected, shot_ids, timestamps, shot_counts):
                continue
            selected.append(idx)
            if shot_ids is not None and FINAL_MAX_PER_SHOT > 0:
                shot_counts[shot_ids[idx]] += 1
            if len(selected) >= k:
                break

    return sorted(selected)
