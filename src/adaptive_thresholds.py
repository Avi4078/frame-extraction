"""
Adaptive threshold estimation for scene cuts and quality filters.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.config import (
    ADAPTIVE_COLORFULNESS_MAX,
    ADAPTIVE_COLORFULNESS_MIN,
    ADAPTIVE_COLORFULNESS_PERCENTILE,
    ADAPTIVE_CONTRAST_HIGH_SPREAD,
    ADAPTIVE_CONTRAST_KEEP_HIGH,
    ADAPTIVE_CONTRAST_KEEP_LOW,
    ADAPTIVE_CONTRAST_LOW_SPREAD,
    ADAPTIVE_CONTRAST_MAX,
    ADAPTIVE_CONTRAST_MIN,
    ADAPTIVE_CUT_MAD_MULTIPLIER,
    ADAPTIVE_CUT_MAX,
    ADAPTIVE_CUT_MIN,
    ADAPTIVE_DARK_MAX,
    ADAPTIVE_DARK_MIN,
    ADAPTIVE_DARK_PERCENTILE,
    COLORFULNESS_THRESHOLD,
    CONTRAST_PERCENTILE,
    CUT_THRESHOLD,
    DARK_THRESHOLD,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_percentile(values: List[float], q: float, default: float) -> float:
    if not values:
        return default
    arr = np.asarray(values, dtype=np.float32)
    return float(np.percentile(arr, q))


def estimate_adaptive_thresholds(
    cut_scores: List[float],
    brightness_scores: List[float],
    contrast_scores: List[float],
    colorfulness_scores: List[float],
) -> Dict[str, float]:
    """
    Estimate robust thresholds from current-video statistics.

    Returns thresholds for:
    - cut_threshold
    - dark_threshold
    - contrast_percentile
    - colorfulness_threshold
    """
    # Defaults first.
    out = {
        "cut_threshold": float(CUT_THRESHOLD),
        "dark_threshold": float(DARK_THRESHOLD),
        "contrast_percentile": float(CONTRAST_PERCENTILE),
        "colorfulness_threshold": float(COLORFULNESS_THRESHOLD),
    }

    # Need a minimum sample size to avoid unstable estimation.
    if len(cut_scores) < 16:
        return out

    # Adaptive cut threshold from robust center + MAD.
    cut_arr = np.asarray(cut_scores, dtype=np.float32)
    cut_med = float(np.median(cut_arr))
    cut_mad = float(np.median(np.abs(cut_arr - cut_med)))
    cut_dyn = cut_med + ADAPTIVE_CUT_MAD_MULTIPLIER * cut_mad
    out["cut_threshold"] = _clamp(cut_dyn, ADAPTIVE_CUT_MIN, ADAPTIVE_CUT_MAX)

    # Adaptive darkness from low-percentile brightness.
    dark_dyn = _safe_percentile(brightness_scores, ADAPTIVE_DARK_PERCENTILE, DARK_THRESHOLD)
    out["dark_threshold"] = _clamp(dark_dyn, ADAPTIVE_DARK_MIN, ADAPTIVE_DARK_MAX)

    # Adaptive contrast keep-percent from spread.
    if contrast_scores:
        c = np.asarray(contrast_scores, dtype=np.float32)
        spread = float(np.percentile(c, 90) - np.percentile(c, 10))

        if spread <= ADAPTIVE_CONTRAST_LOW_SPREAD:
            keep_percent = ADAPTIVE_CONTRAST_KEEP_HIGH
        elif spread >= ADAPTIVE_CONTRAST_HIGH_SPREAD:
            keep_percent = ADAPTIVE_CONTRAST_KEEP_LOW
        else:
            # Linear interpolation in between.
            alpha = (spread - ADAPTIVE_CONTRAST_LOW_SPREAD) / max(
                ADAPTIVE_CONTRAST_HIGH_SPREAD - ADAPTIVE_CONTRAST_LOW_SPREAD,
                1e-6,
            )
            keep_percent = ADAPTIVE_CONTRAST_KEEP_HIGH + alpha * (
                ADAPTIVE_CONTRAST_KEEP_LOW - ADAPTIVE_CONTRAST_KEEP_HIGH
            )
        out["contrast_percentile"] = _clamp(
            keep_percent, ADAPTIVE_CONTRAST_MIN, ADAPTIVE_CONTRAST_MAX
        )

    # Adaptive colorfulness floor from percentile.
    color_dyn = _safe_percentile(
        colorfulness_scores,
        ADAPTIVE_COLORFULNESS_PERCENTILE,
        COLORFULNESS_THRESHOLD,
    )
    out["colorfulness_threshold"] = _clamp(
        color_dyn,
        ADAPTIVE_COLORFULNESS_MIN,
        ADAPTIVE_COLORFULNESS_MAX,
    )

    return out
