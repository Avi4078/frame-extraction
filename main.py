"""
FrameCuration Engine - Main pipeline.

Two-pass architecture:
- Pass 1: metric extraction on streamed frames
- Stage filters + selection
- Pass 2: targeted frame reads for outputs and visualizations
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from src.adaptive_thresholds import estimate_adaptive_thresholds
from src.config import (
    ADAPTIVE_THRESHOLDS_ENABLED,
    ANALYSIS_SCALE,
    BLUR_THRESHOLD,
    COLORFULNESS_THRESHOLD,
    CONTRAST_PERCENTILE,
    CUT_THRESHOLD,
    DARK_THRESHOLD,
    EDGE_DENSITY_MAX,
    EDGE_DENSITY_MIN,
    MOTION_THRESHOLD,
    POST_CUT_SKIP_SECONDS,
    QA_DEBUG,
    QA_SAMPLES_PER_STAGE,
    STAGE8_VIZ_MAX_CLUSTERS,
    STAGE8_VIZ_MAX_PER_CLUSTER,
)
from src.deduplication import (
    cluster_indices_by_phash,
    deduplicate,
    save_similarity_cluster_sheet,
    save_similarity_scatter,
)
from src.final_selection import select_final
from src.frame_stream import get_video_info, stream_frames
from src.output import (
    print_funnel_report,
    save_final_contact_sheet,
    save_funnel_chart,
    save_metadata_jsonl,
    save_selected_frames,
)
from src.per_shot_selection import (
    compute_feature_vectors,
    compute_quality_scores,
    select_per_shot,
)
from src.qa_debug import sample_stage_rejections, save_all_qa_sheets_from_frames
from src.scene_segmentation import (
    bhattacharyya_distance,
    compute_histogram,
    plot_cut_scores,
    save_contact_sheet,
)
from src.semantic_similarity import (
    build_diversity_report,
    compute_semantic_similarity,
    save_semantic_heatmap,
    save_semantic_scatter,
)
from src.temporal_stability import plot_motion_scores
from src.blur_filter import plot_sharpness_histogram, save_best_worst_frames
from src.brightness_contrast import plot_brightness_contrast_scatter
from src.colorfulness import plot_colorfulness_histogram
from src.clutter_filter import plot_edge_density_histogram, save_clean_busy_examples


def parse_args():
    parser = argparse.ArgumentParser(description="FrameCuration Engine")
    parser.add_argument("video_path", type=str, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--viz_dir", type=str, default="visualizations", help="Visualization directory")
    parser.add_argument("--qa_seed", type=int, default=0, help="Seed for QA sampling")
    return parser.parse_args()


def _best_worst_indices(values: list[float], n: int = 5) -> tuple[list[int], list[int]]:
    if not values:
        return [], []
    order = np.argsort(values)
    worst = order[:n].tolist()
    best = order[-n:][::-1].tolist()
    return best, worst


def _read_specific_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Read selected frame indices with mostly sequential decoding."""
    targets = sorted(set(i for i in frame_indices if i >= 0))
    if not targets:
        return {}

    result: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)

    try:
        target_iter = iter(targets)
        next_target = next(target_iter, None)
        idx = 0

        while next_target is not None:
            while idx < next_target:
                if not cap.grab():
                    return result
                idx += 1

            ret, bgr = cap.read()
            if not ret:
                break

            result[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            idx += 1
            next_target = next(target_iter, None)
    finally:
        cap.release()

    return result


def main():
    args = parse_args()

    if not os.path.isfile(args.video_path):
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    qa_dir = os.path.join(args.viz_dir, "qa_rejected")
    if QA_DEBUG:
        os.makedirs(qa_dir, exist_ok=True)

    print("=" * 60)
    print("  FrameCuration Engine")
    print(f"  Analysis scale: {ANALYSIS_SCALE}x")
    print("=" * 60)

    start_time = time.time()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video_path}")
        sys.exit(1)
    info = get_video_info(cap)
    cap.release()

    fps = info["fps"]
    total_expected = info["total_frames"]
    skip_frames = int(POST_CUT_SKIP_SECONDS * fps)

    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {fps:.2f}")
    if info.get("fps_fallback_used"):
        print(
            "  WARNING: source FPS metadata "
            f"({info.get('fps_reported', 0.0):.3f}) invalid; using fallback {fps:.1f}"
        )
    print(f"  Duration: {info['duration_minutes']:.2f} min ({total_expected:,} frames)")

    # Pass 1: stream and compute metrics.
    print("\nPass 1 - Computing per-frame metrics...")

    cut_scores: list[float] = []
    motion_scores: list[float] = []
    sharpness_scores: list[float] = []
    brightness_scores: list[float] = []
    contrast_scores: list[float] = []
    colorfulness_scores_list: list[float] = []
    edge_densities: list[float] = []
    timestamps: list[float] = []

    prev_hist = None
    prev_gray = None

    for frame in tqdm(
        stream_frames(args.video_path),
        total=total_expected if total_expected > 0 else None,
        desc="  Analyzing",
        unit="frame",
        ascii=True,
    ):
        if ANALYSIS_SCALE != 1.0:
            w = max(1, int(frame.width * ANALYSIS_SCALE))
            h = max(1, int(frame.height * ANALYSIS_SCALE))
            rgb_small = cv2.resize(frame.rgb, (w, h), interpolation=cv2.INTER_AREA)
            gray_small = cv2.resize(frame.gray, (w, h), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = frame.rgb
            gray_small = frame.gray

        bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
        timestamps.append(frame.timestamp)

        hist = compute_histogram(bgr_small)
        if prev_hist is None:
            cut_scores.append(0.0)
        else:
            cut_scores.append(bhattacharyya_distance(prev_hist, hist))
        prev_hist = hist

        if prev_gray is None:
            motion_scores.append(0.0)
        else:
            diff = np.abs(gray_small.astype(np.float32) - prev_gray.astype(np.float32))
            motion_scores.append(float(np.mean(diff)))
        prev_gray = gray_small.copy()

        lap = cv2.Laplacian(gray_small, cv2.CV_64F)
        sharpness_scores.append(float(np.var(lap)))

        brightness_scores.append(float(np.mean(gray_small)))
        contrast_scores.append(float(np.std(gray_small)))

        r_chan = rgb_small[:, :, 0].astype(np.float32)
        g_chan = rgb_small[:, :, 1].astype(np.float32)
        b_chan = rgb_small[:, :, 2].astype(np.float32)
        rg = r_chan - g_chan
        yb = 0.5 * (r_chan + g_chan) - b_chan
        colorfulness_scores_list.append(
            float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
        )

        edges = cv2.Canny(gray_small, 50, 150)
        edge_densities.append(float(np.count_nonzero(edges)) / max(edges.size, 1))

    total_frames = len(timestamps)
    print(f"  Analyzed {total_frames:,} frames")

    if total_frames == 0:
        print("No frames decoded; exiting.")
        return

    # Adaptive thresholds.
    if ADAPTIVE_THRESHOLDS_ENABLED:
        adaptive = estimate_adaptive_thresholds(
            cut_scores,
            brightness_scores,
            contrast_scores,
            colorfulness_scores_list,
        )
        cut_threshold = adaptive["cut_threshold"]
        dark_threshold = adaptive["dark_threshold"]
        contrast_percentile = adaptive["contrast_percentile"]
        colorfulness_threshold = adaptive["colorfulness_threshold"]
    else:
        cut_threshold = CUT_THRESHOLD
        dark_threshold = DARK_THRESHOLD
        contrast_percentile = CONTRAST_PERCENTILE
        colorfulness_threshold = COLORFULNESS_THRESHOLD

    print("\nThresholds in use:")
    print(f"  Cut threshold: {cut_threshold:.3f}")
    print(f"  Dark threshold: {dark_threshold:.2f}")
    print(f"  Contrast keep percentile: {contrast_percentile:.1f}")
    print(f"  Colorfulness threshold: {colorfulness_threshold:.2f}")

    # Stage filters.
    print("\nApplying filters...")
    qa_rejected = defaultdict(list)

    # Stage 1: scene cuts + post-cut exclusion.
    cut_indices = [i for i in range(1, total_frames) if cut_scores[i] > cut_threshold]
    cut_set = set(cut_indices)

    is_excluded = [False] * total_frames
    shot_ids = [0] * total_frames
    current_shot = 0

    for i in range(total_frames):
        if i in cut_set:
            current_shot += 1
            for j in range(i, min(i + skip_frames, total_frames)):
                is_excluded[j] = True
        shot_ids[i] = current_shot
        if is_excluded[i]:
            qa_rejected["post_cut"].append(i)

    n_shots = current_shot + 1
    post_cut_remaining = sum(1 for x in is_excluded if not x)
    print(
        f"  Stage 1: {len(cut_indices)} cuts -> {n_shots} shots, "
        f"{post_cut_remaining:,} remain"
    )

    # Stage 2: motion.
    is_stable = [m < MOTION_THRESHOLD for m in motion_scores]
    for i, ok in enumerate(is_stable):
        if not ok:
            qa_rejected["motion"].append(i)

    motion_remaining = sum(1 for i in range(total_frames) if is_stable[i] and not is_excluded[i])
    print(f"  Stage 2: {motion_remaining:,} after motion filter")

    # Stage 3: blur.
    is_sharp = [s > BLUR_THRESHOLD for s in sharpness_scores]
    for i, ok in enumerate(is_sharp):
        if not ok:
            qa_rejected["blur"].append(i)

    blur_remaining = sum(
        1 for i in range(total_frames) if is_sharp[i] and is_stable[i] and not is_excluded[i]
    )
    print(f"  Stage 3: {blur_remaining:,} after blur filter")

    # Stage 4: brightness + contrast.
    contrast_cutoff = (
        float(np.percentile(contrast_scores, 100.0 - contrast_percentile))
        if contrast_scores
        else 0.0
    )
    is_bright = [
        (brightness_scores[i] >= dark_threshold and contrast_scores[i] >= contrast_cutoff)
        for i in range(total_frames)
    ]
    for i, ok in enumerate(is_bright):
        if not ok:
            qa_rejected["brightness_contrast"].append(i)

    bc_remaining = sum(
        1
        for i in range(total_frames)
        if is_bright[i] and is_sharp[i] and is_stable[i] and not is_excluded[i]
    )
    print(f"  Stage 4: {bc_remaining:,} after brightness/contrast filter")

    # Stage 5: colorfulness.
    is_colorful = [c >= colorfulness_threshold for c in colorfulness_scores_list]
    for i, ok in enumerate(is_colorful):
        if not ok:
            qa_rejected["colorfulness"].append(i)

    color_remaining = sum(
        1
        for i in range(total_frames)
        if is_colorful[i] and is_bright[i] and is_sharp[i] and is_stable[i] and not is_excluded[i]
    )
    print(f"  Stage 5: {color_remaining:,} after colorfulness filter")

    # Stage 6: clutter.
    is_clean = [EDGE_DENSITY_MIN < d < EDGE_DENSITY_MAX for d in edge_densities]
    for i, ok in enumerate(is_clean):
        if not ok:
            qa_rejected["clutter"].append(i)

    clutter_remaining = sum(
        1
        for i in range(total_frames)
        if is_clean[i]
        and is_colorful[i]
        and is_bright[i]
        and is_sharp[i]
        and is_stable[i]
        and not is_excluded[i]
    )
    print(f"  Stage 6: {clutter_remaining:,} after clutter filter")

    candidate_mask = [
        (not is_excluded[i])
        and is_stable[i]
        and is_sharp[i]
        and is_bright[i]
        and is_colorful[i]
        and is_clean[i]
        for i in range(total_frames)
    ]

    # Selection stages.
    print("\nSelection stages...")

    quality_scores = compute_quality_scores(
        sharpness_scores,
        contrast_scores,
        colorfulness_scores_list,
        edge_densities,
    )
    feature_vectors = compute_feature_vectors(
        sharpness_scores,
        contrast_scores,
        colorfulness_scores_list,
        edge_densities,
    )

    per_shot_indices = select_per_shot(
        quality_scores,
        shot_ids,
        candidate_mask,
        timestamps,
        fps,
        feature_vectors=feature_vectors,
    )
    print(f"  Stage 7: {len(per_shot_indices):,} frames after per-shot selection")

    print(f"  Stage 8: Reading {len(per_shot_indices)} frames for dedup hashing...")
    dedup_frames_rgb = _read_specific_frames(args.video_path, per_shot_indices)
    similarity_clusters = cluster_indices_by_phash(per_shot_indices, dedup_frames_rgb)

    deduped_indices = deduplicate(per_shot_indices, dedup_frames_rgb, quality_scores, shot_ids)
    print(f"  Stage 8: {len(deduped_indices):,} frames after deduplication")

    semantic_indices = list(deduped_indices)
    semantic_similarity = compute_semantic_similarity(dedup_frames_rgb, semantic_indices)

    final_indices = select_final(
        deduped_indices,
        quality_scores,
        info["duration_minutes"],
        shot_ids=shot_ids,
        timestamps=timestamps,
        semantic_indices=semantic_indices,
        semantic_similarity=semantic_similarity,
    )
    print(f"  Stage 9: {len(final_indices)} final stills selected")

    # Stage 9 semantic visuals and report.
    sem_scatter_path = os.path.join(args.viz_dir, "stage9_semantic_scatter.png")
    sem_heatmap_path = os.path.join(args.viz_dir, "stage9_semantic_heatmap.png")
    sem_report_path = os.path.join(args.viz_dir, "stage9_diversity_report.json")

    save_semantic_scatter(
        semantic_indices,
        semantic_similarity,
        quality_scores,
        shot_ids,
        sem_scatter_path,
    )
    save_semantic_heatmap(semantic_similarity, semantic_indices, sem_heatmap_path)
    diversity_report = build_diversity_report(
        final_indices,
        shot_ids,
        timestamps,
        semantic_indices,
        semantic_similarity,
        save_path=sem_report_path,
    )
    print(
        "  Diversity: "
        f"shots={diversity_report['n_unique_shots']}, "
        f"mean_sim={diversity_report['mean_pairwise_similarity']:.3f}, "
        f"redundant_pairs={diversity_report['redundant_pairs']}"
    )

    # Build visualization index sets.
    sharp_best_indices, sharp_worst_indices = _best_worst_indices(sharpness_scores, n=5)
    edge_sorted = np.argsort(edge_densities)
    clean_example_indices = edge_sorted[:5].tolist()
    busy_example_indices = edge_sorted[-5:][::-1].tolist()

    qa_samples = {}
    qa_sample_indices: list[int] = []
    if QA_DEBUG:
        qa_samples = sample_stage_rejections(
            qa_rejected,
            n_samples=QA_SAMPLES_PER_STAGE,
            seed=args.qa_seed,
        )
        qa_sample_indices = sorted({idx for sample in qa_samples.values() for idx in sample})

    post_selection_needed = set(final_indices)
    post_selection_needed.update(cut_indices)
    post_selection_needed.update(sharp_best_indices)
    post_selection_needed.update(sharp_worst_indices)
    post_selection_needed.update(clean_example_indices)
    post_selection_needed.update(busy_example_indices)
    post_selection_needed.update(qa_sample_indices)

    missing_indices = sorted(idx for idx in post_selection_needed if idx not in dedup_frames_rgb)
    if missing_indices:
        print(f"\nPass 2 - Reading {len(missing_indices)} additional frames...")
        pass2_frames_rgb = _read_specific_frames(args.video_path, missing_indices)
    else:
        pass2_frames_rgb = {}

    all_read_frames_rgb = dict(dedup_frames_rgb)
    all_read_frames_rgb.update(pass2_frames_rgb)

    # Save outputs.
    save_selected_frames(all_read_frames_rgb, final_indices, timestamps, quality_scores, args.output_dir)
    save_metadata_jsonl(
        final_indices,
        timestamps,
        shot_ids,
        sharpness_scores,
        motion_scores,
        brightness_scores,
        contrast_scores,
        colorfulness_scores_list,
        edge_densities,
        quality_scores,
        args.output_dir,
    )
    save_final_contact_sheet(
        all_read_frames_rgb,
        final_indices,
        quality_scores,
        os.path.join(args.viz_dir, "final_contact_sheet.png"),
    )

    # Core visualizations.
    print("\nGenerating visualizations...")
    plot_cut_scores(
        cut_scores,
        cut_indices,
        fps,
        os.path.join(args.viz_dir, "stage1_cut_scores.png"),
        threshold=cut_threshold,
    )
    if cut_indices:
        save_contact_sheet(
            all_read_frames_rgb,
            cut_indices,
            os.path.join(args.viz_dir, "stage1_contact_sheet.png"),
        )

    plot_motion_scores(
        motion_scores,
        is_stable,
        fps,
        MOTION_THRESHOLD,
        os.path.join(args.viz_dir, "stage2_motion.png"),
    )
    plot_sharpness_histogram(
        sharpness_scores,
        BLUR_THRESHOLD,
        os.path.join(args.viz_dir, "stage3_sharpness.png"),
    )
    save_best_worst_frames(
        all_read_frames_rgb,
        sharpness_scores,
        os.path.join(args.viz_dir, "stage3_sharpness_best_worst.png"),
        n=5,
        best_indices=sharp_best_indices,
        worst_indices=sharp_worst_indices,
    )
    plot_brightness_contrast_scatter(
        brightness_scores,
        contrast_scores,
        is_bright,
        dark_threshold,
        contrast_cutoff,
        os.path.join(args.viz_dir, "stage4_brightness_contrast.png"),
    )
    plot_colorfulness_histogram(
        colorfulness_scores_list,
        colorfulness_threshold,
        os.path.join(args.viz_dir, "stage5_colorfulness.png"),
    )
    plot_edge_density_histogram(
        edge_densities,
        EDGE_DENSITY_MIN,
        EDGE_DENSITY_MAX,
        os.path.join(args.viz_dir, "stage6_edge_density.png"),
    )
    save_clean_busy_examples(
        all_read_frames_rgb,
        edge_densities,
        os.path.join(args.viz_dir, "stage6_clean_busy_examples.png"),
        clean_indices=clean_example_indices,
        busy_indices=busy_example_indices,
    )
    save_similarity_cluster_sheet(
        clusters=similarity_clusters,
        frames_rgb=dedup_frames_rgb,
        quality_scores=quality_scores,
        save_path=os.path.join(args.viz_dir, "stage8_similarity_clusters.png"),
        max_clusters=STAGE8_VIZ_MAX_CLUSTERS,
        max_per_cluster=STAGE8_VIZ_MAX_PER_CLUSTER,
    )
    save_similarity_scatter(
        indices=per_shot_indices,
        frames_rgb=dedup_frames_rgb,
        quality_scores=quality_scores,
        clusters=similarity_clusters,
        shot_ids=shot_ids,
        save_path=os.path.join(args.viz_dir, "stage8_similarity_scatter.png"),
    )

    if QA_DEBUG:
        print("  Saving QA rejection sheets...")
        save_all_qa_sheets_from_frames(
            stage_rejections=qa_rejected,
            sampled_by_stage=qa_samples,
            frames=all_read_frames_rgb,
            qa_dir=qa_dir,
        )

    # Funnel report.
    funnel_data = [
        {"stage": "Raw frames", "count": total_frames},
        {"stage": "Post-cut exclusion", "count": post_cut_remaining},
        {"stage": "Motion filter", "count": motion_remaining},
        {"stage": "Blur filter", "count": blur_remaining},
        {"stage": "Brightness/contrast", "count": bc_remaining},
        {"stage": "Colorfulness", "count": color_remaining},
        {"stage": "Clutter filter", "count": clutter_remaining},
        {"stage": "Per-shot top-K", "count": len(per_shot_indices)},
        {"stage": "Deduplicated", "count": len(deduped_indices)},
        {"stage": "Final selection", "count": len(final_indices)},
    ]
    print_funnel_report(funnel_data, total_frames)
    save_funnel_chart(funnel_data, os.path.join(args.viz_dir, "funnel_chart.png"))

    elapsed = time.time() - start_time
    print(f"\nComplete in {elapsed:.1f}s")
    print(f"  {len(final_indices)} stills -> {args.output_dir}/")
    print(f"  Visualizations -> {args.viz_dir}/")
    if QA_DEBUG:
        print(f"  QA sheets -> {qa_dir}/")


if __name__ == "__main__":
    main()
