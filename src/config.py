"""
FrameCuration Engine - Configuration and thresholds.

All tunable parameters in one place.
"""

# Stage 1: Scene segmentation
CUT_THRESHOLD = 0.20               # Static fallback cut threshold
POST_CUT_SKIP_SECONDS = 0.4        # Seconds to skip after a detected cut

# Stage 2: Temporal stability
MOTION_THRESHOLD = 12.0            # MAD between consecutive frames

# Stage 3: Blur filtering
BLUR_THRESHOLD = 120.0             # Variance of Laplacian

# Stage 4: Brightness and contrast
DARK_THRESHOLD = 20.0              # Static fallback minimum mean grayscale brightness
CONTRAST_PERCENTILE = 80           # Keep top X percent by contrast (fallback)

# Stage 5: Colorfulness
COLORFULNESS_THRESHOLD = 20.0      # Static fallback Hasler and Susstrunk minimum

# Stage 6: Clutter and simplicity
EDGE_DENSITY_MIN = 0.005           # Minimum edge density
EDGE_DENSITY_MAX = 0.20            # Maximum edge density

# Adaptive thresholds (Phase 2)
ADAPTIVE_THRESHOLDS_ENABLED = True
ADAPTIVE_CUT_MAD_MULTIPLIER = 5.0
ADAPTIVE_CUT_MIN = 0.20
ADAPTIVE_CUT_MAX = 0.60
ADAPTIVE_DARK_PERCENTILE = 15.0
ADAPTIVE_DARK_MIN = 35.0
ADAPTIVE_DARK_MAX = 75.0
ADAPTIVE_CONTRAST_LOW_SPREAD = 20.0
ADAPTIVE_CONTRAST_HIGH_SPREAD = 45.0
ADAPTIVE_CONTRAST_KEEP_HIGH = 70.0
ADAPTIVE_CONTRAST_KEEP_LOW = 45.0
ADAPTIVE_CONTRAST_MIN = 35.0
ADAPTIVE_CONTRAST_MAX = 85.0
ADAPTIVE_COLORFULNESS_PERCENTILE = 40.0
ADAPTIVE_COLORFULNESS_MIN = 25.0
ADAPTIVE_COLORFULNESS_MAX = 95.0

# Stage 7: Per-shot selection
# k = clamp(ceil(shot_duration_sec / SHOT_K_DIVISOR), SHOT_K_MIN, SHOT_K_MAX)
SHOT_K_DIVISOR = 10                # Divide shot duration by this to get K
SHOT_K_MIN = 1                     # Minimum frames per shot
SHOT_K_MAX = 3                     # Base maximum frames per shot
MIN_TEMPORAL_DISTANCE = 1.0        # Minimum seconds between selected frames in a shot

# Stage 7 diversity boost (Phase 3)
STAGE7_DIVERSITY_ENABLED = True
STAGE7_DIVERSITY_LONG_SHOT_SECONDS = 12.0
STAGE7_DIVERSITY_STD_THRESHOLD = 0.13
STAGE7_DIVERSITY_STD_STEP = 0.05
STAGE7_DIVERSITY_BONUS_K_MAX = 2
STAGE7_DIVERSITY_MAX_K = 6
STAGE7_MIN_FEATURE_DISTANCE = 0.18
STAGE7_ALLOW_BACKFILL = True

# Quality score weights
W_SHARPNESS = 0.30
W_CONTRAST = 0.15
W_COLORFULNESS = 0.20
W_EDGE_DENSITY = 0.15              # Subtracted as penalty
W_HERO_BIAS = 0.10                 # Hero frame center bias weight

# Hero frame bias: center_weight = exp(-((t - shot_center)^2) / sigma^2)
HERO_SIGMA_FRACTION = 0.45         # sigma = HERO_SIGMA_FRACTION * shot_duration

# Stage 8: Deduplication
WITHIN_SHOT_PHASH_THRESHOLD = 7    # pHash Hamming distance < this => duplicate (same shot)
ACROSS_SHOT_PHASH_THRESHOLD = 5    # pHash Hamming distance < this => duplicate (cross shot)
STAGE8_VIZ_MAX_CLUSTERS = 50       # Cluster rows in contact sheet (0 = all)
STAGE8_VIZ_MAX_PER_CLUSTER = 8     # Frames per cluster row (0 = all)

# Stage 9: Final selection
TOP_PER_CLUSTER = 1                # Keep top-1 per cluster
FINAL_K_SHORT = 50                 # Top-K for short videos (<5 min)
FINAL_K_LONG = 100                 # Top-K for long videos (>=5 min)
SHORT_VIDEO_MINUTES = 5

# Diversity-aware final rerank (Phase 5)
FINAL_RERANK_ENABLED = True
FINAL_RERANK_LAMBDA = 0.45
FINAL_RERANK_CANDIDATE_MULTIPLIER = 3
FINAL_MAX_PER_SHOT = 2             # 0 disables per-shot cap
FINAL_MIN_TIME_GAP_SEC = 0.0       # 0 disables global temporal gap in final set

# Semantic similarity module (Phase 4 and 6)
SEMANTIC_RESIZE = 224
SEMANTIC_SSIM_WEIGHT = 0.40
SEMANTIC_COLOR_WEIGHT = 0.30
SEMANTIC_ORB_WEIGHT = 0.30
SEMANTIC_CLUSTER_THRESHOLD = 0.72
SEMANTIC_HEATMAP_MAX_POINTS = 120
SEMANTIC_REDUNDANCY_THRESHOLD = 0.80

# Histogram bins for scene segmentation
HIST_BINS = 64

# QA Debug mode
QA_DEBUG = True                    # Set False to disable rejected-frame sampling
QA_SAMPLES_PER_STAGE = 8           # Number of rejected frames to save per stage

# Performance
ANALYSIS_SCALE = 0.5               # Downscale factor for metric computation (0.5 = half res)

# Face landmark quality (Stage 7 boost)
FACE_DETECTION_ENABLED = True
W_FACE_QUALITY = 0.20              # Weight in composite quality score
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MIN_SIZE_FRACTION = 0.03      # Ignore faces smaller than 3% of frame area
FACE_EAR_BLINK_THRESHOLD = 0.21    # Below this = eyes closed / blinking
FACE_YAW_PENALTY_DEG = 45.0        # Start penalizing beyond 45° yaw
FACE_PITCH_PENALTY_DEG = 30.0      # Start penalizing beyond 30° pitch
