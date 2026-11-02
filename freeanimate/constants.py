"""Shared constants, presets, and small helpers for FreeAnimate AI."""

from __future__ import annotations

APP_NAME = "FreeAnimate AI"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Duration presets (minutes) — section 3 of the master prompt
# ---------------------------------------------------------------------------
DURATION_CHOICES = {
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    5: 25,
    6: 30,
    7: 40,
    8: 45,
    9: None,  # custom
}

MIN_CUSTOM_DURATION_MIN = 1
MAX_CUSTOM_DURATION_MIN = 90

# ---------------------------------------------------------------------------
# Long-form film mode limits — master prompt section 15
# ---------------------------------------------------------------------------
# Film mode is a separate production path (``freeanimate film``) with its own
# enforced limits.  The short-video presets above are left untouched so the
# existing ``create``/``make`` flows keep their documented behaviour.
FILM_MIN_DURATION_MIN = 5          # minimum long-video target
FILM_DEFAULT_DURATION_MIN = 15     # default production length
FILM_MAX_DURATION_MIN = 25         # maximum supported duration
FILM_PREVIEW_DEFAULT_SECONDS = 45  # representative short preview before render
FILM_TARGET_SCENE_SECONDS = 40.0   # scene pacing target for long videos
FILM_MAX_SCENES = 240              # hard cap so bad counts cannot explode renders
FILM_DIALOGUE_AUDIO_SHARE = 0.35   # cap on prompt dialogue share of the runtime

# ---------------------------------------------------------------------------
# Export presets
# ---------------------------------------------------------------------------
RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
DEFAULT_RESOLUTION = "1080p"
FRAME_RATES = (24, 25, 30)
DEFAULT_FPS = 24

# ---------------------------------------------------------------------------
# Narration estimates
# ---------------------------------------------------------------------------
DEFAULT_WPM = 130  # spoken words per minute
MIN_WPM = 80
MAX_WPM = 220

# Scene pacing
MIN_SCENE_SECONDS = 4.0
MAX_SCENE_SECONDS = 45.0
TARGET_SCENE_SECONDS = 22.0  # aim for roughly this per scene

# Duration tolerance: warn if final video deviates more than this fraction
DEFAULT_DURATION_TOLERANCE = 0.15  # +/- 15%

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MUSIC_VOLUME = 0.18  # background music relative level
DUCK_THRESHOLD = 0.02
DUCK_RATIO = 8.0

# ---------------------------------------------------------------------------
# Animation / camera movement vocabulary (Mode B + D)
# ---------------------------------------------------------------------------
CAMERA_MOVES = (
    "push_in",
    "pull_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "drift_diagonal",
    "slow_zoom_center",
)

TRANSITIONS = ("fade", "crossfade", "none")

ANIMATION_MODES = {
    "illustration": "Animated illustrations (Ken Burns over generated artwork)",
    "motion_graphics": "Procedural motion graphics (diagrams, timelines, text cards)",
    "hybrid": "Hybrid (illustrations + motion graphics chosen per scene)",
    "ai_video": "AI-generated video clips when a compatible free backend exists",
}
DEFAULT_ANIMATION_MODE = "hybrid"

# ---------------------------------------------------------------------------
# Pipeline stage labels — section 16 progress reporting
# ---------------------------------------------------------------------------
PIPELINE_STAGES = (
    "Preparing script",
    "Planning scenes",
    "Generating narration",
    "Creating subtitles",
    "Preparing visual assets",
    "Rendering video",
    "Assembling final video",
    "Validating output",
    "Generating thumbnails",
)

# ---------------------------------------------------------------------------
# YouTube thumbnails — section 12
# ---------------------------------------------------------------------------
THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720
THUMBNAIL_JPEG_QUALITY = 90
THUMBNAIL_COUNT = 3

# ---------------------------------------------------------------------------
# Windows-safe defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_X264_PRESET = "veryfast"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_CONTAINER = "mp4"
