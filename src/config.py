from dataclasses import dataclass
from pathlib import Path

DEFAULT_ZUMA_DIR = Path(r"C:\Games\Zuma Deluxe")
DEFAULT_DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"

LOGICAL_W = 640
LOGICAL_H = 480
TOP_HUD_HEIGHT = 32
BALL_DIAMETER = 32.0
BALL_RADIUS = 16.0

COLORS = ("red", "yellow", "green", "blue", "purple", "white")
POWERUPS = ("bomb", "backwards", "slow", "accuracy")

COLOR_BGR = {
    "red": (40, 40, 235),
    "yellow": (30, 225, 240),
    "green": (60, 210, 60),
    "blue": (240, 140, 40),
    "purple": (220, 70, 190),
    "white": (240, 240, 240),
}

COLOR_HEX = {
    "red": "#EB2828",
    "yellow": "#F0E11E",
    "green": "#3CD23C",
    "blue": "#288CF0",
    "purple": "#BE46DC",
    "white": "#F0F0F0",
}

POWERUP_BADGES = {
    "bomb": "B",
    "backwards": "R",
    "slow": "S",
    "accuracy": "A",
}

# Verified circular Hue centers & bounds on OpenCV [0, 180) scale
BALL_HUE_CENTERS = {
    "red": 0.0,
    "yellow": 26.0,
    "green": 63.0,
    "blue": 109.0,
    "purple": 150.0,
}

# (low, high) inclusive intervals in OpenCV H [0..179]
HUE_RANGES = {
    "red": ((0, 14), (166, 180)),
    "yellow": ((15, 39),),
    "green": ((40, 84),),
    "blue": ((85, 130),),
    "purple": ((131, 165),),
}

WHITE_MAX_S = 45
WHITE_MIN_V = 140
WHITE_RATIO_MIN = 0.28
WHITE_MAX_COLORED_S = 95

# Stricter white check on the frog so gray stone skin never reads as a White ball
FROG_WHITE_MAX_S = 44
FROG_WHITE_MIN_V = 165
FROG_WHITE_RATIO_MIN = 0.30
FROG_COLOR_RATIO_MIN = 0.72
FROG_COLOR_MIN_MEDIAN_S = 110.0

COLOR_MIN_S = 55
COLOR_MIN_V = 60
MIN_VALID_COLOR_PIXELS = 12
MIN_VALID_DOT_PIXELS = 6

# Verified against 24-angle rotated powerup sprites (min 0.663) vs 298 normal frames (max 0.592)
POWERUP_MATCH_THRESHOLD = 0.63

# Map detection & curve scanning thresholds
MAP_THUMB_W = 80
MAP_THUMB_H = 60
MAP_MATCH_TRIM_RATIO = 0.60
MAP_MATCH_MAX_MAE = 15.0
SPACE_MEDIAN_MAX_V = 42.0
SPACE_P90_MAX_V = 165.0
MAP_CONFIRM_STREAK = 3

CURVE_SCAN_STEP_PX = 4.0
CURVE_MIN_VISIBILITY = 0.40
CURVE_PRESENCE_DIFF = 24.0
CURVE_PRESENCE_DIFF_ANIMATED = 34.0
CURVE_NMS_MIN_SPACING_PX = 27.5
BALL_FG_DIFF_THRESH = 18
BALL_MIN_FG_RATIO = 0.65
BALL_MIN_QUADRANT_FG_RATIO = 0.44
SKULL_END_EXCLUSION_PX = 48.0
SEGMENT_GAP_PX = 44.0
RAYCAST_CLEARANCE_RADIUS = BALL_RADIUS * 2.25
ADJACENT_DIFF_COLOR_CLEARANCE_RADIUS = BALL_RADIUS * 1.60

# Physics & timing defaults
# levels.xml firespeed="6" px/tick at ~100 ticks/sec (train speed=".5" matches ~50 px/s observed)
BULLET_SPEED_PX_PER_SEC = 600.0
PIPELINE_LATENCY_SEC = 0.03
SWAP_EXTRA_LATENCY_SEC = 0.085
SHOOT_COOLDOWN_SEC = 0.12
DUMP_RELOAD_FLOOR_SEC = 0.20
DYNAMIC_COOLDOWN_PAD_SEC = 0.03
DYNAMIC_COOLDOWN_COMBO_PAD_SEC = 0.10
COMBO_WAIT_SEC = 0.35

MAX_LEAD_PX = 34.0
MAX_LEAD_SINGLE_OR_ENDGAME_PX = 12.0
TRACKER_MAX_VEL_PX = 105.0
TRACKER_MIN_VEL_PX = -120.0
TRACKER_MAX_STEP_DS_PX = 20.0

BLAST_RADIUS_NORMAL_PX = 56.0
BLAST_RADIUS_BOMB_PX = 95.0
BLAST_EXCLUSION_SEC = 0.55

SWAP_COOLDOWN_SEC = 0.16
SWAP_RELOAD_FLOOR_SEC = 0.28
SWAP_VERIFY_TIMEOUT_SEC = 0.22
SAME_TARGET_RADIUS_PX = 56.0

JUNK_WAIT_MAX_SEC = 0.28
DANGER_PROGRESS_THRESHOLD = 0.72
COIN_SAFE_MAX_PROGRESS = 0.82
ENDGAME_BALL_COUNT = 12
COIN_MIN_BALL_COUNT = 7
COIN_MAX_BALL_COUNT = 28
MANUAL_MOUSE_OVERRIDE_PX = 90.0

LEVEL_START_MIN_CONTIGUOUS = 3
LEVEL_START_CONFIRM_FRAMES = 4
LEVEL_CLEAR_ZERO_FRAMES = 25
LEVEL_CLEARED_RECOVER_FRAMES = 8
LEVEL_ROLLIN_MIN_WAIT_SEC = 1.15
LEVEL_ROLLIN_MAX_WAIT_SEC = 2.60
LEVEL_ROLLIN_CALM_VEL_PX = 82.0
LEVEL_ROLLIN_MAX_GROWTH_PER_SEC = 8.0
LEVEL_ROLLIN_CALM_FRAMES = 8

# Strategy scoring weights
WEIGHT_DEFENSE_POWER = 3.0
WEIGHT_DEFENSE_SCALE = 300.0
WEIGHT_COMBO_CASCADE = 85.0
WEIGHT_BALL_REMOVED = 65.0
WEIGHT_BLOCKER_CLEAR = 25.0
WEIGHT_COIN_BONUS = 28.0
WEIGHT_SWAP_PENALTY = -4.0
WEIGHT_JUNK_SPLIT_PENALTY = -30.0
WEIGHT_SELF_COMBO_SETUP = 45.0
WEIGHT_CASCADE_PREP_SETUP = 28.0
WEIGHT_DISTANCE_PENALTY = 0.02

POWERUP_BONUSES = {
    "bomb": 220.0,
    "backwards": 190.0,
    "slow": 165.0,
    "accuracy": 120.0,
}

# Diagnostics & hotkeys
RING_BUFFER_SECONDS = 5.0
RING_BUFFER_FPS = 15.0
MISS_VERIFY_DELAY_SEC = 1.15
AUTO_MISS_COOLDOWN_SEC = 1.5
MAX_DEBUG_DUMPS = 60

EMERGENCY_STOP_HOTKEYS = ("f12", "ctrl+shift+0")
DEBUG_DUMP_HOTKEY = "f9"


@dataclass
class RuntimeConfig:
    auto_play: bool = False
    show_overlay: bool = True
    allow_swap: bool = True
    prioritize_powerups: bool = True
    grab_coins: bool = True
    lead_aim_multiplier: float = 1.0
    shot_cooldown_sec: float = SHOOT_COOLDOWN_SEC
    game_dir: Path = DEFAULT_ZUMA_DIR
