from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config
from level_data import CurveData, GameAssets, LevelInfo, build_powerup_matrices
from utils import circular_hue_dist, circular_hue_mean

_POWERUP_MATRIX_CACHE: Dict[int, Dict[str, Tuple[np.ndarray, List[str]]]] = {}
_CIRCLE_MASK_CACHE: Dict[Tuple[int, int, bool], np.ndarray] = {}
_RING_OFFSETS_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_RING_QUADRANTS_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_CORE_MASK_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_MEAN_BGR_KERNEL = np.array([[0.3333, 0.3333, 0.3334]], dtype=np.float32)


@dataclass
class DetectedBall:
    curve_idx: int
    s: float
    x: float
    y: float
    color: str
    powerup: Optional[str]
    confidence: float
    visibility: float


@dataclass
class FrogState:
    cur_color: Optional[str]
    next_color: Optional[str]
    angle_rad: float
    cur_confidence: float = 0.0
    next_confidence: float = 0.0


@dataclass
class DetectedCoin:
    x: float
    y: float
    confidence: float


def _match_hue_bucket(hue: float) -> Tuple[Optional[str], float]:
    for color, intervals in config.HUE_RANGES.items():
        for lo, hi in intervals:
            if lo <= hue <= hi:
                center = config.BALL_HUE_CENTERS[color]
                dist = circular_hue_dist(hue, center)
                conf = max(0.25, 1.0 - dist / 35.0)
                return color, conf
    # Nearest fallback
    best_c = min(
        config.BALL_HUE_CENTERS,
        key=lambda c: circular_hue_dist(hue, config.BALL_HUE_CENTERS[c]),
    )
    return best_c, 0.30


def _get_circle_mask(h_img: int, w_img: int, is_dot: bool) -> Tuple[np.ndarray, int]:
    key = (h_img, w_img, is_dot)
    min_px = config.MIN_VALID_DOT_PIXELS if is_dot else config.MIN_VALID_COLOR_PIXELS
    cached = _CIRCLE_MASK_CACHE.get(key)
    if cached is not None:
        return cached, min_px

    yy, xx = np.ogrid[:h_img, :w_img]
    cx = (w_img - 1) / 2.0
    cy = (h_img - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if is_dot:
        r_max = min(h_img, w_img) * 0.40
        mask = dist <= max(2.5, r_max)
    else:
        mask = (dist >= 4.0) & (dist <= min(14.0, min(h_img, w_img) * 0.45))

    if len(_CIRCLE_MASK_CACHE) >= 32:
        _CIRCLE_MASK_CACHE.clear()
    _CIRCLE_MASK_CACHE[key] = mask
    return mask, min_px


def classify_ball_color(
    patch_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    is_dot: bool = False,
    strict_white: bool = False,
    patch_hsv: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], float]:
    """Classify a 32x32 ball patch (or 15x15 swap dot patch) into one of the 6 Zuma colors."""
    if patch_bgr is None or patch_bgr.size == 0:
        return None, 0.0

    h_img, w_img = patch_bgr.shape[:2]
    mask, min_px = _get_circle_mask(h_img, w_img, is_dot)
    if valid_mask is not None and valid_mask.shape == (h_img, w_img):
        mask = mask & valid_mask

    total_px = int(np.count_nonzero(mask))
    if total_px < min_px:
        return None, 0.0

    hsv = (
        patch_hsv
        if (patch_hsv is not None and patch_hsv.shape[:2] == (h_img, w_img))
        else cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    )
    H = hsv[:, :, 0][mask]
    S = hsv[:, :, 1][mask]
    V = hsv[:, :, 2][mask]

    w_max_s = config.FROG_WHITE_MAX_S if strict_white else config.WHITE_MAX_S
    w_min_v = config.FROG_WHITE_MIN_V if strict_white else config.WHITE_MIN_V
    w_ratio_min = config.FROG_WHITE_RATIO_MIN if strict_white else config.WHITE_RATIO_MIN

    white_mask = (S < w_max_s) & (V > w_min_v)
    white_ratio = float(np.count_nonzero(white_mask)) / float(total_px)

    color_mask = (S > config.COLOR_MIN_S) & (V > config.COLOR_MIN_V)
    num_color = int(np.count_nonzero(color_mask))
    s_med = float(np.median(S[color_mask])) if num_color > 0 else 0.0

    if white_ratio >= w_ratio_min and (num_color < min_px or s_med < config.WHITE_MAX_COLORED_S):
        if strict_white and not is_dot and float(np.median(S)) > 60.0:
            return None, 0.0
        conf = min(1.0, 0.55 + white_ratio * 0.6)
        return "white", conf

    if num_color < min_px:
        return None, 0.0

    color_ratio = float(num_color) / float(total_px)
    if not is_dot:
        if strict_white:
            if color_ratio < config.FROG_COLOR_RATIO_MIN or s_med < config.FROG_COLOR_MIN_MEDIAN_S:
                return None, 0.0
        elif color_ratio < 0.40:
            return None, 0.0

    mean_h = circular_hue_mean(H[color_mask])
    color, hue_conf = _match_hue_bucket(mean_h)
    conf = min(1.0, hue_conf * (0.4 + 0.6 * min(1.0, color_ratio * 1.4)))
    return color, conf


def _resolve_powerup_matrices(
    color: str,
    powerup_templates: Union[
        Dict[str, Dict[str, List[np.ndarray]]],
        Dict[str, Tuple[np.ndarray, List[str]]],
    ],
) -> Optional[Tuple[np.ndarray, List[str]]]:
    entry = powerup_templates.get(color)
    if entry is None:
        return None
    if isinstance(entry, tuple):
        return entry
    cache_key = id(powerup_templates)
    cached = _POWERUP_MATRIX_CACHE.get(cache_key)
    if cached is None:
        if len(_POWERUP_MATRIX_CACHE) >= 8:
            _POWERUP_MATRIX_CACHE.clear()
        cached = build_powerup_matrices(powerup_templates)  # type: ignore[arg-type]
        _POWERUP_MATRIX_CACHE[cache_key] = cached
    return cached.get(color)


def classify_powerup(
    patch_bgr: np.ndarray,
    color: Optional[str],
    powerup_templates: Union[
        Dict[str, Dict[str, List[np.ndarray]]],
        Dict[str, Tuple[np.ndarray, List[str]]],
    ],
) -> Tuple[Optional[str], float]:
    """Identify Bomb, Backwards, Slow, or Accuracy overlay inside a 32x32 ball patch via 2-stage BLAS matching."""
    if color is None or color not in powerup_templates:
        return None, 0.0
    if patch_bgr is None or patch_bgr.shape[0] < 22 or patch_bgr.shape[1] < 22:
        return None, 0.0

    resolved = _resolve_powerup_matrices(color, powerup_templates)
    if resolved is None:
        return None, 0.0
    mat_T, labels = resolved

    h, w = patch_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    roi_u8 = patch_bgr[max(0, cy - 11):min(h, cy + 11), max(0, cx - 11):min(w, cx + 11)]
    if roi_u8.shape[0] < 18 or roi_u8.shape[1] < 18:
        return None, 0.0

    roi = np.ascontiguousarray(roi_u8, dtype=np.float32)
    win = sliding_window_view(roi, (18, 18, 3))[:, :, 0]  # (sh, sw, 18, 18, 3)

    # Stage 1: coarse 2px shift stride (3x3 = 9 shifts for 22x22 ROI)
    win9 = win[::2, ::2]
    n9 = win9.shape[0] * win9.shape[1]
    P9 = np.ascontiguousarray(win9.reshape(n9, 972))
    ch_sums9 = win9.sum(axis=(2, 3)).reshape(n9, 3)
    sq_sums9 = np.einsum("ij,ij->i", P9, P9)
    norm_sq9 = sq_sums9 - (ch_sums9 ** 2).sum(axis=1) * (1.0 / 324.0)
    if float(norm_sq9[n9 // 2]) < 64.0 * 972.0:
        return None, 0.0
    inv_norms9 = 1.0 / np.sqrt(np.maximum(norm_sq9, 1e-6))
    scores9 = (P9 @ mat_T) * inv_norms9[:, None]
    best9 = float(scores9.max())
    if best9 < 0.56:
        return None, max(0.0, best9)

    # Stage 2: all 1px shifts (5x5 = 25 shifts for 22x22 ROI)
    n_all = win.shape[0] * win.shape[1]
    P_all = np.ascontiguousarray(win.reshape(n_all, 972))
    ch_sums = win.sum(axis=(2, 3)).reshape(n_all, 3)
    sq_sums = np.einsum("ij,ij->i", P_all, P_all)
    norm_sq = sq_sums - (ch_sums ** 2).sum(axis=1) * (1.0 / 324.0)
    inv_norms = 1.0 / np.sqrt(np.maximum(norm_sq, 1e-6))
    scores = (P_all @ mat_T) * inv_norms[:, None]

    max_per_tpl = scores.max(axis=0)
    best_idx = int(np.argmax(max_per_tpl))
    best_score = float(max_per_tpl[best_idx])

    if best_score >= config.POWERUP_MATCH_THRESHOLD:
        return labels[best_idx], best_score
    return None, max(0.0, best_score)


def classify_powerups_batch(
    patches: List[np.ndarray],
    colors: List[Optional[str]],
    powerup_templates: Union[
        Dict[str, Dict[str, List[np.ndarray]]],
        Dict[str, Tuple[np.ndarray, List[str]]],
    ],
) -> List[Tuple[Optional[str], float]]:
    """Batch-classify power-ups grouped by color via 2-stage BLAS matrix multiplication."""
    results: List[Tuple[Optional[str], float]] = [(None, 0.0)] * len(patches)
    if not patches:
        return results

    by_col: Dict[str, List[int]] = {}
    for idx, (patch, col) in enumerate(zip(patches, colors)):
        if col is not None and col in powerup_templates and patch is not None and patch.shape[0] >= 22 and patch.shape[1] >= 22:
            by_col.setdefault(col, []).append(idx)

    for col, idxs in by_col.items():
        resolved = _resolve_powerup_matrices(col, powerup_templates)
        if resolved is None:
            continue
        mat_T, labels = resolved

        roi_list = []
        for idx in idxs:
            p = patches[idx]
            cy, cx = p.shape[0] // 2, p.shape[1] // 2
            roi_list.append(p[cy - 11:cy + 11, cx - 11:cx + 11])
        rois = np.stack(roi_list, axis=0).astype(np.float32)  # (K, 22, 22, 3)

        win = sliding_window_view(rois, (1, 18, 18, 3))[:, :, :, 0, 0]  # (K, 5, 5, 18, 18, 3)
        win9 = win[:, ::2, ::2]  # (K, 3, 3, 18, 18, 3)
        k_cnt = len(idxs)
        p9 = np.ascontiguousarray(win9.reshape(k_cnt * 9, 972))
        ch_sums9 = win9.sum(axis=(3, 4)).reshape(k_cnt * 9, 3)
        sq_sums9 = np.einsum("ij,ij->i", p9, p9)
        norm_sq9 = sq_sums9 - (ch_sums9 ** 2).sum(axis=1) * (1.0 / 324.0)
        inv_norms9 = np.where(norm_sq9 >= 64.0 * 972.0, 1.0 / np.sqrt(np.maximum(norm_sq9, 1e-6)), 0.0)
        scores9 = ((p9 @ mat_T) * inv_norms9[:, None]).reshape(k_cnt, 9, -1)

        max_per_ball = scores9.max(axis=1)  # (K, N_tpl)
        best_idx_per_ball = np.argmax(max_per_ball, axis=1)
        best_sc_per_ball = max_per_ball[np.arange(k_cnt), best_idx_per_ball]

        for k_i, orig_i in enumerate(idxs):
            sc9 = float(best_sc_per_ball[k_i])
            if sc9 >= config.POWERUP_MATCH_THRESHOLD:
                results[orig_i] = (labels[int(best_idx_per_ball[k_i])], sc9)
            elif sc9 >= 0.56:
                w25 = win[k_i].reshape(25, 18, 18, 3)
                p25 = np.ascontiguousarray(w25.reshape(25, 972))
                cs25 = w25.sum(axis=(1, 2)).reshape(25, 3)
                sq25 = np.einsum("ij,ij->i", p25, p25)
                ns25 = sq25 - (cs25 ** 2).sum(axis=1) * (1.0 / 324.0)
                in25 = np.where(ns25 > 1e-6, 1.0 / np.sqrt(np.maximum(ns25, 1e-6)), 0.0)
                sc25 = (p25 @ mat_T) * in25[:, None]
                m25 = sc25.max(axis=0)
                b_i = int(np.argmax(m25))
                b_s = float(m25[b_i])
                if b_s >= config.POWERUP_MATCH_THRESHOLD:
                    results[orig_i] = (labels[b_i], b_s)
                else:
                    results[orig_i] = (None, max(0.0, b_s))
            else:
                results[orig_i] = (None, max(0.0, sc9))

    return results


def detect_map(
    frame_bgr: np.ndarray,
    assets: GameAssets,
    hint_map_id: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """Match live 640x480 frame against known level thumbnails (or space starfield)."""
    if frame_bgr is None or frame_bgr.shape[:2] != (config.LOGICAL_H, config.LOGICAL_W):
        return None, 999.0

    body = frame_bgr[config.TOP_HUD_HEIGHT:, :]
    thumb = cv2.resize(body, (config.MAP_THUMB_W, config.MAP_THUMB_H), interpolation=cv2.INTER_AREA)
    thumb_i16 = thumb.astype(np.int16)

    if hint_map_id is not None:
        hint_lvl = assets.levels.get(hint_map_id)
        if hint_lvl is not None and hint_lvl.thumb_bgr is not None:
            diff_h = np.abs(thumb_i16 - hint_lvl.thumb_bgr.astype(np.int16)).mean(axis=2).ravel()
            kh = max(1, int(len(diff_h) * config.MAP_MATCH_TRIM_RATIO))
            hint_mae = float(np.mean(np.partition(diff_h, kh - 1)[:kh]))
            if hint_mae <= (config.MAP_MATCH_MAX_MAE * 0.85):
                return hint_map_id, hint_mae

    best_id: Optional[str] = None
    best_mae = 999.0

    for lvl_id, lvl in assets.levels.items():
        if lvl.thumb_bgr is None:
            continue
        diff = np.abs(thumb_i16 - lvl.thumb_bgr.astype(np.int16)).mean(axis=2).ravel()
        k = max(1, int(len(diff) * config.MAP_MATCH_TRIM_RATIO))
        trimmed_mae = float(np.mean(np.partition(diff, k - 1)[:k]))
        if trimmed_mae < best_mae:
            best_mae = trimmed_mae
            best_id = lvl_id

    if best_mae <= config.MAP_MATCH_MAX_MAE:
        return best_id, best_mae

    if "space" in assets.levels:
        gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
        p25 = float(np.percentile(gray, 25))
        med = float(np.median(gray))
        p90 = float(np.percentile(gray, 90))
        p99 = float(np.percentile(gray, 99))
        hud_gray = cv2.cvtColor(frame_bgr[:config.TOP_HUD_HEIGHT, :], cv2.COLOR_BGR2GRAY)
        hud_mean = float(np.mean(hud_gray))
        if (
            p25 <= 18.0
            and med <= config.SPACE_MEDIAN_MAX_V
            and p90 <= config.SPACE_P90_MAX_V
            and p99 >= 55.0
            and hud_mean >= 30.0
        ):
            return "space", med

    return None, best_mae


def is_pause_or_dialog_open(frame_bgr: np.ndarray, level: LevelInfo) -> bool:
    """Detect whether a pause/options/stage-complete modal dialog covers the center board."""
    if level.bg_bgr is None or frame_bgr.shape[:2] != (config.LOGICAL_H, config.LOGICAL_W):
        return False

    # Center box where PopCap dialogbox / STATS panel appears
    center_live = frame_bgr[125:355, 165:475].astype(np.int16)
    center_bg = level.bg_bgr[125:355, 165:475].astype(np.int16)
    center_diff = np.abs(center_live - center_bg).mean(axis=2)
    center_changed_ratio = float(np.mean(center_diff > 28.0))
    if center_changed_ratio < 0.45:
        return False

    hsv_c = cv2.cvtColor(frame_bgr[125:355, 165:475], cv2.COLOR_BGR2HSV)
    # Green STATS level-complete panel (H~56, S~78, V~100; works even on dark slate/green maps)
    green_dialog_mask = (
        (hsv_c[:, :, 0] >= 45)
        & (hsv_c[:, :, 0] <= 72)
        & (hsv_c[:, :, 1] >= 45)
        & (hsv_c[:, :, 1] <= 150)
        & (hsv_c[:, :, 2] >= 55)
        & (hsv_c[:, :, 2] <= 155)
    )
    if float(np.mean(green_dialog_mask)) > 0.50:
        return True

    if center_changed_ratio < 0.72:
        return False

    # Check if outer corners are dimmed compared to background (Zuma dims screen behind dialogs)
    corner_live = cv2.cvtColor(frame_bgr[45:105, 15:95], cv2.COLOR_BGR2GRAY)
    corner_bg = cv2.cvtColor(level.bg_bgr[45:105, 15:95], cv2.COLOR_BGR2GRAY)
    bg_mean = max(10.0, float(np.mean(corner_bg)))
    live_mean = float(np.mean(corner_live))
    if (live_mean / bg_mean) < 0.72:
        return True

    # Also detect warm brownish dialogbox frame pixels even without dimming
    tan_mask = (hsv_c[:, :, 0] >= 10) & (hsv_c[:, :, 0] <= 28) & (hsv_c[:, :, 1] >= 40) & (hsv_c[:, :, 2] >= 40)
    return float(np.mean(tan_mask)) > 0.45 and center_changed_ratio > 0.80


def _visible_cutout_mask(
    level: LevelInfo,
    draw_priority: bool,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Optional[np.ndarray]:
    if draw_priority or level.cutout_canvas is None:
        return None
    return ~level.cutout_canvas[y0:y1, x0:x1]


def _get_ring_offsets(ball_ring_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    key = id(ball_ring_mask)
    cached = _RING_OFFSETS_CACHE.get(key)
    if cached is not None:
        return cached
    wy, wx = np.where(ball_ring_mask)
    half = ball_ring_mask.shape[0] // 2
    ring_dy = (wy - half).astype(np.int32)[::2]
    ring_dx = (wx - half).astype(np.int32)[::2]
    if len(_RING_OFFSETS_CACHE) >= 8:
        _RING_OFFSETS_CACHE.clear()
    _RING_OFFSETS_CACHE[key] = (ring_dy, ring_dx)
    return ring_dy, ring_dx


def _get_ring_quadrants(
    ball_ring_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the 32x32 ball ring mask into 4 geometric quadrants to reject beams/text."""
    key = id(ball_ring_mask)
    cached = _RING_QUADRANTS_CACHE.get(key)
    if cached is not None:
        return cached
    h, w = ball_ring_mask.shape
    yy, xx = np.ogrid[:h, :w]
    hx, hy = w // 2, h // 2
    quads = (
        ball_ring_mask & (xx >= hx) & (yy < hy),
        ball_ring_mask & (xx < hx) & (yy < hy),
        ball_ring_mask & (xx < hx) & (yy >= hy),
        ball_ring_mask & (xx >= hx) & (yy >= hy),
    )
    if len(_RING_QUADRANTS_CACHE) >= 8:
        _RING_QUADRANTS_CACHE.clear()
    _RING_QUADRANTS_CACHE[key] = quads
    return quads


def _get_core_mask(shape: Tuple[int, int]) -> np.ndarray:
    """Return cached r<=7px interior disc mask to reject hollow floating COMBO/score text."""
    cached = _CORE_MASK_CACHE.get(shape)
    if cached is not None:
        return cached
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    core = ((xx - (w - 1) * 0.5) ** 2 + (yy - (h - 1) * 0.5) ** 2) <= 49.0
    if len(_CORE_MASK_CACHE) >= 8:
        _CORE_MASK_CACHE.clear()
    _CORE_MASK_CACHE[shape] = core
    return core


def _compute_diff_map(frame_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """Compute fast 1-channel uint8 mean absdiff across B, G, R in C++."""
    d3 = cv2.absdiff(frame_bgr, bg_bgr)
    return cv2.transform(d3, _MEAN_BGR_KERNEL)


def scan_curve_balls(
    frame_bgr: np.ndarray,
    level: LevelInfo,
    curve: CurveData,
    curve_idx: int,
    assets: GameAssets,
    active_blasts: Sequence[Any] = (),
    diff_map: Optional[np.ndarray] = None,
    frame_hsv: Optional[np.ndarray] = None,
) -> List[DetectedBall]:
    """Scan a single .dat curve in 1D arc-length steps and extract ball centers + colors."""
    scan_limit = max(0.0, curve.total_length - config.SKULL_END_EXCLUSION_PX)
    if scan_limit <= 0.0:
        return []

    sample_s = np.arange(0.0, scan_limit, config.CURVE_SCAN_STEP_PX, dtype=np.float32)
    sample_indices = np.searchsorted(curve.arc_len, sample_s, side="left")
    sample_indices = np.clip(sample_indices, 0, len(curve.points) - 1)

    diff_thr = config.CURVE_PRESENCE_DIFF_ANIMATED if level.has_animated_bg else config.CURVE_PRESENCE_DIFF
    if level.bg_bgr is not None and diff_map is None:
        diff_map = _compute_diff_map(frame_bgr, level.bg_bgr)

    pts = curve.points[sample_indices]
    px_arr = np.rint(pts[:, 0]).astype(np.int32)
    py_arr = np.rint(pts[:, 1]).astype(np.int32)

    valid = (
        (~curve.in_tunnel[sample_indices])
        & (curve.visibility[sample_indices] >= config.CURVE_MIN_VISIBILITY)
        & (px_arr >= 16)
        & (px_arr < config.LOGICAL_W - 16)
        & (py_arr >= config.TOP_HUD_HEIGHT + 8)
        & (py_arr < config.LOGICAL_H - 16)
    )
    valid_idx = np.where(valid)[0]
    scores = np.zeros(len(sample_indices), dtype=np.float32)
    ring_mask = assets.ball_ring_mask

    if len(valid_idx) > 0:
        if diff_map is not None:
            ring_dy, ring_dx = _get_ring_offsets(ring_mask)
            gy = py_arr[valid_idx, None] + ring_dy[None, :]
            gx = px_arr[valid_idx, None] + ring_dx[None, :]
            raw_scores = np.minimum(diff_map[gy, gx], 52).sum(axis=1, dtype=np.int32) * (1.0 / float(len(ring_dy)))

            # Refine exact unoccluded cutout score only where bridge cover actually overlaps
            if level.cutout_canvas is not None:
                cutout = level.cutout_canvas
                for k_i, si in enumerate(valid_idx):
                    pt_idx = int(sample_indices[si])
                    if curve.draw_priority[pt_idx]:
                        continue
                    px = int(px_arr[si])
                    py = int(py_arr[si])
                    x0, y0, x1, y1 = px - 16, py - 16, px + 16, py + 16
                    if cutout[y0:y1, x0:x1].any():
                        mask = ring_mask & (~cutout[y0:y1, x0:x1])
                        if int(np.count_nonzero(mask)) < config.MIN_VALID_COLOR_PIXELS:
                            raw_scores[k_i] = 0.0
                        else:
                            raw_scores[k_i] = float(np.minimum(diff_map[y0:y1, x0:x1][mask], 52).mean())

            scores[valid_idx] = np.where(raw_scores >= diff_thr, raw_scores, 0.0)
        else:
            # Space map: check bright/saturated pixels in the ring against dark background
            if frame_hsv is None:
                frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            for si in valid_idx:
                pt_idx = int(sample_indices[si])
                px = int(px_arr[si])
                py = int(py_arr[si])
                x0, y0 = px - 16, py - 16
                x1, y1 = px + 16, py + 16
                vis_cut = _visible_cutout_mask(level, bool(curve.draw_priority[pt_idx]), x0, y0, x1, y1)
                mask = (ring_mask & vis_cut) if vis_cut is not None else ring_mask
                if vis_cut is not None and int(np.count_nonzero(mask)) < config.MIN_VALID_COLOR_PIXELS:
                    continue
                patch_hsv = frame_hsv[y0:y1, x0:x1]
                S_p = patch_hsv[:, :, 1]
                V_p = patch_hsv[:, :, 2]
                colored_fg = (S_p > 115) & (V_p > 105)
                white_fg = (S_p < 75) & (V_p > 150)
                c_ratio = float(colored_fg[mask].mean())
                w_ratio = float(white_fg[mask].mean())
                if c_ratio >= 0.55:
                    ball_fg = colored_fg
                    fg_ratio = c_ratio
                elif w_ratio >= 0.52:
                    ball_fg = white_fg
                    fg_ratio = w_ratio
                else:
                    continue
                if float(np.percentile(V_p[mask], 75)) < 130.0:
                    continue
                if vis_cut is None or bool(vis_cut.all()):
                    q1, q2, q3, q4 = _get_ring_quadrants(ring_mask)
                    if min(float(ball_fg[q].mean()) for q in (q1, q2, q3, q4)) < 0.25:
                        continue
                scores[si] = fg_ratio * 100.0

    # Non-maximum suppression along 1D arc length (~32px ball spacing)
    candidates = np.where(scores > 0.0)[0].tolist()
    candidates.sort(key=lambda idx: float(scores[idx]), reverse=True)

    picked_si: List[int] = []
    min_samples_sep = config.CURVE_NMS_MIN_SPACING_PX / config.CURVE_SCAN_STEP_PX
    for si in candidates:
        if all(abs(si - p) >= min_samples_sep for p in picked_si):
            picked_si.append(si)

    if not picked_si:
        return []

    picked_si.sort()
    if frame_hsv is None:
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    ball_patches: List[np.ndarray] = []
    ball_colors: List[Optional[str]] = []
    ball_meta: List[Tuple[int, float, float, float, str, float]] = []

    for si in picked_si:
        best_pt_idx = int(sample_indices[si])
        px = int(px_arr[si])
        py = int(py_arr[si])
        x0, y0 = max(0, px - 16), max(0, py - 16)
        x1, y1 = min(config.LOGICAL_W, px + 16), min(config.LOGICAL_H, py + 16)
        if (x1 - x0) != 32 or (y1 - y0) != 32:
            continue

        vis_cut = _visible_cutout_mask(level, bool(curve.draw_priority[best_pt_idx]), x0, y0, x1, y1)
        if diff_map is not None:
            fg_map = diff_map[y0:y1, x0:x1] > config.BALL_FG_DIFF_THRESH
            if vis_cut is None or bool(vis_cut.all()):
                fg_ratio = float(fg_map[ring_mask].mean())
                if fg_ratio < config.BALL_MIN_FG_RATIO:
                    continue
                core_mask = _get_core_mask(ring_mask.shape)
                if float(fg_map[core_mask].mean()) < 0.55:
                    continue
                q1, q2, q3, q4 = _get_ring_quadrants(ring_mask)
                min_q = min(float(fg_map[q].mean()) for q in (q1, q2, q3, q4))
                if min_q < config.BALL_MIN_QUADRANT_FG_RATIO:
                    continue
            else:
                vis_ring = ring_mask & vis_cut
                if int(np.count_nonzero(vis_ring)) < config.MIN_VALID_COLOR_PIXELS:
                    continue
                if float(fg_map[vis_ring].mean()) < config.BALL_MIN_FG_RATIO:
                    continue

        patch = frame_bgr[y0:y1, x0:x1]
        patch_hsv = frame_hsv[y0:y1, x0:x1]
        color, conf = classify_ball_color(patch, valid_mask=vis_cut, is_dot=False, patch_hsv=patch_hsv)
        if color is None or conf < 0.35:
            continue

        # 3-point parabolic sub-step arc-length refinement reduces 4px scan quantization jitter
        s_val = float(curve.arc_len[best_pt_idx])
        x_val = float(curve.points[best_pt_idx, 0])
        y_val = float(curve.points[best_pt_idx, 1])
        if 0 < si < (len(scores) - 1) and scores[si - 1] > 0.0 and scores[si + 1] > 0.0:
            denom = 2.0 * (2.0 * float(scores[si]) - float(scores[si - 1]) - float(scores[si + 1]))
            if denom > 1e-4:
                delta_si = max(-0.5, min(0.5, (float(scores[si - 1]) - float(scores[si + 1])) / denom))
                s_val = max(0.0, min(curve.total_length, float(sample_s[si]) + delta_si * config.CURVE_SCAN_STEP_PX))
                x_val, y_val = curve.point_at_arc_len(s_val)

        ball_patches.append(patch)
        ball_colors.append(color)
        ball_meta.append((best_pt_idx, s_val, x_val, y_val, color, conf))

    if not ball_meta:
        return []

    pu_lookup = assets.powerup_matrices if assets.powerup_matrices else assets.powerup_templates
    pu_results = classify_powerups_batch(ball_patches, ball_colors, pu_lookup)

    balls: List[DetectedBall] = []
    for (best_pt_idx, s_val, x_val, y_val, color, conf), (pu, _) in zip(ball_meta, pu_results):
        balls.append(
            DetectedBall(
                curve_idx=curve_idx,
                s=s_val,
                x=x_val,
                y=y_val,
                color=color,
                powerup=pu,
                confidence=conf,
                visibility=float(curve.visibility[best_pt_idx]),
            )
        )

    return balls


def _sample_patch(frame_bgr: np.ndarray, cx: float, cy: float, half_size: int) -> Optional[np.ndarray]:
    ix, iy = int(round(cx)), int(round(cy))
    x0, y0 = ix - half_size, iy - half_size
    x1, y1 = ix + half_size, iy + half_size
    if x0 < 0 or y0 < 0 or x1 > frame_bgr.shape[1] or y1 > frame_bgr.shape[0]:
        return None
    return frame_bgr[y0:y1, x0:x1]


def detect_frog_balls(
    frame_bgr: np.ndarray,
    frog_xy: Tuple[float, float],
    cursor_logical: Optional[Tuple[float, float]] = None,
    frame_hsv: Optional[np.ndarray] = None,
) -> FrogState:
    """Detect the frog's loaded mouth ball and back preview swap dot."""
    gx, gy = frog_xy
    angle = 0.0

    # Fast cursor-guided path
    if cursor_logical is not None:
        dx = cursor_logical[0] - gx
        dy = cursor_logical[1] - gy
        norm = float(np.hypot(dx, dy))
        if norm >= 6.0:
            ux, uy = dx / norm, dy / norm
            angle = float(np.arctan2(uy, ux))
            dot_patch = _sample_patch(frame_bgr, gx - ux * 25.0, gy - uy * 25.0, 7)
            dot_hsv = _sample_patch(frame_hsv, gx - ux * 25.0, gy - uy * 25.0, 7) if frame_hsv is not None else None
            n_col, n_conf = (
                classify_ball_color(dot_patch, is_dot=True, strict_white=True, patch_hsv=dot_hsv)
                if dot_patch is not None
                else (None, 0.0)
            )
            for mouth_r in (24.0, 21.5):
                mouth_patch = _sample_patch(frame_bgr, gx + ux * mouth_r, gy + uy * mouth_r, 13)
                mouth_hsv = (
                    _sample_patch(frame_hsv, gx + ux * mouth_r, gy + uy * mouth_r, 13)
                    if frame_hsv is not None
                    else None
                )
                c_col, c_conf = (
                    classify_ball_color(mouth_patch, is_dot=False, strict_white=True, patch_hsv=mouth_hsv)
                    if mouth_patch is not None
                    else (None, 0.0)
                )
                if c_col is not None and c_conf >= 0.45:
                    return FrogState(
                        cur_color=c_col,
                        next_color=n_col,
                        angle_rad=angle,
                        cur_confidence=c_conf,
                        next_confidence=n_conf,
                    )

    # 36-angle fallback around frog center (handles cursor outside window or startup)
    if frame_hsv is None:
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    best_score = -1.0
    best_state = FrogState(cur_color=None, next_color=None, angle_rad=0.0)
    for mouth_r in (24.0, 21.5):
        for deg in range(0, 360, 10):
            rad = float(np.deg2rad(deg))
            ux, uy = float(np.cos(rad)), float(np.sin(rad))
            mouth_patch = _sample_patch(frame_bgr, gx + ux * mouth_r, gy + uy * mouth_r, 13)
            if mouth_patch is None:
                continue
            mouth_hsv = _sample_patch(frame_hsv, gx + ux * mouth_r, gy + uy * mouth_r, 13)
            c_col, c_conf = classify_ball_color(mouth_patch, is_dot=False, strict_white=True, patch_hsv=mouth_hsv)
            if c_col is None or c_conf < 0.38:
                continue
            dot_patch = _sample_patch(frame_bgr, gx - ux * 25.0, gy - uy * 25.0, 7)
            dot_hsv = _sample_patch(frame_hsv, gx - ux * 25.0, gy - uy * 25.0, 7)
            n_col, n_conf = (
                classify_ball_color(dot_patch, is_dot=True, strict_white=True, patch_hsv=dot_hsv)
                if dot_patch is not None
                else (None, 0.0)
            )
            total = c_conf + 0.5 * n_conf
            if total > best_score:
                best_score = total
                best_state = FrogState(
                    cur_color=c_col,
                    next_color=n_col,
                    angle_rad=rad,
                    cur_confidence=c_conf,
                    next_confidence=n_conf,
                )
        if best_state.cur_color is not None:
            break

    return best_state


def detect_coins(frame_bgr: np.ndarray, level: LevelInfo) -> List[DetectedCoin]:
    """Detect active bonus coins at the level's exact <TreasurePoint> coordinates."""
    coins: List[DetectedCoin] = []
    if not level.treasure_points:
        return coins

    for tx, ty in level.treasure_points:
        patch = _sample_patch(frame_bgr, tx, ty, 12)
        if patch is None:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gold_mask = (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 155)
        gold_px = int(np.sum(gold_mask))
        if gold_px < 14:
            continue

        if level.bg_bgr is not None:
            bg_patch = _sample_patch(level.bg_bgr, tx, ty, 12)
            if bg_patch is not None:
                diff = float(np.mean(np.abs(patch.astype(np.int16) - bg_patch.astype(np.int16))))
                if diff < 18.0:
                    continue

        coins.append(DetectedCoin(x=tx, y=ty, confidence=min(1.0, gold_px / 120.0)))

    return coins


class VisionPipeline:
    """Stateful CV pipeline with 3-frame map confirmation hysteresis."""

    def __init__(self, assets: GameAssets):
        self.assets = assets
        self._confirmed_map: Optional[str] = None
        self._candidate_map: Optional[str] = None
        self._candidate_streak: int = 0
        self._miss_streak: int = 0

    def reset(self) -> None:
        self._confirmed_map = None
        self._candidate_map = None
        self._candidate_streak = 0
        self._miss_streak = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        cursor_logical: Optional[Tuple[float, float]] = None,
        active_blasts: Sequence[Any] = (),
    ) -> Tuple[Optional[LevelInfo], bool, Dict[int, List[DetectedBall]], FrogState, List[DetectedCoin]]:
        raw_map, _ = detect_map(frame_bgr, self.assets, hint_map_id=self._confirmed_map)
        if raw_map is None:
            self._miss_streak += 1
            self._candidate_streak = 0
            if self._miss_streak >= 3:
                self._confirmed_map = None
                return None, True, {}, FrogState(None, None, 0.0), []
            confirmed = self.assets.levels.get(self._confirmed_map) if self._confirmed_map else None
            return confirmed, True, {}, FrogState(None, None, 0.0), []

        self._miss_streak = 0
        if raw_map == self._confirmed_map:
            self._candidate_map = raw_map
            self._candidate_streak = config.MAP_CONFIRM_STREAK
        else:
            if raw_map == self._candidate_map:
                self._candidate_streak += 1
            else:
                self._candidate_map = raw_map
                self._candidate_streak = 1
            if self._candidate_streak >= config.MAP_CONFIRM_STREAK or self._confirmed_map is None:
                self._confirmed_map = raw_map

        level = self.assets.levels.get(self._confirmed_map)
        if level is None:
            return None, True, {}, FrogState(None, None, 0.0), []

        paused = is_pause_or_dialog_open(frame_bgr, level)
        if paused:
            return level, True, {}, FrogState(None, None, 0.0), []

        diff_map = _compute_diff_map(frame_bgr, level.bg_bgr) if level.bg_bgr is not None else None
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        detected_by_curve: Dict[int, List[DetectedBall]] = {}
        for idx, curve in enumerate(level.curves):
            detected_by_curve[idx] = scan_curve_balls(
                frame_bgr,
                level,
                curve,
                idx,
                self.assets,
                active_blasts=active_blasts,
                diff_map=diff_map,
                frame_hsv=frame_hsv,
            )

        frog = detect_frog_balls(frame_bgr, level.frog_xy, cursor_logical, frame_hsv=frame_hsv)
        coins = detect_coins(frame_bgr, level)
        return level, False, detected_by_curve, frog, coins
