import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
from level_data import CurveData, LevelInfo
from tracker import BlastZone, TrackState, TrackedBall
from utils import point_segment_dist, ray_circle_first_hit
from vision import DetectedCoin, FrogState


@dataclass
class ShotDecision:
    action: str                        # "shoot", "swap_shoot", "wait", "none"
    mode: str                          # "CONTROL", "DEFENSE", "PAUSED"
    aim_xy: Optional[Tuple[float, float]]
    shot_color: Optional[str]
    target_ball_id: Optional[int]
    target_curve_idx: Optional[int]
    target_s: Optional[float]
    expected_removed: int
    cascades: int
    score: float
    reason: str
    blocked: bool = False
    blocker_xy: Optional[Tuple[float, float]] = None
    is_coin: bool = False
    is_junk_dump: bool = False
    is_bomb: bool = False
    will_rollback: bool = False
    is_backwards: bool = False


def simulate_insert(colors: List[str], insert_idx: int, shot_color: str) -> Tuple[int, int]:
    """Simulate inserting shot_color at insert_idx; return (total_balls_removed, cascades)."""
    seq = list(colors[:insert_idx]) + [shot_color] + list(colors[insert_idx:])
    cursor = insert_idx
    total_removed = 0
    cascades = 0

    while 0 <= cursor < len(seq):
        c = seq[cursor]
        left = cursor
        while left - 1 >= 0 and seq[left - 1] == c:
            left -= 1
        right = cursor
        while right + 1 < len(seq) and seq[right + 1] == c:
            right += 1

        run_len = right - left + 1
        if run_len >= 3:
            total_removed += run_len
            if total_removed > run_len:
                cascades += 1
            seq = seq[:left] + seq[right + 1:]
            if not seq or left == 0 or left >= len(seq):
                break
            # In Zuma, a gap closes and triggers a cascade only if both sides share color
            if seq[left - 1] != seq[left]:
                break
            cursor = left
        else:
            break

    return total_removed, cascades


def predict_gap_rollback(colors: List[str], insert_idx: int, shot_color: str) -> bool:
    """Return True when popping at insert_idx leaves matching end colors that trigger a Zuma pullback."""
    seq = list(colors[:insert_idx]) + [shot_color] + list(colors[insert_idx:])
    left = insert_idx
    while left - 1 >= 0 and seq[left - 1] == shot_color:
        left -= 1
    right = insert_idx
    while right + 1 < len(seq) and seq[right + 1] == shot_color:
        right += 1
    if (right - left + 1) < 3:
        return False
    return left > 0 and (right + 1) < len(seq) and seq[left - 1] == seq[right + 1]


def compute_lead_aim(
    frog_xy: Tuple[float, float],
    curve: CurveData,
    s0: float,
    velocity: float,
    bullet_speed: float = config.BULLET_SPEED_PX_PER_SEC,
    latency: float = config.PIPELINE_LATENCY_SEC,
    lead_mult: float = 1.0,
    max_lead_px: float = config.MAX_LEAD_PX,
) -> Tuple[float, Tuple[float, float], float]:
    """2-iteration ballistic intercept along 1D curve arc length with safe lead clamping."""
    fx, fy = frog_xy
    s_pred = float(s0)
    px, py = curve.point_at_arc_len(s_pred)

    for _ in range(2):
        dist = math.hypot(px - fx, py - fy)
        t_hit = (dist / max(bullet_speed, 100.0) + latency) * lead_mult
        delta_s = max(-max_lead_px, min(max_lead_px, velocity * t_hit))
        s_pred = max(0.0, min(curve.total_length, s0 + delta_s))
        px, py = curve.point_at_arc_len(s_pred)

    final_dist = math.hypot(px - fx, py - fy)
    return s_pred, (px, py), final_dist


def split_into_segments(
    curve_balls: List[TrackedBall],
    gap_px: float = config.SEGMENT_GAP_PX,
) -> List[List[TrackedBall]]:
    """Split sorted balls on a curve into contiguous trains separated by empty gaps."""
    if not curve_balls:
        return []
    segments: List[List[TrackedBall]] = [[curve_balls[0]]]
    for b in curve_balls[1:]:
        if (b.s - segments[-1][-1].s) > gap_px:
            segments.append([b])
        else:
            segments[-1].append(b)
    return segments


def check_line_of_sight(
    frog_xy: Tuple[float, float],
    aim_xy: Tuple[float, float],
    all_balls: List[TrackedBall],
    ignore_ball_ids: set[int],
    clearance_radius: float = config.RAYCAST_CLEARANCE_RADIUS,
    curves: Optional[List[CurveData]] = None,
    predicted_pts: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    adj_diff_ids: Optional[set[int]] = None,
    adj_clearance_radius: float = config.ADJACENT_DIFF_COLOR_CLEARANCE_RADIUS,
) -> Tuple[bool, Optional[TrackedBall]]:
    """Cast a ball-width capsule from frog_xy to aim_xy across all curves (including moving blockers)."""
    fx, fy = frog_xy
    ax, ay = aim_xy
    dx = ax - fx
    dy = ay - fy
    target_dist = math.hypot(dx, dy)
    if target_dist <= 1e-6:
        return True, None
    dir_unit = (dx / target_dist, dy / target_dist)

    closest_blocker: Optional[TrackedBall] = None
    closest_t = target_dist

    for b in all_balls:
        if b.ball_id in ignore_ball_ids or b.in_tunnel_memory:
            continue
        is_adj_diff = (adj_diff_ids is not None and b.ball_id in adj_diff_ids)
        r_check = adj_clearance_radius if is_adj_diff else clearance_radius
        max_t = (target_dist - 2.0) if is_adj_diff else (target_dist - config.BALL_RADIUS * 0.5)

        if is_adj_diff and predicted_pts is not None and b.ball_id in predicted_pts:
            # Touching neighbors travel alongside the lead-shifted target; test at bullet impact time.
            hit_t = ray_circle_first_hit(frog_xy, dir_unit, predicted_pts[b.ball_id][0], r_check)
        else:
            hit_t = ray_circle_first_hit(frog_xy, dir_unit, (b.x, b.y), r_check)
            if hit_t is None:
                if predicted_pts is not None:
                    pred_list = predicted_pts.get(b.ball_id)
                    if pred_list:
                        for pred_xy in pred_list:
                            hit_t = ray_circle_first_hit(frog_xy, dir_unit, pred_xy, r_check)
                            if hit_t is not None:
                                break
                elif curves is not None and abs(b.velocity) > 5.0 and 0 <= b.curve_idx < len(curves):
                    b_dist = math.hypot(b.x - fx, b.y - fy)
                    t_block = b_dist / max(config.BULLET_SPEED_PX_PER_SEC, 100.0) + config.PIPELINE_LATENCY_SEC
                    pred_s = max(0.0, min(curves[b.curve_idx].total_length, b.s + max(-20.0, min(20.0, b.velocity * t_block))))
                    bx_pred, by_pred = curves[b.curve_idx].point_at_arc_len(pred_s)
                    hit_t = ray_circle_first_hit(frog_xy, dir_unit, (bx_pred, by_pred), r_check)
        eff_cap = min(closest_t, max_t)
        if hit_t is not None and 24.0 < hit_t < eff_cap:
            closest_t = hit_t
            closest_blocker = b

    if closest_blocker is not None:
        return False, closest_blocker
    return True, None


def find_offscreen_dump_point(
    frog_xy: Tuple[float, float],
    all_balls: List[TrackedBall],
    predicted_pts: Optional[Dict[int, List[Tuple[float, float]]]] = None,
) -> Optional[Tuple[float, float]]:
    """Find an unobstructed straight shot off the playable area, avoiding moving trains."""
    fx, fy = frog_xy
    min_x, max_x = 8.0, float(config.LOGICAL_W - 8)
    min_y, max_y = float(config.TOP_HUD_HEIGHT + 8), float(config.LOGICAL_H - 8)
    visible = [b for b in all_balls if not b.in_tunnel_memory]

    exit_rays: List[Tuple[float, float]] = []
    for deg in range(0, 360, 2):
        rad = math.radians(deg)
        ux, uy = math.cos(rad), math.sin(rad)
        t_candidates = []
        if ux > 1e-6:
            t_candidates.append((max_x - fx) / ux)
        elif ux < -1e-6:
            t_candidates.append((min_x - fx) / ux)
        if uy > 1e-6:
            t_candidates.append((max_y - fy) / uy)
        elif uy < -1e-6:
            t_candidates.append((min_y - fy) / uy)
        t_edge = min((t for t in t_candidates if t > 0.0), default=420.0)
        exit_rays.append((fx + ux * t_edge, fy + uy * t_edge))

    pts_to_avoid: List[Tuple[float, float]] = [(b.x, b.y) for b in visible]
    if predicted_pts:
        for b in visible:
            pts_to_avoid.extend(predicted_pts.get(b.ball_id, ()))

    for clear_radius in (
        config.BALL_RADIUS * 2.15,
        config.BALL_RADIUS * 1.95,
        config.BALL_RADIUS * 1.65,
    ):
        best_pt: Optional[Tuple[float, float]] = None
        best_margin = -1.0
        for aim_xy in exit_rays:
            clear, _ = check_line_of_sight(
                frog_xy,
                aim_xy,
                all_balls,
                set(),
                clearance_radius=clear_radius,
                predicted_pts=predicted_pts,
            )
            if clear:
                min_d = min(
                    (point_segment_dist(pt, frog_xy, aim_xy) for pt in pts_to_avoid),
                    default=100.0,
                )
                if min_d > best_margin:
                    best_margin = min_d
                    best_pt = aim_xy
        if best_pt is not None:
            return best_pt

    return None


def _in_active_blast(curve_idx: int, s: float, active_blasts: List[BlastZone]) -> bool:
    return any(b.curve_idx == curve_idx and abs(s - b.s) <= b.radius_px for b in active_blasts)


def _pop_danger_score(mode: str, run_progress: float) -> float:
    """Scale progress urgency so longer chains win unless a group enters the skull danger zone."""
    base = run_progress ** config.WEIGHT_DEFENSE_POWER
    if mode == "DEFENSE":
        if run_progress >= config.DANGER_PROGRESS_THRESHOLD:
            return base * (config.WEIGHT_DEFENSE_SCALE * 2.5)
        return base * (config.WEIGHT_DEFENSE_SCALE * 0.55)
    return base * (config.WEIGHT_DEFENSE_SCALE * 0.35)


class StrategySolver:
    """Two-mode raycast, ballistic lead-aim, swap-ball, and junk-fallback decision engine."""

    def __init__(self):
        self._junk_wait_started: Optional[float] = None
        self._recent_board_colors: Dict[str, float] = {}

    def reset(self) -> None:
        self._junk_wait_started = None
        self._recent_board_colors.clear()

    def decide(
        self,
        level: LevelInfo,
        track_state: TrackState,
        frog: FrogState,
        coins: List[DetectedCoin],
        cfg: config.RuntimeConfig,
        now: float,
    ) -> ShotDecision:
        mode = "DEFENSE" if track_state.max_progress >= config.DANGER_PROGRESS_THRESHOLD else "CONTROL"
        all_balls = track_state.all_balls
        visible_balls = [b for b in all_balls if not b.in_tunnel_memory]

        if frog.cur_color is None or not visible_balls:
            return ShotDecision(
                action="none",
                mode=mode,
                aim_xy=None,
                shot_color=frog.cur_color,
                target_ball_id=None,
                target_curve_idx=None,
                target_s=None,
                expected_removed=0,
                cascades=0,
                score=0.0,
                reason="waiting_for_balls",
            )

        num_visible = len(visible_balls)
        is_endgame = (num_visible <= config.ENDGAME_BALL_COUNT)
        board_colors = {b.color for b in visible_balls}
        for c in board_colors:
            self._recent_board_colors[c] = now

        colors_to_try: List[Tuple[str, str]] = [("shoot", frog.cur_color)]
        if cfg.allow_swap and frog.next_color is not None and frog.next_color != frog.cur_color:
            colors_to_try.append(("swap_shoot", frog.next_color))

        fx, fy = level.frog_xy
        predicted_pts: Dict[int, List[Tuple[float, float]]] = {}
        inv_speed = 1.0 / max(config.BULLET_SPEED_PX_PER_SEC, 100.0)
        for b in all_balls:
            if not b.in_tunnel_memory and abs(b.velocity) > 5.0 and 0 <= b.curve_idx < len(level.curves):
                b_dist = math.hypot(b.x - fx, b.y - fy)
                t_block = b_dist * inv_speed + config.PIPELINE_LATENCY_SEC
                curve_b = level.curves[b.curve_idx]
                pred_s1 = max(0.0, min(curve_b.total_length, b.s + max(-config.MAX_LEAD_PX, min(config.MAX_LEAD_PX, b.velocity * t_block))))
                pred_s2 = max(0.0, min(curve_b.total_length, b.s + max(-36.0, min(36.0, b.velocity * (t_block + 0.12)))))
                predicted_pts[b.ball_id] = [
                    curve_b.point_at_arc_len(pred_s1),
                    curve_b.point_at_arc_len(pred_s2),
                ]

        clear_matches: List[ShotDecision] = []
        blocked_matches: List[ShotDecision] = []
        blocker_bonus: Dict[int, float] = {}
        setup_candidates: List[ShotDecision] = []
        setup_relaxed_candidates: List[ShotDecision] = []
        seam_candidates: List[ShotDecision] = []
        has_blocked_pair_for_cur_or_next = False

        # Pass 1: evaluate every visible ball slot per contiguous segment on every curve
        for curve_idx, curve_balls in track_state.balls_by_curve.items():
            if curve_idx >= len(level.curves):
                continue
            curve = level.curves[curve_idx]
            segments = split_into_segments(curve_balls)
            has_safe_rear_seg = any(
                bool(s and s[-1].progress < config.DANGER_PROGRESS_THRESHOLD)
                for s in segments
            )
            max_lead = (
                config.MAX_LEAD_SINGLE_OR_ENDGAME_PX
                if len(curve_balls) <= 5
                else config.MAX_LEAD_PX
            )

            for seg_idx, seg in enumerate(segments):
                seq_colors = [b.color for b in seg]

                for bi, ball in enumerate(seg):
                    if ball.in_tunnel_memory:
                        continue

                    # Contiguous same-color run around bi inside this segment
                    li = bi
                    while li - 1 >= 0 and seg[li - 1].color == ball.color:
                        li -= 1
                    ri = bi
                    while ri + 1 < len(seg) and seg[ri + 1].color == ball.color:
                        ri += 1
                    run_len = ri - li + 1

                    # Visible bounds of the contiguous same-color run (only vis_li represents the run for same-color shots)
                    vis_li = li
                    while vis_li <= ri and seg[vis_li].in_tunnel_memory:
                        vis_li += 1
                    vis_ri = ri
                    while vis_ri >= li and seg[vis_ri].in_tunnel_memory:
                        vis_ri -= 1
                    is_run_rep = (bi == vis_li)

                    # Ignore ONLY same-color balls in this contiguous run; check touching diff-color neighbors via adj_diff_ids
                    same_run_ignore = {seg[k].ball_id for k in range(li, ri + 1)}
                    adj_diff_ids: set[int] = set()
                    has_left_diff = (li - 1 >= 0 and abs(seg[li].s - seg[li - 1].s) <= 38.0)
                    has_right_diff = (ri + 1 < len(seg) and abs(seg[ri + 1].s - seg[ri].s) <= 38.0)
                    if has_left_diff:
                        adj_diff_ids.add(seg[li - 1].ball_id)
                    if has_right_diff:
                        adj_diff_ids.add(seg[ri + 1].ball_id)

                    def _clamp_s(val: float) -> float:
                        return max(0.0, min(curve.total_length, val))

                    # Build aim candidates shifted away from oblique different-colored touching neighbors
                    aim_s_candidates: List[Tuple[float, set[int]]] = []
                    if run_len >= 2:
                        if is_run_rep and vis_li <= vis_ri:
                            s_mid = (seg[vis_li].s + seg[vis_ri].s) * 0.5
                            if has_left_diff and has_right_diff:
                                s_seq = [s_mid, _clamp_s(s_mid + 4.0), _clamp_s(s_mid - 4.0)]
                            elif has_left_diff:
                                # Keep aim centered inside the run so lead never overshoots past the front ball into empty track
                                s_seq = [s_mid, _clamp_s(s_mid + 3.0), seg[vis_li].s]
                            elif has_right_diff:
                                s_seq = [s_mid, _clamp_s(s_mid - 3.0), seg[vis_ri].s]
                            else:
                                s_seq = [s_mid, seg[vis_li].s, seg[vis_ri].s]
                            for s_c in s_seq:
                                aim_s_candidates.append((s_c, same_run_ignore))
                    else:
                        if has_left_diff and has_right_diff:
                            single_offsets = (0.0, 3.0, -3.0)
                        elif has_left_diff:
                            single_offsets = (0.0, 4.0) if bi == (len(seg) - 1) else (0.0, 5.0, 9.0)
                        elif has_right_diff:
                            single_offsets = (0.0, -4.0) if bi == 0 else (0.0, -5.0, -9.0)
                        else:
                            single_offsets = (0.0,) if (bi == 0 or bi == (len(seg) - 1)) else (0.0, 6.0, -6.0)
                        for off_s in single_offsets:
                            aim_s_candidates.append((_clamp_s(ball.s + off_s), same_run_ignore))

                    def _make_setup_decision(
                        a_name: str,
                        s_col: str,
                        a_xy: Tuple[float, float],
                        s_pred_val: float,
                    ) -> ShotDecision:
                        d_frog = math.hypot(a_xy[0] - level.frog_xy[0], a_xy[1] - level.frog_xy[1])
                        d_pen = d_frog * (
                            config.WEIGHT_DISTANCE_PENALTY
                            if is_endgame
                            else config.WEIGHT_DISTANCE_PENALTY * 2.0
                        )
                        pu_setup = (
                            config.POWERUP_BONUSES.get(ball.powerup, 0.0) * 0.75
                            if (cfg.prioritize_powerups and ball.powerup)
                            else 0.0
                        )
                        self_combo_bonus = (
                            config.WEIGHT_SELF_COMBO_SETUP
                            if (a_name == "shoot" and frog.next_color == s_col)
                            else 0.0
                        )
                        cascade_prep_bonus = 0.0
                        endgame_close_bonus = 0.0
                        if is_endgame:
                            endgame_close_bonus = max(0.0, (1.0 - d_frog / 400.0) * 20.0)
                        elif li - 1 >= 0 and ri + 1 < len(seg) and seg[li - 1].color == seg[ri + 1].color:
                            nbr_color = seg[li - 1].color
                            l_cnt = 1
                            while li - 1 - l_cnt >= 0 and seg[li - 1 - l_cnt].color == nbr_color:
                                l_cnt += 1
                            r_cnt = 1
                            while ri + 1 + r_cnt < len(seg) and seg[ri + 1 + r_cnt].color == nbr_color:
                                r_cnt += 1
                            if (l_cnt + r_cnt) >= 3:
                                cascade_prep_bonus = config.WEIGHT_CASCADE_PREP_SETUP

                        if mode == "DEFENSE":
                            in_danger_seg = bool(seg and seg[-1].progress >= config.DANGER_PROGRESS_THRESHOLD)
                            if has_safe_rear_seg and in_danger_seg and not (cfg.prioritize_powerups and ball.powerup):
                                progress_term = -35.0 + ball.progress * 10.0
                            else:
                                progress_term = ball.progress * 15.0
                        else:
                            progress_term = ball.progress * 15.0

                        setup_score = (
                            12.0
                            + progress_term
                            + pu_setup
                            + self_combo_bonus
                            + cascade_prep_bonus
                            + endgame_close_bonus
                            + (ball.progress * 50.0 if (is_endgame and mode == "CONTROL") else 0.0)
                            + swap_pen
                            - d_pen
                        )
                        return ShotDecision(
                            action=a_name,
                            mode=mode,
                            aim_xy=a_xy,
                            shot_color=s_col,
                            target_ball_id=ball.ball_id,
                            target_curve_idx=curve_idx,
                            target_s=s_pred_val,
                            expected_removed=0,
                            cascades=0,
                            score=setup_score,
                            reason="setup_pair",
                        )

                    for action_name, shot_color in colors_to_try:
                        swap_pen = config.WEIGHT_SWAP_PENALTY if action_name == "swap_shoot" else 0.0
                        shot_latency = config.PIPELINE_LATENCY_SEC + (
                            config.SWAP_EXTRA_LATENCY_SEC if action_name == "swap_shoot" else 0.0
                        )
                        removed, cascades = simulate_insert(seq_colors, bi, shot_color)
                        is_same_color = (ball.color == shot_color)
                        if is_same_color and removed < 3 and action_name == "swap_shoot":
                            continue
                        if is_same_color and run_len >= 2 and not is_run_rep:
                            continue
                        if (
                            is_same_color
                            and removed < 3
                            and mode == "CONTROL"
                            and has_left_diff
                            and has_right_diff
                        ):
                            is_fast_single = abs(ball.velocity) >= 65.0
                            is_sharp_curve = False
                            if 20.0 < ball.s < (curve.total_length - 20.0):
                                p0 = curve.point_at_arc_len(ball.s - 20.0)
                                p2 = curve.point_at_arc_len(ball.s + 20.0)
                                v1x, v1y = ball.x - p0[0], ball.y - p0[1]
                                v2x, v2y = p2[0] - ball.x, p2[1] - ball.y
                                n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
                                if n1 > 1e-3 and n2 > 1e-3:
                                    is_sharp_curve = ((v1x * v2x + v1y * v2y) / (n1 * n2)) < 0.866
                            if is_fast_single or is_sharp_curve:
                                continue

                        in_pair_or_more = (run_len >= 2)

                        # Resolve first valid clear (or best blocked) aim point for this slot
                        chosen_s_pred = None
                        chosen_aim_xy = None
                        chosen_clear = False
                        chosen_blocker = None

                        cand_list = (
                            aim_s_candidates
                            if is_same_color
                            else [(ball.s, same_run_ignore)]
                        )
                        # Cap brief combo-rollback lead, but use full negative speed during sustained backwards/Zuma reverse
                        if ball.velocity < 0.0:
                            aim_vel = ball.velocity if ball.reverse_streak >= 5 else max(-18.0, ball.velocity)
                        else:
                            aim_vel = ball.velocity
                        for s_try, ign in cand_list:
                            s_p, a_xy, _ = compute_lead_aim(
                                level.frog_xy,
                                curve,
                                s_try,
                                aim_vel,
                                latency=shot_latency,
                                lead_mult=cfg.lead_aim_multiplier,
                                max_lead_px=max_lead,
                            )
                            if curve.is_tunnel_at(s_p) or curve.visibility_at(s_p) < 0.50:
                                continue
                            if _in_active_blast(curve_idx, s_p, track_state.active_blasts):
                                continue

                            clr, blk = check_line_of_sight(
                                level.frog_xy,
                                a_xy,
                                all_balls,
                                ign,
                                curves=level.curves,
                                predicted_pts=predicted_pts,
                                adj_diff_ids=adj_diff_ids,
                            )
                            if chosen_aim_xy is None or clr:
                                chosen_s_pred = s_p
                                chosen_aim_xy = a_xy
                                chosen_clear = clr
                                chosen_blocker = blk
                            if clr:
                                break

                        # Fallback physical-clearance pass for reachable same-color targets (singles OR pairs)
                        if is_same_color and not chosen_clear and shot_color in board_colors:
                            min_rel_vis = 0.20 if num_visible <= 6 else 0.35
                            if run_len >= 2 and is_run_rep and vis_li <= vis_ri:
                                s_mid_rel = (seg[vis_li].s + seg[vis_ri].s) * 0.5
                                rel_s_candidates = (
                                    s_mid_rel,
                                    _clamp_s(s_mid_rel + 8.0),
                                    _clamp_s(s_mid_rel - 8.0),
                                    seg[vis_li].s,
                                    seg[vis_ri].s,
                                )
                            else:
                                if has_left_diff and has_right_diff:
                                    rel_offsets = (0.0, 4.0, -4.0)
                                elif has_left_diff:
                                    rel_offsets = (0.0, 6.0)
                                elif has_right_diff:
                                    rel_offsets = (0.0, -6.0)
                                else:
                                    rel_offsets = (0.0, 8.0, -8.0)
                                rel_s_candidates = tuple(_clamp_s(ball.s + off_s) for off_s in rel_offsets)
                            for s_rel in rel_s_candidates:
                                s_pr, a_xyr, _ = compute_lead_aim(
                                    level.frog_xy,
                                    curve,
                                    s_rel,
                                    aim_vel,
                                    latency=shot_latency,
                                    lead_mult=cfg.lead_aim_multiplier,
                                    max_lead_px=max_lead,
                                )
                                if curve.is_tunnel_at(s_pr) or curve.visibility_at(s_pr) < min_rel_vis:
                                    continue
                                if _in_active_blast(curve_idx, s_pr, track_state.active_blasts):
                                    continue
                                r_clear, _ = check_line_of_sight(
                                    level.frog_xy,
                                    a_xyr,
                                    all_balls,
                                    same_run_ignore,
                                    clearance_radius=config.BALL_RADIUS * 2.02,
                                    curves=level.curves,
                                    predicted_pts=predicted_pts,
                                    adj_diff_ids=adj_diff_ids,
                                    adj_clearance_radius=config.BALL_RADIUS * (1.78 if removed >= 3 else 1.45),
                                )
                                if r_clear:
                                    rel_dec = _make_setup_decision(action_name, shot_color, a_xyr, s_pr)
                                    if removed >= 3:
                                        rel_rb = predict_gap_rollback(seq_colors, bi, shot_color)
                                        rel_dec.score += (
                                            removed * config.WEIGHT_BALL_REMOVED
                                            + cascades * config.WEIGHT_COMBO_CASCADE
                                            + _pop_danger_score(mode, seg[ri].progress)
                                            + (seg[ri].progress * 95.0 if is_endgame else 0.0)
                                            + (12.0 if (rel_rb and mode == "CONTROL") else 0.0)
                                        )
                                        rel_dec.expected_removed = removed
                                        rel_dec.cascades = cascades
                                        rel_dec.will_rollback = rel_rb
                                        rel_dec.reason = f"match_{removed}_casc_{cascades}_relaxed"
                                    setup_relaxed_candidates.append(rel_dec)
                                    break

                        if chosen_aim_xy is None or chosen_s_pred is None:
                            continue

                        dist_from_frog = math.hypot(
                            chosen_aim_xy[0] - level.frog_xy[0],
                            chosen_aim_xy[1] - level.frog_xy[1],
                        )
                        dist_pen = dist_from_frog * config.WEIGHT_DISTANCE_PENALTY if mode == "CONTROL" else 0.0

                        if removed >= 3 and is_same_color:
                            run_progress = seg[ri].progress
                            danger = _pop_danger_score(mode, run_progress)
                            will_rb = predict_gap_rollback(seq_colors, bi, shot_color)

                            pu_bonus = 0.0
                            is_bomb = False
                            is_backwards = False
                            if cfg.prioritize_powerups:
                                for k in range(li, ri + 1):
                                    pu = seg[k].powerup
                                    if pu:
                                        pu_bonus = max(pu_bonus, config.POWERUP_BONUSES.get(pu, 0.0))
                                        if pu == "bomb":
                                            is_bomb = True
                                        elif pu == "backwards":
                                            is_backwards = True

                            score = (
                                removed * config.WEIGHT_BALL_REMOVED
                                + cascades * config.WEIGHT_COMBO_CASCADE
                                + danger
                                + pu_bonus
                                + swap_pen
                                + (run_progress * 95.0 if is_endgame else 0.0)
                                + (12.0 if (will_rb and mode == "CONTROL") else 0.0)
                                - dist_pen
                            )

                            reason_str = f"match_{removed}_casc_{cascades}"
                            if pu_bonus > 0:
                                reason_str += "_pu"

                            dec = ShotDecision(
                                action=action_name,
                                mode=mode,
                                aim_xy=chosen_aim_xy,
                                shot_color=shot_color,
                                target_ball_id=ball.ball_id,
                                target_curve_idx=curve_idx,
                                target_s=chosen_s_pred,
                                expected_removed=removed,
                                cascades=cascades,
                                score=score,
                                reason=reason_str,
                                blocked=not chosen_clear,
                                blocker_xy=(chosen_blocker.x, chosen_blocker.y) if chosen_blocker else None,
                                is_bomb=is_bomb,
                                will_rollback=will_rb,
                                is_backwards=is_backwards,
                            )
                            if chosen_clear:
                                clear_matches.append(dec)
                            else:
                                blocked_matches.append(dec)
                                has_blocked_pair_for_cur_or_next = True
                                if chosen_blocker is not None:
                                    blocker_bonus[chosen_blocker.ball_id] = max(
                                        blocker_bonus.get(chosen_blocker.ball_id, 0.0),
                                        config.WEIGHT_BLOCKER_CLEAR + (30.0 if mode == "DEFENSE" else 0.0),
                                    )
                        elif chosen_clear:
                            if is_same_color:
                                if shot_color not in board_colors:
                                    continue
                                setup_candidates.append(
                                    _make_setup_decision(action_name, shot_color, chosen_aim_xy, chosen_s_pred)
                                )
                            else:
                                split_pen = config.WEIGHT_JUNK_SPLIT_PENALTY if in_pair_or_more else 0.0
                                seam_score = -10.0 + split_pen + swap_pen - (ball.progress * 15.0)
                                seam_candidates.append(
                                    ShotDecision(
                                        action=action_name,
                                        mode=mode,
                                        aim_xy=chosen_aim_xy,
                                        shot_color=shot_color,
                                        target_ball_id=ball.ball_id,
                                        target_curve_idx=curve_idx,
                                        target_s=chosen_s_pred,
                                        expected_removed=0,
                                        cascades=0,
                                        score=seam_score,
                                        reason="safe_seam_insert" if split_pen == 0.0 else "forced_insert",
                                    )
                                )

        # Also check if any contiguous pair exists inside a segment (or tunnel) so junk hold knows
        if not has_blocked_pair_for_cur_or_next:
            for curve_balls in track_state.balls_by_curve.values():
                for seg in split_into_segments(curve_balls):
                    for i in range(len(seg) - 1):
                        if seg[i].color == seg[i + 1].color:
                            if seg[i].color in {c for _, c in colors_to_try}:
                                has_blocked_pair_for_cur_or_next = True
                                break

        # Apply unblocking bonus to clear matches that happen to clear an inner-coil blocker
        for dec in clear_matches:
            if dec.target_ball_id is not None and dec.target_ball_id in blocker_bonus:
                dec.score += blocker_bonus[dec.target_ball_id]
                dec.reason += "_unblock"

        # Check for unobstructed coin opportunity (uses physical 31.2px raycast clearance)
        best_coin: Optional[ShotDecision] = None
        if (
            cfg.grab_coins
            and track_state.max_progress < config.COIN_SAFE_MAX_PROGRESS
            and coins
        ):
            for coin in coins:
                clear, _ = check_line_of_sight(
                    level.frog_xy,
                    (coin.x, coin.y),
                    all_balls,
                    set(),
                    clearance_radius=config.BALL_RADIUS * 2.15,
                    predicted_pts=predicted_pts,
                )
                if clear:
                    best_coin = ShotDecision(
                        action="shoot",
                        mode=mode,
                        aim_xy=(coin.x, coin.y),
                        shot_color=frog.cur_color,
                        target_ball_id=None,
                        target_curve_idx=None,
                        target_s=None,
                        expected_removed=0,
                        cascades=0,
                        score=config.WEIGHT_COIN_BONUS,
                        reason="grab_coin",
                        is_coin=True,
                    )
                    break

        # 1a. Best strict-clearance 3+ pop (or swap+pop) always beats everything else
        if clear_matches:
            self._junk_wait_started = None
            clear_matches.sort(key=lambda d: d.score, reverse=True)
            return clear_matches[0]

        # 1b. Relaxed-clearance 3+ pop (e.g. tight gap or emerging ball) still beats coins & setup pairs
        relaxed_matches = [d for d in setup_relaxed_candidates if d.expected_removed >= 3]
        if relaxed_matches:
            for dec in relaxed_matches:
                if dec.target_ball_id is not None and dec.target_ball_id in blocker_bonus:
                    dec.score += blocker_bonus[dec.target_ball_id]
            relaxed_matches.sort(key=lambda d: d.score, reverse=True)
            self._junk_wait_started = None
            return relaxed_matches[0]

        # 2. Unobstructed coin beats non-pop setup/dump shots
        if best_coin is not None:
            self._junk_wait_started = None
            return best_coin

        # 3. Setup pair onto any reachable same-color ball (strict first, then relaxed physical clearance)
        relaxed_setups = [d for d in setup_relaxed_candidates if d.expected_removed < 3]
        chosen_setups = setup_candidates if setup_candidates else relaxed_setups
        if chosen_setups:
            for dec in chosen_setups:
                if dec.target_ball_id is not None and dec.target_ball_id in blocker_bonus:
                    dec.score += blocker_bonus[dec.target_ball_id] * 0.5
            chosen_setups.sort(key=lambda d: d.score, reverse=True)
            self._junk_wait_started = None
            return chosen_setups[0]

        # Guard against 1-frame occlusion flickers before treating cur_color as an orphan
        recently_on_board = (now - self._recent_board_colors.get(frog.cur_color, -10.0)) < 0.30 and num_visible > 3
        if frog.cur_color not in board_colors and recently_on_board:
            if self._junk_wait_started is None:
                self._junk_wait_started = now
            if (now - self._junk_wait_started) < config.JUNK_WAIT_MAX_SEC:
                return ShotDecision(
                    action="wait",
                    mode=mode,
                    aim_xy=None,
                    shot_color=frog.cur_color,
                    target_ball_id=None,
                    target_curve_idx=None,
                    target_s=None,
                    expected_removed=0,
                    cascades=0,
                    score=0.0,
                    reason="hold_for_flickered_color",
                )
            self._junk_wait_started = None

        # Immediate Orphan-Color Dump: never hold an orphan color that has 0 balls on the board
        if frog.cur_color not in board_colors:
            dump_pt = find_offscreen_dump_point(level.frog_xy, all_balls, predicted_pts=predicted_pts)
            if dump_pt is not None:
                self._junk_wait_started = None
                return ShotDecision(
                    action="shoot",
                    mode=mode,
                    aim_xy=dump_pt,
                    shot_color=frog.cur_color,
                    target_ball_id=None,
                    target_curve_idx=None,
                    target_s=None,
                    expected_removed=0,
                    cascades=0,
                    score=-5.0,
                    reason="junk_offscreen_dump",
                    is_junk_dump=True,
                )

        # Tier B: Brief hold (~0.28s) in endgame or if cur_color itself has a blocked 3+ pop;
        # otherwise throw away unplayable balls immediately instead of stalling in CONTROL mode.
        has_blocked_endgame_match = is_endgame and any(c in board_colors for _, c in colors_to_try)
        has_blocked_cur_pop = any(d.shot_color == frog.cur_color for d in blocked_matches)
        if (mode == "CONTROL" and has_blocked_cur_pop) or has_blocked_endgame_match:
            if self._junk_wait_started is None:
                self._junk_wait_started = now
            if (now - self._junk_wait_started) < config.JUNK_WAIT_MAX_SEC:
                preview_blocked = blocked_matches[0] if blocked_matches else None
                return ShotDecision(
                    action="wait",
                    mode=mode,
                    aim_xy=preview_blocked.aim_xy if preview_blocked else None,
                    shot_color=frog.cur_color,
                    target_ball_id=preview_blocked.target_ball_id if preview_blocked else None,
                    target_curve_idx=preview_blocked.target_curve_idx if preview_blocked else None,
                    target_s=preview_blocked.target_s if preview_blocked else None,
                    expected_removed=0,
                    cascades=0,
                    score=0.0,
                    reason="hold_for_blocked_pair",
                    blocked=True if preview_blocked else False,
                    blocker_xy=preview_blocked.blocker_xy if preview_blocked else None,
                )

        # Tier C: Unobstructed off-screen dump (to cycle frog ball cleanly without splitting pairs)
        dump_pt = find_offscreen_dump_point(level.frog_xy, all_balls, predicted_pts=predicted_pts)
        if dump_pt is not None:
            self._junk_wait_started = None
            return ShotDecision(
                action="shoot",
                mode=mode,
                aim_xy=dump_pt,
                shot_color=frog.cur_color,
                target_ball_id=None,
                target_curve_idx=None,
                target_s=None,
                expected_removed=0,
                cascades=0,
                score=-5.0,
                reason="junk_offscreen_dump",
                is_junk_dump=True,
            )

        # Tier C2: Temporarily surrounded — wait for an offscreen gap before dirty seam insert
        if seam_candidates:
            dump_wait_max = max(config.JUNK_WAIT_MAX_SEC, 0.65) if mode == "CONTROL" else config.JUNK_WAIT_MAX_SEC
            if self._junk_wait_started is None:
                self._junk_wait_started = now
            if (now - self._junk_wait_started) < dump_wait_max:
                return ShotDecision(
                    action="wait",
                    mode=mode,
                    aim_xy=None,
                    shot_color=frog.cur_color,
                    target_ball_id=None,
                    target_curve_idx=None,
                    target_s=None,
                    expected_removed=0,
                    cascades=0,
                    score=0.0,
                    reason="wait_for_dump_gap",
                )
            self._junk_wait_started = None

        # Tier C3: Completely surrounded — pick safest non-pair-splitting seam
        if seam_candidates:
            seam_candidates.sort(key=lambda d: d.score, reverse=True)
            return seam_candidates[0]

        # Everything blocked (no clear shots at all): display best blocked preview for HUD
        if blocked_matches:
            blocked_matches.sort(key=lambda d: d.score, reverse=True)
            b = blocked_matches[0]
            return ShotDecision(
                action="wait",
                mode=mode,
                aim_xy=b.aim_xy,
                shot_color=b.shot_color,
                target_ball_id=b.target_ball_id,
                target_curve_idx=b.target_curve_idx,
                target_s=b.target_s,
                expected_removed=0,
                cascades=0,
                score=b.score,
                reason="all_targets_blocked",
                blocked=True,
                blocker_xy=b.blocker_xy,
            )

        return ShotDecision(
            action="none",
            mode=mode,
            aim_xy=None,
            shot_color=frog.cur_color,
            target_ball_id=None,
            target_curve_idx=None,
            target_s=None,
            expected_removed=0,
            cascades=0,
            score=0.0,
            reason="no_valid_target",
        )
