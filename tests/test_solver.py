import math

import config
from input_controller import AutoInputController, compute_dynamic_cooldown
from level_data import load_game_assets
from solver import (
    ShotDecision,
    StrategySolver,
    check_line_of_sight,
    compute_lead_aim,
    find_offscreen_dump_point,
    predict_gap_rollback,
    simulate_insert,
    split_into_segments,
)
from tracker import BlastZone, TrackState, TrackedBall
from vision import DetectedCoin, FrogState
from worker import LevelLifecycleGate, _compute_tail_progress


def _make_ball(curve, curve_idx, ball_id, s, color, velocity=0.0, powerup=None, in_tunnel_memory=False):
    x, y = curve.point_at_arc_len(s)
    return TrackedBall(
        ball_id=ball_id,
        curve_idx=curve_idx,
        s=s,
        x=x,
        y=y,
        color=color,
        powerup=powerup,
        velocity=velocity,
        progress=s / max(curve.total_length, 1.0),
        visibility=curve.visibility_at(s),
        in_tunnel_memory=in_tunnel_memory,
    )


def test_simulate_insert_combo_cascade():
    # A A B B A + B at index 2 -> B B B (3) pops, leaving A A A (3) -> 6 removed, 1 cascade
    removed, cascades = simulate_insert(["red", "red", "blue", "blue", "red"], 2, "blue")
    assert removed == 6
    assert cascades == 1


def test_lead_aim_projects_ahead_of_moving_target():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    s0 = 800.0
    s_pred, _, _ = compute_lead_aim(lvl.frog_xy, curve, s0, velocity=100.0, lead_mult=1.0)
    assert s_pred > s0 + 10.0

    # Extreme velocity spike must stay bounded by max_lead_px
    s_capped, _, _ = compute_lead_aim(lvl.frog_xy, curve, s0, velocity=5000.0, lead_mult=1.0)
    assert (s_capped - s0) <= config.MAX_LEAD_PX + 1e-6


def test_solver_raycast_blocking_and_swap_decision():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(allow_swap=True)

    # Outer pair of blues at ~600px, and an inner green ball directly between frog and outer pair
    b1 = _make_ball(curve, 0, 1, 580.0, "blue")
    b2 = _make_ball(curve, 0, 2, 612.0, "blue")

    # Place a single yellow ball unobstructed elsewhere, plus a red pair unobstructed
    b3 = _make_ball(curve, 0, 3, 1400.0, "red")
    b4 = _make_ball(curve, 0, 4, 1432.0, "red")

    # Place an inner green ball right on the line between frog_xy and b1!
    fx, fy = lvl.frog_xy
    mid_x = (fx + b1.x) * 0.5
    mid_y = (fy + b1.y) * 0.5
    blocker = TrackedBall(99, 0, 1800.0, mid_x, mid_y, "green", None, 0.0, 0.7, 1.0)

    # Cur = blue (blocked!), Next = red (open 3-match!) -> should pick swap_shoot!
    frog = FrogState(cur_color="blue", next_color="red", angle_rad=0.0)
    st = TrackState({0: [b1, b2, b3, b4, blocker]}, [b1, b2, b3, b4, blocker], [], 0.7)
    dec = solver.decide(lvl, st, frog, [], cfg, now=1.0)
    assert dec.action == "swap_shoot"
    assert dec.shot_color == "red"
    assert dec.expected_removed >= 3


def test_solver_avoids_blast_zone_and_prioritizes_bomb_and_defense():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig()

    # Pair 1 at s=500 inside active blast zone; Pair 2 at s=1200 with a Bomb!
    p1a = _make_ball(curve, 0, 1, 500.0, "red")
    p1b = _make_ball(curve, 0, 2, 532.0, "red")
    p2a = _make_ball(curve, 0, 3, 1200.0, "red", powerup="bomb")
    p2b = _make_ball(curve, 0, 4, 1232.0, "red")

    blast = BlastZone(curve_idx=0, s=510.0, radius_px=60.0, expires_at=5.0)
    frog = FrogState(cur_color="red", next_color="red", angle_rad=0.0)
    st = TrackState({0: [p1a, p1b, p2a, p2b]}, [p1a, p1b, p2a, p2b], [blast], 0.45)

    dec = solver.decide(lvl, st, frog, [], cfg, now=1.0)
    assert dec.action == "shoot"
    assert abs(dec.target_s - 1200.0) < 60.0
    assert dec.is_bomb is True

    # Defense override: high-progress pair near skull beats far low-progress pair
    far_pair = [_make_ball(curve, 0, 10, 300.0, "green"), _make_ball(curve, 0, 11, 332.0, "green")]
    danger_s = curve.total_length * 0.85
    near_pair = [_make_ball(curve, 0, 12, danger_s - 32.0, "green"), _make_ball(curve, 0, 13, danger_s, "green")]
    st_def = TrackState({0: far_pair + near_pair}, far_pair + near_pair, [], 0.85)
    dec_def = solver.decide(lvl, st_def, FrogState("green", "blue", 0.0), [], cfg, now=2.0)
    assert dec_def.mode == "DEFENSE"
    assert dec_def.target_s > danger_s - 60.0


def test_solver_3_tier_junk_fallback_and_coin_grab():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(grab_coins=True)

    # Single green + blue pair (cur = yellow has NO matches and NO coins)
    b1 = _make_ball(curve, 0, 1, 500.0, "green")
    b2 = _make_ball(curve, 0, 2, 532.0, "blue")
    b3 = _make_ball(curve, 0, 3, 564.0, "blue")

    # Tier 1: Setup shot if single yellow exists
    b_yel = _make_ball(curve, 0, 4, 900.0, "yellow")
    st1 = TrackState({0: [b1, b2, b3, b_yel]}, [b1, b2, b3, b_yel], [], 0.3)
    dec1 = solver.decide(lvl, st1, FrogState("yellow", "purple", 0.0), [], cfg, now=1.0)
    assert dec1.action == "shoot"
    assert dec1.reason == "setup_pair"

    # Priority 2: When no pop exists (only setup_pair), open coin beats setup_pair
    tx, ty = lvl.treasure_points[2]
    dec_coin = solver.decide(lvl, st1, FrogState("yellow", "purple", 0.0), [DetectedCoin(tx, ty, 0.9)], cfg, now=1.05)
    assert dec_coin.is_coin is True

    # Priority 1: When frog CAN pop something (clear_matches), pop beats coin (powerups rank highest)
    dec_pop = solver.decide(lvl, st1, FrogState("blue", "purple", 0.0), [DetectedCoin(tx, ty, 0.9)], cfg, now=1.10)
    assert dec_pop.is_coin is False
    assert dec_pop.expected_removed >= 3

    # Priority 3: Relaxed physical-clearance setup pairs onto same-color ball (33px gap) instead of dumping offscreen
    fx, fy = lvl.frog_xy
    ux = (b_yel.x - fx) / max(math.hypot(b_yel.x - fx, b_yel.y - fy), 1.0)
    uy = (b_yel.y - fy) / max(math.hypot(b_yel.x - fx, b_yel.y - fy), 1.0)
    mid_x = (fx + b_yel.x) * 0.5
    mid_y = (fy + b_yel.y) * 0.5
    # Place non-blocking red ball 33.5px perpendicular to ray (blocked under strict 36px, clear under relaxed 29.6px)
    tight_nbr = TrackedBall(90, 0, 120.0, mid_x - uy * 33.5, mid_y + ux * 33.5, "red", None, 0.0, 0.05, 1.0)
    st_tight = TrackState({0: [tight_nbr, b_yel]}, [tight_nbr, b_yel], [], 0.3)
    dec_relaxed = solver.decide(lvl, st_tight, FrogState("yellow", "purple", 0.0), [], cfg, now=1.15)
    assert dec_relaxed.action == "shoot"
    assert dec_relaxed.reason == "setup_pair"
    assert dec_relaxed.is_junk_dump is False

    # Side-by-side touching neighbors at perp=24.5px (zero inner blockers) still pair under relaxed 23.2px,
    # whereas a hairpin neighbor folding directly in front at perp=18px is blocked and dumps cleanly
    touch_l = TrackedBall(91, 0, b_yel.s - 32.0, b_yel.x - ux * 20.0 - uy * 24.5, b_yel.y - uy * 20.0 + ux * 24.5, "red", None, 0.0, 0.3, 1.0)
    touch_r = TrackedBall(92, 0, b_yel.s + 32.0, b_yel.x - ux * 20.0 + uy * 24.5, b_yel.y - uy * 20.0 - ux * 24.5, "green", None, 0.0, 0.31, 1.0)
    st_angled = TrackState({0: [touch_l, b_yel, touch_r]}, [touch_l, b_yel, touch_r], [], 0.31)
    dec_angled = solver.decide(lvl, st_angled, FrogState("yellow", "purple", 0.0), [], cfg, now=1.20)
    assert dec_angled.action == "shoot"
    assert dec_angled.reason == "setup_pair"
    assert dec_angled.is_junk_dump is False

    hairpin_front = TrackedBall(93, 0, b_yel.s + 32.0, b_yel.x - ux * 26.0, b_yel.y - uy * 26.0, "green", None, 0.0, 0.31, 1.0)
    st_hairpin = TrackState({0: [b_yel, hairpin_front]}, [b_yel, hairpin_front], [], 0.31)
    # Endgame hold fires first; once JUNK_WAIT_MAX_SEC elapses, it dumps rather than clipping the hairpin neighbor
    assert solver.decide(lvl, st_hairpin, FrogState("yellow", "purple", 0.0), [], cfg, now=1.25).action == "wait"
    dec_hairpin = solver.decide(lvl, st_hairpin, FrogState("yellow", "purple", 0.0), [], cfg, now=1.60)
    assert dec_hairpin.reason != "setup_pair"
    assert dec_hairpin.is_junk_dump is True

    # Tier 2 -> Tier 3: Fully blocked pair triggers brief hold, then off-screen dump after JUNK_WAIT_MAX_SEC
    fx, fy = lvl.frog_xy
    blocker1 = TrackedBall(5, 0, 1600.0, (fx + b2.x) * 0.5, (fy + b2.y) * 0.5, "red", None, 0.0, 0.5, 1.0)
    blocker2 = TrackedBall(6, 0, 1632.0, (fx + b3.x) * 0.5, (fy + b3.y) * 0.5, "purple", None, 0.0, 0.51, 1.0)
    st_hold = TrackState({0: [b1, b2, b3, blocker1, blocker2]}, [b1, b2, b3, blocker1, blocker2], [], 0.51)

    dec_wait = solver.decide(lvl, st_hold, FrogState("blue", "blue", 0.0), [], cfg, now=10.0)
    assert dec_wait.action == "wait"
    assert dec_wait.reason == "hold_for_blocked_pair"

    dec_dump = solver.decide(lvl, st_hold, FrogState("blue", "blue", 0.0), [], cfg, now=10.0 + config.JUNK_WAIT_MAX_SEC + 0.1)
    assert dec_dump.action == "shoot"
    assert dec_dump.is_junk_dump is True


def test_solver_segment_gap_no_fake_match():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig()

    # Two single yellow balls separated by a 150px gap must NOT be treated as a match_3 or gap_pull
    y1 = _make_ball(curve, 0, 1, 500.0, "yellow")
    y2 = _make_ball(curve, 0, 2, 650.0, "yellow")
    st_gap = TrackState({0: [y1, y2]}, [y1, y2], [], 0.3)
    assert len(split_into_segments([y1, y2])) == 2
    dec_gap = solver.decide(lvl, st_gap, FrogState("yellow", "yellow", 0.0), [], cfg, now=1.0)
    assert dec_gap.expected_removed == 0
    assert dec_gap.reason == "setup_pair"


def test_level_lifecycle_waits_for_balls_and_stops_on_clear():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    curve = assets.levels["spiral"].curves[0]
    gate = LevelLifecycleGate()
    assert gate.state == "WAITING_FOR_BALLS"

    # Single stray sparkle frame or mid-screen "LEVEL X-Y" banner text at s=1800 must NOT start gameplay
    stray = _make_ball(curve, 0, 1, 100.0, "yellow")
    for k in range(5):
        assert gate.update(TrackState({0: [stray]}, [stray], [], 0.05), now=0.1 * k) == "WAITING_FOR_BALLS"

    banner1 = _make_ball(curve, 0, 2, 1800.0, "yellow")
    banner2 = _make_ball(curve, 0, 3, 1832.0, "red")
    banner3 = _make_ball(curve, 0, 4, 1864.0, "blue")
    for k in range(6):
        assert gate.update(TrackState({0: [banner1, banner2, banner3]}, [banner1, banner2, banner3], [], 0.65), now=1.0 + 0.05 * k) == "WAITING_FOR_BALLS"

    # 4 frames with >= 3 contiguous entrance balls enters LOADING_ROLLIN (waits out initial fast speed)
    b1 = _make_ball(curve, 0, 1, 100.0, "yellow", velocity=100.0)
    b2 = _make_ball(curve, 0, 2, 132.0, "red", velocity=100.0)
    b3 = _make_ball(curve, 0, 3, 164.0, "blue", velocity=100.0)
    t_cur = 2.0
    for _ in range(config.LEVEL_START_CONFIRM_FRAMES):
        st = gate.update(TrackState({0: [b1, b2, b3]}, [b1, b2, b3], [], 0.06), now=t_cur)
        t_cur += 0.05
    assert st == "LOADING_ROLLIN"

    # While still rolling in fast past LEVEL_ROLLIN_MIN_WAIT_SEC, must stay LOADING_ROLLIN
    for _ in range(10):
        t_cur += 0.15
        assert gate.update(TrackState({0: [b1, b2, b3]}, [b1, b2, b3], [], 0.12), now=t_cur) == "LOADING_ROLLIN"

    # Once train slows to normal gameplay speed and stabilizes, transitions to PLAYING
    c1 = _make_ball(curve, 0, 1, 300.0, "yellow", velocity=45.0)
    c2 = _make_ball(curve, 0, 2, 332.0, "red", velocity=45.0)
    c3 = _make_ball(curve, 0, 3, 364.0, "blue", velocity=45.0)
    t_cur += 0.05
    gate.update(TrackState({0: [c1, c2, c3]}, [c1, c2, c3], [], 0.15), now=t_cur)
    for _ in range(config.LEVEL_ROLLIN_CALM_FRAMES):
        t_cur += 0.05
        st = gate.update(TrackState({0: [c1, c2, c3]}, [c1, c2, c3], [], 0.15), now=t_cur)
    assert st == "PLAYING"

    # Brief empty gap (e.g. final ball under a bridge) must NOT stop autoplay prematurely
    for _ in range(6):
        t_cur += 0.05
        assert gate.update(TrackState({0: []}, [], [], 0.0), now=t_cur) == "PLAYING"

    # Full LEVEL_CLEAR_ZERO_FRAMES empty frames transitions to LEVEL_CLEARED
    for _ in range(config.LEVEL_CLEAR_ZERO_FRAMES):
        t_cur += 0.05
        st = gate.update(TrackState({0: []}, [], [], 0.0), now=t_cur)
    assert st == "LEVEL_CLEARED"

    # Stray post-clear fireworks balls keep it LEVEL_CLEARED
    fw = _make_ball(curve, 0, 9, 1200.0, "blue")
    for _ in range(5):
        t_cur += 0.05
        assert gate.update(TrackState({0: [fw]}, [fw], [], 0.4), now=t_cur) == "LEVEL_CLEARED"

    # Re-emerging final 1-2 balls from a long bridge recover back to PLAYING
    for _ in range(config.LEVEL_CLEARED_RECOVER_FRAMES - 5):
        t_cur += 0.05
        st_rec = gate.update(TrackState({0: [fw]}, [fw], [], 0.4), now=t_cur)
    assert st_rec == "PLAYING"

    # Re-clear and verify next level fresh 3-ball entrance enters LOADING_ROLLIN before PLAYING
    for _ in range(config.LEVEL_CLEAR_ZERO_FRAMES):
        t_cur += 0.05
        gate.update(TrackState({0: []}, [], [], 0.0), now=t_cur)
    assert gate.state == "LEVEL_CLEARED"

    for _ in range(config.LEVEL_START_CONFIRM_FRAMES):
        t_cur += 0.05
        st = gate.update(TrackState({0: [b1, b2, b3]}, [b1, b2, b3], [], 0.06), now=t_cur)
    assert st == "LOADING_ROLLIN"


def test_level_lifecycle_recovers_midlevel_past_entrance():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    curve = assets.levels["spiral"].curves[0]
    gate = LevelLifecycleGate()
    assert gate.state == "WAITING_FOR_BALLS"

    mid = [
        _make_ball(curve, 0, 10 + i, 1200.0 + i * 32.0, col, velocity=40.0)
        for i, col in enumerate(["yellow", "red", "blue", "green"])
    ]
    t_cur = 1.0
    for _ in range(config.LEVEL_CLEARED_RECOVER_FRAMES):
        t_cur += 0.05
        st = gate.update(TrackState({0: mid}, mid, [], 0.45), now=t_cur)
    assert st == "PLAYING"

    # Tail progress: [None] while near spawn or < 5%, [progress] once >= 5%; supports 2-curve twin maps
    at_spawn = [_make_ball(curve, 0, 1, 20.0, "red"), _make_ball(curve, 0, 2, 1200.0, "blue")]
    assert _compute_tail_progress(assets.levels["spiral"], TrackState({0: at_spawn}, at_spawn, [], 0.45), "PLAYING") == [None]
    below_5pct = [_make_ball(curve, 0, 2, 100.0, "red"), _make_ball(curve, 0, 3, 1200.0, "blue")]
    assert _compute_tail_progress(assets.levels["spiral"], TrackState({0: below_5pct}, below_5pct, [], 0.45), "PLAYING") == [None]
    past_spawn = [_make_ball(curve, 0, 2, 220.0, "red"), _make_ball(curve, 0, 3, 1200.0, "blue")]
    tail_vals = _compute_tail_progress(assets.levels["spiral"], TrackState({0: past_spawn}, past_spawn, [], 0.45), "PLAYING")
    assert len(tail_vals) == 1 and tail_vals[0] is not None and abs(tail_vals[0] - past_spawn[0].progress) < 1e-6

    twin_lvl = assets.levels["snakepit"]
    c0, c1 = twin_lvl.curves[0], twin_lvl.curves[1]
    t0 = _make_ball(c0, 0, 10, 20.0, "red")
    t1 = _make_ball(c1, 1, 11, 500.0, "blue")
    twin_tails = _compute_tail_progress(twin_lvl, TrackState({0: [t0], 1: [t1]}, [t0, t1], [], t1.progress), "PLAYING")
    assert len(twin_tails) == 2 and twin_tails[0] is None and twin_tails[1] is not None


def test_solver_raycast_strict_clearance_and_no_cross_gap_ignore():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    fx, fy = lvl.frog_xy

    # Target at 200px to the right; place a blocker 31px off the ray line (between 28.8 and 32.8)
    target = TrackedBall(1, 0, 500.0, fx + 200.0, fy, "blue", None, 0.0, 0.2, 1.0)
    blocker = TrackedBall(2, 0, 900.0, fx + 100.0, fy + 31.0, "red", None, 0.0, 0.5, 1.0)
    clear, hit = check_line_of_sight(lvl.frog_xy, (target.x, target.y), [target, blocker], {target.ball_id})
    assert clear is False
    assert hit is blocker

    # Verify a blocker across a gap in curve_balls is NOT ignored even if adjacent in list order
    solver = StrategySolver()
    b1 = _make_ball(curve, 0, 10, 500.0, "blue")
    b2 = _make_ball(curve, 0, 11, 532.0, "blue")
    tx, ty = curve.point_at_arc_len(516.0)
    # Adjacent in curve_balls after a 400px gap, sitting directly between frog and b1/b2!
    blk = TrackedBall(12, 0, 950.0, (fx + tx) * 0.5, (fy + ty) * 0.5, "red", None, 0.0, 0.5, 1.0)
    st = TrackState({0: [b1, b2, blk]}, [b1, b2, blk], [], 0.5)
    dec = solver.decide(lvl, st, FrogState("blue", "blue", 0.0), [], config.RuntimeConfig(), now=1.0)
    assert dec.action == "wait" or dec.target_ball_id != b1.ball_id


def test_dynamic_shot_cooldown_scales_with_distance_and_combos():
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)
    frog_xy = (320.0, 240.0)

    close_dec = ShotDecision("shoot", "CONTROL", (350.0, 240.0), "blue", 1, 0, 100.0, 3, 0, 50.0, "match_3")
    far_dec = ShotDecision("shoot", "CONTROL", (620.0, 240.0), "blue", 2, 0, 100.0, 3, 0, 50.0, "match_3")
    combo_dec = ShotDecision("shoot", "CONTROL", (620.0, 240.0), "blue", 3, 0, 100.0, 6, 1, 90.0, "match_6")
    def_combo_dec = ShotDecision("shoot", "DEFENSE", (620.0, 240.0), "blue", 4, 0, 100.0, 6, 1, 90.0, "match_6")

    w_close = compute_dynamic_cooldown(frog_xy, close_dec.aim_xy, close_dec, cfg)
    w_far = compute_dynamic_cooldown(frog_xy, far_dec.aim_xy, far_dec, cfg)
    w_combo = compute_dynamic_cooldown(frog_xy, combo_dec.aim_xy, combo_dec, cfg)
    w_def = compute_dynamic_cooldown(frog_xy, def_combo_dec.aim_xy, def_combo_dec, cfg)

    assert abs(w_close - 0.12) < 1e-6
    assert w_far > 0.35
    assert abs((w_combo - w_far) - config.DYNAMIC_COOLDOWN_COMBO_PAD_SEC) < 1e-6
    assert abs(w_def - w_far) < 1e-6


def test_smart_setup_combos_and_target_aware_cooldown():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)
    fx, fy = lvl.frog_xy

    # 1. Midgame cascade-prep setup_pair ([R, R, B, R]) should beat ordinary setup_pair
    filler = [_make_ball(curve, 0, 50 + k, 100.0 + k * 32.0, "purple" if k % 2 == 0 else "yellow") for k in range(8)]
    b_ord = _make_ball(curve, 0, 1, 400.0, "blue")
    r1 = _make_ball(curve, 0, 2, 668.0, "red")
    r2 = _make_ball(curve, 0, 3, 700.0, "red")
    b_casc = _make_ball(curve, 0, 4, 732.0, "blue")
    r3 = _make_ball(curve, 0, 5, 764.0, "red")
    all_mid = filler + [b_ord, r1, r2, b_casc, r3]
    st_casc = TrackState({0: all_mid}, all_mid, [], 0.25)
    dec_casc = solver.decide(lvl, st_casc, FrogState("blue", "green", 0.0), [], cfg, now=1.0)
    assert dec_casc.reason == "setup_pair"
    assert dec_casc.target_ball_id == b_casc.ball_id

    # 2. 2-ball self-combo setup_pair (cur == next == yellow) vs swap green setup: cur shoot beats swap
    y1 = _make_ball(curve, 0, 10, 600.0, "yellow")
    g1 = _make_ball(curve, 0, 11, 700.0, "green")
    st_self = TrackState({0: [y1, g1]}, [y1, g1], [], 0.25)
    dec_self = solver.decide(lvl, st_self, FrogState("yellow", "yellow", 0.0), [], cfg, now=2.0)
    assert dec_self.action == "shoot"
    assert dec_self.reason == "setup_pair"
    assert dec_self.score > 45.0

    # 3. Bullet-flight + swap readiness gating
    ctrl = AutoInputController()
    far_target_a = ShotDecision("shoot", "CONTROL", (620.0, 240.0), "blue", 1, 0, 100.0, 3, 0, 50.0, "match_3")
    same_target_a = ShotDecision("shoot", "CONTROL", (630.0, 240.0), "blue", 2, 0, 100.0, 3, 0, 50.0, "match_3")
    other_target_b = ShotDecision("shoot", "CONTROL", (80.0, 240.0), "red", 3, 0, 200.0, 3, 0, 50.0, "match_3")
    swap_b = ShotDecision("swap_shoot", "CONTROL", (80.0, 240.0), "red", 3, 0, 200.0, 3, 0, 50.0, "match_3")

    ctrl._last_shot_t = 10.0
    ctrl._last_aim_xy = far_target_a.aim_xy
    ctrl._last_landing_t = 10.0 + compute_dynamic_cooldown(lvl.frog_xy, far_target_a.aim_xy, far_target_a, cfg)
    ctrl._last_was_combo = False
    ctrl._last_was_dump = False

    # While a track bullet is still in flight, do NOT fire a second track bullet,
    # EXCEPT open off-track coins which fire immediately without waiting for landing
    coin_dec = ShotDecision("shoot", "CONTROL", (80.0, 95.0), "blue", None, None, None, 0, 0, 28.0, "grab_coin", is_coin=True)
    assert ctrl.can_fire_now(same_target_a, cfg, now=10.13) is False
    assert ctrl.can_fire_now(other_target_b, cfg, now=10.13) is False
    assert ctrl.can_fire_now(coin_dec, cfg, now=10.13) is True

    # Ring completely encloses frog except for a narrow 28px-halfwidth (56px total) gap straight right
    surround_ring = []
    for k, deg_w in enumerate(range(14, 348, 10)):
        rad_w = math.radians(deg_w)
        surround_ring.append(
            TrackedBall(200 + k, 0, 100.0 + k * 25.0, fx + math.cos(rad_w) * 140.0, fy + math.sin(rad_w) * 140.0, "red", None, 0.0, 0.2, 1.0)
        )
    gap_pair = [
        TrackedBall(290, 0, 900.0, fx + 140.0, fy - 28.0, "red", None, 0.0, 0.5, 1.0),
        TrackedBall(291, 0, 932.0, fx + 140.0, fy + 28.0, "blue", None, 0.0, 0.51, 1.0),
    ]
    dump_xy = find_offscreen_dump_point(lvl.frog_xy, surround_ring + gap_pair)
    assert dump_xy is not None
    assert dump_xy[0] > fx + 140.0

    # After landing, regular shoots pass at reload floor, while swap_shoot respects SWAP_RELOAD_FLOOR_SEC
    ctrl._last_shot_t = 10.0
    ctrl._last_landing_t = 10.15
    assert ctrl.can_fire_now(other_target_b, cfg, now=10.16) is True
    assert ctrl.can_fire_now(swap_b, cfg, now=10.16) is False
    assert ctrl.can_fire_now(swap_b, cfg, now=10.29) is True

    # If previous shot was an offscreen dump (_last_was_dump=True), distant targets fire at reload floor
    ctrl._last_shot_t = 12.0
    ctrl._last_landing_t = 12.40
    ctrl._last_was_dump = True
    assert ctrl.can_fire_now(other_target_b, cfg, now=12.13) is True
    assert compute_dynamic_cooldown(lvl.frog_xy, other_target_b.aim_xy, other_target_b, cfg, last_aim_xy=far_target_a.aim_xy, last_was_combo=False) == 0.12


def test_oblique_neighbor_not_ignored_and_orphan_endgame_dump():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)
    fx, fy = lvl.frog_xy

    # Adjacent different-colored neighbor sitting closer along oblique ray (perp=22 < 25.6) blocks
    target_b = TrackedBall(1, 0, 500.0, fx + 200.0, fy, "blue", None, 0.0, 0.2, 1.0)
    nbr_r = TrackedBall(2, 0, 468.0, fx + 170.0, fy + 22.0, "red", None, 0.0, 0.19, 1.0)
    clear_obs, blk = check_line_of_sight(
        lvl.frog_xy,
        (target_b.x, target_b.y),
        [target_b, nbr_r],
        {target_b.ball_id},
        adj_diff_ids={nbr_r.ball_id},
    )
    assert clear_obs is False
    assert blk is nbr_r

    # Normal ~25-deg neighbor (perp=29.0 > 25.6) does NOT falsely block setup_pair
    nbr_angle = TrackedBall(3, 0, 468.0, fx + 186.0, fy + 29.0, "red", None, 0.0, 0.19, 1.0)
    clear_angle, blk_angle = check_line_of_sight(
        lvl.frog_xy,
        (target_b.x, target_b.y),
        [target_b, nbr_angle],
        {target_b.ball_id},
        adj_diff_ids={nbr_angle.ball_id},
    )
    assert clear_angle is True
    assert blk_angle is None

    # Orphan color in endgame immediately dumps offscreen instead of holding
    g1 = _make_ball(curve, 0, 10, 500.0, "green")
    g2 = _make_ball(curve, 0, 11, 532.0, "green")
    st_orphan = TrackState({0: [g1, g2]}, [g1, g2], [], 0.2)
    dec_orphan = solver.decide(lvl, st_orphan, FrogState("blue", "red", 0.0), [], cfg, now=1.0)
    assert dec_orphan.action == "shoot"
    assert dec_orphan.is_junk_dump is True


def test_swap_shoot_extra_latency_and_endgame_blocked_hold():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)
    fx, fy = lvl.frog_xy

    # 1. swap_shoot uses +SWAP_EXTRA_LATENCY_SEC lead vs direct shoot on a moving train
    b1 = _make_ball(curve, 0, 1, 700.0, "blue", velocity=50.0)
    b2 = _make_ball(curve, 0, 2, 732.0, "blue", velocity=50.0)
    b3 = _make_ball(curve, 0, 3, 764.0, "red", velocity=50.0)
    b4 = _make_ball(curve, 0, 4, 796.0, "yellow", velocity=50.0)
    b5 = _make_ball(curve, 0, 5, 828.0, "green", velocity=50.0)
    b6 = _make_ball(curve, 0, 6, 860.0, "purple", velocity=50.0)
    st_move = TrackState({0: [b1, b2, b3, b4, b5, b6]}, [b1, b2, b3, b4, b5, b6], [], 0.4)

    dec_direct = solver.decide(lvl, st_move, FrogState("blue", "red", 0.0), [], cfg, now=1.0)
    dec_swap = solver.decide(lvl, st_move, FrogState("red", "blue", 0.0), [], cfg, now=1.1)
    assert dec_direct.action == "shoot"
    assert dec_swap.action == "swap_shoot"
    assert dec_swap.target_s > dec_direct.target_s + 2.5

    # 2. Endgame single-ball hold: holds up to JUNK_WAIT_MAX_SEC instead of dumping immediately
    solver.reset()
    tx, ty = curve.point_at_arc_len(600.0)
    dx_t, dy_t = tx - fx, ty - fy
    d_t = math.hypot(dx_t, dy_t)
    ux, uy = dx_t / d_t, dy_t / d_t
    px, py = -uy, ux
    target_b = _make_ball(curve, 0, 20, 600.0, "blue")
    blk_a = TrackedBall(21, 0, 900.0, fx + ux * (d_t * 0.5), fy + uy * (d_t * 0.5), "red", None, 0.0, 0.50, 1.0)
    blk_b = TrackedBall(22, 0, 932.0, fx + ux * (d_t * 0.5) + px * 28.0, fy + uy * (d_t * 0.5) + py * 28.0, "red", None, 0.0, 0.51, 1.0)
    blk_c = TrackedBall(23, 0, 964.0, fx + ux * (d_t * 0.5) - px * 28.0, fy + uy * (d_t * 0.5) - py * 28.0, "red", None, 0.0, 0.52, 1.0)
    st_end = TrackState({0: [target_b, blk_a, blk_b, blk_c]}, [target_b, blk_a, blk_b, blk_c], [], 0.52)

    dec_hold = solver.decide(lvl, st_end, FrogState("blue", "green", 0.0), [], cfg, now=5.0)
    assert dec_hold.action == "wait"
    assert dec_hold.reason == "hold_for_blocked_pair"

    # After JUNK_WAIT_MAX_SEC expires, falls through cleanly
    dec_after = solver.decide(lvl, st_end, FrogState("blue", "green", 0.0), [], cfg, now=5.35)
    assert dec_after.action == "shoot"
    assert dec_after.is_junk_dump is True


def test_pair_aims_at_run_center_and_prefers_narrow_dump_over_seam():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)
    fx, fy = lvl.frog_xy

    # 1. 3-ball run bounded by different-colored neighbors aims at run center, NOT front edge
    b_left = _make_ball(curve, 0, 1, 568.0, "yellow")
    p0 = _make_ball(curve, 0, 2, 600.0, "purple")
    p1 = _make_ball(curve, 0, 3, 632.0, "purple")
    p2 = _make_ball(curve, 0, 4, 664.0, "purple")
    b_right = _make_ball(curve, 0, 5, 696.0, "green")
    st_run = TrackState({0: [b_left, p0, p1, p2, b_right]}, [b_left, p0, p1, p2, b_right], [], 0.35)

    dec_run = solver.decide(lvl, st_run, FrogState("purple", "red", 0.0), [], cfg, now=1.0)
    assert dec_run.action == "shoot"
    assert abs(dec_run.target_s - p1.s) < 8.0

    # 2. Physical 32.5px-halfwidth (65px total) escape gap straight right dumps offscreen instead of seam insert
    narrow_ring = []
    for k, deg_w in enumerate(range(16, 346, 10)):
        rad_w = math.radians(deg_w)
        narrow_ring.append(
            TrackedBall(300 + k, 0, 100.0 + k * 25.0, fx + math.cos(rad_w) * 140.0, fy + math.sin(rad_w) * 140.0, "red", None, 0.0, 0.2, 1.0)
        )
    narrow_gap = [
        TrackedBall(390, 0, 900.0, fx + 140.0, fy - 32.5, "red", None, 0.0, 0.48, 1.0),
        TrackedBall(391, 0, 932.0, fx + 140.0, fy + 32.5, "blue", None, 0.0, 0.49, 1.0),
    ]
    all_narrow = narrow_ring + narrow_gap
    dump_xy = find_offscreen_dump_point(lvl.frog_xy, all_narrow)
    assert dump_xy is not None
    assert dump_xy[0] > fx + 140.0

    st_narrow = TrackState({0: all_narrow}, all_narrow, [], 0.49)
    dec_narrow = solver.decide(lvl, st_narrow, FrogState("white", "white", 0.0), [], cfg, now=2.0)
    assert dec_narrow.action == "shoot"
    assert dec_narrow.is_junk_dump is True

    # 3. Zero escape gap in CONTROL mode briefly waits for dump gap before dirty seam insert
    solver.reset()
    full_ring = []
    for k, deg_w in enumerate(range(0, 360, 10)):
        rad_w = math.radians(deg_w)
        full_ring.append(
            TrackedBall(400 + k, 0, 100.0 + k * 20.0, fx + math.cos(rad_w) * 160.0, fy + math.sin(rad_w) * 160.0, "red" if k % 2 == 0 else "blue", None, 0.0, 0.2, 1.0)
        )
    # Place one unobstructed inner-curve ball inside the ring so seam_candidates is non-empty
    inner_target = _make_ball(curve, 0, 499, curve.total_length - 300.0, "red")
    full_ring.append(inner_target)
    st_closed = TrackState({0: full_ring}, full_ring, [], 0.25)

    dec_wait_gap = solver.decide(lvl, st_closed, FrogState("white", "white", 0.0), [], cfg, now=3.0)
    assert dec_wait_gap.action == "wait"
    assert dec_wait_gap.reason == "wait_for_dump_gap"

    dec_still_wait = solver.decide(lvl, st_closed, FrogState("white", "white", 0.0), [], cfg, now=3.35)
    assert dec_still_wait.action == "wait"

    dec_seam = solver.decide(lvl, st_closed, FrogState("white", "white", 0.0), [], cfg, now=3.70)
    assert dec_seam.action == "shoot"
    assert dec_seam.reason in ("safe_seam_insert", "forced_insert")


def test_defense_setup_prefers_rear_segment_over_skull_front():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)

    # Rear safe segment (~0.18 progress) vs danger lead segment (~0.82 progress) separated by gap
    safe_purple = _make_ball(curve, 0, 1, 500.0, "purple")
    safe_tail = _make_ball(curve, 0, 2, 532.0, "green")

    d_base = curve.total_length * 0.80
    d1 = _make_ball(curve, 0, 10, d_base, "red")
    d2 = _make_ball(curve, 0, 11, d_base + 32.0, "yellow")
    danger_purple = _make_ball(curve, 0, 12, d_base + 64.0, "purple")
    d4 = _make_ball(curve, 0, 13, d_base + 96.0, "green")
    d5 = _make_ball(curve, 0, 14, d_base + 128.0, "blue")

    all_b = [safe_purple, safe_tail, d1, d2, danger_purple, d4, d5]
    st_def = TrackState({0: all_b}, all_b, [], d5.progress)

    dec = solver.decide(lvl, st_def, FrogState("purple", "red", 0.0), [], cfg, now=1.0)
    assert dec.mode == "DEFENSE"
    assert dec.action == "shoot"
    assert dec.reason == "setup_pair"
    assert dec.target_ball_id == safe_purple.ball_id

    # When DEFENSE mode has only ONE contiguous train (no safe rear segment), setup_pair prefers the
    # close front-half ball (~130px away) instead of shooting 380px across the entire map at the tail
    gf = assets.levels["groovefest"]
    gf_curve = gf.curves[0]
    gf_chain = []
    gf_s = 1464.0
    for k in range(35):
        if abs(gf_s - 1496.0) < 16.0:
            col = "yellow"
        elif abs(gf_s - 2432.0) < 16.0:
            col = "yellow"
        else:
            col = ("red", "green", "blue")[k % 3]
        gf_chain.append(_make_ball(gf_curve, 0, 500 + k, gf_s, col, velocity=-50.0))
        gf_s += 32.0
    st_gf = TrackState({0: gf_chain}, gf_chain, [], gf_chain[-1].progress)
    dec_gf = solver.decide(gf, st_gf, FrogState("yellow", "purple", 0.0), [], cfg, now=2.0)
    assert dec_gf.mode == "DEFENSE"
    assert dec_gf.action == "shoot"
    assert dec_gf.reason == "setup_pair"
    assert math.hypot(dec_gf.aim_xy[0] - gf.frog_xy[0], dec_gf.aim_xy[1] - gf.frog_xy[1]) < 160.0


def test_relaxed_match_3_beats_setup_pair_and_coin_and_swap_cooldown():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12, grab_coins=True)
    fx, fy = lvl.frog_xy

    # Single open red ball (strict setup_pair) + tight-gap blue pair (passes relaxed 29.6px, fails strict 36px) + open coin
    red_single = _make_ball(curve, 0, 1, 500.0, "red")
    b1 = _make_ball(curve, 0, 2, 900.0, "blue")
    b2 = _make_ball(curve, 0, 3, 932.0, "blue")
    mx, my = (fx + (b1.x + b2.x) * 0.5) * 0.5, (fy + (b1.y + b2.y) * 0.5) * 0.5
    dx, dy = (b1.x + b2.x) * 0.5 - fx, (b1.y + b2.y) * 0.5 - fy
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    # Place non-matching ball 33px off the ray (blocked under strict 36px, clear under relaxed 29.6px)
    tight_nbr = TrackedBall(4, 0, 1800.0, mx - uy * 33.0, my + ux * 33.0, "green", None, 0.0, 0.65, 1.0)
    st = TrackState({0: [red_single, b1, b2, tight_nbr]}, [red_single, b1, b2, tight_nbr], [], 0.65)
    tx, ty = lvl.treasure_points[2]

    dec = solver.decide(lvl, st, FrogState("red", "blue", 0.0), [DetectedCoin(tx, ty, 0.9)], cfg, now=1.0)
    assert dec.action == "swap_shoot"
    assert dec.shot_color == "blue"
    assert dec.expected_removed >= 3
    assert dec.is_coin is False

    # swap_shoot dynamic cooldown includes SWAP_EXTRA_LATENCY_SEC
    shoot_dec = ShotDecision("shoot", "CONTROL", (600.0, 240.0), "blue", 1, 0, 100.0, 3, 0, 50.0, "match_3")
    swap_dec = ShotDecision("swap_shoot", "CONTROL", (600.0, 240.0), "blue", 1, 0, 100.0, 3, 0, 50.0, "match_3")
    w_shoot = compute_dynamic_cooldown((320.0, 240.0), shoot_dec.aim_xy, shoot_dec, cfg)
    w_swap = compute_dynamic_cooldown((320.0, 240.0), swap_dec.aim_xy, swap_dec, cfg)
    assert abs((w_swap - w_shoot) - config.SWAP_EXTRA_LATENCY_SEC) < 1e-6

    # Same-color ball in a 32.5px separate-train gap still pairs (>31.2px physical clearance),
    # while a 25px separate-train gap is NOT shot through and dumps off-screen instead
    dx_r, dy_r = red_single.x - fx, red_single.y - fy
    dr = math.hypot(dx_r, dy_r)
    ux_r, uy_r = dx_r / dr, dy_r / dr
    mx_r, my_r = (fx + red_single.x) * 0.5, (fy + red_single.y) * 0.5
    gap_nbr = TrackedBall(5, 0, 1800.0, mx_r - uy_r * 32.5, my_r + ux_r * 32.5, "green", None, 0.0, 0.65, 1.0)
    st_gap32 = TrackState({0: [red_single, gap_nbr]}, [red_single, gap_nbr], [], 0.65)
    dec_pair32 = solver.decide(lvl, st_gap32, FrogState("red", "purple", 0.0), [], cfg, now=2.0)
    assert dec_pair32.action == "shoot"
    assert dec_pair32.reason == "setup_pair"
    assert dec_pair32.is_junk_dump is False

    tight_inner = TrackedBall(6, 0, 1800.0, mx_r - uy_r * 25.0, my_r + ux_r * 25.0, "green", None, 0.0, 0.65, 1.0)
    st_gap25 = TrackState({0: [red_single, tight_inner]}, [red_single, tight_inner], [], 0.65)
    assert solver.decide(lvl, st_gap25, FrogState("red", "purple", 0.0), [], cfg, now=2.1).action == "wait"
    dec_block25 = solver.decide(lvl, st_gap25, FrogState("red", "purple", 0.0), [], cfg, now=2.45)
    assert dec_block25.reason != "setup_pair"
    assert dec_block25.is_junk_dump is True

    # Off-screen dump avoids a gap that a moving ball is predicted to roll across
    gap_ring = []
    for k, deg_w in enumerate(range(12, 350, 10)):
        rad_w = math.radians(deg_w)
        gap_ring.append(
            TrackedBall(600 + k, 0, 100.0 + k * 25.0, fx + math.cos(rad_w) * 140.0, fy + math.sin(rad_w) * 140.0, "red", None, 0.0, 0.2, 1.0)
        )
    moving_closer = TrackedBall(690, 0, 900.0, fx + 140.0, fy - 35.0, "blue", None, 80.0, 0.45, 1.0)
    dump_pred = find_offscreen_dump_point(
        lvl.frog_xy,
        gap_ring + [moving_closer],
        predicted_pts={moving_closer.ball_id: [(fx + 140.0, fy)]},
    )
    assert dump_pred is None


def test_longer_pop_prioritized_over_closer_or_midtrack_3_pop_and_moving_adj_neighbor_clear():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    solver = StrategySolver()
    cfg = config.RuntimeConfig(shot_cooldown_sec=0.12)

    # 1. CONTROL mode: rear 4-ball pop (s=500, progress=0.18, dist=206px) beats closer mid-track 3-ball pop (s=1450, progress=0.52, dist=145px)
    rear_4 = [
        _make_ball(curve, 0, 1, 500.0, "red"),
        _make_ball(curve, 0, 2, 532.0, "red"),
        _make_ball(curve, 0, 3, 564.0, "red"),
    ]
    mid_3 = [
        _make_ball(curve, 0, 4, 1450.0, "red"),
        _make_ball(curve, 0, 5, 1482.0, "red"),
    ]
    st_ctrl = TrackState({0: rear_4 + mid_3}, rear_4 + mid_3, [], mid_3[-1].progress)
    dec_ctrl = solver.decide(lvl, st_ctrl, FrogState("red", "green", 0.0), [], cfg, now=1.0)
    assert dec_ctrl.action == "shoot"
    assert dec_ctrl.expected_removed == 4
    assert dec_ctrl.target_ball_id == rear_4[0].ball_id

    # 2. DEFENSE mode: outside skull danger zone, rear 4-ball pop STILL beats mid-track 3-ball pop (progress=0.53 < 0.72)
    danger_dummy = [_make_ball(curve, 0, 6, curve.total_length * 0.80, "purple")]
    st_def_mid = TrackState({0: rear_4 + mid_3 + danger_dummy}, rear_4 + mid_3 + danger_dummy, [], danger_dummy[0].progress)
    dec_def_mid = solver.decide(lvl, st_def_mid, FrogState("red", "green", 0.0), [], cfg, now=2.0)
    assert dec_def_mid.mode == "DEFENSE"
    assert dec_def_mid.expected_removed == 4
    assert dec_def_mid.target_ball_id == rear_4[0].ball_id

    # 3. DEFENSE mode skull emergency: 3-ball pop AT the skull danger zone (progress=0.80 >= 0.72) beats rear 4-ball pop
    danger_s = curve.total_length * 0.79
    skull_3 = [
        _make_ball(curve, 0, 7, danger_s, "red"),
        _make_ball(curve, 0, 8, danger_s + 32.0, "red"),
    ]
    st_def_skull = TrackState({0: rear_4 + skull_3}, rear_4 + skull_3, [], skull_3[-1].progress)
    dec_def_skull = solver.decide(lvl, st_def_skull, FrogState("red", "green", 0.0), [], cfg, now=3.0)
    assert dec_def_skull.mode == "DEFENSE"
    assert dec_def_skull.expected_removed == 3
    assert dec_def_skull.target_ball_id == skull_3[0].ball_id

    # 4. Loopy fast-moving flanking neighbor clearance (miss_20261122_115945_160 geometry):
    # Moving target flanked by touching neighbors pairs via setup_pair instead of dumping off-screen
    loopy = assets.levels["loopy"]
    lcurve = loopy.curves[0]
    vel = 78.0
    moving_seg = [
        _make_ball(lcurve, 0, 100, 1289.2, "blue", velocity=vel),
        _make_ball(lcurve, 0, 101, 1316.2, "red", velocity=vel),
        _make_ball(lcurve, 0, 102, 1351.4, "yellow", velocity=vel),
        _make_ball(lcurve, 0, 103, 1383.8, "red", velocity=vel),
        _make_ball(lcurve, 0, 104, 1416.2, "yellow", velocity=vel),
    ]
    loopy_lead = [_make_ball(lcurve, 0, 105, lcurve.total_length * 0.78, "blue", velocity=vel)]
    st_loopy = TrackState({0: moving_seg + loopy_lead}, moving_seg + loopy_lead, [], loopy_lead[0].progress)
    dec_loopy = solver.decide(loopy, st_loopy, FrogState("red", "red", 0.0), [], cfg, now=4.0)
    assert dec_loopy.action == "shoot"
    assert dec_loopy.reason == "setup_pair"
    assert dec_loopy.is_junk_dump is False

    # 4b. In CONTROL mode, skip risky swap_shoot setup_pair and fast flanked single-ball setup_pair -> dump cleanly
    solver.reset()
    cols_13 = ["blue", "red", "yellow", "purple", "yellow", "blue", "purple", "yellow", "blue", "purple", "yellow", "blue", "purple"]
    fast_seg = [
        _make_ball(lcurve, 0, 110 + i, 1250.0 + i * 32.0, col, velocity=75.0)
        for i, col in enumerate(cols_13)
    ]
    st_fast_ctrl = TrackState({0: fast_seg}, fast_seg, [], 0.25)
    dec_no_swap_setup = solver.decide(loopy, st_fast_ctrl, FrogState("green", "red", 0.0), [], cfg, now=4.2)
    assert dec_no_swap_setup.is_junk_dump is True
    dec_skip_fast_setup = solver.decide(loopy, st_fast_ctrl, FrogState("red", "green", 0.0), [], cfg, now=4.4)
    assert dec_skip_fast_setup.is_junk_dump is True

    # 4c. Endgame bonus optimization: 50%-progress cluster clears BEFORE 5%-progress cluster so 5% is left for last
    solver.reset()
    early_5 = [_make_ball(curve, 0, 210, curve.total_length * 0.05, "red"), _make_ball(curve, 0, 211, curve.total_length * 0.05 + 32.0, "red")]
    mid_50 = [_make_ball(curve, 0, 212, curve.total_length * 0.50, "red"), _make_ball(curve, 0, 213, curve.total_length * 0.50 + 32.0, "red")]
    st_end_bonus = TrackState({0: early_5 + mid_50}, early_5 + mid_50, [], mid_50[-1].progress)
    dec_end_bonus = solver.decide(lvl, st_end_bonus, FrogState("red", "blue", 0.0), [], cfg, now=5.0)
    assert dec_end_bonus.action == "shoot"
    assert dec_end_bonus.target_ball_id in (mid_50[0].ball_id, mid_50[1].ball_id)

    # 5. Moving lead-edge pair (open front ahead) aims from run center, never overshooting past front edge
    sp = assets.levels["spaceinvaders"]
    sc = sp.curves[0]
    p_tail = _make_ball(sc, 0, 200, 125.0, "purple", velocity=70.0)
    w_a = _make_ball(sc, 0, 201, 155.0, "white", velocity=70.0)
    w_b = _make_ball(sc, 0, 202, 187.0, "white", velocity=70.0)
    st_lead = TrackState({0: [p_tail, w_a, w_b]}, [p_tail, w_a, w_b], [], w_b.progress)
    dec_lead = solver.decide(sp, st_lead, FrogState("white", "yellow", 0.0), [], cfg, now=5.0)
    assert dec_lead.action == "shoot"
    assert dec_lead.target_s <= w_b.s + 10.0

    # 6. Even in DEFENSE mode, surrounded orphan waits briefly for a dump gap before random seam insert
    solver.reset()
    fx, fy = lvl.frog_xy
    ring_def = []
    for k, deg_w in enumerate(range(0, 360, 10)):
        rad_w = math.radians(deg_w)
        ring_def.append(
            TrackedBall(700 + k, 0, 100.0 + k * 20.0, fx + math.cos(rad_w) * 160.0, fy + math.sin(rad_w) * 160.0, "red" if k % 2 == 0 else "blue", None, 0.0, 0.2, 1.0)
        )
    inner_def = _make_ball(curve, 0, 799, curve.total_length * 0.80, "red")
    ring_def.append(inner_def)
    st_ring_def = TrackState({0: ring_def}, ring_def, [], inner_def.progress)
    dec_def_wait = solver.decide(lvl, st_ring_def, FrogState("white", "white", 0.0), [], cfg, now=6.0)
    assert dec_def_wait.mode == "DEFENSE"
    assert dec_def_wait.action == "wait"
    assert dec_def_wait.reason == "wait_for_dump_gap"

    # 7. Closer front-train ball blocking the ray and ±8px offsets (< 32.32px) must block shot (no magneting)
    solver.reset()
    target_back = _make_ball(curve, 0, 800, 500.0, "red")
    dx_b, dy_b = target_back.x - fx, target_back.y - fy
    db = math.hypot(dx_b, dy_b)
    ux_b, uy_b = dx_b / db, dy_b / db
    front_magnet = TrackedBall(801, 0, 1800.0, (fx + target_back.x) * 0.5 - uy_b * 28.0, (fy + target_back.y) * 0.5 + ux_b * 28.0, "green", None, 0.0, 0.65, 1.0)
    st_mag = TrackState({0: [target_back, front_magnet]}, [target_back, front_magnet], [], 0.65)
    assert solver.decide(lvl, st_mag, FrogState("red", "purple", 0.0), [], cfg, now=7.0).action == "wait"
    dec_mag = solver.decide(lvl, st_mag, FrogState("red", "purple", 0.0), [], cfg, now=7.35)
    assert dec_mag.reason != "setup_pair"
    assert dec_mag.is_junk_dump is True

    # 8. Second yellow in stock must NOT double-shoot an already-popping yellow-yellow pair
    y_a = _make_ball(curve, 0, 810, 800.0, "yellow", velocity=45.0)
    y_b = _make_ball(curve, 0, 811, 832.0, "yellow", velocity=45.0)
    g_sep = _make_ball(curve, 0, 812, 864.0, "green", velocity=45.0)
    y_blast = BlastZone(0, 820.0, config.BLAST_RADIUS_NORMAL_PX, expires_at=8.0 + config.BLAST_EXCLUSION_SEC)
    st_popping = TrackState({0: [y_a, y_b, g_sep]}, [y_a, y_b, g_sep], [y_blast], 0.35)
    dec_no_dbl = solver.decide(lvl, st_popping, FrogState("yellow", "purple", 0.0), [], cfg, now=8.2)
    assert dec_no_dbl.target_ball_id != y_a.ball_id

    # 9. Coin cooldown prevents spamming the same coin while bullet is still in flight
    ctrl_coin = AutoInputController()
    coin_shot = ShotDecision("shoot", "CONTROL", (80.0, 95.0), "blue", None, None, None, 0, 0, 28.0, "grab_coin", is_coin=True)
    coin_dist = math.hypot(coin_shot.aim_xy[0] - lvl.frog_xy[0], coin_shot.aim_xy[1] - lvl.frog_xy[1])
    coin_wait = max(0.0, coin_dist - config.BALL_RADIUS) / config.BULLET_SPEED_PX_PER_SEC
    ctrl_coin._last_shot_t = 20.0
    ctrl_coin._last_landing_t = 20.0 + config.DUMP_RELOAD_FLOOR_SEC
    ctrl_coin._last_was_dump = True
    ctrl_coin._last_coin_xy = coin_shot.aim_xy
    ctrl_coin._last_coin_landing_t = 20.0 + coin_wait + 0.18
    assert ctrl_coin.can_fire_now(coin_shot, cfg, now=20.22) is False
    assert ctrl_coin.can_fire_now(coin_shot, cfg, now=20.0 + coin_wait + 0.20) is True

    # 10. Reversing train pair does not over-lead backward onto trailing neighbor
    tr_lvl = assets.levels["triangle"]
    tr_c = tr_lvl.curves[0]
    g_tr = _make_ball(tr_c, 0, 900, 1290.0, "green", velocity=-40.0)
    b1_tr = _make_ball(tr_c, 0, 901, 1322.0, "blue", velocity=-40.0)
    b2_tr = _make_ball(tr_c, 0, 902, 1354.0, "blue", velocity=-40.0)
    st_rev = TrackState({0: [g_tr, b1_tr, b2_tr]}, [g_tr, b1_tr, b2_tr], [], b2_tr.progress)
    dec_rev = solver.decide(tr_lvl, st_rev, FrogState("blue", "red", 0.0), [], cfg, now=9.0)
    assert dec_rev.action == "shoot"
    assert dec_rev.target_s > b1_tr.s + 6.0

    # 11. CONTROL mode midgame immediately dumps unplayable cur_color instead of stalling for blocked next_color pair
    solver.reset()
    pad_balls = [_make_ball(curve, 0, 950 + k, 100.0 + k * 32.0, "green" if k % 2 == 0 else "yellow") for k in range(14)]
    blk_g1 = _make_ball(curve, 0, 970, 900.0, "purple")
    blk_g2 = _make_ball(curve, 0, 971, 932.0, "purple")
    blk_wall = TrackedBall(972, 0, 1800.0, (fx + blk_g1.x) * 0.5, (fy + blk_g1.y) * 0.5, "white", None, 0.0, 0.55, 1.0)
    st_dump_ctrl = TrackState({0: pad_balls + [blk_g1, blk_g2, blk_wall]}, pad_balls + [blk_g1, blk_g2, blk_wall], [], 0.55)
    dec_dump_ctrl = solver.decide(lvl, st_dump_ctrl, FrogState("red", "purple", 0.0), [], cfg, now=10.0)
    assert dec_dump_ctrl.mode == "CONTROL"
    assert dec_dump_ctrl.action == "shoot"
    assert dec_dump_ctrl.is_junk_dump is True

    # 12. Predict same-color gap rollback even when cascades == 0 (1 + 1 matching end balls)
    assert predict_gap_rollback(["red", "yellow", "yellow", "red"], 1, "yellow") is True
    assert predict_gap_rollback(["blue", "yellow", "yellow", "red"], 1, "yellow") is False
    rb_seg = [
        _make_ball(curve, 0, 980, 700.0, "red"),
        _make_ball(curve, 0, 981, 732.0, "yellow"),
        _make_ball(curve, 0, 982, 764.0, "yellow"),
        _make_ball(curve, 0, 983, 796.0, "red"),
    ]
    st_rb = TrackState({0: rb_seg}, rb_seg, [], 0.35)
    dec_rb = solver.decide(lvl, st_rb, FrogState("yellow", "purple", 0.0), [], cfg, now=11.0)
    assert dec_rb.will_rollback is True
    w_rb_def = compute_dynamic_cooldown(lvl.frog_xy, dec_rb.aim_xy, ShotDecision(**{**dec_rb.__dict__, "mode": "DEFENSE"}), cfg)
    w_no_rb = compute_dynamic_cooldown(lvl.frog_xy, dec_rb.aim_xy, ShotDecision(**{**dec_rb.__dict__, "mode": "DEFENSE", "will_rollback": False}), cfg)
    assert abs((w_rb_def - w_no_rb) - config.DYNAMIC_COOLDOWN_COMBO_PAD_SEC) < 1e-6

    # 13. Sustained backwards/Zuma reverse (reverse_streak >= 5) uses full negative lead,
    #     and shooting a backwards powerup adds DYNAMIC_COOLDOWN_COMBO_PAD_SEC
    sustained_seg = [
        _make_ball(tr_c, 0, 990, 1290.0, "green", velocity=-60.0),
        _make_ball(tr_c, 0, 991, 1322.0, "blue", velocity=-60.0),
        _make_ball(tr_c, 0, 992, 1354.0, "blue", velocity=-60.0),
    ]
    for b in sustained_seg:
        b.reverse_streak = 6
    st_sust = TrackState({0: sustained_seg}, sustained_seg, [], sustained_seg[-1].progress)
    dec_sust = solver.decide(tr_lvl, st_sust, FrogState("blue", "red", 0.0), [], cfg, now=12.0)
    assert dec_sust.target_s < dec_rev.target_s - 4.0

    sustained_seg[1].powerup = "backwards"
    dec_bw = solver.decide(tr_lvl, st_sust, FrogState("blue", "red", 0.0), [], cfg, now=12.1)
    assert dec_bw.is_backwards is True
    w_bw = compute_dynamic_cooldown(tr_lvl.frog_xy, dec_bw.aim_xy, dec_bw, cfg)
    w_plain = compute_dynamic_cooldown(tr_lvl.frog_xy, dec_bw.aim_xy, ShotDecision(**{**dec_bw.__dict__, "is_backwards": False}), cfg)
    assert abs((w_bw - w_plain) - config.DYNAMIC_COOLDOWN_COMBO_PAD_SEC) < 1e-6
