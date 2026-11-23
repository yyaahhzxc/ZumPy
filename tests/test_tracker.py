import config
from level_data import load_game_assets
from tracker import BallTracker
from vision import DetectedBall


def test_tracker_velocity_forward_pause_and_reverse():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["spiral"]
    curve = lvl.curves[0]
    tracker = BallTracker()

    def make_det(s_vals, colors=("red", "red", "blue")):
        out = []
        for s_v, col in zip(s_vals, colors):
            x, y = curve.point_at_arc_len(s_v)
            out.append(DetectedBall(0, s_v, x, y, col, None, 0.9, 1.0))
        return {0: out}

    # Moving forward at ~60 px/s (6 px every 0.1s)
    t = 10.0
    s0 = [300.0, 332.0, 364.0]
    for step in range(5):
        st = tracker.update(lvl, make_det([v + step * 6.0 for v in s0]), t + step * 0.1)
    assert st.all_balls[0].velocity > 35.0

    # Combo snap-back reversing (-20 px per 0.1s)
    t2 = t + 0.5
    cur_s = [v + 24.0 for v in s0]
    for step in range(1, 5):
        st = tracker.update(lvl, make_det([v - step * 18.0 for v in cur_s]), t2 + step * 0.1)
    assert st.all_balls[0].velocity < -40.0
    assert st.all_balls[0].reverse_streak >= 2


def test_tracker_tunnel_memory_and_blast_zones():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["underover"]
    curve = lvl.curves[0]
    tracker = BallTracker()

    # Find first tunnel entry index
    tunnel_idxs = [i for i, flag in enumerate(curve.in_tunnel) if flag]
    assert tunnel_idxs
    enter_s = float(curve.arc_len[tunnel_idxs[0]])

    # Frame 1: ball just before tunnel entrance
    pre_s = max(0.0, enter_s - 12.0)
    x1, y1 = curve.point_at_arc_len(pre_s)
    tracker.update(lvl, {0: [DetectedBall(0, pre_s, x1, y1, "blue", None, 0.9, 1.0)]}, 1.0)

    # Frame 2: ball moves forward toward tunnel
    near_s = max(0.0, enter_s - 4.0)
    x2, y2 = curve.point_at_arc_len(near_s)
    tracker.update(lvl, {0: [DetectedBall(0, near_s, x2, y2, "blue", None, 0.9, 1.0)]}, 1.1)

    # Frame 3: ball disappears under tunnel -> ghost memory preserved!
    st3 = tracker.update(lvl, {0: []}, 1.2)
    assert len(st3.all_balls) == 1
    assert st3.all_balls[0].in_tunnel_memory is True
    assert st3.all_balls[0].color == "blue"

    # Blast zone expiration (including impact_at delay for distant shots)
    tracker.add_blast_zone(0, 500.0, is_bomb=True, now=5.0, duration_sec=0.4, flight_delay_sec=0.2)
    assert len(tracker.get_active_vision_blasts(5.05)) == 0
    assert len(tracker.get_active_vision_blasts(5.16)) == 1
    assert len(tracker.get_active_blasts(5.65)) == 0


def test_tracker_filters_inflight_bullets_and_rejects_60fps_insertion_spikes():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    lvl = assets.levels["groovefest"]
    curve = lvl.curves[0]
    tracker = BallTracker()

    # 1. Stationary 60-fps train with a sudden 1-frame 16px insertion jump must not spike velocity
    base_s = [600.0, 632.0, 664.0, 696.0]
    for step in range(4):
        dets = [
            DetectedBall(0, sv, *curve.point_at_arc_len(sv), col, None, 0.9, 1.0)
            for sv, col in zip(base_s, ("red", "yellow", "blue", "white"))
        ]
        tracker.update(lvl, {0: dets}, 1.0 + step * 0.016)
    shifted_s = [600.0, 632.0, 680.0, 712.0]
    st_jump = tracker.update(
        lvl,
        {
            0: [
                DetectedBall(0, sv, *curve.point_at_arc_len(sv), col, None, 0.9, 1.0)
                for sv, col in zip(shifted_s, ("red", "yellow", "blue", "white"))
            ]
        },
        1.064,
    )
    assert max(abs(b.velocity) for b in st_jump.all_balls) < 10.0

    # 2. Active in-flight bullet crossing a curve segment along its shot ray is ignored
    tracker.reset(lvl.id)
    fx, fy = lvl.frog_xy
    el = 0.25
    travel = (el - config.PIPELINE_LATENCY_SEC) * config.BULLET_SPEED_PX_PER_SEC
    aim_xy = None
    bx = by = s_near = 0.0
    for idx_c in range(len(curve.points)):
        px, py = float(curve.points[idx_c, 0]), float(curve.points[idx_c, 1])
        dist_p = ((px - fx) ** 2 + (py - fy) ** 2) ** 0.5
        if abs(dist_p - travel) < 15.0:
            aim_xy = (px, py)
            bx, by = px, py
            s_near = float(curve.arc_len[idx_c])
            break
    assert aim_xy is not None
    # Real shot continues well past the crossing point (bullet is mid-flight, not landing)
    ext = (travel + 150.0) / max(travel, 1.0)
    aim_xy = (fx + (aim_xy[0] - fx) * ext, fy + (aim_xy[1] - fy) * ext)

    tracker.add_inflight_bullet(lvl.frog_xy, aim_xy, is_swap=False, now=2.0)
    st_bullet = tracker.update(lvl, {0: [DetectedBall(0, s_near, bx, by, "yellow", None, 0.9, 1.0)]}, 2.0 + el)
    assert len(st_bullet.all_balls) == 0

    # A ball detected right at the aim point (the shot ball joining the chain) is NOT hidden
    tracker.reset(lvl.id)
    tracker.add_inflight_bullet(lvl.frog_xy, (bx, by), is_swap=False, now=3.0)
    st_land = tracker.update(lvl, {0: [DetectedBall(0, s_near, bx, by, "yellow", None, 0.9, 1.0)]}, 3.0 + el)
    assert len(st_land.all_balls) == 1
