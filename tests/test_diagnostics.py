import json
import numpy as np

import config
from diagnostics import DiagnosticsManager
from solver import ShotDecision


def test_diagnostics_ring_buffer_manual_dump_and_auto_miss(tmp_path):
    diag = DiagnosticsManager(debug_root=tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    summary = {
        "map_id": "spiral",
        "mode": "CONTROL",
        "cur_color": "blue",
        "next_color": "red",
        "frog_xy": (327.0, 233.0),
        "aim_xy": (450.0, 200.0),
        "reason": "match_3_casc_0",
        "ball_count": 10,
        "balls": [{"x": 450.0, "y": 200.0, "color": "blue", "powerup": None}],
    }

    for i in range(5):
        diag.record_frame(frame, summary, now=1.0 + i * 0.1)
    assert len(diag.ring) == 5

    manual_dir = diag.dump_manual("manual")
    assert manual_dir is not None
    assert (manual_dir / "telemetry.json").is_file()
    assert (manual_dir / "frame_00.png").is_file()

    # Shot that succeeds (ball count drops from 10 -> 7) should NOT trigger miss dump
    dec = ShotDecision("shoot", "CONTROL", (450.0, 200.0), "blue", 1, 0, 500.0, 3, 0, 50.0, "match_3")
    diag.register_shot(frame, dec, pre_ball_count=10, summary=summary, now=2.0)
    assert diag.check_pending_shots(frame, current_ball_count=7, state_summary=summary, now=3.25) is None
    assert diag.auto_miss_count == 0

    # Shot that misfires (ball count stays >= pre_ball_count) SHOULD trigger auto miss dump
    diag.register_shot(frame, dec, pre_ball_count=10, summary=summary, now=3.0)
    miss_dir = diag.check_pending_shots(frame, current_ball_count=11, state_summary=summary, now=4.25)
    assert miss_dir is not None
    assert (miss_dir / "pre_shot.png").is_file()
    assert (miss_dir / "aim_overlay.png").is_file()
    assert (miss_dir / "post_shot.png").is_file()
    assert (miss_dir / "miss_report.json").is_file()
    report_data = json.loads((miss_dir / "miss_report.json").read_text())
    assert report_data["misfire_type"] == "pop_failed"
    assert "intended" in report_data["intended_vs_actual"]
    assert "actual" in report_data["intended_vs_actual"]
    assert diag.auto_miss_count == 1

    # setup_pair that forms a touching pair near aim_xy should NOT trigger auto miss
    setup_dec = ShotDecision("shoot", "CONTROL", (450.0, 200.0), "blue", 1, 0, 500.0, 0, 0, 20.0, "setup_pair")
    post_paired = {
        **summary,
        "balls": [
            {"x": 450.0, "y": 200.0, "color": "blue", "powerup": None},
            {"x": 482.0, "y": 200.0, "color": "blue", "powerup": None},
        ],
    }
    diag.register_shot(frame, setup_dec, pre_ball_count=5, summary=summary, now=10.0)
    assert diag.check_pending_shots(frame, current_ball_count=6, state_summary=post_paired, now=11.25) is None

    # setup_pair that misses near aim_xy SHOULD log setup_pair_failed even if an unrelated pair exists elsewhere
    post_isolated = {
        **summary,
        "balls": [
            {"x": 100.0, "y": 100.0, "color": "blue", "powerup": None},
            {"x": 132.0, "y": 100.0, "color": "blue", "powerup": None},
            {"x": 450.0, "y": 200.0, "color": "blue", "powerup": None},
        ],
    }
    diag.register_shot(frame, setup_dec, pre_ball_count=5, summary=summary, now=11.0)
    setup_miss = diag.check_pending_shots(frame, current_ball_count=6, state_summary=post_isolated, now=12.25)
    assert setup_miss is not None
    assert json.loads((setup_miss / "miss_report.json").read_text())["misfire_type"] == "setup_pair_failed"

    # Clean offscreen dump (even if matching color exists blocked elsewhere, count unchanged) should NOT trigger auto miss
    pre_with_white = {**summary, "balls": [{"x": 450.0, "y": 200.0, "color": "white", "powerup": None}]}
    dump_white = ShotDecision("shoot", "CONTROL", (630.0, 240.0), "white", None, None, None, 0, 0, -5.0, "junk_offscreen_dump", is_junk_dump=True)
    diag.register_shot(frame, dump_white, pre_ball_count=5, summary=pre_with_white, now=18.0)
    assert diag.check_pending_shots(frame, current_ball_count=5, state_summary=pre_with_white, now=19.25) is None

    # Offscreen dump that actually hits track (ball count & shot_color count increase) SHOULD log dump_hit_track
    post_hit_white = {
        **summary,
        "balls": [
            {"x": 450.0, "y": 200.0, "color": "white", "powerup": None},
            {"x": 600.0, "y": 240.0, "color": "white", "powerup": None},
        ],
    }
    diag.register_shot(frame, dump_white, pre_ball_count=5, summary=pre_with_white, now=20.0)
    dump_miss = diag.check_pending_shots(frame, current_ball_count=6, state_summary=post_hit_white, now=21.25)
    assert dump_miss is not None
    assert json.loads((dump_miss / "miss_report.json").read_text())["misfire_type"] == "dump_hit_track"
