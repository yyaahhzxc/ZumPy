import copy
import math
import threading
import time
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore

import config
from diagnostics import DiagnosticsManager
from input_controller import AutoInputController
from level_data import GameAssets, LevelInfo
from solver import ShotDecision, StrategySolver, split_into_segments
from tracker import BallTracker, TrackState
from vision import VisionPipeline
from window_capture import ZumaWindowCapture


def _compute_tail_progress(
    level: Optional[LevelInfo],
    track_state: Optional[TrackState],
    life_state: str,
) -> List[Optional[float]]:
    """Return per-curve rear-most ball progress once no balls remain near that curve's spawn entrance."""
    if level is None or track_state is None or life_state != "PLAYING":
        return []
    result: List[Optional[float]] = []
    for idx, curve in enumerate(level.curves):
        curve_balls = track_state.balls_by_curve.get(idx, ())
        vis_curve = [b for b in curve_balls if not b.in_tunnel_memory]
        if not vis_curve:
            result.append(None)
            continue
        first_vis_s = 0.0
        s = 0.0
        while s < curve.total_length:
            if curve.visibility_at(s) >= 0.5:
                first_vis_s = s
                break
            s += 8.0
        spawn_cutoff = first_vis_s + 52.0
        if any(b.s <= spawn_cutoff for b in curve_balls):
            result.append(None)
        else:
            tail_p = float(min(b.progress for b in vis_curve))
            result.append(tail_p if tail_p >= 0.05 else None)
    return result


class LevelLifecycleGate:
    """Prevents aiming/shooting before real balls finish fast roll-in or after a level is cleared."""

    def __init__(self):
        self.state: str = "WAITING_FOR_BALLS"
        self._start_streak: int = 0
        self._midlevel_streak: int = 0
        self._empty_streak: int = 0
        self._restart_streak: int = 0
        self._recover_streak: int = 0
        self._rollin_started_t: float = 0.0
        self._rollin_calm_streak: int = 0
        self._rollin_empty_streak: int = 0
        self._prev_vis_count: int = 0
        self._prev_max_s: float = 0.0
        self._prev_t: float = 0.0

    def reset(self) -> None:
        self.state = "WAITING_FOR_BALLS"
        self._start_streak = 0
        self._midlevel_streak = 0
        self._empty_streak = 0
        self._restart_streak = 0
        self._recover_streak = 0
        self._rollin_started_t = 0.0
        self._rollin_calm_streak = 0
        self._rollin_empty_streak = 0
        self._prev_vis_count = 0
        self._prev_max_s = 0.0
        self._prev_t = 0.0

    def _enter_rollin(self, now: float, vis_count: int, max_s: float = 0.0) -> None:
        self.state = "LOADING_ROLLIN"
        self._rollin_started_t = now
        self._rollin_calm_streak = 0
        self._rollin_empty_streak = 0
        self._start_streak = 0
        self._midlevel_streak = 0
        self._empty_streak = 0
        self._restart_streak = 0
        self._recover_streak = 0
        self._prev_vis_count = vis_count
        self._prev_max_s = max_s
        self._prev_t = now

    def update(self, track_state: TrackState, now: Optional[float] = None) -> str:
        if now is None:
            now = time.monotonic()

        visible_balls = [b for b in track_state.all_balls if not b.in_tunnel_memory]
        max_s = max((b.s for b in visible_balls), default=0.0)
        entrance_contig = 0
        max_contig = 0
        for balls in track_state.balls_by_curve.values():
            vis_curve = [b for b in balls if not b.in_tunnel_memory]
            for seg in split_into_segments(vis_curve):
                if len(seg) > max_contig:
                    max_contig = len(seg)
                if seg and seg[0].s < 550.0 and len(seg) > entrance_contig:
                    entrance_contig = len(seg)

        if self.state == "WAITING_FOR_BALLS":
            # Require balls to actually enter near the track start (ignores preview / "LEVEL X-Y" banner text)
            if entrance_contig >= config.LEVEL_START_MIN_CONTIGUOUS:
                self._start_streak += 1
                self._midlevel_streak = 0
                if self._start_streak >= config.LEVEL_START_CONFIRM_FRAMES:
                    self._enter_rollin(now, len(visible_balls), max_s)
            elif max_contig >= 4:
                # Recover mid-level if lifecycle was reset after balls already rolled past s=550
                self._start_streak = 0
                self._midlevel_streak += 1
                if self._midlevel_streak >= config.LEVEL_CLEARED_RECOVER_FRAMES:
                    self.state = "PLAYING"
                    self._empty_streak = 0
                    self._midlevel_streak = 0
            else:
                self._start_streak = 0
                self._midlevel_streak = 0
        elif self.state == "LOADING_ROLLIN":
            if len(track_state.all_balls) == 0:
                self._rollin_empty_streak += 1
                if self._rollin_empty_streak >= config.LEVEL_CLEAR_ZERO_FRAMES:
                    self.reset()
                return self.state
            self._rollin_empty_streak = 0

            dt = max(now - self._prev_t, 1e-3)
            vis_count = len(visible_balls)
            growth_rate = max(0, vis_count - self._prev_vis_count) / dt
            lead_vel = max(0.0, (max_s - self._prev_max_s) / dt)
            max_vel = max(lead_vel, max((b.velocity for b in visible_balls), default=0.0))
            elapsed = now - self._rollin_started_t
            self._prev_vis_count = vis_count
            self._prev_max_s = max_s
            self._prev_t = now

            if track_state.max_progress >= config.DANGER_PROGRESS_THRESHOLD or elapsed >= config.LEVEL_ROLLIN_MAX_WAIT_SEC:
                self.state = "PLAYING"
                self._empty_streak = 0
            elif (
                elapsed >= config.LEVEL_ROLLIN_MIN_WAIT_SEC
                and max_vel <= config.LEVEL_ROLLIN_CALM_VEL_PX
                and growth_rate <= config.LEVEL_ROLLIN_MAX_GROWTH_PER_SEC
            ):
                self._rollin_calm_streak += 1
                if self._rollin_calm_streak >= config.LEVEL_ROLLIN_CALM_FRAMES:
                    self.state = "PLAYING"
                    self._empty_streak = 0
            else:
                self._rollin_calm_streak = 0
        elif self.state == "PLAYING":
            if len(track_state.all_balls) == 0:
                self._empty_streak += 1
                if self._empty_streak >= config.LEVEL_CLEAR_ZERO_FRAMES:
                    self.state = "LEVEL_CLEARED"
                    self._restart_streak = 0
            else:
                self._empty_streak = 0
        elif self.state == "LEVEL_CLEARED":
            # Only resume via LOADING_ROLLIN once a fresh contiguous train rolls in near the start,
            # or recover to PLAYING if 1..4 real final balls re-emerge from a long tunnel/bridge.
            if entrance_contig >= config.LEVEL_START_MIN_CONTIGUOUS:
                self._restart_streak += 1
                self._recover_streak = 0
                if self._restart_streak >= config.LEVEL_START_CONFIRM_FRAMES:
                    self._enter_rollin(now, len(visible_balls), max_s)
            elif 1 <= len(visible_balls) <= 4:
                self._restart_streak = 0
                self._recover_streak += 1
                if self._recover_streak >= config.LEVEL_CLEARED_RECOVER_FRAMES:
                    self.state = "PLAYING"
                    self._empty_streak = 0
                    self._recover_streak = 0
            else:
                self._restart_streak = 0
                self._recover_streak = 0

        return self.state


class ZumaWorker(QtCore.QThread):
    """Dedicated real-time capture -> CV -> track -> solve -> overlay/input worker thread."""

    state_updated = QtCore.pyqtSignal(object)
    hook_changed = QtCore.pyqtSignal(bool)
    auto_paused_signal = QtCore.pyqtSignal(str)
    dump_created_signal = QtCore.pyqtSignal(str)

    def __init__(self, assets: GameAssets, capture: ZumaWindowCapture, cfg: config.RuntimeConfig):
        super().__init__()
        self.assets = assets
        self.capture = capture
        self._cfg_lock = threading.Lock()
        self._cfg = copy.copy(cfg)

        self.vision = VisionPipeline(assets)
        self.tracker = BallTracker()
        self.solver = StrategySolver()
        self.lifecycle = LevelLifecycleGate()
        self.input_ctrl = AutoInputController()
        self.diagnostics = DiagnosticsManager()

        self._is_running = True
        self._manual_dump_requested = False
        self._was_hooked = False
        self._last_map_id: Optional[str] = None
        self._prog_sum: float = 0.0
        self._prog_count: int = 0
        self._last_avg_sample_t: float = 0.0
        self._last_nonzero_progress: Optional[float] = None
        self._last_nonzero_t: float = 0.0
        self._last_clear_progress: Optional[float] = None
        self._last_clear_latched_t: float = -10.0
        self._zero_progress_streak: int = 0
        self._prev_life_state: str = "WAITING_FOR_BALLS"

    def update_config(self, **kwargs: Any) -> None:
        with self._cfg_lock:
            for k, v in kwargs.items():
                if hasattr(self._cfg, k):
                    setattr(self._cfg, k, v)
            if "auto_play" in kwargs and not kwargs["auto_play"]:
                self.input_ctrl.reset()

    def get_config_snapshot(self) -> config.RuntimeConfig:
        with self._cfg_lock:
            return copy.copy(self._cfg)

    def request_manual_dump(self) -> None:
        self._manual_dump_requested = True

    def stop(self) -> None:
        self._is_running = False

    def _build_overlay_payload(
        self,
        rect,
        level: Optional[LevelInfo],
        paused: bool,
        track_state: Optional[TrackState],
        frog_cur: Optional[str],
        frog_next: Optional[str],
        coins,
        decision: Optional[ShotDecision],
        fps: float,
        latency_ms: float,
        auto_play: bool,
        show_hud: bool,
    ) -> Dict[str, Any]:
        curves_draw = []
        if level is not None:
            for c in level.curves:
                curves_draw.append({
                    "points": c.points_list if c.points_list else c.points.tolist(),
                    "in_tunnel": c.in_tunnel_list if c.in_tunnel_list else c.in_tunnel.tolist(),
                })

        blasts_draw = []
        balls_draw = []
        if level is not None and track_state is not None:
            for b in track_state.all_balls:
                balls_draw.append({
                    "id": b.ball_id,
                    "x": b.x,
                    "y": b.y,
                    "color": b.color,
                    "powerup": b.powerup,
                    "velocity": round(b.velocity, 1),
                    "progress": round(b.progress, 3),
                    "in_tunnel_memory": b.in_tunnel_memory,
                })
            for bz in track_state.active_blasts:
                if 0 <= bz.curve_idx < len(level.curves):
                    bx, by = level.curves[bz.curve_idx].point_at_arc_len(bz.s)
                    blasts_draw.append({"x": bx, "y": by, "radius_px": bz.radius_px})

        return {
            "rect": rect,
            "map_id": level.id if level else None,
            "map_name": level.disp_name if level else None,
            "is_paused": paused,
            "frog_xy": level.frog_xy if level else None,
            "cur_color": frog_cur,
            "next_color": frog_next,
            "curves": curves_draw,
            "balls": balls_draw,
            "blasts": blasts_draw,
            "coins": [{"x": c.x, "y": c.y} for c in coins],
            "action": decision.action if decision else "none",
            "mode": decision.mode if decision else ("PAUSED" if paused else "CONTROL"),
            "aim_xy": decision.aim_xy if decision else None,
            "blocked": decision.blocked if decision else False,
            "blocker_xy": decision.blocker_xy if decision else None,
            "reason": decision.reason if decision else "idle",
            "is_coin": decision.is_coin if decision else False,
            "is_junk_dump": decision.is_junk_dump if decision else False,
            "fps": fps,
            "latency_ms": latency_ms,
            "auto_play": auto_play,
            "show_hud": show_hud,
            "auto_miss_count": self.diagnostics.auto_miss_count,
            "max_progress": float(track_state.max_progress) if track_state else 0.0,
            "avg_progress": (self._prog_sum / self._prog_count) if self._prog_count > 0 else 0.0,
            "last_clear_progress": self._last_clear_progress,
            "tail_progress": _compute_tail_progress(level, track_state, self.lifecycle.state),
        }

    def run(self) -> None:
        fps_ema = 0.0
        last_loop_t = time.perf_counter()

        while self._is_running:
            t_start = time.perf_counter()
            dt_loop = max(1e-4, t_start - last_loop_t)
            last_loop_t = t_start
            inst_fps = 1.0 / dt_loop
            fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

            cfg = self.get_config_snapshot()
            hooked = self.capture.is_hooked()
            if not hooked:
                hooked = self.capture.find_window()
            if hooked != self._was_hooked:
                self._was_hooked = hooked
                self.hook_changed.emit(hooked)

            if not hooked:
                self.state_updated.emit(None)
                time.sleep(0.25)
                continue

            grab = self.capture.grab_frame()
            if grab is None:
                self.state_updated.emit(None)
                time.sleep(0.08)
                continue

            frame_bgr, rect = grab
            cursor_screen = self.capture.get_cursor_screen_pos()
            cursor_logical = (
                self.capture.screen_to_logical(cursor_screen[0], cursor_screen[1], rect)
                if cursor_screen is not None
                else None
            )

            now = time.monotonic()
            is_fg = self.capture.is_foreground()
            active_blasts = self.tracker.get_active_vision_blasts(now)
            level, paused, detected_by_curve, frog, coins = self.vision.process_frame(
                frame_bgr,
                cursor_logical=cursor_logical,
                active_blasts=active_blasts,
            )

            if level is None or paused:
                if level is None:
                    if self._last_nonzero_progress is not None and self._zero_progress_streak >= 1:
                        self._last_clear_progress = self._last_nonzero_progress
                        self._last_nonzero_progress = None
                        self._last_clear_latched_t = now
                    self._zero_progress_streak = 0
                    self._last_map_id = None
                    self.tracker.reset(None)
                    self.solver.reset()
                    self.lifecycle.reset()
                payload = self._build_overlay_payload(
                    rect, level, True, None, None, None, [], None,
                    fps_ema, (time.perf_counter() - t_start) * 1000.0, cfg.auto_play,
                    show_hud=False,
                )
                self.state_updated.emit(payload)
                time.sleep(0.04)
                continue

            if self._last_map_id != level.id:
                self._last_map_id = level.id
                self.tracker.reset(level.id)
                self.solver.reset()
                self.lifecycle.reset()
                self._prog_sum = 0.0
                self._prog_count = 0
                self._last_avg_sample_t = 0.0
                self._last_nonzero_progress = None
                self._zero_progress_streak = 0
                self._prev_life_state = "WAITING_FOR_BALLS"

            track_state = self.tracker.update(level, detected_by_curve, now)
            life_state = self.lifecycle.update(track_state, now)

            vis_count = sum(1 for b in track_state.all_balls if not b.in_tunnel_memory)
            if life_state == "PLAYING" and vis_count >= 3 and track_state.max_progress > 0.0:
                prog = float(track_state.max_progress)
                if self._prog_count == 0 or (now - self._last_avg_sample_t) >= 1.0:
                    self._prog_sum += prog
                    self._prog_count += 1
                    self._last_avg_sample_t = now
                self._last_nonzero_progress = prog
                self._last_nonzero_t = now
                self._zero_progress_streak = 0
            elif vis_count == 0 or life_state == "LEVEL_CLEARED":
                if vis_count == 0:
                    self._zero_progress_streak += 1
                if self._last_nonzero_progress is not None and (
                    life_state == "LEVEL_CLEARED" or self._zero_progress_streak >= 2
                ):
                    self._last_clear_progress = self._last_nonzero_progress
                    self._last_nonzero_progress = None
                    self._last_clear_latched_t = now
            self._prev_life_state = life_state

            if life_state != "PLAYING":
                self.input_ctrl.reset()
                idle_reason = {
                    "WAITING_FOR_BALLS": "waiting_for_balls",
                    "LOADING_ROLLIN": "loading_balls",
                    "LEVEL_CLEARED": "level_cleared",
                }.get(life_state, "waiting_for_balls")
                idle_dec = ShotDecision(
                    action="none",
                    mode=life_state,
                    aim_xy=None,
                    shot_color=frog.cur_color,
                    target_ball_id=None,
                    target_curve_idx=None,
                    target_s=None,
                    expected_removed=0,
                    cascades=0,
                    score=0.0,
                    reason=idle_reason,
                )
                payload = self._build_overlay_payload(
                    rect,
                    level,
                    False,
                    track_state,
                    frog.cur_color,
                    frog.next_color,
                    [],
                    idle_dec,
                    fps_ema,
                    (time.perf_counter() - t_start) * 1000.0,
                    cfg.auto_play,
                    show_hud=is_fg,
                )
                self.state_updated.emit(payload)
                time.sleep(0.016)
                continue

            decision = self.solver.decide(level, track_state, frog, coins, cfg, now)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            payload = self._build_overlay_payload(
                rect,
                level,
                False,
                track_state,
                frog.cur_color,
                frog.next_color,
                coins,
                decision,
                fps_ema,
                latency_ms,
                cfg.auto_play,
                show_hud=is_fg,
            )

            diag_summary = {
                "map_id": payload["map_id"],
                "mode": payload["mode"],
                "cur_color": payload["cur_color"],
                "next_color": payload["next_color"],
                "frog_xy": payload["frog_xy"],
                "aim_xy": payload["aim_xy"],
                "reason": payload["reason"],
                "ball_count": len(track_state.all_balls),
                "balls": payload["balls"],
            }
            self.diagnostics.record_frame(frame_bgr, diag_summary, now)

            visible_ball_count = sum(1 for b in track_state.all_balls if not b.in_tunnel_memory)
            miss_dir = self.diagnostics.check_pending_shots(frame_bgr, visible_ball_count, diag_summary, now)
            if miss_dir is not None:
                payload["auto_miss_count"] = self.diagnostics.auto_miss_count
                self.dump_created_signal.emit(f"Auto-miss saved: {miss_dir.name}")

            if self._manual_dump_requested:
                self._manual_dump_requested = False
                out = self.diagnostics.dump_manual("manual")
                if out is not None:
                    self.dump_created_signal.emit(f"Saved F9 dump: {out.name}")
                else:
                    self.dump_created_signal.emit("No gameplay frames buffered yet for F9 dump.")

            act_status, override_msg = self.input_ctrl.execute(
                decision, frog, level.frog_xy, rect, is_fg, paused, cfg, now
            )
            if act_status == "override" and override_msg:
                self.update_config(auto_play=False)
                payload["auto_play"] = False
                self.auto_paused_signal.emit(override_msg)
            elif act_status == "fired":
                flight_delay = 0.0
                if decision.aim_xy is not None:
                    flight_delay = (
                        math.hypot(decision.aim_xy[0] - level.frog_xy[0], decision.aim_xy[1] - level.frog_xy[1])
                        / max(config.BULLET_SPEED_PX_PER_SEC, 100.0)
                        + (config.SWAP_EXTRA_LATENCY_SEC if decision.action == "swap_shoot" else 0.0)
                    )
                    self.tracker.add_inflight_bullet(
                        level.frog_xy,
                        decision.aim_xy,
                        decision.action == "swap_shoot",
                        now,
                    )
                if (decision.expected_removed >= 3 or decision.is_bomb) and decision.target_curve_idx is not None and decision.target_s is not None:
                    self.tracker.add_blast_zone(
                        decision.target_curve_idx,
                        decision.target_s,
                        decision.is_bomb,
                        now,
                        flight_delay_sec=flight_delay,
                    )
                self.diagnostics.register_shot(frame_bgr, decision, visible_ball_count, diag_summary, now)

            self.state_updated.emit(payload)

            # Cap loop gently around ~50-60 FPS to keep CPU headroom for Zuma
            elapsed = time.perf_counter() - t_start
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)
