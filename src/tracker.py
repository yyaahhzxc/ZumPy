from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import config
from level_data import CurveData, LevelInfo
from vision import DetectedBall


@dataclass
class BlastZone:
    curve_idx: int
    s: float
    radius_px: float
    expires_at: float
    impact_at: float = 0.0


@dataclass
class TrackedBall:
    ball_id: int
    curve_idx: int
    s: float
    x: float
    y: float
    color: str
    powerup: Optional[str]
    velocity: float          # px/sec along arc length
    progress: float          # s / total_length in [0, 1]
    visibility: float
    in_tunnel_memory: bool = False
    last_seen: float = 0.0
    seen_frames: int = 1
    reverse_streak: int = 0


@dataclass
class TrackState:
    balls_by_curve: Dict[int, List[TrackedBall]]
    all_balls: List[TrackedBall]
    active_blasts: List[BlastZone]
    max_progress: float


class BallTracker:
    """1D curve-coordinate ball tracker with live speed, tunnel memory, and blast exclusion zones."""

    def __init__(self):
        self._level_id: Optional[str] = None
        self._next_id: int = 1
        self._prev_by_curve: Dict[int, List[TrackedBall]] = {}
        self._tunnel_ghosts: Dict[int, List[TrackedBall]] = {}
        self._blasts: List[BlastZone] = []
        self._inflight_bullets: List[Tuple[float, Tuple[float, float], Tuple[float, float], float]] = []
        self._last_update_t: Optional[float] = None

    def reset(self, level_id: Optional[str] = None) -> None:
        self._level_id = level_id
        self._prev_by_curve.clear()
        self._tunnel_ghosts.clear()
        self._blasts.clear()
        self._inflight_bullets.clear()
        self._last_update_t = None

    def add_inflight_bullet(
        self,
        frog_xy: Tuple[float, float],
        aim_xy: Tuple[float, float],
        is_swap: bool,
        now: float,
    ) -> None:
        """Record a recently fired projectile ray so crossing bullets are ignored by the curve scanner."""
        lat = config.PIPELINE_LATENCY_SEC + (config.SWAP_EXTRA_LATENCY_SEC if is_swap else 0.0)
        self._inflight_bullets.append((now, frog_xy, aim_xy, lat))

    def add_blast_zone(
        self,
        curve_idx: int,
        s: float,
        is_bomb: bool,
        now: float,
        duration_sec: float = config.BLAST_EXCLUSION_SEC,
        flight_delay_sec: float = 0.0,
    ) -> None:
        radius = config.BLAST_RADIUS_BOMB_PX if is_bomb else config.BLAST_RADIUS_NORMAL_PX
        self._blasts.append(
            BlastZone(
                curve_idx=curve_idx,
                s=float(s),
                radius_px=radius,
                expires_at=now + flight_delay_sec + duration_sec,
                impact_at=now + flight_delay_sec * 0.75,
            )
        )

    def get_active_blasts(self, now: float) -> List[BlastZone]:
        self._blasts = [b for b in self._blasts if b.expires_at > now]
        return list(self._blasts)

    def get_active_vision_blasts(self, now: float) -> List[BlastZone]:
        """Return blast zones that have reached impact so explosion sparkles do not register as balls."""
        return [b for b in self.get_active_blasts(now) if now >= b.impact_at]

    def _is_inflight_bullet(self, det: DetectedBall, now: float) -> bool:
        for fired_at, fxy, axy, lat in self._inflight_bullets:
            el = now - fired_at
            if el < 0.0 or el > 0.85:
                continue
            dx = axy[0] - fxy[0]
            dy = axy[1] - fxy[1]
            ray_len = math.hypot(dx, dy)
            if ray_len <= 1e-6:
                continue
            ux, uy = dx / ray_len, dy / ray_len
            rx, ry = det.x - fxy[0], det.y - fxy[1]
            proj = rx * ux + ry * uy
            perp = abs(rx * uy - ry * ux)
            d_est = max(0.0, el - lat) * config.BULLET_SPEED_PX_PER_SEC
            # Only mid-flight: never hide the shot ball once it reaches (and joins) the target chain
            if proj > ray_len - 28.0:
                continue
            if perp <= 26.0 and proj >= 18.0 and abs(proj - d_est) <= 60.0:
                return True
        return False

    def _match_curve(
        self,
        curve_idx: int,
        curve: CurveData,
        detected: List[DetectedBall],
        dt: float,
        now: float,
    ) -> List[TrackedBall]:
        self._inflight_bullets = [b for b in self._inflight_bullets if 0.0 <= (now - b[0]) <= 0.85]
        prev_balls = self._prev_by_curve.get(curve_idx, []) + self._tunnel_ghosts.get(curve_idx, [])
        used_prev: set[int] = set()
        tracked: List[TrackedBall] = []

        # Match detected balls against previous frame (same color, plausible 1D delta)
        for det in detected:
            best_idx = -1
            best_dist = float("inf")
            for pi, pb in enumerate(prev_balls):
                if pi in used_prev or pb.color != det.color:
                    continue
                ds = det.s - pb.s
                if -52.0 <= ds <= 42.0:
                    ad = abs(ds)
                    if ad < best_dist:
                        best_dist = ad
                        best_idx = pi

            if best_idx >= 0:
                used_prev.add(best_idx)
                pb = prev_balls[best_idx]
                seen_frames = pb.seen_frames + 1
                ds_step = det.s - pb.s
                if pb.in_tunnel_memory:
                    vel = pb.velocity
                else:
                    # Reject single-frame insertion/snap jumps (keep prior speed) but stay as responsive as before
                    max_phys_step = min(config.TRACKER_MAX_STEP_DS_PX, max(6.0, 220.0 * dt))
                    if abs(ds_step) <= max_phys_step:
                        inst_vel = ds_step / max(dt, 1e-3)
                        inst_vel = max(config.TRACKER_MIN_VEL_PX, min(config.TRACKER_MAX_VEL_PX, inst_vel))
                        alpha = 0.65 if (inst_vel >= 25.0 and pb.velocity < 15.0) else 0.50
                        vel = (1.0 - alpha) * pb.velocity + alpha * inst_vel
                    else:
                        vel = pb.velocity
                bid = pb.ball_id
                rev_streak = (pb.reverse_streak + 1) if vel < -15.0 else 0
            else:
                if self._is_inflight_bullet(det, now):
                    continue
                vel = 0.0
                seen_frames = 1
                rev_streak = 0
                bid = self._next_id
                self._next_id += 1

            prog = det.s / max(curve.total_length, 1.0)
            tracked.append(
                TrackedBall(
                    ball_id=bid,
                    curve_idx=curve_idx,
                    s=det.s,
                    x=det.x,
                    y=det.y,
                    color=det.color,
                    powerup=det.powerup,
                    velocity=vel,
                    progress=max(0.0, min(1.0, prog)),
                    visibility=det.visibility,
                    in_tunnel_memory=False,
                    last_seen=now,
                    seen_frames=seen_frames,
                    reverse_streak=rev_streak,
                )
            )

        tracked.sort(key=lambda b: b.s)

        # Smooth velocity toward segment median across touching balls to reject single-ball spikes
        if len(tracked) >= 2:
            seg_start = 0
            for i in range(1, len(tracked) + 1):
                is_end = (i == len(tracked)) or ((tracked[i].s - tracked[i - 1].s) > 42.0)
                if is_end:
                    seg = tracked[seg_start:i]
                    if len(seg) >= 2:
                        avg_v = sum(b.velocity for b in seg) / len(seg)
                        for b in seg:
                            b.velocity = 0.5 * b.velocity + 0.5 * avg_v
                    seg_start = i

        # Tunnel continuity: retain persistent balls that just rolled into an underground tunnel (t1 == 1)
        new_ghosts: List[TrackedBall] = []
        for pi, pb in enumerate(prev_balls):
            if pi in used_prev:
                continue
            if (now - pb.last_seen) > 4.0:
                continue

            advance_v = pb.velocity if pb.velocity > 5.0 else 28.0
            est_s = pb.s + advance_v * dt
            if est_s >= curve.total_length:
                continue

            in_tun_now = curve.is_tunnel_at(est_s)
            near_tun_ahead = any(
                curve.is_tunnel_at(min(curve.total_length, pb.s + off))
                for off in (8.0, 20.0, 34.0)
            )
            if pb.in_tunnel_memory and in_tun_now:
                keep = True
            elif (not pb.in_tunnel_memory) and near_tun_ahead and pb.seen_frames >= 2:
                keep = True
            else:
                keep = False

            if keep:
                px, py = curve.point_at_arc_len(est_s)
                prog = est_s / max(curve.total_length, 1.0)
                new_ghosts.append(
                    TrackedBall(
                        ball_id=pb.ball_id,
                        curve_idx=curve_idx,
                        s=est_s,
                        x=px,
                        y=py,
                        color=pb.color,
                        powerup=pb.powerup,
                        velocity=pb.velocity,
                        progress=max(0.0, min(1.0, prog)),
                        visibility=0.0,
                        in_tunnel_memory=True,
                        last_seen=pb.last_seen,
                        seen_frames=pb.seen_frames,
                    )
                )

        self._prev_by_curve[curve_idx] = tracked
        self._tunnel_ghosts[curve_idx] = new_ghosts

        combined = sorted(tracked + new_ghosts, key=lambda b: b.s)
        return combined

    def update(
        self,
        level: LevelInfo,
        detected_by_curve: Dict[int, List[DetectedBall]],
        now: float,
    ) -> TrackState:
        if self._level_id != level.id:
            self.reset(level.id)

        dt = 0.033 if self._last_update_t is None else max(1e-3, min(0.25, now - self._last_update_t))
        self._last_update_t = now

        active_blasts = self.get_active_blasts(now)
        balls_by_curve: Dict[int, List[TrackedBall]] = {}
        all_balls: List[TrackedBall] = []

        for idx, curve in enumerate(level.curves):
            dets = detected_by_curve.get(idx, [])
            combined = self._match_curve(idx, curve, dets, dt, now)
            balls_by_curve[idx] = combined
            all_balls.extend(combined)

        vis_progs = [
            b.progress
            for b in all_balls
            if (not b.in_tunnel_memory) or (now - b.last_seen) <= 1.0
        ]
        if vis_progs:
            max_prog = max(vis_progs)
        elif all_balls:
            max_prog = max(b.progress for b in all_balls)
        else:
            max_prog = 0.0

        return TrackState(
            balls_by_curve=balls_by_curve,
            all_balls=all_balls,
            active_blasts=active_blasts,
            max_progress=max_prog,
        )
