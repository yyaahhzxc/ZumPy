import json
import math
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from solver import ShotDecision, point_segment_dist


@dataclass
class RingEntry:
    timestamp: float
    frame_bgr: np.ndarray
    summary: Dict[str, Any]


@dataclass
class PendingShot:
    shot_time: float
    pre_frame_bgr: np.ndarray
    aim_frame_bgr: np.ndarray
    decision: ShotDecision
    pre_ball_count: int
    telemetry: Dict[str, Any]


def annotate_debug_frame(frame_bgr: np.ndarray, summary: Dict[str, Any]) -> np.ndarray:
    """Render compact ball/frog/aim annotations onto a debug frame copy."""
    img = frame_bgr.copy()
    for b in summary.get("balls", []):
        col = config.COLOR_BGR.get(b.get("color", ""), (200, 200, 200))
        cx, cy = int(round(b["x"])), int(round(b["y"]))
        cv2.circle(img, (cx, cy), 15, col, 2)
        if b.get("powerup"):
            cv2.putText(
                img,
                config.POWERUP_BADGES.get(b["powerup"], "?"),
                (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    frog_xy = summary.get("frog_xy")
    aim_xy = summary.get("aim_xy")
    if frog_xy and aim_xy:
        cv2.line(
            img,
            (int(round(frog_xy[0])), int(round(frog_xy[1]))),
            (int(round(aim_xy[0])), int(round(aim_xy[1]))),
            (0, 255, 120),
            2,
        )
        cv2.circle(img, (int(round(aim_xy[0])), int(round(aim_xy[1]))), 10, (0, 255, 255), 2)

    header = (
        f"map={summary.get('map_id')} mode={summary.get('mode')} "
        f"cur={summary.get('cur_color')} next={summary.get('next_color')} "
        f"reason={summary.get('reason')}"
    )
    cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class DiagnosticsManager:
    """Manages 5-second rolling debug ring buffer, F9 manual dumps, and auto misfire detection."""

    def __init__(self, debug_root: Path = config.DEFAULT_DEBUG_DIR):
        self.debug_root = Path(debug_root)
        maxlen = max(10, int(config.RING_BUFFER_SECONDS * config.RING_BUFFER_FPS))
        self.ring: Deque[RingEntry] = deque(maxlen=maxlen)
        self._last_record_t: float = 0.0
        self._record_interval: float = 1.0 / max(1.0, config.RING_BUFFER_FPS)
        self._pending: List[PendingShot] = []
        self._last_auto_miss_t: float = -1e9
        self.auto_miss_count: int = 0
        self.manual_dump_count: int = 0

    def _prune_old_dumps(self) -> None:
        if not self.debug_root.is_dir():
            return
        dirs = sorted(
            [p for p in self.debug_root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        while len(dirs) > config.MAX_DEBUG_DUMPS:
            old = dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)

    def _make_dump_dir(self, prefix: str) -> Path:
        self.debug_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out = self.debug_root / f"{prefix}_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def record_frame(self, frame_bgr: np.ndarray, summary: Dict[str, Any], now: float) -> None:
        if (now - self._last_record_t) < self._record_interval:
            return
        self._last_record_t = now
        self.ring.append(
            RingEntry(
                timestamp=now,
                frame_bgr=frame_bgr.copy(),
                summary=dict(summary),
            )
        )

    def dump_manual(self, tag: str = "manual") -> Optional[Path]:
        if not self.ring:
            return None
        out_dir = self._make_dump_dir(tag)
        entries = list(self.ring)
        # Save up to ~15 evenly spaced frames across the 5s buffer so disk writes stay fast
        step = max(1, len(entries) // 15)
        sampled = entries[::step]
        if entries[-1] is not sampled[-1]:
            sampled.append(entries[-1])

        telemetry = []
        for idx, entry in enumerate(sampled):
            annotated = annotate_debug_frame(entry.frame_bgr, entry.summary)
            cv2.imwrite(str(out_dir / f"frame_{idx:02d}.png"), annotated)
            telemetry.append({"idx": idx, "timestamp": entry.timestamp, **entry.summary})

        (out_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")
        self.manual_dump_count += 1
        self._prune_old_dumps()
        return out_dir

    def register_shot(
        self,
        frame_bgr: np.ndarray,
        decision: ShotDecision,
        pre_ball_count: int,
        summary: Dict[str, Any],
        now: float,
    ) -> None:
        """Track all fired shots so pop/setup misfires, seam inserts, and track-clipping dumps/coins are logged."""
        if decision.action not in ("shoot", "swap_shoot"):
            return
        aim_annot = annotate_debug_frame(frame_bgr, summary)
        self._pending.append(
            PendingShot(
                shot_time=now,
                pre_frame_bgr=frame_bgr.copy(),
                aim_frame_bgr=aim_annot,
                decision=decision,
                pre_ball_count=pre_ball_count,
                telemetry=dict(summary),
            )
        )

    @staticmethod
    def _did_shot_misfire(
        ps: PendingShot,
        current_ball_count: int,
        state_summary: Dict[str, Any],
    ) -> Tuple[bool, str]:
        dec = ps.decision
        # Ignore level-clear / next-level transition frames (e.g. 1 -> 67 balls)
        if current_ball_count == 0 or current_ball_count > ps.pre_ball_count + 10:
            return False, ""

        if dec.expected_removed >= 3:
            if current_ball_count < ps.pre_ball_count:
                return False, ""
            # Follow-up shots can add balls before we check, so also confirm the targeted
            # same-color group near the aim point actually vanished (local pop evidence).
            if dec.aim_xy is not None and dec.shot_color:
                ax, ay = dec.aim_xy
                r2 = 70.0 * 70.0

                def _near_count(balls: List[Dict[str, Any]]) -> int:
                    return sum(
                        1
                        for b in balls
                        if b.get("color") == dec.shot_color
                        and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r2
                    )

                pre_near = _near_count(ps.telemetry.get("balls", []))
                post_near = _near_count(state_summary.get("balls", []))
                if pre_near >= 2 and post_near < pre_near:
                    return False, ""
            return True, "pop_failed"

        if dec.reason in ("safe_seam_insert", "forced_insert"):
            return True, "unmatched_seam_insert"

        pre_balls = ps.telemetry.get("balls", [])
        post_balls = state_summary.get("balls", [])
        col = dec.shot_color

        if dec.is_junk_dump or dec.is_coin:
            if col and current_ball_count > ps.pre_ball_count:
                fxy = ps.telemetry.get("frog_xy")
                if fxy is not None and dec.aim_xy is not None:
                    pre_ray = sum(
                        1
                        for b in pre_balls
                        if b.get("color") == col and point_segment_dist((b["x"], b["y"]), fxy, dec.aim_xy) <= 55.0
                    )
                    post_ray = sum(
                        1
                        for b in post_balls
                        if b.get("color") == col and point_segment_dist((b["x"], b["y"]), fxy, dec.aim_xy) <= 55.0
                    )
                    if post_ray > pre_ray:
                        return True, ("coin_hit_track" if dec.is_coin else "dump_hit_track")
                else:
                    pre_cnt = sum(1 for b in pre_balls if b.get("color") == col)
                    post_cnt = sum(1 for b in post_balls if b.get("color") == col)
                    if post_cnt > pre_cnt:
                        return True, ("coin_hit_track" if dec.is_coin else "dump_hit_track")
            return False, ""

        if dec.reason == "setup_pair":
            if current_ball_count >= ps.pre_ball_count and col:
                def _count_pairs(balls_list: List[Dict[str, Any]]) -> int:
                    return sum(
                        1
                        for i, a in enumerate(balls_list)
                        for b in balls_list[i + 1:]
                        if ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) <= (38.5 * 38.5)
                    )

                if dec.aim_xy is not None:
                    ax, ay = dec.aim_xy
                    r85_sq = 85.0 * 85.0
                    post_85 = [
                        b
                        for b in post_balls
                        if b.get("color") == col and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r85_sq
                    ]
                    pre_85 = [
                        b
                        for b in pre_balls
                        if b.get("color") == col and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r85_sq
                    ]
                    if _count_pairs(post_85) > _count_pairs(pre_85) or len(post_85) < len(pre_85):
                        return False, ""

                    r2 = 240.0 * 240.0
                    same_col = [
                        b
                        for b in post_balls
                        if b.get("color") == col and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r2
                    ]
                    pre_same_col = [
                        b
                        for b in pre_balls
                        if b.get("color") == col and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r2
                    ]
                    # If the targeted ball already popped on a quick follow-up shot, don't flag setup_pair_failed
                    if len(same_col) < len(pre_same_col):
                        return False, ""
                    if _count_pairs(same_col) <= _count_pairs(pre_same_col):
                        return True, "setup_pair_failed"
                else:
                    same_col = [b for b in post_balls if b.get("color") == col]
                    if _count_pairs(same_col) == 0:
                        return True, "setup_pair_failed"
            return False, ""

        return False, ""

    @staticmethod
    def _summarize_intended_vs_actual(
        ps: PendingShot,
        current_ball_count: int,
        state_summary: Dict[str, Any],
        misfire_type: str,
    ) -> Dict[str, Any]:
        dec = ps.decision
        col = dec.shot_color or "?"
        if dec.expected_removed >= 3:
            intended = f"pop {dec.expected_removed} {col} balls ({dec.reason})"
        elif dec.reason == "setup_pair":
            intended = f"pair {col} ball ({dec.action})"
        elif dec.is_junk_dump:
            intended = f"dump {col} offscreen"
        elif dec.is_coin:
            intended = "grab coin cleanly"
        else:
            intended = f"{dec.reason} ({col})"

        pre_balls = ps.telemetry.get("balls", [])
        post_balls = state_summary.get("balls", [])
        pre_near = post_near = 0
        if dec.aim_xy is not None and dec.shot_color:
            ax, ay = dec.aim_xy
            r2 = 155.0 * 155.0
            pre_near = sum(
                1 for b in pre_balls if b.get("color") == dec.shot_color and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r2
            )
            post_near = sum(
                1 for b in post_balls if b.get("color") == dec.shot_color and ((b["x"] - ax) ** 2 + (b["y"] - ay) ** 2) <= r2
            )

        return {
            "intended": intended,
            "actual": f"balls {ps.pre_ball_count}->{current_ball_count}, {col}_near_aim {pre_near}->{post_near}",
            "misplaced": misfire_type in ("setup_pair_failed", "unmatched_seam_insert", "dump_hit_track", "coin_hit_track")
            or (misfire_type == "pop_failed" and post_near > pre_near),
        }

    def check_pending_shots(
        self,
        frame_bgr: np.ndarray,
        current_ball_count: int,
        state_summary: Dict[str, Any],
        now: float,
    ) -> Optional[Path]:
        """Verify pending shots after MISS_VERIFY_DELAY_SEC and auto-dump on apparent misfire."""
        remaining: List[PendingShot] = []
        saved_dir: Optional[Path] = None

        for ps in self._pending:
            elapsed = now - ps.shot_time
            verify_delay = config.MISS_VERIFY_DELAY_SEC
            if ps.decision.target_ball_id is not None and ps.decision.aim_xy is not None:
                fxy = ps.telemetry.get("frog_xy")
                if fxy:
                    dist = math.hypot(ps.decision.aim_xy[0] - fxy[0], ps.decision.aim_xy[1] - fxy[1])
                    extra = config.SWAP_EXTRA_LATENCY_SEC if ps.decision.action == "swap_shoot" else 0.0
                    verify_delay = max(verify_delay, dist / max(config.BULLET_SPEED_PX_PER_SEC, 100.0) + extra + 0.65)
            if elapsed < verify_delay:
                remaining.append(ps)
                continue

            misfired, misfire_type = self._did_shot_misfire(ps, current_ball_count, state_summary)
            if misfired and (now - self._last_auto_miss_t) >= config.AUTO_MISS_COOLDOWN_SEC and saved_dir is None:
                self._last_auto_miss_t = now
                out_dir = self._make_dump_dir("miss")
                cv2.imwrite(str(out_dir / "pre_shot.png"), ps.pre_frame_bgr)
                cv2.imwrite(str(out_dir / "aim_overlay.png"), ps.aim_frame_bgr)
                cv2.imwrite(str(out_dir / "post_shot.png"), annotate_debug_frame(frame_bgr, state_summary))
                report = {
                    "misfire_type": misfire_type,
                    "intended_vs_actual": self._summarize_intended_vs_actual(
                        ps, current_ball_count, state_summary, misfire_type
                    ),
                    "shot_time": ps.shot_time,
                    "checked_at": now,
                    "pre_ball_count": ps.pre_ball_count,
                    "post_ball_count": current_ball_count,
                    "decision": asdict(ps.decision),
                    "pre_telemetry": ps.telemetry,
                    "post_telemetry": state_summary,
                }
                (out_dir / "miss_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                self.auto_miss_count += 1
                self._prune_old_dumps()
                saved_dir = out_dir

        self._pending = remaining
        return saved_dir
