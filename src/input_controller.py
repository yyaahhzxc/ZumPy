import ctypes
from ctypes import wintypes
import math
import time
from typing import Optional, Tuple

import config
from solver import ShotDecision
from vision import FrogState
from window_capture import WindowRect, ZumaWindowCapture

# Win32 SendInput constants and 64-bit compatible structures
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


def _screen_to_abs_virtual(sx: int, sy: int) -> Tuple[int, int]:
    user32 = ctypes.windll.user32
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = max(2, user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    vh = max(2, user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    nx = int(round((sx - vx) * 65535.0 / float(vw - 1)))
    ny = int(round((sy - vy) * 65535.0 / float(vh - 1)))
    nx = max(0, min(65535, nx))
    ny = max(0, min(65535, ny))
    return nx, ny


def _send_mouse_events(*events: Tuple[int, int, int]) -> None:
    arr = (INPUT * len(events))()
    for i, (dx, dy, flags) in enumerate(events):
        arr[i].type = INPUT_MOUSE
        arr[i].union.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
    ctypes.windll.user32.SendInput(len(events), ctypes.byref(arr), ctypes.sizeof(INPUT))


def send_aim_and_left_click(sx: int, sy: int) -> None:
    """Move cursor instantaneously and left-click using Win32 SendInput."""
    nx, ny = _screen_to_abs_virtual(sx, sy)
    move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send_mouse_events((nx, ny, move_flags))
    time.sleep(0.012)
    _send_mouse_events(
        (nx, ny, move_flags | MOUSEEVENTF_LEFTDOWN),
        (nx, ny, move_flags | MOUSEEVENTF_LEFTUP),
    )


def send_aim_and_right_click(sx: int, sy: int) -> None:
    """Move cursor inside Zuma's client area and right-click to swap balls."""
    nx, ny = _screen_to_abs_virtual(sx, sy)
    move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send_mouse_events((nx, ny, move_flags))
    time.sleep(0.008)
    _send_mouse_events(
        (nx, ny, move_flags | MOUSEEVENTF_RIGHTDOWN),
        (nx, ny, move_flags | MOUSEEVENTF_RIGHTUP),
    )


def send_swap_and_left_click(sx: int, sy: int) -> None:
    """Move cursor, right-click to swap frog ball, wait for swap animation, and left-click to fire."""
    nx, ny = _screen_to_abs_virtual(sx, sy)
    move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send_mouse_events((nx, ny, move_flags))
    time.sleep(0.008)
    _send_mouse_events((nx, ny, move_flags | MOUSEEVENTF_RIGHTDOWN))
    time.sleep(0.014)
    _send_mouse_events((nx, ny, move_flags | MOUSEEVENTF_RIGHTUP))
    time.sleep(0.065)
    _send_mouse_events((nx, ny, move_flags | MOUSEEVENTF_LEFTDOWN))
    time.sleep(0.012)
    _send_mouse_events((nx, ny, move_flags | MOUSEEVENTF_LEFTUP))


def compute_dynamic_cooldown(
    frog_xy: Tuple[float, float],
    aim_xy: Tuple[float, float],
    decision: ShotDecision,
    cfg: config.RuntimeConfig,
    last_aim_xy: Optional[Tuple[float, float]] = None,
    last_was_combo: bool = False,
) -> float:
    """Compute distance-aware cooldown so close or switched-target shots fire fast while same-spot shots wait for landing."""
    dist = math.hypot(aim_xy[0] - frog_xy[0], aim_xy[1] - frog_xy[1])
    flight_sec = max(0.0, dist - config.BALL_RADIUS) / max(config.BULLET_SPEED_PX_PER_SEC, 100.0)
    swap_pad = config.SWAP_EXTRA_LATENCY_SEC if decision.action == "swap_shoot" else 0.0
    full_wait = max(cfg.shot_cooldown_sec, flight_sec + config.DYNAMIC_COOLDOWN_PAD_SEC + swap_pad)
    if decision.will_rollback or decision.is_backwards or (decision.mode != "DEFENSE" and (decision.cascades > 0 or decision.is_bomb)):
        full_wait += config.DYNAMIC_COOLDOWN_COMBO_PAD_SEC

    if last_aim_xy is not None and not last_was_combo:
        same_target = (
            math.hypot(aim_xy[0] - last_aim_xy[0], aim_xy[1] - last_aim_xy[1])
            < config.SAME_TARGET_RADIUS_PX
        )
        if not same_target:
            return cfg.shot_cooldown_sec

    return full_wait


class AutoInputController:
    """Handles safe automated aiming, atomic right-click swaps, target-aware cooldown, and manual override."""

    def __init__(self):
        self._last_shot_t: float = 0.0
        self._next_allowed_t: float = 0.0
        self._last_sent_screen_pos: Optional[Tuple[int, int]] = None
        self._pending_swap_target_color: Optional[str] = None
        self._pending_swap_expires: float = 0.0
        self._last_aim_xy: Optional[Tuple[float, float]] = None
        self._last_landing_t: float = 0.0
        self._last_was_combo: bool = False
        self._last_was_dump: bool = False
        self._last_coin_xy: Optional[Tuple[float, float]] = None
        self._last_coin_landing_t: float = 0.0

    def reset(self) -> None:
        self._last_sent_screen_pos = None
        self._pending_swap_target_color = None
        self._pending_swap_expires = 0.0
        self._last_aim_xy = None
        self._last_landing_t = 0.0
        self._last_was_combo = False
        self._last_was_dump = False
        self._last_coin_xy = None
        self._last_coin_landing_t = 0.0
        self._next_allowed_t = 0.0

    def can_fire_now(self, decision: ShotDecision, cfg: config.RuntimeConfig, now: float) -> bool:
        """Check whether reload floor, swap readiness, and previous track-shot landing allow firing."""
        min_reload = (
            max(cfg.shot_cooldown_sec, config.SWAP_RELOAD_FLOOR_SEC)
            if decision.action == "swap_shoot"
            else cfg.shot_cooldown_sec
        )
        if now < (self._last_shot_t + min_reload):
            return False
        if decision.action not in ("shoot", "swap_shoot") or decision.blocked or decision.aim_xy is None:
            return False
        if decision.is_coin and now < self._last_coin_landing_t and self._last_coin_xy is not None:
            dx_c = decision.aim_xy[0] - self._last_coin_xy[0]
            dy_c = decision.aim_xy[1] - self._last_coin_xy[1]
            if (dx_c * dx_c + dy_c * dy_c) < (config.SAME_TARGET_RADIUS_PX * config.SAME_TARGET_RADIUS_PX):
                return False
        if now < self._last_landing_t:
            if decision.is_coin:
                if self._last_aim_xy is not None:
                    dx = decision.aim_xy[0] - self._last_aim_xy[0]
                    dy = decision.aim_xy[1] - self._last_aim_xy[1]
                    if (dx * dx + dy * dy) < (config.SAME_TARGET_RADIUS_PX * config.SAME_TARGET_RADIUS_PX):
                        return False
            else:
                if not self._last_was_dump or self._last_was_combo:
                    return False
                if self._last_aim_xy is not None:
                    dx = decision.aim_xy[0] - self._last_aim_xy[0]
                    dy = decision.aim_xy[1] - self._last_aim_xy[1]
                    if (dx * dx + dy * dy) < (config.SAME_TARGET_RADIUS_PX * config.SAME_TARGET_RADIUS_PX):
                        return False
        return True

    def check_manual_override(self, rect: WindowRect) -> bool:
        """Return True if the user moved the mouse far away from PyZuma's last sent position."""
        if self._last_sent_screen_pos is None:
            return False
        cur = ZumaWindowCapture.get_cursor_screen_pos()
        if cur is None:
            return False
        dx = cur[0] - self._last_sent_screen_pos[0]
        dy = cur[1] - self._last_sent_screen_pos[1]
        thresh = config.MANUAL_MOUSE_OVERRIDE_PX * max(1.0, rect.scale_x)
        return (dx * dx + dy * dy) > (thresh * thresh)

    def execute(
        self,
        decision: ShotDecision,
        frog: FrogState,
        frog_xy: Tuple[float, float],
        rect: WindowRect,
        is_foreground: bool,
        is_paused: bool,
        cfg: config.RuntimeConfig,
        now: float,
    ) -> Tuple[str, Optional[str]]:
        """
        Attempt to execute decision if Auto-Play is active and guards pass.
        Returns (status_code, optional_override_reason).
        status_code: "off", "unfocused", "paused", "override", "waiting", "fired"
        """
        if not cfg.auto_play:
            self.reset()
            return "off", None

        if is_paused:
            self._pending_swap_target_color = None
            return "paused", None

        if not is_foreground:
            self._pending_swap_target_color = None
            self._last_sent_screen_pos = None
            return "unfocused", None

        if self.check_manual_override(rect):
            self.reset()
            return "override", "Paused: Manual mouse override"

        if not self.can_fire_now(decision, cfg, now):
            return "waiting", None

        sx, sy = ZumaWindowCapture.logical_to_screen(decision.aim_xy[0], decision.aim_xy[1], rect)
        if decision.action == "swap_shoot":
            send_swap_and_left_click(sx, sy)
            frog.cur_color, frog.next_color = frog.next_color, frog.cur_color
        else:
            send_aim_and_left_click(sx, sy)

        self._last_sent_screen_pos = (sx, sy)
        self._last_shot_t = now
        self._last_aim_xy = decision.aim_xy
        self._last_was_dump = bool(decision.is_junk_dump or decision.is_coin)
        self._last_was_combo = bool(
            decision.will_rollback
            or decision.is_backwards
            or (decision.mode != "DEFENSE" and (decision.cascades > 0 or decision.is_bomb))
        )

        if decision.is_coin:
            coin_dist = math.hypot(decision.aim_xy[0] - frog_xy[0], decision.aim_xy[1] - frog_xy[1])
            coin_flight = max(0.0, coin_dist - config.BALL_RADIUS) / max(config.BULLET_SPEED_PX_PER_SEC, 100.0)
            self._last_coin_xy = decision.aim_xy
            self._last_coin_landing_t = now + coin_flight + 0.18

        if self._last_was_dump:
            wait_sec = max(cfg.shot_cooldown_sec, config.DUMP_RELOAD_FLOOR_SEC)
        else:
            wait_sec = compute_dynamic_cooldown(frog_xy, decision.aim_xy, decision, cfg)
        self._last_landing_t = now + wait_sec
        self._next_allowed_t = now + (
            wait_sec
            if (self._last_was_combo or not self._last_was_dump)
            else max(cfg.shot_cooldown_sec, config.DUMP_RELOAD_FLOOR_SEC)
        )
        return "fired", None
