import math
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def get_base_dir() -> Path:
    """Resolve base directory cleanly for both script execution and PyInstaller."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


class LockedObject:
    """Thread-safe wrapper adapted from PyLex 2.0."""

    def __init__(self, initial: Any = None):
        self.lock = threading.Lock()
        self.obj: Optional[Any] = initial

    def set(self, obj: Any) -> None:
        if not self.lock.locked():
            raise RuntimeError("Tried to set LockedObject value without acquiring lock first")
        self.obj = obj

    def get(self) -> Optional[Any]:
        if not self.lock.locked():
            raise RuntimeError("Tried to get LockedObject value without acquiring lock first")
        return self.obj

    def value_is_set(self) -> bool:
        with self.lock:
            return self.obj is not None


def circular_hue_mean(hues: np.ndarray) -> float:
    """Compute circular mean hue on OpenCV [0, 180) scale."""
    if hues.size == 0:
        return 0.0
    angles = hues.astype(np.float64) * (2.0 * math.pi / 180.0)
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    deg = math.degrees(math.atan2(mean_sin, mean_cos))
    return (deg / 2.0) % 180.0


def circular_hue_dist(h1: float, h2: float) -> float:
    """Shortest distance between two OpenCV hues in [0, 180)."""
    d = abs(float(h1) - float(h2)) % 180.0
    return min(d, 180.0 - d)


def ray_circle_first_hit(
    origin: Tuple[float, float],
    dir_unit: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> Optional[float]:
    """Return distance t >= 0 where ray (origin + t * dir_unit) first hits circle, else None."""
    ox, oy = origin
    dx, dy = dir_unit
    cx, cy = center
    fx = ox - cx
    fy = oy - cy

    b = 2.0 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - (radius * radius)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5
    if t1 >= 0.0:
        return t1
    if t2 >= 0.0:
        return t2
    return None


def point_segment_dist(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Shortest Euclidean distance from point p to segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def find_file_case_insensitive(
    folder: Path,
    stem: str,
    exts: Tuple[str, ...],
) -> Optional[Path]:
    """Find a file matching stem + any ext case-insensitively inside folder."""
    if not folder.is_dir():
        return None
    want_names = {f"{stem.lower()}{ext.lower()}" for ext in exts}
    for entry in folder.iterdir():
        if entry.is_file() and entry.name.lower() in want_names:
            return entry
    return None
