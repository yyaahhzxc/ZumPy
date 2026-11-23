import threading
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Callable, Optional, Tuple

import cv2
import mss
import numpy as np
import win32api
import win32con
import win32gui
import win32process

import config
from utils import LockedObject


ZUMA_PROCESS_NAMES = frozenset({"zuma.exe", "popcapgame1.exe", "zumadeluxe.exe"})


def is_zuma_process(proc_name: str) -> bool:
    """Return True only for actual Zuma Deluxe / PopCap game executables."""
    return proc_name.strip().lower() in ZUMA_PROCESS_NAMES


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    width: int
    height: int

    @property
    def scale_x(self) -> float:
        return self.width / float(config.LOGICAL_W)

    @property
    def scale_y(self) -> float:
        return self.height / float(config.LOGICAL_H)


class ZumaWindowCapture:
    """Win32 window tracker + fast mss client-area capture for Zuma Deluxe."""

    def __init__(self):
        self.hwnd = LockedObject(None)
        self._monitor_thread: Optional[threading.Thread] = None
        self._thread_local = threading.local()

    def _get_sct(self) -> mss.mss:
        sct = getattr(self._thread_local, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._thread_local.sct = sct
        return sct

    def is_hooked(self) -> bool:
        return self.hwnd.value_is_set()

    def find_window(self, on_close: Optional[Callable[[], None]] = None) -> bool:
        """Locate visible Zuma Deluxe window and start exit monitor thread."""
        with self.hwnd.lock:
            existing = self.hwnd.obj
        if existing and win32gui.IsWindow(existing):
            return True

        found_hwnd: Optional[int] = None

        def _enum_cb(hwnd: int, _extra) -> None:
            nonlocal found_hwnd
            if found_hwnd is not None:
                return
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd) or ""
            title_l = title.lower()
            if "pyzuma" in title_l:
                return

            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc_name = ""
            try:
                handle = win32api.OpenProcess(
                    win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                    False,
                    pid,
                )
                try:
                    proc_name = Path(win32process.GetModuleFileNameEx(handle, 0)).name.lower()
                finally:
                    win32api.CloseHandle(handle)
            except Exception:
                proc_name = ""

            if is_zuma_process(proc_name):
                rect = win32gui.GetClientRect(hwnd)
                if (rect[2] - rect[0]) >= 300 and (rect[3] - rect[1]) >= 200:
                    found_hwnd = hwnd

        win32gui.EnumWindows(_enum_cb, None)
        if found_hwnd is None:
            return False

        with self.hwnd.lock:
            self.hwnd.obj = found_hwnd

        def _monitor_loop() -> None:
            while True:
                with self.hwnd.lock:
                    cur = self.hwnd.obj
                if cur is None or win32gui.IsWindow(cur) == 0:
                    with self.hwnd.lock:
                        self.hwnd.obj = None
                    if on_close is not None:
                        on_close()
                    break
                if not threading.main_thread().is_alive():
                    break
                sleep(1.0)

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._monitor_thread.start()
        return True

    def get_client_rect(self) -> Optional[WindowRect]:
        with self.hwnd.lock:
            hwnd = self.hwnd.obj
        if not hwnd or not win32gui.IsWindow(hwnd):
            return None
        try:
            if win32gui.IsIconic(hwnd):
                return None
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            w = right - left
            h = bottom - top
            if w <= 10 or h <= 10:
                return None
            sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
            return WindowRect(left=int(sx), top=int(sy), width=int(w), height=int(h))
        except Exception:
            return None

    def is_foreground(self) -> bool:
        with self.hwnd.lock:
            hwnd = self.hwnd.obj
        if not hwnd:
            return False
        try:
            return win32gui.GetForegroundWindow() == hwnd
        except Exception:
            return False

    def focus(self) -> None:
        with self.hwnd.lock:
            hwnd = self.hwnd.obj
        if not hwnd:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def grab_frame(self) -> Optional[Tuple[np.ndarray, WindowRect]]:
        """Capture game client area and return (640x480 BGR image, WindowRect)."""
        rect = self.get_client_rect()
        if rect is None:
            return None
        sct = self._get_sct()
        mon = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        try:
            raw = np.asarray(sct.grab(mon), dtype=np.uint8)
        except Exception:
            return None
        bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        if bgr.shape[1] != config.LOGICAL_W or bgr.shape[0] != config.LOGICAL_H:
            bgr = cv2.resize(bgr, (config.LOGICAL_W, config.LOGICAL_H), interpolation=cv2.INTER_AREA)
        return bgr, rect

    @staticmethod
    def logical_to_screen(x: float, y: float, rect: WindowRect) -> Tuple[int, int]:
        sx = int(round(rect.left + x * rect.scale_x))
        sy = int(round(rect.top + y * rect.scale_y))
        return sx, sy

    @staticmethod
    def screen_to_logical(sx: float, sy: float, rect: WindowRect) -> Tuple[float, float]:
        x = (float(sx) - rect.left) / max(rect.scale_x, 1e-6)
        y = (float(sy) - rect.top) / max(rect.scale_y, 1e-6)
        return x, y

    @staticmethod
    def get_cursor_screen_pos() -> Optional[Tuple[int, int]]:
        try:
            pt = win32api.GetCursorPos()
            return int(pt[0]), int(pt[1])
        except Exception:
            return None
