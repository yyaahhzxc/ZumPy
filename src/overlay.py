import ctypes
from typing import Any, Dict, Optional

import win32api
import win32con
import win32gui
from PyQt6 import QtCore, QtGui, QtWidgets

import config

OVERLAY_COLORKEY_RGB = (1, 1, 2)
OVERLAY_COLORKEY_QCOLOR = QtGui.QColor(*OVERLAY_COLORKEY_RGB)


class HudOverlay(QtWidgets.QWidget):
    """Transparent, always-on-top, click-through PyQt6 HUD pinned to Zuma's client area."""

    def __init__(self):
        super().__init__()
        self._state: Optional[Dict[str, Any]] = None
        self._enabled: bool = True

        self.setWindowTitle("PyZuma HUD Overlay")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def _apply_win32_overlay_styles(self) -> None:
        try:
            hwnd = int(self.winId())
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)
            win32gui.SetLayeredWindowAttributes(
                hwnd,
                win32api.RGB(*OVERLAY_COLORKEY_RGB),
                255,
                win32con.LWA_COLORKEY,
            )
            # WDA_EXCLUDEFROMCAPTURE (0x11): visible to the user, excluded from mss screen capture
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except Exception:
            pass

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._apply_win32_overlay_styles()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled and self.isVisible():
            self.hide()

    def update_state(self, state: Optional[Dict[str, Any]]) -> None:
        self._state = state
        if (
            not self._enabled
            or state is None
            or state.get("rect") is None
            or not state.get("show_hud", False)
        ):
            if self.isVisible():
                self.hide()
            return

        rect = state["rect"]
        dpr = max(1.0, float(self.devicePixelRatioF()))
        screen = QtGui.QGuiApplication.screenAt(QtCore.QPoint(int(rect.left / dpr), int(rect.top / dpr)))
        if screen is not None:
            dpr = max(1.0, float(screen.devicePixelRatio()))

        target_geo = QtCore.QRect(
            int(round(rect.left / dpr)),
            int(round(rect.top / dpr)),
            max(1, int(round(rect.width / dpr))),
            max(1, int(round(rect.height / dpr))),
        )
        if self.geometry() != target_geo:
            self.setGeometry(target_geo)
        if not self.isVisible():
            self.show()
            self._apply_win32_overlay_styles()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self._enabled or self._state is None:
            return
        rect = self._state.get("rect")
        if rect is None or not self._state.get("show_hud", False):
            return

        sx_scale = self.width() / float(config.LOGICAL_W)
        sy_scale = self.height() / float(config.LOGICAL_H)
        painter = QtGui.QPainter(self)
        # Disable antialiasing under LWA_COLORKEY so edges never blend with RGB(1, 1, 2)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), OVERLAY_COLORKEY_QCOLOR)

        def _pt(x: float, y: float) -> QtCore.QPointF:
            return QtCore.QPointF(x * sx_scale, y * sy_scale)

        # 1. Draw curve paths (light gray = open, dashed red = tunnel)
        for curve_draw in self._state.get("curves", []):
            pts = curve_draw.get("points", [])
            tunnels = curve_draw.get("in_tunnel", [])
            for i in range(0, len(pts) - 8, 8):
                p1 = _pt(pts[i][0], pts[i][1])
                p2 = _pt(pts[i + 8][0], pts[i + 8][1])
                if tunnels and tunnels[i]:
                    pen = QtGui.QPen(QtGui.QColor(220, 65, 65), 2.0, QtCore.Qt.PenStyle.DashLine)
                else:
                    pen = QtGui.QPen(QtGui.QColor(150, 150, 150), 1.0, QtCore.Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

        # 2. Active blast exclusion zones (wireframe ring only)
        for blast in self._state.get("blasts", []):
            bx, by = blast["x"], blast["y"]
            br = blast["radius_px"]
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 165, 40), 2.0, QtCore.Qt.PenStyle.DashDotLine))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(_pt(bx, by), br * sx_scale, br * sy_scale)

        # 3. Bonus coins
        for coin in self._state.get("coins", []):
            cp = _pt(coin["x"], coin["y"])
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 215, 0), 2.5))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(cp, 15.0 * sx_scale, 15.0 * sy_scale)

        # 4. Tracked balls (r=16.5 sits outside the r in [4, 14] vision ring)
        painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold))
        for b in self._state.get("balls", []):
            hex_c = config.COLOR_HEX.get(b["color"], "#CCCCCC")
            qcol = QtGui.QColor(hex_c)
            if b.get("in_tunnel_memory"):
                pen = QtGui.QPen(qcol, 2.0, QtCore.Qt.PenStyle.DotLine)
            else:
                pen = QtGui.QPen(qcol, 2.0)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            center = _pt(b["x"], b["y"])
            r_px = 16.5 * sx_scale
            painter.drawEllipse(center, r_px, r_px)

            pu = b.get("powerup")
            if pu:
                badge = config.POWERUP_BADGES.get(pu, "?")
                painter.setPen(QtGui.QColor(255, 255, 255))
                painter.drawText(
                    QtCore.QRectF(center.x() - 10, center.y() - 10, 20, 20),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    badge,
                )

        # 5. Frog current + swap color badges at frog center (away from mouth/dot r=24..25 ring)
        frog_xy = self._state.get("frog_xy")
        cur_col = self._state.get("cur_color")
        next_col = self._state.get("next_color")
        if frog_xy:
            fp = _pt(frog_xy[0], frog_xy[1])
            if cur_col in config.COLOR_HEX:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.5))
                painter.setBrush(QtGui.QColor(config.COLOR_HEX[cur_col]))
                painter.drawEllipse(fp, 5, 5)
            if next_col in config.COLOR_HEX:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.0))
                painter.setBrush(QtGui.QColor(config.COLOR_HEX[next_col]))
                painter.drawEllipse(QtCore.QPointF(fp.x() + 10, fp.y()), 3, 3)

        # 6. Aiming laser
        aim_xy = self._state.get("aim_xy")
        action = self._state.get("action", "none")
        blocked = self._state.get("blocked", False)
        blocker_xy = self._state.get("blocker_xy")

        if frog_xy and aim_xy:
            p_from = _pt(frog_xy[0], frog_xy[1])
            p_to = _pt(aim_xy[0], aim_xy[1])
            if blocked:
                pen = QtGui.QPen(QtGui.QColor(255, 65, 65), 2.0, QtCore.Qt.PenStyle.DashLine)
            elif action == "swap_shoot":
                pen = QtGui.QPen(QtGui.QColor(255, 220, 40), 2.0, QtCore.Qt.PenStyle.SolidLine)
            elif self._state.get("is_coin") or self._state.get("is_junk_dump"):
                pen = QtGui.QPen(QtGui.QColor(50, 225, 255), 2.0, QtCore.Qt.PenStyle.SolidLine)
            else:
                pen = QtGui.QPen(QtGui.QColor(60, 255, 120), 2.0, QtCore.Qt.PenStyle.SolidLine)

            painter.setPen(pen)
            painter.drawLine(p_from, p_to)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(p_to, 11.0 * sx_scale, 11.0 * sy_scale)

            if blocked and blocker_xy:
                bp = _pt(blocker_xy[0], blocker_xy[1])
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 3.0))
                painter.drawLine(QtCore.QPointF(bp.x() - 8, bp.y() - 8), QtCore.QPointF(bp.x() + 8, bp.y() + 8))
                painter.drawLine(QtCore.QPointF(bp.x() - 8, bp.y() + 8), QtCore.QPointF(bp.x() + 8, bp.y() - 8))

        # Note: Status text is shown in the Control Panel rather than a pill box over the board,
        # so the overlay never occludes top-left curve loops (e.g., loopy / underover) in BitBlt captures.
