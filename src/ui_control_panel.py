from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

import config
from level_data import load_game_assets
from overlay import HudOverlay
from window_capture import ZumaWindowCapture
from worker import ZumaWorker

DARK_STYLESHEET = """
QWidget {
    background-color: #161922;
    color: #E6EAF2;
    font-family: "Segoe UI", sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2A3040;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #9FB0CC;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #262C3D;
    border: 1px solid #38425B;
    border-radius: 5px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #313950;
}
QPushButton:disabled {
    background-color: #1D212D;
    color: #636D83;
}
QCheckBox {
    spacing: 6px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #272E3F;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    background: #4C8DFF;
}
"""


class ControlPanel(QtWidgets.QWidget):
    """Dark-theme PyZuma control panel & telemetry dashboard."""

    def __init__(self, worker: ZumaWorker, overlay: HudOverlay, capture: ZumaWindowCapture):
        super().__init__()
        self.worker = worker
        self.overlay = overlay
        self.capture = capture

        self.setWindowTitle("PyZuma — Zuma Deluxe Assistant")
        self.setMinimumWidth(360)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header / Status
        self.status_label = QtWidgets.QLabel("Searching for Zuma Deluxe...")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._set_status_style(False)
        root.addWidget(self.status_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_find = QtWidgets.QPushButton("Find Zuma")
        self.btn_focus = QtWidgets.QPushButton("Focus Game")
        self.btn_folder = QtWidgets.QPushButton("Game Folder")
        btn_row.addWidget(self.btn_find)
        btn_row.addWidget(self.btn_focus)
        btn_row.addWidget(self.btn_folder)
        root.addLayout(btn_row)

        # Live Telemetry
        telem_group = QtWidgets.QGroupBox("Live Telemetry")
        form = QtWidgets.QFormLayout(telem_group)
        form.setSpacing(6)

        self.lbl_map = QtWidgets.QLabel("—")
        self.lbl_mode = QtWidgets.QLabel("PAUSED")
        self.lbl_skull_pct = QtWidgets.QLabel("0%")
        self.lbl_skull_pct.setStyleSheet("color: #7CFFA8; font-weight: bold;")
        self.lbl_skull_avg = QtWidgets.QLabel("0%")
        self.lbl_skull_avg.setStyleSheet("color: #7CFFA8; font-weight: bold;")
        self.lbl_skull_last = QtWidgets.QLabel("—")
        self.lbl_skull_last.setStyleSheet("color: #D9E2EC; font-weight: bold;")
        self.lbl_tail_pct = QtWidgets.QLabel("—")
        self.lbl_tail_pct.setStyleSheet("color: #D9E2EC; font-weight: bold;")
        self.lbl_balls = QtWidgets.QLabel("0")
        self.lbl_perf = QtWidgets.QLabel("0 FPS / 0.0 ms")
        self.lbl_reason = QtWidgets.QLabel("idle")
        self.lbl_reason.setWordWrap(True)
        self.lbl_misses = QtWidgets.QLabel("0")

        frog_row = QtWidgets.QHBoxLayout()
        self.swatch_cur = QtWidgets.QLabel("?")
        self.swatch_cur.setFixedSize(56, 22)
        self.swatch_cur.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.swatch_next = QtWidgets.QLabel("?")
        self.swatch_next.setFixedSize(56, 22)
        self.swatch_next.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        frog_row.addWidget(QtWidgets.QLabel("Cur:"))
        frog_row.addWidget(self.swatch_cur)
        frog_row.addSpacing(8)
        frog_row.addWidget(QtWidgets.QLabel("Swap:"))
        frog_row.addWidget(self.swatch_next)
        frog_row.addStretch()

        form.addRow("Map:", self.lbl_map)
        form.addRow("Mode:", self.lbl_mode)
        form.addRow("Latest Ball to Skull:", self.lbl_skull_pct)
        form.addRow("Avg Latest to Skull:", self.lbl_skull_avg)
        form.addRow("Last Ball at Clear:", self.lbl_skull_last)
        form.addRow("Train Tail (Other End):", self.lbl_tail_pct)
        form.addRow("Frog:", frog_row)
        form.addRow("Balls:", self.lbl_balls)
        form.addRow("Speed:", self.lbl_perf)
        form.addRow("Action:", self.lbl_reason)
        form.addRow("Auto-Misses:", self.lbl_misses)
        root.addWidget(telem_group)

        # Strategy & Safety Controls
        ctrl_group = QtWidgets.QGroupBox("Controls & Safety")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(8)

        self.btn_auto = QtWidgets.QPushButton("Enable Auto-Play (F12 to Stop)")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(False)
        self._style_auto_button(False)
        ctrl_layout.addWidget(self.btn_auto)

        self.chk_overlay = QtWidgets.QCheckBox("Show HUD Overlay")
        self.chk_overlay.setChecked(True)
        self.chk_swap = QtWidgets.QCheckBox("Allow Right-Click Ball Swapping")
        self.chk_swap.setChecked(True)
        self.chk_powerups = QtWidgets.QCheckBox("Prioritize Power-Ups (Bomb/Back/Slow/Acc)")
        self.chk_powerups.setChecked(True)
        self.chk_coins = QtWidgets.QCheckBox("Grab Safe Coins")
        self.chk_coins.setChecked(True)

        for chk in (self.chk_overlay, self.chk_swap, self.chk_powerups, self.chk_coins):
            ctrl_layout.addWidget(chk)

        # Sliders
        self.lbl_lead_val = QtWidgets.QLabel("Lead-Aim Multiplier: 1.00x")
        self.slider_lead = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_lead.setRange(50, 160)
        self.slider_lead.setValue(100)

        self.lbl_cd_val = QtWidgets.QLabel(f"Min Cooldown (Dynamic): {config.SHOOT_COOLDOWN_SEC:.2f}s")
        self.slider_cd = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_cd.setRange(12, 50)
        self.slider_cd.setValue(int(round(config.SHOOT_COOLDOWN_SEC * 100)))

        ctrl_layout.addWidget(self.lbl_lead_val)
        ctrl_layout.addWidget(self.slider_lead)
        ctrl_layout.addWidget(self.lbl_cd_val)
        ctrl_layout.addWidget(self.slider_cd)

        self.btn_dump = QtWidgets.QPushButton("Snapshot Debug (F9)")
        ctrl_layout.addWidget(self.btn_dump)
        root.addWidget(ctrl_group)

        self.lbl_notice = QtWidgets.QLabel("Overlay-first safe mode ready. Run Zuma in Windowed mode.")
        self.lbl_notice.setStyleSheet("color: #8D99AE; font-size: 11px;")
        self.lbl_notice.setWordWrap(True)
        root.addWidget(self.lbl_notice)

    def _set_status_style(self, hooked: bool) -> None:
        if hooked:
            self.status_label.setText("Zuma Deluxe Hooked")
            self.status_label.setStyleSheet(
                "background-color: #1B3B2B; color: #7CFFA8; padding: 6px; border-radius: 5px; font-weight: bold;"
            )
        else:
            self.status_label.setText("Zuma Window Not Found")
            self.status_label.setStyleSheet(
                "background-color: #3D1E24; color: #FF8C94; padding: 6px; border-radius: 5px; font-weight: bold;"
            )

    def _style_auto_button(self, active: bool) -> None:
        if active:
            self.btn_auto.setText("AUTO-PLAY ACTIVE — Press F12 to Stop")
            self.btn_auto.setStyleSheet(
                "background-color: #1F6F43; border: 1px solid #36C978; color: #FFFFFF; padding: 8px; font-weight: bold;"
            )
        else:
            self.btn_auto.setText("Enable Auto-Play (F12 to Stop)")
            self.btn_auto.setStyleSheet(
                "background-color: #2B3347; border: 1px solid #4A5878; color: #E6EAF2; padding: 8px; font-weight: bold;"
            )

    def _set_swatch(self, lbl: QtWidgets.QLabel, color_name: Optional[str]) -> None:
        if color_name in config.COLOR_HEX:
            bg = config.COLOR_HEX[color_name]
            fg = "#111111" if color_name in ("yellow", "white", "green") else "#FFFFFF"
            lbl.setText(color_name.upper())
            lbl.setStyleSheet(
                f"background-color: {bg}; color: {fg}; border-radius: 4px; font-weight: bold; font-size: 11px;"
            )
        else:
            lbl.setText("?")
            lbl.setStyleSheet("background-color: #222736; color: #7C879E; border-radius: 4px;")

    def _connect_signals(self) -> None:
        self.btn_find.clicked.connect(self._on_find_clicked)
        self.btn_focus.clicked.connect(self.capture.focus)
        self.btn_folder.clicked.connect(self._on_folder_clicked)

        self.btn_auto.toggled.connect(self._on_auto_toggled)
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        self.chk_swap.toggled.connect(lambda v: self.worker.update_config(allow_swap=v))
        self.chk_powerups.toggled.connect(lambda v: self.worker.update_config(prioritize_powerups=v))
        self.chk_coins.toggled.connect(lambda v: self.worker.update_config(grab_coins=v))

        self.slider_lead.valueChanged.connect(self._on_lead_changed)
        self.slider_cd.valueChanged.connect(self._on_cd_changed)
        self.btn_dump.clicked.connect(self.worker.request_manual_dump)

        self.worker.hook_changed.connect(self._on_hook_changed)
        self.worker.state_updated.connect(self._on_state_updated)
        self.worker.auto_paused_signal.connect(self._on_auto_paused)
        self.worker.dump_created_signal.connect(self._on_notice)

    @QtCore.pyqtSlot()
    def _on_find_clicked(self) -> None:
        found = self.capture.find_window()
        self._set_status_style(found)
        if not found:
            self.lbl_notice.setText("Launch Zuma Deluxe in Windowed mode, then click Find Zuma.")

    @QtCore.pyqtSlot()
    def _on_folder_clicked(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Zuma Deluxe Folder", str(config.DEFAULT_ZUMA_DIR))
        if not chosen:
            return
        try:
            assets = load_game_assets(Path(chosen))
            self.worker.assets = assets
            self.worker.vision.assets = assets
            self.worker.vision.reset()
            self.lbl_notice.setText(f"Loaded {len(assets.levels)} maps from {chosen}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Zuma Folder", f"Could not load levels.xml:\n{exc}")

    @QtCore.pyqtSlot(bool)
    def _on_auto_toggled(self, checked: bool) -> None:
        self._style_auto_button(checked)
        self.worker.update_config(auto_play=checked)
        if checked:
            self.lbl_notice.setText("Auto-Play ON. Focus Zuma window to run; F12 or moving mouse stops it.")
        else:
            self.lbl_notice.setText("Auto-Play OFF (HUD overlay mode).")

    @QtCore.pyqtSlot(bool)
    def _on_overlay_toggled(self, checked: bool) -> None:
        self.overlay.set_enabled(checked)
        self.worker.update_config(show_overlay=checked)

    @QtCore.pyqtSlot(int)
    def _on_lead_changed(self, val: int) -> None:
        mult = val / 100.0
        self.lbl_lead_val.setText(f"Lead-Aim Multiplier: {mult:.2f}x")
        self.worker.update_config(lead_aim_multiplier=mult)

    @QtCore.pyqtSlot(int)
    def _on_cd_changed(self, val: int) -> None:
        cd = val / 100.0
        self.lbl_cd_val.setText(f"Min Cooldown (Dynamic): {cd:.2f}s")
        self.worker.update_config(shot_cooldown_sec=cd)

    @QtCore.pyqtSlot(bool)
    def _on_hook_changed(self, hooked: bool) -> None:
        self._set_status_style(hooked)

    @QtCore.pyqtSlot(object)
    def _on_state_updated(self, payload: Optional[Dict[str, Any]]) -> None:
        self.overlay.update_state(payload)
        if payload is None:
            return

        self.lbl_map.setText(payload.get("map_name") or "Menu / Transition")
        mode = "PAUSED" if payload.get("is_paused") else payload.get("mode", "CONTROL")
        self.lbl_mode.setText(mode)
        def _pct_style(val: int) -> str:
            if val >= int(round(config.DANGER_PROGRESS_THRESHOLD * 100)):
                return "color: #FF6B6B; font-weight: bold;"
            if val >= 50:
                return "color: #FFD166; font-weight: bold;"
            return "color: #7CFFA8; font-weight: bold;"

        pct = int(round(max(0.0, min(1.0, float(payload.get("max_progress", 0.0)))) * 100.0))
        self.lbl_skull_pct.setText(f"{pct}%")
        self.lbl_skull_pct.setStyleSheet(_pct_style(pct))

        avg_pct = int(round(max(0.0, min(1.0, float(payload.get("avg_progress", 0.0)))) * 100.0))
        self.lbl_skull_avg.setText(f"{avg_pct}%")
        self.lbl_skull_avg.setStyleSheet(_pct_style(avg_pct))

        last_clear = payload.get("last_clear_progress")
        if last_clear is None:
            self.lbl_skull_last.setText("—")
            self.lbl_skull_last.setStyleSheet("color: #D9E2EC; font-weight: bold;")
        else:
            last_pct = int(round(max(0.0, min(1.0, float(last_clear))) * 100.0))
            self.lbl_skull_last.setText(f"{last_pct}%")
            self.lbl_skull_last.setStyleSheet(_pct_style(last_pct))

        tail_list = payload.get("tail_progress") or []
        if not tail_list:
            self.lbl_tail_pct.setText("—")
            self.lbl_tail_pct.setStyleSheet("color: #D9E2EC; font-weight: bold;")
        else:
            parts = []
            any_active = False
            for tp in tail_list:
                if tp is None:
                    parts.append("—")
                else:
                    any_active = True
                    parts.append(f"{int(round(max(0.0, min(1.0, float(tp))) * 100.0))}%")
            self.lbl_tail_pct.setText(" | ".join(parts))
            self.lbl_tail_pct.setStyleSheet(
                "color: #7CFFA8; font-weight: bold;" if any_active else "color: #D9E2EC; font-weight: bold;"
            )
        self._set_swatch(self.swatch_cur, payload.get("cur_color"))
        self._set_swatch(self.swatch_next, payload.get("next_color"))
        self.lbl_balls.setText(str(len(payload.get("balls", []))))
        self.lbl_perf.setText(f"{payload.get('fps', 0.0):.0f} FPS / {payload.get('latency_ms', 0.0):.1f} ms")
        self.lbl_reason.setText(payload.get("reason", "idle"))
        self.lbl_misses.setText(str(payload.get("auto_miss_count", 0)))

    @QtCore.pyqtSlot(str)
    def _on_auto_paused(self, msg: str) -> None:
        if self.btn_auto.isChecked():
            self.btn_auto.setChecked(False)
        self.lbl_notice.setText(msg)

    @QtCore.pyqtSlot(str)
    def _on_notice(self, msg: str) -> None:
        self.lbl_notice.setText(msg)

    @QtCore.pyqtSlot()
    def emergency_stop(self) -> None:
        if self.btn_auto.isChecked():
            self.btn_auto.setChecked(False)
        else:
            self.worker.update_config(auto_play=False)
        self.lbl_notice.setText("Emergency stop triggered (F12).")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.worker.stop()
        self.worker.wait(1500)
        self.overlay.close()
        super().closeEvent(event)
