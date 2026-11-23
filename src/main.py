import sys

import keyboard
from PyQt6 import QtCore, QtWidgets

import config
from level_data import load_game_assets
from overlay import HudOverlay
from ui_control_panel import ControlPanel
from window_capture import ZumaWindowCapture
from worker import ZumaWorker


class HotkeyBridge(QtCore.QObject):
    """Safely marshals global keyboard hotkey callbacks onto the Qt main thread."""

    stop_triggered = QtCore.pyqtSignal()
    dump_triggered = QtCore.pyqtSignal()


def main() -> None:
    if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QtWidgets.QApplication(sys.argv)
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    capture = ZumaWindowCapture()
    cfg = config.RuntimeConfig()

    overlay = HudOverlay()
    worker = ZumaWorker(assets, capture, cfg)
    panel = ControlPanel(worker, overlay, capture)

    bridge = HotkeyBridge()
    bridge.stop_triggered.connect(panel.emergency_stop)
    bridge.dump_triggered.connect(worker.request_manual_dump)

    for hk in config.EMERGENCY_STOP_HOTKEYS:
        try:
            keyboard.add_hotkey(hk, bridge.stop_triggered.emit)
        except Exception:
            pass
    try:
        keyboard.add_hotkey(config.DEBUG_DUMP_HOTKEY, bridge.dump_triggered.emit)
    except Exception:
        pass

    panel.show()
    worker.start()
    exit_code = app.exec()

    try:
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
