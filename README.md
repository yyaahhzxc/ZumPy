# ZumPy

An external Win32 + OpenCV computer-vision solver and transparent HUD overlay for **Zuma Deluxe** on Windows.

ZumPy reads Zuma Deluxe's own curve geometry (`levels.xml` + `.dat` paths) in read-only mode, captures the game window in real time, tracks every sphere and coin on the board, and renders a click-through overlay with live telemetry and optional auto-aim/auto-shoot assistance.

## Features

- **Real-time Level & Curve Detection**: Automatically identifies the active map (including twin-curve maps) from `levels.xml` and `.dat` path assets without modifying game files.
- **Vision & Tracking (`src/vision.py`, `src/tracker.py`)**: Detects spheres across all 6 colors, powerup badges (`B` bomb, `R` reverse, `S` slow, `A` accuracy), frog current/next balls, and bonus coins with velocity-smoothed motion tracking and tunnel/blast awareness.
- **Strategy Solver (`src/solver.py`)**:
  - Evaluates pops (`3+` matches and cascade chains), powerups, and endgame cleanup.
  - Switches dynamically between `CONTROL` and `DEFENSE` modes based on skull proximity.
  - Supports `swap_shoot`, `setup_pair`, `grab_coin`, and safe offscreen junk dumps with moving-blocker line-of-sight checks.
- **Transparent HUD Overlay & Control Panel (`src/overlay.py`, `src/ui_control_panel.py`)**:
  - Click-through `WS_EX_TRANSPARENT` overlay showing ball detections, aim line, and predicted target point.
  - Live telemetry including **Mode**, **Skull Progress**, **Avg Latest to Skull**, **Last Ball at Clear**, and **Train Tail (Other End)**.
- **Diagnostics & Auto-Miss Logger (`src/diagnostics.py`)**: Captures pre/aim/post snapshots and `miss_report.json` (`intended_vs_actual`) to `debug/` whenever a shot misfires.
- **Safety Controls**: Global emergency stop hotkeys (`F12`, `Ctrl+Shift+0`), manual mouse-override pause, and `F9` manual debug snapshot.

## Requirements

- Windows 10/11
- Python 3.10+
- *Zuma Deluxe* installed at `C:\Games\Zuma Deluxe` (configurable via `DEFAULT_ZUMA_DIR` in `src/config.py`)

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Running ZumPy

1. Launch **Zuma Deluxe** (windowed mode recommended).
2. Start ZumPy from the project root:

```powershell
python src/main.py
```

3. Use the Control Panel to toggle **HUD Overlay**, **Auto Play**, **Grab Coins**, **Dump Junk**, and **Prioritize Powerups**.

### Hotkeys

| Hotkey | Action |
| --- | --- |
| `F12` or `Ctrl+Shift+0` | Emergency stop (disables Auto Play immediately) |
| `F9` | Save a manual debug snapshot to `debug/` |

## Project Structure

```text
src/
  config.py             # Tunable thresholds, colors, timings, and hotkeys
  diagnostics.py        # Rolling frame buffer, F9 dumps, and auto-miss verification
  input_controller.py   # Win32 SendInput aim/click/swap actuation and cooldowns
  level_data.py         # Read-only parser for Zuma Deluxe levels.xml and .dat curves
  main.py               # Application entry point and global hotkey bridge
  overlay.py            # Transparent PyQt6 click-through HUD overlay
  solver.py             # Line-of-sight raycasting and shot-selection solver
  tracker.py            # Frame-to-frame ball ID, velocity, and blast-zone tracking
  ui_control_panel.py   # Control panel UI and live telemetry readout
  utils.py              # Geometry and color helpers
  vision.py             # Map matching, curve ball scanning, frog & coin detection
  window_capture.py     # Win32 window capture (mss / PrintWindow)
  worker.py             # Main vision -> tracker -> solver -> input worker thread
tests/                  # Unit tests for level data, vision, tracker, solver, diagnostics, and worker
```

## Running Tests

```powershell
python -m pytest tests/ -q
```
