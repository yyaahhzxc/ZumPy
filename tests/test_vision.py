import os
import cv2
import numpy as np
from PIL import Image

import config
from level_data import load_game_assets
from vision import (
    classify_ball_color,
    classify_powerup,
    detect_coins,
    detect_frog_balls,
    detect_map,
    is_pause_or_dialog_open,
    scan_curve_balls,
)


def test_classify_all_real_spinning_ball_frames():
    img_dir = config.DEFAULT_ZUMA_DIR / "images"
    color_files = {
        "blue": "baBallBlue.gif",
        "green": "baBallGreen.gif",
        "purple": "baBallPurple.gif",
        "red": "baBallRed.gif",
        "white": "baBallWhite.gif",
        "yellow": "baBallYellow.gif",
    }

    total = 0
    correct = 0
    for expected_col, fname in color_files.items():
        sheet = np.array(Image.open(img_dir / fname).convert("RGB"))
        w = sheet.shape[1]
        n_frames = sheet.shape[0] // w
        for fi in range(n_frames):
            rgb = sheet[fi * w:(fi + 1) * w, :w]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pred, conf = classify_ball_color(bgr, is_dot=False)
            total += 1
            if pred == expected_col and conf >= 0.25:
                correct += 1

    assert total >= 290
    assert correct / total >= 0.99, f"Expected >=99% accuracy across {total} frames, got {correct}/{total}"


def test_classify_all_real_swap_dots():
    img_dir = config.DEFAULT_ZUMA_DIR / "images"
    dotz = np.array(Image.open(img_dir / "baDotz.gif").convert("RGB"))
    expected = ["blue", "yellow", "red", "green", "purple", "white"]
    for i, exp in enumerate(expected):
        tile_rgb = dotz[:, i * 15:(i + 1) * 15]
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        pred, _ = classify_ball_color(tile_bgr, is_dot=True, strict_white=True)
        assert pred == exp, f"Dot {i} expected {exp}, got {pred}"

    # Gray frog stone patch or empty pink/olive reloading tongue must NOT trigger a frog ball under strict_white=True
    gray_stone = np.full((32, 32, 3), 145, dtype=np.uint8)
    pred_stone, _ = classify_ball_color(gray_stone, is_dot=False, strict_white=True)
    assert pred_stone is None

    empty_tongue = np.full((26, 26, 3), (120, 130, 145), dtype=np.uint8)
    empty_tongue[8:18, 8:18] = (110, 115, 165)  # faint pink tongue center (low S ~ 85, low color_ratio)
    pred_tongue, _ = classify_ball_color(empty_tongue, is_dot=False, strict_white=True)
    assert pred_tongue is None

    # Real White ball sprites tucked 6px onto frog-jaw background must still read as "white"
    white_sheet = cv2.cvtColor(np.array(Image.open(img_dir / "baBallWhite.gif").convert("RGB")), cv2.COLOR_RGB2BGR)
    alpha = np.array(Image.open(img_dir / "_ballalpha.gif").convert("L")).astype(np.float32)[:, :, None] / 255.0
    for fi in range(white_sheet.shape[0] // 32):
        jaw = np.full((32, 32, 3), (115, 130, 120), dtype=np.float32)
        spr = white_sheet[fi * 32:(fi + 1) * 32, :32].astype(np.float32)
        comp = (spr * alpha + jaw * (1.0 - alpha)).astype(np.uint8)
        patch26 = np.full((26, 26, 3), (115, 130, 120), dtype=np.uint8)
        patch26[:, :23] = comp[3:29, 9:32]
        pred_w, _ = classify_ball_color(patch26, is_dot=False, strict_white=True)
        assert pred_w == "white", f"White frame {fi} shifted 6px failed under strict_white: got {pred_w}"


def test_classify_all_24_powerups_and_zero_false_positives():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    img_dir = config.DEFAULT_ZUMA_DIR / "images"
    cap_color = {"blue": "Blue", "green": "Green", "purple": "Purple", "red": "Red", "white": "White", "yellow": "Yellow"}
    cap_ptype = {"bomb": "Bomb", "backwards": "Backwards", "slow": "Slow", "accuracy": "Accuracy"}

    for col, cname in cap_color.items():
        for pt, pname in cap_ptype.items():
            rgb = np.array(Image.open(img_dir / f"ba{pname}{cname}.gif").convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for ang in (0, 45, 90, 135, 225):
                if ang == 0:
                    test_bgr = bgr
                else:
                    M = cv2.getRotationMatrix2D((15.5, 15.5), ang, 1.0)
                    test_bgr = cv2.warpAffine(bgr, M, (32, 32), borderMode=cv2.BORDER_REPLICATE)
                pred_pt, score = classify_powerup(test_bgr, col, assets.powerup_templates)
                assert pred_pt == pt, f"Failed {pname}{cname}@{ang}: got {pred_pt} ({score:.3f})"

        sheet = np.array(Image.open(img_dir / f"baBall{cname}.gif").convert("RGB"))
        n_frames = sheet.shape[0] // 32
        for fi in range(n_frames):
            bgr = cv2.cvtColor(sheet[fi * 32:(fi + 1) * 32, :32], cv2.COLOR_RGB2BGR)
            pred_pt, score = classify_powerup(bgr, col, assets.powerup_templates)
            assert pred_pt is None, f"False positive powerup {pred_pt} ({score:.3f}) on {cname} frame {fi}"


def test_detect_map_with_cluttered_balls_and_space_and_menu():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    for map_id in ("spiral", "underover", "blackswirley"):
        lvl = assets.levels[map_id]
        frame = lvl.bg_bgr.copy()
        curve = lvl.curves[0]
        # Paste 30% cluttered colored circles along the curve
        for i in range(100, min(len(curve.points), 1300), 34):
            cx, cy = int(curve.points[i, 0]), int(curve.points[i, 1])
            cv2.circle(frame, (cx, cy), 15, (0, 0, 220), -1)
        det_id, mae = detect_map(frame, assets)
        assert det_id == map_id, f"Expected {map_id}, got {det_id} (mae={mae:.2f})"

    # Dark space frame (with visible Zuma top HUD bar + frog + stars/balls)
    space_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    space_frame[:config.TOP_HUD_HEIGHT, :] = (70, 90, 100)
    space_frame[config.TOP_HUD_HEIGHT::25, ::30] = (200, 200, 220)  # stars
    cv2.circle(space_frame, (246, 188), 42, (90, 140, 90), -1)  # frog
    for bx in range(120, 320, 32):
        cv2.circle(space_frame, (bx, 300), 15, (40, 200, 40), -1)
    det_space, _ = detect_map(space_frame, assets)
    assert det_space == "space"

    # Cluttered 70-ball space frame (p90 > 75 due to many bright balls) MUST still match space
    cluttered_space = space_frame.copy()
    scurve = assets.levels["space"].curves[0]
    for i in range(0, min(len(scurve.points), 2240), 32):
        cx, cy = int(scurve.points[i, 0]), int(scurve.points[i, 1])
        cv2.circle(cluttered_space, (cx, cy), 15, (200, 180, 40), -1)
    assert detect_map(cluttered_space, assets)[0] == "space"

    # Dim moving purple aurora/nebula in space (even with white "+130" floating text) must NOT trigger phantom purple balls
    aurora_space = space_frame.copy()
    pt_near_skull = scurve.points[int(len(scurve.points) * 0.90)]
    ax, ay = int(pt_near_skull[0]), int(pt_near_skull[1])
    cv2.circle(aurora_space, (ax, ay), 36, (95, 45, 80), -1)
    cv2.putText(aurora_space, "SLOWDOWN", (ax - 30, ay + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    aurora_balls = scan_curve_balls(aurora_space, assets.levels["space"], scurve, 0, assets)
    assert not any(b.color == "purple" for b in aurora_balls)

    # Pure black fade or dark IDE window without Zuma top HUD bar MUST NOT match space
    pure_black = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detect_map(pure_black, assets)[0] is None

    dark_ide = np.full((480, 640, 3), 24, dtype=np.uint8)
    dark_ide[60:440:30, 40:600:40] = (140, 140, 140)
    assert detect_map(dark_ide, assets)[0] is None

    # Bright uniform menu screen should reject
    menu_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    det_menu, _ = detect_map(menu_frame, assets)
    assert det_menu is None


def test_scan_curve_coins_frog_and_pause_dialog():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    img_dir = config.DEFAULT_ZUMA_DIR / "images"
    lvl = assets.levels["spiral"]
    frame = lvl.bg_bgr.copy()

    # Alpha mask for clean ball pasting
    alpha = np.array(Image.open(img_dir / "_ballalpha.gif").convert("L")) > 128

    def paste_ball(bgr_patch, cx, cy):
        x0, y0 = int(cx) - 16, int(cy) - 16
        roi = frame[y0:y0 + 32, x0:x0 + 32]
        roi[alpha] = bgr_patch[alpha]

    # Paste 3 real balls spaced 34px apart along spiral curve
    curve = lvl.curves[0]
    s_targets = [400.0, 434.0, 468.0]
    paste_cols = ["Red", "Blue", "White"]
    for s_val, cname in zip(s_targets, paste_cols):
        px, py = curve.point_at_arc_len(s_val)
        sheet = np.array(Image.open(img_dir / f"baBall{cname}.gif").convert("RGB"))
        bgr = cv2.cvtColor(sheet[:32, :32], cv2.COLOR_RGB2BGR)
        paste_ball(bgr, px, py)

    dets = scan_curve_balls(frame, lvl, curve, 0, assets)
    assert len(dets) == 3
    assert [d.color for d in dets] == ["red", "blue", "white"]

    # Paste gold coin at first TreasurePoint
    tx, ty = lvl.treasure_points[0]
    cv2.circle(frame, (int(tx), int(ty)), 10, (0, 215, 255), -1)
    coins = detect_coins(frame, lvl)
    assert len(coins) >= 1

    # Paste frog mouth (green) + swap dot (yellow) along cursor vector (aim right)
    gx, gy = lvl.frog_xy
    green_bgr = cv2.cvtColor(np.array(Image.open(img_dir / "baBallGreen.gif").convert("RGB"))[:32, :32], cv2.COLOR_RGB2BGR)
    paste_ball(green_bgr, gx + 25.0, gy)
    dotz = np.array(Image.open(img_dir / "baDotz.gif").convert("RGB"))
    yel_dot = cv2.cvtColor(dotz[:, 15:30], cv2.COLOR_RGB2BGR)
    dx0, dy0 = int(round(gx - 22.0)) - 7, int(round(gy)) - 7
    frame[dy0:dy0 + 15, dx0:dx0 + 15] = yel_dot
    frog = detect_frog_balls(frame, lvl.frog_xy, cursor_logical=(gx + 100.0, gy))
    assert frog.cur_color == "green"
    assert frog.next_color == "yellow"

    # Normal gameplay should not falsely trigger pause detector
    assert is_pause_or_dialog_open(frame, lvl) is False

    # Dimmed background + centered dialogbox should trigger pause detector
    paused_frame = (frame * 0.45).astype(np.uint8)
    cv2.rectangle(paused_frame, (150, 110), (490, 370), (65, 105, 145), -1)
    assert is_pause_or_dialog_open(paused_frame, lvl) is True

    # Green STATS level-complete panel (even without dark corners) should also trigger pause detector
    stats_frame = lvl.bg_bgr.copy()
    cv2.rectangle(stats_frame, (100, 90), (540, 420), (45, 95, 45), -1)
    assert is_pause_or_dialog_open(stats_frame, lvl) is True


def test_phantom_cyan_beam_and_sparkle_text_rejected():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    img_dir = config.DEFAULT_ZUMA_DIR / "images"
    lvl = assets.levels["snakepit"]
    frame = lvl.bg_bgr.copy()

    # Alpha-composite one real Red ball along curve 0
    alpha = np.array(Image.open(img_dir / "_ballalpha.gif").convert("L")).astype(np.float32)[:, :, None] / 255.0
    red_spr = cv2.cvtColor(np.array(Image.open(img_dir / "baBallRed.gif").convert("RGB"))[:32, :32], cv2.COLOR_RGB2BGR).astype(np.float32)
    rx, ry = lvl.curves[0].point_at_arc_len(600.0)
    ix, iy = int(round(rx)) - 16, int(round(ry)) - 16
    roi = frame[iy:iy + 32, ix:ix + 32].astype(np.float32)
    frame[iy:iy + 32, ix:ix + 32] = (red_spr * alpha + roi * (1.0 - alpha)).astype(np.uint8)

    # Blend a translucent cyan Accuracy sight beam across an inner curve segment
    bx, by = lvl.curves[0].point_at_arc_len(lvl.curves[0].total_length * 0.90)
    overlay = frame.copy()
    cv2.line(overlay, (int(lvl.frog_xy[0]), int(lvl.frog_xy[1])), (int(bx) + 30, int(by) + 80), (245, 225, 45), 9)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    dets = []
    for ci, c in enumerate(lvl.curves):
        dets.extend(scan_curve_balls(frame, lvl, c, ci, assets))
    assert len(dets) == 1
    assert dets[0].color == "red"

    # Also verify live miss captures if present on disk
    debug_root = config.DEFAULT_DEBUG_DIR
    live_cases = [
        ("miss_20261122_094651_389", "snakepit", (429.0, 172.0), "blue", 2),
        ("miss_20261122_094631_616", "snakepit", (325.0, 356.0), "blue", 23),
        ("miss_20261122_094347_030", "loopy", (488.0, 158.0), "red", 22),
        ("miss_20261122_113046_581", "longrange", (553.0, 75.0), "blue", 11),
    ]
    for folder, mid, phantom_xy, phantom_col, min_real in live_cases:
        img_path = debug_root / folder / "pre_shot.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        mlvl = assets.levels[mid]
        mdets = []
        for ci, c in enumerate(mlvl.curves):
            mdets.extend(scan_curve_balls(img, mlvl, c, ci, assets))
        assert len(mdets) >= min_real, f"{folder}: expected >= {min_real} real balls, got {len(mdets)}"
        near_phantom = [
            b for b in mdets
            if b.color == phantom_col and ((b.x - phantom_xy[0]) ** 2 + (b.y - phantom_xy[1]) ** 2) < 20.0 ** 2
        ]
        assert not near_phantom, f"{folder}: phantom {phantom_col} ball still detected near {phantom_xy}"

    # Verify adjacent White ball squeezed next to bright neighbors is NOT suppressed by 1D NMS
    tilt_path = debug_root / "miss_20261122_114214_635" / "pre_shot.png"
    if tilt_path.exists():
        timg = cv2.imread(str(tilt_path))
        tlvl = assets.levels["tiltspiral"]
        tdets = []
        for ci, c in enumerate(tlvl.curves):
            tdets.extend(scan_curve_balls(timg, tlvl, c, ci, assets))
        whites = [b for b in tdets if b.color == "white"]
        assert len(whites) >= 4
        assert any(((b.x - 460.0) ** 2 + (b.y - 72.0) ** 2) < 15.0 ** 2 for b in whites)
