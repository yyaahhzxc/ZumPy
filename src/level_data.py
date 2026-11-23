import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config
from utils import find_file_case_insensitive


@dataclass
class CutoutLayer:
    name: str
    x: int
    y: int
    priority: int
    mask: np.ndarray  # (h, w) bool where True = opaque foreground cover


@dataclass
class CurveData:
    name: str
    points: np.ndarray          # (N, 2) float32 in 640x480 logical space
    in_tunnel: np.ndarray       # (N,) bool (t1 != 0)
    draw_priority: np.ndarray   # (N,) bool (t2 != 0: True means ball draws OVER bridges)
    arc_len: np.ndarray         # (N,) float32 cumulative distance in px
    tangents: np.ndarray        # (N, 2) float32 unit tangent vectors
    visibility: np.ndarray      # (N,) float32 fraction [0..1] of ball ring visible
    total_length: float
    points_list: List[List[float]] = field(default_factory=list)
    in_tunnel_list: List[bool] = field(default_factory=list)

    def index_at_arc_len(self, s: float) -> int:
        if len(self.arc_len) == 0:
            return 0
        idx = int(np.searchsorted(self.arc_len, s, side="left"))
        return max(0, min(len(self.arc_len) - 1, idx))

    def point_at_arc_len(self, s: float) -> Tuple[float, float]:
        idx = self.index_at_arc_len(s)
        pt = self.points[idx]
        return float(pt[0]), float(pt[1])

    def is_tunnel_at(self, s: float) -> bool:
        idx = self.index_at_arc_len(s)
        return bool(self.in_tunnel[idx])

    def visibility_at(self, s: float) -> float:
        idx = self.index_at_arc_len(s)
        return float(self.visibility[idx])


@dataclass
class LevelInfo:
    id: str
    disp_name: str
    frog_xy: Tuple[float, float]
    curves: List[CurveData]
    treasure_points: List[Tuple[float, float]]
    cutouts: List[CutoutLayer]
    cutout_canvas: Optional[np.ndarray]  # (480, 640) bool union of bridge masks
    bg_bgr: Optional[np.ndarray]         # (480, 640, 3) uint8
    thumb_bgr: Optional[np.ndarray]      # (MAP_THUMB_H, MAP_THUMB_W, 3) uint8
    has_animated_bg: bool = False
    is_space: bool = False


@dataclass
class GameAssets:
    zuma_dir: Path
    levels: Dict[str, LevelInfo]
    ball_alpha_mask: np.ndarray                                # (32, 32) bool
    ball_ring_mask: np.ndarray                                 # (32, 32) bool (r in [4, 14])
    powerup_templates: Dict[str, Dict[str, List[np.ndarray]]]  # color -> ptype -> 24x (18, 18, 3) BGR
    dialog_ref_gray: Optional[np.ndarray] = None
    powerup_matrices: Dict[str, Tuple[np.ndarray, List[str]]] = field(default_factory=dict)


_ATTR_TOKEN_RE = re.compile(r'([A-Za-z0-9_:-]+)\s*=\s*("[^"]*"|\'[^\']*\'|[^\s/>]+)')


def sanitize_levels_xml(raw_text: str) -> str:
    """Clean PopCap's non-strict levels.xml (unquoted attrs and duplicate keys)."""
    cleaned = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)

    def _clean_tag(match: re.Match) -> str:
        full = match.group(0)
        if full.startswith("<?") or full.startswith("</"):
            return full
        self_closing = full.endswith("/>")
        inner = full[1:-2 if self_closing else -1].strip()
        parts = re.split(r"\s+", inner, maxsplit=1)
        tag_name = parts[0]
        if len(parts) == 1:
            return f"<{tag_name}/>" if self_closing else f"<{tag_name}>"

        attr_str = parts[1]
        attrs: Dict[str, str] = {}
        for k, v in _ATTR_TOKEN_RE.findall(attr_str):
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                val = v[1:-1]
            else:
                val = v
            attrs[k] = val

        rendered = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        end = " />" if self_closing else ">"
        return f"<{tag_name} {rendered}{end}"

    return re.sub(r"<[^>]+>", _clean_tag, cleaned, flags=re.DOTALL)


def _build_ball_ring_mask(size: int = 32, r_min: float = 4.0, r_max: float = 14.0) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    return (dist >= r_min) & (dist <= r_max)


def _compute_curve_visibility(
    points: np.ndarray,
    draw_priority: np.ndarray,
    cutout_canvas: Optional[np.ndarray],
    ball_ring_mask: np.ndarray,
) -> np.ndarray:
    n = len(points)
    visibility = np.ones(n, dtype=np.float32)
    if cutout_canvas is None or n == 0:
        return visibility

    h_img, w_img = cutout_canvas.shape
    ring_area = float(np.sum(ball_ring_mask))
    if ring_area <= 0.0:
        return visibility

    half = ball_ring_mask.shape[0] // 2
    for i in range(n):
        if draw_priority[i]:
            continue
        cx = int(round(float(points[i, 0])))
        cy = int(round(float(points[i, 1])))
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + 32, y0 + 32

        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(w_img, x1), min(h_img, y1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue

        mx0, my0 = ix0 - x0, iy0 - y0
        mx1, my1 = mx0 + (ix1 - ix0), my0 + (iy1 - iy0)
        covered = cutout_canvas[iy0:iy1, ix0:ix1] & ball_ring_mask[my0:my1, mx0:mx1]
        vis = 1.0 - float(np.sum(covered)) / ring_area
        visibility[i] = max(0.0, min(1.0, vis))

    return visibility


def parse_dat_curve(
    dat_path: Path,
    cutout_canvas: Optional[np.ndarray],
    ball_ring_mask: np.ndarray,
) -> CurveData:
    """Parse a Zuma Deluxe binary .dat curve file into 640x480 logical coordinates."""
    data = dat_path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Invalid .dat curve file (too short): {dat_path}")

    sec_size = struct.unpack_from("<I", data, 12)[0]
    off = 16 + sec_size
    count = struct.unpack_from("<I", data, off)[0]
    sx, sy = struct.unpack_from("<ff", data, off + 4)

    pts = np.empty((count, 2), dtype=np.float32)
    in_tunnel = np.zeros(count, dtype=bool)
    draw_priority = np.zeros(count, dtype=bool)

    x, y = float(sx), float(sy)
    pts[0] = (x, y)

    pos = off + 12
    for i in range(1, count):
        t1, t2, dx, dy = struct.unpack_from("<BBbb", data, pos)
        pos += 4
        x += dx / 100.0
        y += dy / 100.0
        pts[i] = (x, y)
        in_tunnel[i] = (t1 != 0)
        draw_priority[i] = (t2 != 0)

    if count > 1:
        in_tunnel[0] = in_tunnel[1]
        draw_priority[0] = draw_priority[1]

    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    arc_len = np.zeros(count, dtype=np.float32)
    if count > 1:
        arc_len[1:] = np.cumsum(seg_lens)

    tangents = np.zeros((count, 2), dtype=np.float32)
    if count > 1:
        norms = np.maximum(seg_lens[:, None], 1e-6)
        unit_diffs = diffs / norms
        tangents[:-1] = unit_diffs
        tangents[-1] = unit_diffs[-1]

    visibility = _compute_curve_visibility(pts, draw_priority, cutout_canvas, ball_ring_mask)
    total_length = float(arc_len[-1]) if count > 0 else 0.0

    return CurveData(
        name=dat_path.stem.lower(),
        points=pts,
        in_tunnel=in_tunnel,
        draw_priority=draw_priority,
        arc_len=arc_len,
        tangents=tangents,
        visibility=visibility,
        total_length=total_length,
        points_list=pts.tolist(),
        in_tunnel_list=in_tunnel.tolist(),
    )


def _load_cutouts(level_folder: Path, elem: ET.Element) -> Tuple[List[CutoutLayer], Optional[np.ndarray]]:
    cutouts: List[CutoutLayer] = []
    canvas: Optional[np.ndarray] = None

    for ce in elem.findall("Cutout"):
        img_name = ce.attrib.get("image", "")
        if not img_name:
            continue
        cx = int(float(ce.attrib.get("x", "0")))
        cy = int(float(ce.attrib.get("y", "0")))
        pri = int(float(ce.attrib.get("pri", "1")))

        gif_path = find_file_case_insensitive(
            level_folder, f"_{img_name}", (".gif", ".png")
        ) or find_file_case_insensitive(
            level_folder, img_name, (".gif", ".png")
        )
        if gif_path is None:
            continue

        mask_arr = np.array(Image.open(gif_path).convert("L")) > 128
        cutouts.append(CutoutLayer(name=img_name, x=cx, y=cy, priority=pri, mask=mask_arr))

        if canvas is None:
            canvas = np.zeros((config.LOGICAL_H, config.LOGICAL_W), dtype=bool)
        h_m, w_m = mask_arr.shape
        x0, y0 = max(0, cx), max(0, cy)
        x1, y1 = min(config.LOGICAL_W, cx + w_m), min(config.LOGICAL_H, cy + h_m)
        if x0 < x1 and y0 < y1:
            canvas[y0:y1, x0:x1] |= mask_arr[y0 - cy:y1 - cy, x0 - cx:x1 - cx]

    return cutouts, canvas


def _load_powerup_templates(images_dir: Path) -> Dict[str, Dict[str, List[np.ndarray]]]:
    color_cap = {
        "red": "Red",
        "yellow": "Yellow",
        "green": "Green",
        "blue": "Blue",
        "purple": "Purple",
        "white": "White",
    }
    ptype_cap = {
        "bomb": "Bomb",
        "backwards": "Backwards",
        "slow": "Slow",
        "accuracy": "Accuracy",
    }

    templates: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for color_key, c_name in color_cap.items():
        templates[color_key] = {}
        for p_key, p_name in ptype_cap.items():
            path = find_file_case_insensitive(images_dir, f"ba{p_name}{c_name}", (".gif", ".png"))
            if path is None:
                continue
            rgb = np.array(Image.open(path).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rot_list: List[np.ndarray] = []
            for ang in range(0, 360, 15):
                if ang == 0:
                    rot_list.append(bgr[7:25, 7:25].copy())
                else:
                    M = cv2.getRotationMatrix2D((15.5, 15.5), ang, 1.0)
                    rot_bgr = cv2.warpAffine(bgr, M, (32, 32), borderMode=cv2.BORDER_REPLICATE)
                    rot_list.append(rot_bgr[7:25, 7:25].copy())
            templates[color_key][p_key] = rot_list
    return templates


def build_powerup_matrices(
    powerup_templates: Dict[str, Dict[str, List[np.ndarray]]],
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Precompute per-color zero-mean unit-norm float32 template matrices (972, N) for BLAS scoring."""
    matrices: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for color_key, pmap in powerup_templates.items():
        rows: List[np.ndarray] = []
        labels: List[str] = []
        for p_key, tlist in pmap.items():
            for tpl in tlist:
                v = tpl.astype(np.float32)
                v = v - v.mean(axis=(0, 1), keepdims=True)
                flat = v.reshape(-1)
                norm = float(np.linalg.norm(flat))
                if norm > 1e-6:
                    flat = flat / norm
                rows.append(flat)
                labels.append(p_key)
        if rows:
            mat_T = np.ascontiguousarray(np.stack(rows, axis=0).T, dtype=np.float32)
            matrices[color_key] = (mat_T, labels)
    return matrices


def make_thumb_bgr(bg_bgr: np.ndarray) -> np.ndarray:
    """Create matching thumbnail from the playable region below the top menu bar."""
    body = bg_bgr[config.TOP_HUD_HEIGHT:, :]
    return cv2.resize(body, (config.MAP_THUMB_W, config.MAP_THUMB_H), interpolation=cv2.INTER_AREA)


def load_game_assets(zuma_dir: Path = config.DEFAULT_ZUMA_DIR) -> GameAssets:
    """Load levels.xml, all .dat curves, backgrounds, cutout masks, and sprites."""
    levels_dir = zuma_dir / "levels"
    images_dir = zuma_dir / "images"
    xml_path = levels_dir / "levels.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Could not find Zuma levels.xml at {xml_path}")

    raw_xml = xml_path.read_text(encoding="utf-8", errors="replace")
    root = ET.fromstring(sanitize_levels_xml(raw_xml))

    ball_alpha_path = find_file_case_insensitive(images_dir, "_ballalpha", (".gif", ".png"))
    if ball_alpha_path is not None:
        ball_alpha_mask = np.array(Image.open(ball_alpha_path).convert("L")) > 128
    else:
        yy, xx = np.ogrid[:32, :32]
        ball_alpha_mask = ((xx - 15.5) ** 2 + (yy - 15.5) ** 2) <= (15.5 ** 2)

    ball_ring_mask = _build_ball_ring_mask(32, 4.0, 14.0)
    powerup_templates = _load_powerup_templates(images_dir)

    dialog_ref_gray: Optional[np.ndarray] = None
    dialog_path = find_file_case_insensitive(images_dir, "dialogbox", (".jpg", ".png"))
    if dialog_path is not None:
        d_rgb = np.array(Image.open(dialog_path).convert("RGB"))
        dialog_ref_gray = cv2.cvtColor(d_rgb, cv2.COLOR_RGB2GRAY)

    levels: Dict[str, LevelInfo] = {}
    for ge in root.findall("Graphics"):
        lvl_id = ge.attrib.get("id", "").strip()
        if not lvl_id:
            continue

        disp_name = ge.attrib.get("dispname", lvl_id)
        gx = float(ge.attrib.get("gx", "320"))
        gy = float(ge.attrib.get("gy", "240"))
        is_space = ge.attrib.get("space", "false").lower() == "true"
        has_animated_bg = len(ge.findall("BackgroundAlpha")) > 0

        level_folder = levels_dir / lvl_id
        if not level_folder.is_dir():
            for child in levels_dir.iterdir():
                if child.is_dir() and child.name.lower() == lvl_id.lower():
                    level_folder = child
                    break

        cutouts, cutout_canvas = _load_cutouts(level_folder, ge)

        curves: List[CurveData] = []
        for attr in ("curve", "curve2", "curve3"):
            cname = ge.attrib.get(attr, "").strip()
            if not cname:
                continue
            dat_path = find_file_case_insensitive(level_folder, cname, (".dat",))
            if dat_path is not None:
                curves.append(parse_dat_curve(dat_path, cutout_canvas, ball_ring_mask))

        treasure_points: List[Tuple[float, float]] = []
        for tp in ge.findall("TreasurePoint"):
            tx = float(tp.attrib.get("x", "0"))
            ty = float(tp.attrib.get("y", "0"))
            treasure_points.append((tx, ty))

        bg_bgr: Optional[np.ndarray] = None
        thumb_bgr: Optional[np.ndarray] = None
        img_stem = ge.attrib.get("image", "").strip()
        if img_stem and not is_space:
            img_path = find_file_case_insensitive(level_folder, img_stem, (".jpg", ".png", ".jpeg"))
            if img_path is not None:
                bg_rgb = np.array(Image.open(img_path).convert("RGB"))
                bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
                if bg_bgr.shape[:2] != (config.LOGICAL_H, config.LOGICAL_W):
                    bg_bgr = cv2.resize(bg_bgr, (config.LOGICAL_W, config.LOGICAL_H), interpolation=cv2.INTER_AREA)
                thumb_bgr = make_thumb_bgr(bg_bgr)

        levels[lvl_id.lower()] = LevelInfo(
            id=lvl_id.lower(),
            disp_name=disp_name,
            frog_xy=(gx, gy),
            curves=curves,
            treasure_points=treasure_points,
            cutouts=cutouts,
            cutout_canvas=cutout_canvas,
            bg_bgr=bg_bgr,
            thumb_bgr=thumb_bgr,
            has_animated_bg=has_animated_bg,
            is_space=is_space,
        )

    return GameAssets(
        zuma_dir=zuma_dir,
        levels=levels,
        ball_alpha_mask=ball_alpha_mask,
        ball_ring_mask=ball_ring_mask,
        powerup_templates=powerup_templates,
        dialog_ref_gray=dialog_ref_gray,
        powerup_matrices=build_powerup_matrices(powerup_templates),
    )
