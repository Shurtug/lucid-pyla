"""Identify which gadget is equipped, and when the bot should fire it.

A brawler owns two gadgets but only one is equipped, and the game never tells
the bot which. Without that, gadget handling has to assume the worst: fire only
when a target is present, because the alternative might waste the charge.

The gadget button draws the equipped gadget's artwork, so it can be read off
the screen. Both the button face and the artwork are green, so colour separates
nothing - the signal is SHAPE. Both sides reduce to a binary silhouette which
are compared by intersection-over-union.

Reference icons come from tools/fetch_gadget_icons.py; the trigger table from
tools/gadget_triggers.csv (classified from the game's own descriptions).
"""
import csv
import json
import re

import cv2
import numpy as np

from utils import load_toml_as_dict, resolve_project_path

NORM = 64                 # silhouettes are compared at this resolution
# Calibrated against live frames: genuine matches led the runner-up by
# 0.141-0.206, while deliberately matching a button against the WRONG brawler's
# icons still produced leads of ~0.064. 0.10 sits in that gap - it accepts every
# real match observed and rejects the accidental ones.
MIN_MARGIN = 0.10         # IoU lead the winner needs before it's believed
MIN_SCORE = 0.30          # ...and an absolute floor, so noise can't "win"

# What each trigger category means for the bot. PLACEMENT and UNCLEAR fall back
# to in-combat, which is the old always-safe behaviour.
CATEGORY_FALLBACK = "IN_COMBAT"

_icon_index = None
_triggers = None
_masks = {}


def _norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _load():
    global _icon_index, _triggers
    if _icon_index is None:
        path = resolve_project_path("cfg", "gadget_icons.json")
        try:
            _icon_index = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _icon_index = {}
    if _triggers is None:
        _triggers = {}
        try:
            with open(resolve_project_path("tools", "gadget_triggers.csv"), encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    _triggers[(row["brawler"].strip().lower(), _norm_name(row["gadget"]))] = \
                        row["category"].strip().upper()
        except Exception:
            pass
    return _icon_index, _triggers


def _normalise(mask):
    """Binary mask -> fixed-size square cropped to its own bounding box, so
    shapes are compared rather than however much padding each carries."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return (cv2.resize(mask, (NORM, NORM), interpolation=cv2.INTER_AREA) > 60).astype(np.uint8)


def _icon_mask(filename):
    if filename not in _masks:
        path = resolve_project_path("api", "assets", "gadget_icons", filename)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            _masks[filename] = None
        else:
            alpha = img[:, :, 3] if img.shape[2] == 4 else np.full(img.shape[:2], 255, np.uint8)
            _masks[filename] = _normalise((alpha > 60).astype(np.uint8) * 255)
    return _masks[filename]


def button_silhouette(frame_bgr):
    """The gadget artwork on the button, as a normalised binary mask.

    The button face and its artwork are both green but split cleanly by
    brightness, with the art the brighter of the two. The outer fifth is
    dropped first to shed the circular frame and charge ring.
    """
    x1, y1, x2, y2 = load_toml_as_dict("cfg/lobby_config.toml")["pixel_counter_crop_area"]["gadget"]
    h, w = frame_bgr.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0
    crop = frame_bgr[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    m = int(min(ch, cw) * 0.20)
    inner = crop[m:ch - m, m:cw - m] if ch > 2 * m and cw > 2 * m else crop
    if inner.size == 0:
        return None
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    art = ((hue > 35) & (hue < 90) & (sat > 60) & (val >= 110)).astype(np.uint8) * 255
    art = cv2.morphologyEx(art, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _normalise(art)


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def identify(frame_bgr, brawler):
    """-> (gadget_name, category, score, margin) or None when undecidable.

    Undecidable is a normal outcome: no icons for that brawler, the button not
    visible, or the two candidates too close to call. Callers should keep their
    existing behaviour in that case rather than guessing.
    """
    index, triggers = _load()
    entries = index.get(str(brawler).lower().strip())
    if not entries:
        return None
    button = button_silhouette(frame_bgr)
    if button is None:
        return None

    scored = []
    for e in entries:
        icon = _icon_mask(e["file"])
        if icon is not None:
            scored.append((_iou(button, icon), e))
    if not scored:
        return None
    scored.sort(key=lambda r: -r[0])

    best_score, best = scored[0]
    margin = best_score - scored[1][0] if len(scored) > 1 else best_score
    if best_score < MIN_SCORE or margin < MIN_MARGIN:
        return None
    category = triggers.get((str(brawler).lower().strip(), _norm_name(best["name"])),
                            CATEGORY_FALLBACK)
    return best["name"], category, best_score, margin
