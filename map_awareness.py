"""Read the game mode and map from the lobby, and look up that map's layout.

Brawl Stars shows both on the banner beside the PLAY button, so the bot can
learn what it's about to play before the match starts - and it re-reads them
every game, which matters because the map rotation changes.

Two things come out of this:
  * the game mode as GROUND TRUTH, instead of guessing it from whether poison
    gas or teammates have been seen
  * the map's exact tile grid (see tools/extract_maps.py), which knows things
    the vision model can't - water blocks movement but not shots, some walls
    are destructible and some aren't

Nothing here moves the bot. It reports what it read; callers decide.
"""
import json
import re

import cv2
import numpy as np

from utils import extract_text_and_positions, resolve_project_path

# Banner crops, in the 1920x1080 layout the rest of the bot is calibrated to.
# The mode sits above the map name in white; the map name is green-on-dark and
# needs upscaling before OCR will read it at all.
MODE_CROP = (800, 915, 1330, 1015)      # x1, y1, x2, y2
MAP_CROP = (860, 995, 1200, 1035)
MAP_UPSCALE = 3

_data = None


def _load():
    global _data
    if _data is None:
        path = resolve_project_path("cfg", "map_data.json")
        try:
            _data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _data = {"tiles": {}, "by_name": {}, "grids": {}}
    return _data


def available():
    """False when map_data.json hasn't been generated - callers should treat
    every lookup as unavailable rather than failing."""
    return bool(_load().get("grids"))


def _crop(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0
    return frame[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]


def _text_in(frame_rgb, box, upscale=1, min_prob=0.25):
    crop = _crop(frame_rgb, box)
    if crop.size == 0:
        return []
    if upscale > 1:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    try:
        return list(extract_text_and_positions(crop, min_prob=min_prob).keys())
    except Exception:
        return []


def read_lobby_banner(frame_rgb):
    """-> (mode, map_name), either possibly None. Expects an RGB frame, as the
    capture pipeline produces."""
    mode = None
    for text in _text_in(frame_rgb, MODE_CROP):
        t = text.strip().lower()
        # the banner also carries a "NEW!" badge and stray glyphs
        if len(t) >= 4 and re.search(r"[a-z]{4}", t) and "new" not in t:
            mode = t
            break

    map_name = None
    known = _load().get("by_name", {})
    candidates = _text_in(frame_rgb, MAP_CROP, upscale=MAP_UPSCALE)
    for text in candidates:
        t = text.strip().lower()
        if t in known:                      # exact hit against a real map name
            map_name = t
            break
    if map_name is None:
        # OCR drops or mangles the odd character; accept a close unique match
        import difflib
        for text in candidates:
            t = text.strip().lower()
            if len(t) < 4:
                continue
            close = difflib.get_close_matches(t, known.keys(), n=2, cutoff=0.82)
            if len(close) == 1:
                map_name = close[0]
                break
    return mode, map_name


def resolve(map_name, mode=None):
    """-> {'map', 'grid', 'width', 'height'} for a map name read off the lobby,
    or None. A name covers several internal maps (solo/duo/trio, themed
    re-skins); the mode narrows it, and they normally share a layout anyway."""
    data = _load()
    entries = data.get("by_name", {}).get((map_name or "").strip().lower())
    if not entries:
        return None

    chosen = entries[0]["map"]
    if mode:
        m = mode.replace(" ", "")
        # location names look like SurvivalTrio6 / SurvivalTeam6 / Survival6
        want = "trio" if "trio" in m else ("team" if "duo" in m else None)
        for e in entries:
            loc = e["location"].lower()
            if want and want in loc:
                chosen = e["map"]
                break
    grid = data.get("grids", {}).get(chosen)
    if not grid:
        return None
    return {"map": chosen, "grid": grid,
            "height": len(grid), "width": max((len(r) for r in grid), default=0)}


def tile_at(resolved, col, row):
    """Tile properties at a grid cell, or None when out of bounds."""
    if not resolved:
        return None
    grid = resolved["grid"]
    if not (0 <= row < len(grid)) or not (0 <= col < len(grid[row])):
        return None
    return _load().get("tiles", {}).get(grid[row][col], {"name": "unknown"})
