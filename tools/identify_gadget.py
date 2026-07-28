"""Work out which gadget is equipped by matching the in-match gadget button.

Each brawler owns two gadgets but only one is equipped, and the game never
tells the bot which. That's why apex.pyla can only fire on cooldown for the
handful of brawlers where BOTH gadgets are safe to fire blind. Reading the
button's artwork settles it.

Reference icons come from tools/fetch_gadget_icons.py (run that first).

This is deliberately a standalone tool rather than wired into the play loop:
matching needs validating against real frames before anything acts on it.

    # against a saved frame
    python tools/identify_gadget.py --image match.png --brawler shelly

    # against whatever the emulator is showing right now
    python tools/identify_gadget.py --live --brawler shelly

    # dump the cropped button so you can eyeball what it's matching on
    python tools/identify_gadget.py --image match.png --brawler shelly --save-crop btn.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_toml_as_dict  # noqa: E402

ICON_DIR = PROJECT_ROOT / "api" / "assets" / "gadget_icons"
INDEX_PATH = PROJECT_ROOT / "cfg" / "gadget_icons.json"
# Reference icons are ~100px with transparent margins; the button art sits
# inside a circular frame, so both get normalised to this before comparing.
NORM = 64


def load_index():
    if not INDEX_PATH.exists():
        sys.exit("no %s - run tools/fetch_gadget_icons.py first" % INDEX_PATH)
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def button_crop(frame):
    """The gadget button region, scaled from the 1920x1080 reference layout."""
    x1, y1, x2, y2 = load_toml_as_dict("cfg/lobby_config.toml")["pixel_counter_crop_area"]["gadget"]
    h, w = frame.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0
    return frame[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]


def _normalise(mask):
    """Binary mask -> fixed-size square, cropped to its own bounding box so
    two silhouettes are compared on shape rather than on how much padding
    each happens to carry."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return (cv2.resize(mask, (NORM, NORM), interpolation=cv2.INTER_AREA) > 60).astype(np.uint8)


def prep_icon(path):
    """Reference PNG -> normalised silhouette (its alpha channel)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    alpha = img[:, :, 3] if img.shape[2] == 4 else np.full(img.shape[:2], 255, np.uint8)
    return _normalise((alpha > 60).astype(np.uint8) * 255)


def prep_button(crop):
    """Button crop -> normalised silhouette of the gadget art.

    Both the button face and the artwork on it are green, so colour can't
    separate them - measured on a live frame the two form a clean bimodal
    split in VALUE, with the art the BRIGHTER of the two. The outer fifth is
    dropped first to shed the circular frame and the charge ring.
    """
    h, w = crop.shape[:2]
    m = int(min(h, w) * 0.20)
    inner = crop[m:h - m, m:w - m] if h > 2 * m and w > 2 * m else crop
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    art = ((hue > 35) & (hue < 90) & (sat > 60) & (val >= 110)).astype(np.uint8) * 255
    art = cv2.morphologyEx(art, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _normalise(art)


def score(button_mask, icon_mask):
    """Silhouette agreement as intersection-over-union, 0..1."""
    if button_mask is None or icon_mask is None:
        return 0.0
    inter = np.logical_and(button_mask, icon_mask).sum()
    union = np.logical_or(button_mask, icon_mask).sum()
    return float(inter / union) if union else 0.0


def identify(frame, brawler, save_crop=None):
    index = load_index()
    key = brawler.lower().strip()
    entries = index.get(key)
    if not entries:
        return None, "no icons for %r (run fetch_gadget_icons.py; 9 newest brawlers aren't on the CDN)" % key

    crop = button_crop(frame)
    if crop.size == 0:
        return None, "gadget button crop was empty - wrong resolution?"
    if save_crop:
        cv2.imwrite(str(save_crop), crop)
    button = prep_button(crop)
    if button is None:
        return None, "no gadget artwork found in the button - is one visible/ready?"

    results = []
    for e in entries:
        icon = prep_icon(ICON_DIR / e["file"])
        if icon is None:
            continue
        results.append((score(button, icon), e))
    if not results:
        return None, "reference icons missing on disk - run fetch_gadget_icons.py"
    results.sort(key=lambda r: -r[0])
    return results, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="a saved frame to test against")
    src.add_argument("--live", action="store_true", help="grab the current emulator frame")
    ap.add_argument("--brawler", required=True, help="brawler key, e.g. shelly")
    ap.add_argument("--save-crop", help="write the cropped button here for inspection")
    args = ap.parse_args()

    if args.live:
        from window_controller import WindowController
        wc = WindowController()
        frame = wc.screenshot()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # scrcpy frames are RGB
    else:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit("could not read %s" % args.image)

    results, err = identify(frame, args.brawler, args.save_crop)
    if err:
        sys.exit(err)

    print("gadget candidates for %s:" % args.brawler)
    for s, e in results:
        print("  %.3f  %-24s %s" % (s, e["name"], e["description"][:60]))
    best, runner = results[0], (results[1] if len(results) > 1 else None)
    margin = best[0] - runner[0] if runner else best[0]
    print()
    print("best guess: %s  (score %.3f, margin %.3f)" % (best[1]["name"], best[0], margin))
    if margin < 0.08:
        print("WARNING: margin is small - not a confident identification")


if __name__ == "__main__":
    main()
