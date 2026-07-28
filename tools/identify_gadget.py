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


def prep_icon(path):
    """Reference PNG -> normalised BGR + a mask of its non-transparent art."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    else:
        alpha = np.full(img.shape[:2], 255, np.uint8)
        bgr = img
    # trim transparent padding so scale is comparable between icons
    ys, xs = np.where(alpha > 40)
    if len(ys):
        bgr = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bgr = cv2.resize(bgr, (NORM, NORM), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (NORM, NORM), interpolation=cv2.INTER_AREA)
    return bgr, alpha


def prep_button(crop):
    """Button crop -> normalised BGR, centre-cropped to drop the frame ring."""
    h, w = crop.shape[:2]
    m = int(min(h, w) * 0.18)          # shave the circular border
    inner = crop[m:h - m, m:w - m] if h > 2 * m and w > 2 * m else crop
    return cv2.resize(inner, (NORM, NORM), interpolation=cv2.INTER_AREA)


def score(button, icon_bgr, icon_alpha):
    """Similarity in 0..1. Hue histogram over the icon's opaque pixels - the
    button tints and scales the art, so colour distribution survives that far
    better than raw pixel correlation."""
    mask = (icon_alpha > 60).astype(np.uint8) * 255
    b_hsv = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)
    i_hsv = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
    hb = cv2.calcHist([b_hsv], [0, 1], mask, [24, 8], [0, 180, 0, 256])
    hi = cv2.calcHist([i_hsv], [0, 1], mask, [24, 8], [0, 180, 0, 256])
    cv2.normalize(hb, hb, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hi, hi, 0, 1, cv2.NORM_MINMAX)
    hist = float(cv2.compareHist(hb, hi, cv2.HISTCMP_CORREL))
    # plus a masked template correlation for shape agreement
    res = cv2.matchTemplate(button, icon_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
    tmpl = float(res.max()) if np.isfinite(res).any() else 0.0
    return max(0.0, 0.6 * hist + 0.4 * tmpl)


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

    results = []
    for e in entries:
        bgr, alpha = prep_icon(ICON_DIR / e["file"])
        if bgr is None:
            continue
        results.append((score(button, bgr, alpha), e))
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
