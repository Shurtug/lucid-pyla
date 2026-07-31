"""Probe: can we work out where the player is inside the map grid?

Run from the project root, against frames collected by play.py's match sampler:

    python tools/localize_probe.py

MEASURED, not assumed:
  * tile pitch at a 1280-wide capture is ~49px across, ~38px down (fitted a
    lattice to detected wall-box centres; edge autocorrelation was useless on
    art this busy - it returned 18-101px with near-zero correlation)
  * searching all 106 showdown maps, 6 of 8 real frames independently picked
    the SAME map, with margins of 0.18-0.34 over the runner-up
  * the 2 that disagreed had margins of 0.002 and 0.007 - correctly unsure
    rather than confidently wrong
  * control: random wall patterns still favour some maps (bias is real), but
    never by more than 0.038

So the runner-up margin separates signal from noise by roughly 10x, and
MIN_MARGIN below is set from that gap rather than guessed. Anything under it
should be treated as "don't know" and the caller should keep doing whatever it
did before.

Still unproven: whether the winning CELL is right, not just the winning map.
That needs ground truth per frame.

How it works: turn what the vision model sees into a small patch in TILE space
centred on the player, then slide that patch over each candidate map's wall
grid, masking out cells we can't see so they neither help nor hurt the score.
"""
import sys, glob, json
import cv2
import numpy as np

sys.path.insert(0, '.')
from detect import Detect
from utils import load_toml_as_dict

PITCH_X_1280 = 49.0        # measured from wall-centre lattice fits
PITCH_Y_1280 = 38.0
BLOCKING = set("MXYCBNTIJEo")   # tiles that read as a wall on screen
MIN_MARGIN = 0.10               # from the control: noise peaks at 0.038, real matches reach 0.34


def load_maps(prefix="Survival"):
    d = json.load(open('cfg/map_data.json', encoding='utf-8'))
    out = {}
    for name, grid in d['grids'].items():
        if not name.startswith(prefix):
            continue
        h = len(grid)
        w = max(len(r) for r in grid)
        a = np.zeros((h, w), np.float32)
        for r, line in enumerate(grid):
            for c, ch in enumerate(line):
                if ch in BLOCKING:
                    a[r, c] = 1.0
        out[name] = a
    return out


def observed_patch(frame_bgr, main_det, tile_det, radius=11):
    """-> (patch, mask) in tile space centred on the player, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    scale = w / 1280.0
    px_pitch = PITCH_X_1280 * scale
    py_pitch = PITCH_Y_1280 * scale

    md = main_det.detect_objects(rgb, conf_tresh=0.5)
    players = md.get('player') or []
    if not players:
        return None
    b = players[0]
    ppos = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    td = tile_det.detect_objects(rgb, conf_tresh=0.45)
    walls = [x for k, v in td.items() if 'bush' not in k for x in v]

    size = radius * 2 + 1
    patch = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), np.uint8)

    # everything on screen is observed; mark the visible window
    for rr in range(size):
        for cc in range(size):
            dx = (cc - radius) * px_pitch
            dy = (rr - radius) * py_pitch
            if abs(dx) <= w / 2 - px_pitch and abs(dy) <= h / 2 - py_pitch:
                mask[rr, cc] = 1

    for wb in walls:
        wx = (wb[0] + wb[2]) / 2.0
        wy = (wb[1] + wb[3]) / 2.0
        cc = int(round((wx - ppos[0]) / px_pitch)) + radius
        rr = int(round((wy - ppos[1]) / py_pitch)) + radius
        if 0 <= rr < size and 0 <= cc < size:
            patch[rr, cc] = 1.0
            mask[rr, cc] = 1
    if mask.sum() < 30 or patch.sum() < 6:
        return None
    return patch, mask


def best_match(patch, mask, maps):
    results = []
    for name, grid in maps.items():
        if grid.shape[0] < patch.shape[0] or grid.shape[1] < patch.shape[1]:
            continue
        res = cv2.matchTemplate(grid, patch, cv2.TM_CCORR_NORMED, mask=mask.astype(np.float32))
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        _, mx, _, loc = cv2.minMaxLoc(res)
        results.append((mx, name, loc))
    results.sort(reverse=True)
    return results


def main():
    classes = load_toml_as_dict('cfg/bot_config.toml')['wall_model_classes']
    tile_det = Detect('models/tileDetector.onnx', classes=classes)
    main_det = Detect('models/mainInGameModel.onnx', classes=['enemy', 'teammate', 'player'])
    maps = load_maps()
    print('candidate showdown maps: %d' % len(maps))

    fs = sorted(glob.glob('debug_frames/match_samples/*.png'))
    picks = []
    for i in [20, 40, 60, 80, 100, 120, 140, 160]:
        frame = cv2.imread(fs[i])
        op = observed_patch(frame, main_det, tile_det)
        if op is None:
            print('frame %3d: no usable patch' % i)
            continue
        patch, mask = op
        res = best_match(patch, mask, maps)
        if not res:
            continue
        top, second = res[0], res[1]
        picks.append(top[1])
        print('frame %3d: walls=%2d -> %-14s score=%.3f  (runner-up %-14s %.3f, margin %.3f)  cell=%s'
              % (i, int(patch.sum()), top[1], top[0], second[1], second[0], top[0] - second[0], top[2]))
    if picks:
        import collections
        c = collections.Counter(picks)
        print()
        print('map agreement across frames:', c.most_common(4))


if __name__ == '__main__':
    main()
