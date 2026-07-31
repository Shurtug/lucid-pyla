"""Work out which grid cell the player is standing in.

The map's exact tile grid is known once the lobby banner has been read
(map_awareness), so this is a matching problem rather than a search: turn the
walls the vision model sees into a patch in TILE space centred on the player,
slide it over that one grid, and take the best alignment.

What the samples showed, and why the code is shaped this way:

  * measured tile pitch is ~49px across and ~38px down at a 1280-wide capture
    (fitted a lattice to detected wall-box centres). The vertical figure is
    smaller because the camera is tilted, not because tiles aren't square.
  * on a Brawl Ball map, searching blind across 13 candidates still picked the
    right one 34 times out of 39, with cell steps of ~3 tiles per 4s - real
    movement. Knowing the map up front removes that 13% error entirely.
  * on Showdown it was far less reliable: 98 candidates, a fully open 60x60
    arena and repetitive terrain meant matches landed all over the map. Hence
    MIN_SCORE below and the motion check - a bad fix should be dropped, not
    acted on.
  * bushes and water were both tried as extra signal and both made it worse.
    Bushes because the game hides them once you're standing in one; water
    because the tile model has no water class, so it is never detected and only
    ever adds cells the vision cannot satisfy.

Nothing here steers the bot. It reports a position and a confidence, and says
None whenever it isn't sure.
"""
import math

import cv2
import numpy as np

# Wall-ish tiles: what reads as an obstacle on screen. Deliberately excludes
# water and rope fences (invisible to the tile model) and bushes.
BLOCKING = set("MXYCBNTIJEo")

PITCH_X_1280 = 49.0
PITCH_Y_1280 = 38.0
PATCH_RADIUS = 11          # tiles either side of the player to compare
MIN_WALLS = 6              # too few walls in view and any alignment "fits"
MIN_SCORE = 0.35           # absolute match quality floor
MAX_TILES_PER_SEC = 4.0    # faster than any brawler; used to reject jumps

_grid_cache = {}


def _grid_array(resolved):
    """Map grid -> binary wall array, cached per map."""
    name = resolved["map"]
    if name not in _grid_cache:
        grid = resolved["grid"]
        h = len(grid)
        w = max(len(r) for r in grid)
        arr = np.zeros((h, w), np.float32)
        for r, line in enumerate(grid):
            for c, ch in enumerate(line):
                if ch in BLOCKING:
                    arr[r, c] = 1.0
        _grid_cache[name] = arr
    return _grid_cache[name]


def observed_patch(player_pos, wall_boxes, frame_shape):
    """Walls on screen -> (patch, mask) in tile space centred on the player."""
    h, w = frame_shape[:2]
    scale = w / 1280.0
    px = PITCH_X_1280 * scale
    py = PITCH_Y_1280 * scale

    size = PATCH_RADIUS * 2 + 1
    patch = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), np.uint8)

    # mark which cells are actually on screen - anything off-frame is unknown
    # and must not count for or against a candidate alignment
    for rr in range(size):
        for cc in range(size):
            if (abs((cc - PATCH_RADIUS) * px) <= w / 2 - px
                    and abs((rr - PATCH_RADIUS) * py) <= h / 2 - py):
                mask[rr, cc] = 1

    seen = 0
    for b in wall_boxes:
        wx = (b[0] + b[2]) / 2.0
        wy = (b[1] + b[3]) / 2.0
        cc = int(round((wx - player_pos[0]) / px)) + PATCH_RADIUS
        rr = int(round((wy - player_pos[1]) / py)) + PATCH_RADIUS
        if 0 <= rr < size and 0 <= cc < size:
            patch[rr, cc] = 1.0
            mask[rr, cc] = 1
            seen += 1
    if seen < MIN_WALLS or mask.sum() < 30:
        return None
    return patch, mask


class Localizer:
    """Tracks the player's grid cell across frames for one match.

    Keeps the last accepted fix so a new one can be sanity-checked against it:
    a brawler cannot cross the map between two frames, so a match that claims
    it did is wrong however good its score looks.
    """

    def __init__(self):
        self.cell = None            # (col, row) last accepted
        self.score = 0.0
        self.updated_at = 0.0
        self.rejected = 0           # consecutive physically-impossible fixes
        self._recent = []           # rolling accept(True)/reject(False) history

    def reset(self):
        self.__init__()

    @property
    def reliable(self):
        """Whether the fixes are holding together on THIS map.

        Measured, not assumed per mode: a run of physically impossible jumps
        means the matcher is latching onto repeated terrain, whatever map it
        is. On a Brawl Ball map the filter rejected nothing across 70 frames;
        on a 60x60 Showdown arena it rejected 40 of them. Anything consuming a
        position should check this rather than trusting every fix.
        """
        if len(self._recent) < 8:
            return False
        return sum(self._recent) / len(self._recent) >= 0.8

    def _note(self, accepted):
        self._recent.append(bool(accepted))
        if len(self._recent) > 20:
            self._recent.pop(0)

    def update(self, resolved, player_pos, wall_boxes, frame_shape, now):
        """-> (col, row) when confident, else None. Safe to call every tick."""
        if not resolved:
            return None
        op = observed_patch(player_pos, wall_boxes, frame_shape)
        if op is None:
            return None
        patch, mask = op
        grid = _grid_array(resolved)
        if grid.shape[0] < patch.shape[0] or grid.shape[1] < patch.shape[1]:
            return None

        res = cv2.matchTemplate(grid, patch, cv2.TM_CCORR_NORMED,
                                mask=mask.astype(np.float32))
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < MIN_SCORE:
            return None
        cell = (loc[0] + PATCH_RADIUS, loc[1] + PATCH_RADIUS)

        if self.cell is not None:
            dt = max(now - self.updated_at, 1e-3)
            dist = math.hypot(cell[0] - self.cell[0], cell[1] - self.cell[1])
            if dist > MAX_TILES_PER_SEC * dt:
                # Nobody moves that fast. Drop it - but if it keeps happening
                # the old fix is the thing that's wrong (a real teleport: died
                # and respawned, or a new match), so give up and re-acquire.
                self.rejected += 1
                self._note(False)
                if self.rejected < 5:
                    return None
                self.cell = None

        self.cell = cell
        self.score = float(score)
        self.updated_at = now
        self.rejected = 0
        self._note(True)
        return cell
