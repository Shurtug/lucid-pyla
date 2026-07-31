"""Replay captured frames through the real decision logic and show what the
bot would do with each one.

This runs the ACTUAL pipeline - the same detection models, the same
Play.loop(), the same .pyla playstyle - with the emulator replaced by a stub
that hands over a still image. So the answers are what the bot would really
have decided on that frame, not a re-implementation that might drift from it.

    # summary line per frame
    python tools/replay_decisions.py debug_frames/match_samples

    # only frames where something interesting happened
    python tools/replay_decisions.py debug_frames/match_samples --acts

    # annotated pictures: detections, the movement arrow, the decision
    python tools/replay_decisions.py debug_frames/match_samples --render out/

    # one specific frame, with everything the playstyle saw
    python tools/replay_decisions.py <file.png> --verbose

Frames saved by play.py's sampler are named
    <time>_<map>_<brawler>_<n>.png
so the brawler and map are taken from the filename; --brawler overrides it.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from play import Play                                    # noqa: E402
from utils import load_toml_as_dict, load_pyla_script, interpret_pyla_code   # noqa: E402
import localization                                       # noqa: E402

BS_W, BS_H = 1920, 1080


class StubWindow:
    """Stands in for the emulator. Only geometry is ever read during a replay;
    anything that would touch the device is a no-op so a replay can never
    send input to a real game."""

    def __init__(self, frame):
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.width_ratio = self.width / BS_W
        self.height_ratio = self.height / BS_H
        self.scale_factor = min(self.width_ratio, self.height_ratio)
        self.joystick_x = 220 * self.width_ratio
        self.joystick_y = 870 * self.height_ratio
        self.aim_drag_radius = load_toml_as_dict("cfg/bot_config.toml").get("aim_drag_radius", 140)
        self.last_frame = frame

    # every input path is deliberately inert
    def press(self, *a, **k): pass
    def move(self, *a, **k): pass
    def release_movement(self, *a, **k): pass
    def aim_swipe(self, *a, **k): pass
    def click(self, *a, **k): pass
    def screenshot(self): return self.last_frame


def parse_name(path):
    """-> (map, brawler) from the sampler's filename, either may be None."""
    m = re.match(r"\d{8}-\d{6}_(.+)_([a-z0-9]+)_\d+\.png$", os.path.basename(path))
    if not m:
        return None, None
    mp = m.group(1)
    return (None if mp == "nomap" else mp), m.group(2)


def build_engine(frame, brawler):
    cfg = load_toml_as_dict("cfg/bot_config.toml")
    _meta, pyla = load_pyla_script(cfg.get("current_playstyle", "apex.pyla"))
    wc = StubWindow(frame)
    play = Play("models/mainInGameModel.onnx", "models/tileDetector.onnx",
                "models/closeTileDetector.onnx", wc, pyla)
    play.current_brawler = brawler
    # replays are one frame at a time; pipelining would hand back the PREVIOUS
    # frame's detections and silently shift every result by one
    play._detect_executor = None
    play.pipeline_inference = False
    return play


def run_frame(play, frame, brawler, now):
    """-> dict describing what the bot decided for this frame."""
    acts = {"attack": 0, "super": 0, "gadget": 0, "hypercharge": 0, "aims": []}
    play.current_brawler = brawler

    def rec_attack(touch_up=True, touch_down=True, aim=None, distance_ratio=1.0):
        acts["attack"] += 1
        if aim:
            acts["aims"].append(tuple(int(v) for v in aim))

    play.attack = rec_attack
    play.use_super = lambda aim=None, distance_ratio=1.0: acts.__setitem__("super", acts["super"] + 1)
    play.use_gadget = lambda: acts.__setitem__("gadget", acts["gadget"] + 1)
    play.use_hypercharge = lambda: acts.__setitem__("hypercharge", acts["hypercharge"] + 1)

    # Capture what the playstyle itself decided. zone and the action list live
    # in the script's own globals, not persistent_data, and loop() throws them
    # away - but they're the whole point of a replay.
    captured = {}

    def get_movement_capturing():
        mv, g = interpret_pyla_code(play._pyla_compiled or play.pyla_code, play.context)
        captured["globals"] = g
        captured["raw_movement"] = mv
        return mv

    play.get_movement = get_movement_capturing

    play.frame = frame
    data = play.get_main_data(frame)
    tiles = play.get_tile_data(frame, data.get("player"))
    walls, bushes = play.process_tile_data(tiles)
    data["wall"], data["bush"] = walls, bushes
    play.last_walls_data, play.last_bushes_data = walls, bushes
    for key in ("player", "enemy", "teammate"):
        data.setdefault(key, [])

    if data["player"]:
        play.update_player_hp(frame, data["player"])

    movement = play.loop(brawler, data, now)

    g = captured.get("globals", {})
    pd = play.persistent_data
    return {
        # raw_movement is what the playstyle asked for; movement is what the
        # engine would actually send after its own smoothing and unstuck pass -
        # they differ, and when they do that IS the interesting part
        "movement": captured.get("raw_movement"),
        "final_movement": movement,
        "zone": g.get("_zone"),
        "playstyle_acts": g.get("_acts") or [],
        "acts": acts,
        "enemies": len(data["enemy"]), "mates": len(data["teammate"]),
        "walls": len(walls), "bushes": len(bushes),
        "player": bool(data["player"]),
        "hp": pd.get("current_hp_pct"), "hp_conf": pd.get("hp_confidence"),
        "ammo": pd.get("current_ammo_segments"),
        "data": data,
    }


def annotate(frame, res, out_path):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    d = res["data"]
    for b in d["wall"]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (110, 110, 110), 1)
    for b in d["bush"]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 160, 60), 1)
    for b in d["teammate"]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 180, 0), 2)
    for b in d["enemy"]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    for b in d["player"]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

    if d["player"] and res["movement"]:
        x1 = int((d["player"][0][0] + d["player"][0][2]) / 2)
        y1 = int((d["player"][0][1] + d["player"][0][3]) / 2)
        mv = res["movement"]
        cv2.arrowedLine(img, (x1, y1), (int(x1 + mv[0] * 1.5), int(y1 + mv[1] * 1.5)),
                        (0, 255, 255), 3, tipLength=0.3)
    for a in res["acts"]["aims"]:
        cv2.drawMarker(img, a, (0, 140, 255), cv2.MARKER_TILTED_CROSS, 22, 2)

    a = res["acts"]
    lines = [
        "zone: %s" % (res["zone"] or "-"),
        "move: %s" % (str(tuple(round(v) for v in res["movement"])) if res["movement"] else "none"),
        "atk=%d super=%d gadget=%d" % (a["attack"], a["super"], a["gadget"]),
        "enemies=%d mates=%d" % (res["enemies"], res["mates"]),
    ]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (12, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(img, t, (12, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a frame, or a folder of them")
    ap.add_argument("--brawler", help="override the brawler from the filename")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--step", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--acts", action="store_true", help="only frames where it acted")
    ap.add_argument("--render", help="write annotated images to this folder")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.png")))[::args.step][:args.limit]
    else:
        files = [args.path]
    if not files:
        sys.exit("no frames found at %s" % args.path)

    first = cv2.imread(files[0])
    if first is None:
        sys.exit("could not read %s" % files[0])
    fmap, fbrawler = parse_name(files[0])
    brawler = args.brawler or fbrawler
    if not brawler:
        sys.exit("could not tell the brawler from the filename - pass --brawler")

    print("frames: %d   brawler: %s   map: %s   playstyle: %s"
          % (len(files), brawler, fmap or "unknown",
             load_toml_as_dict("cfg/bot_config.toml").get("current_playstyle")))
    play = build_engine(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), brawler)

    if args.render:
        Path(args.render).mkdir(parents=True, exist_ok=True)

    print()
    print("%-28s %-22s %-11s %-16s %s" % ("frame", "zone", "move", "actions", "seen"))
    print("-" * 104)
    counts = {}
    # advance a realistic amount between frames: replaying faster than real
    # time makes the engine's movement-smoothing and unstuck timers misbehave
    clock = time.time()
    for f in files:
        frame = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        fm, fb = parse_name(f)
        res = run_frame(play, frame, args.brawler or fb or brawler, clock)
        clock += 2.0
        a = res["acts"]
        acted = a["attack"] or a["super"] or a["gadget"]
        if args.acts and not acted:
            continue
        counts[res["zone"]] = counts.get(res["zone"], 0) + 1
        act_s = ",".join(res["playstyle_acts"]) or "-"
        print("%-28s %-22s %-11s %-16s %s" % (
            os.path.basename(f)[:28], (res["zone"] or "-")[:22],
            "(%s)" % ",".join("%d" % v for v in res["movement"]) if res["movement"] else "none",
            act_s,
            "e=%d m=%d w=%d%s" % (res["enemies"], res["mates"], res["walls"],
                                  "" if res["player"] else " NO-PLAYER")))
        if args.verbose:
            print("      hp=%s conf=%s ammo=%s aims=%s"
                  % (res["hp"], res["hp_conf"], res["ammo"], a["aims"][:3]))
        if args.render:
            annotate(frame, res, Path(args.render) / ("dec_" + os.path.basename(f)))

    print()
    print("zone distribution:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    if args.render:
        print("annotated frames -> %s" % args.render)


if __name__ == "__main__":
    main()
