"""Extract map layouts from a Brawl Stars client so the bot knows the terrain.

The client ships every map as an exact tile grid, plus the table that turns the
name shown in the lobby ("Cavern Churn") into the internal map it refers to.
That's strictly better than reading terrain off the screen: it distinguishes
things the vision model can't, like water (blocks movement but shots fly over)
from a solid wall, and destructible walls from permanent ones.

Join chain, all from local client data - no scraping, no external API:

    lobby text "Cavern Churn"  ->  texts.csv     ->  TID_BATTLE_ROYALE_6
    TID_BATTLE_ROYALE_6        ->  locations.csv ->  Survival_6  (+ variants)
    Survival_6                 ->  maps.csv      ->  60x60 grid

Usage:
    python tools/extract_maps.py <path-to.apk> [--out cfg/map_data.json]
"""
import argparse
import collections
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_tile_props(z):
    """Read the tile legend straight out of the client rather than hardcoding
    it, so a game update that adds terrain is picked up instead of silently
    being treated as open ground.

    The distinctions that matter and which the vision model cannot make:
    water and rope fences block movement but NOT projectiles, and walls differ
    in whether they can be destroyed.
    """
    header, rows = read_csv(z, "assets/csv_logic/tiles.csv")
    ci = {c: header.index(c) for c in
          ("Name", "TileCode", "BlocksMovement", "BlocksProjectiles",
           "IsDestructible", "IsForest")}
    props = {".": {"name": "open"}}
    for row in rows:
        if not row or not row[0].strip():
            continue
        code = row[ci["TileCode"]]
        # '-' marks entries with no map character of their own (themed wall
        # variants that share another code)
        if not code or code.strip() in ("", "-", "."):
            continue

        def flag(col):
            return row[ci[col]].strip().lower() == "true"

        entry = {"name": row[0].strip()}
        if flag("BlocksMovement"):
            entry["blocks_move"] = True
        if flag("BlocksProjectiles"):
            entry["blocks_shots"] = True
        if flag("IsDestructible"):
            entry["destructible"] = True
        if flag("IsForest"):
            entry["cover"] = True
        props[code] = entry
    return props


def read_csv(z, path):
    rows = list(csv.reader(io.StringIO(z.read(path).decode("utf-8", errors="replace"))))
    return rows[0], rows[2:]          # row 1 is the type row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("apk")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "cfg" / "map_data.json"))
    args = ap.parse_args()

    try:
        z = zipfile.ZipFile(args.apk)
    except Exception as exc:
        sys.exit("could not open %s: %s" % (args.apk, exc))

    tile_props = build_tile_props(z)

    texts = {r[0]: r[1] for r in csv.reader(
        io.StringIO(z.read("assets/localization/texts.csv").decode("utf-8", errors="replace")))
        if len(r) >= 2}

    # maps.csv lists one grid ROW per line, with the map name only on its first
    _, map_rows = read_csv(z, "assets/csv_logic/maps.csv")
    grids = collections.defaultdict(list)
    current = None
    for row in map_rows:
        if not row:
            continue
        if row[0].strip():
            current = row[0].strip()
        if current and len(row) > 1:
            grids[current].append(row[1])

    lh, ld = read_csv(z, "assets/csv_logic/locations.csv")
    li = {c: lh.index(c) for c in ("Name", "TID", "Map")}

    # display name -> the internal maps it can mean. One name covers several
    # entries (solo/duo/trio, themed re-skins), so keep them all and let the
    # caller pick using the game mode.
    by_name = collections.defaultdict(list)
    for row in ld:
        if not row or len(row) <= li["Map"]:
            continue
        tid = row[li["TID"]].strip()
        internal = row[li["Map"]].strip()
        display = texts.get(tid, "").strip()
        if not display or internal not in grids:
            continue
        by_name[display.lower()].append({"location": row[li["Name"]].strip(), "map": internal})

    used = {e["map"] for entries in by_name.values() for e in entries}
    out = {
        "tiles": tile_props,
        "by_name": {k: v for k, v in sorted(by_name.items())},
        "grids": {k: grids[k] for k in sorted(used)},
    }
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    unknown = collections.Counter()
    for g in out["grids"].values():
        for line in g:
            for ch in line:
                if ch not in tile_props and not ch.isdigit():
                    unknown[ch] += 1
    size = Path(args.out).stat().st_size / 1e6
    print("named maps: %d   grids kept: %d   (%.1f MB)" % (len(out["by_name"]), len(out["grids"]), size))
    if unknown:
        print("tile characters with no entry in TILE_PROPS: %s" % dict(unknown.most_common(10)))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
