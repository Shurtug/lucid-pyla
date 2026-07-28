"""Download every gadget icon so the bot can tell which one is equipped.

The bot can't ask the game which gadget you picked, so it has to look at the
gadget button. That needs a reference image per gadget. They aren't practically
extractable from the client (the APK's ui.sc is SupercellSWF v6, ahead of the
public tooling), but Brawlify publishes them:

    https://api.brawlapi.com/v1/brawlers   -> per-brawler gadget list + icon URL
    https://cdn.brawlify.com/gadgets/...   -> the PNGs

Writes:
    api/assets/gadget_icons/<brawler>_<n>.png   one per gadget
    cfg/gadget_icons.json                       index: brawler -> [{name, file, id}]

Brawler keys are normalised to the bot's own convention (lowercase, no spaces
or punctuation), matching cfg/brawlers_info.json.

Usage:
    python tools/fetch_gadget_icons.py [--force]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = PROJECT_ROOT / "api" / "assets" / "gadget_icons"
INDEX_PATH = PROJECT_ROOT / "cfg" / "gadget_icons.json"
BRAWLERS_URL = "https://api.brawlapi.com/v1/brawlers"
# Brawlify's edge blocks the default requests UA outright (403).
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"}
TIMEOUT = 25


def norm_key(name):
    """Brawlify display name -> the bot's brawler key ('Mr. P' -> 'mrp')."""
    name = str(name).lower().strip()
    for ch in (" ", "-", ".", "&", "'"):
        name = name.replace(ch, "")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download icons that already exist")
    args = ap.parse_args()

    try:
        resp = requests.get(BRAWLERS_URL, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        brawlers = resp.json()["list"]
    except Exception as exc:
        sys.exit("could not fetch the brawler list: %s" % exc)
    print("fetched %d brawlers" % len(brawlers))

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    known = {}
    if (PROJECT_ROOT / "cfg" / "brawlers_info.json").exists():
        known = json.loads((PROJECT_ROOT / "cfg" / "brawlers_info.json").read_text(encoding="utf-8"))

    index, downloaded, skipped, failed, unmatched = {}, 0, 0, 0, []
    for b in brawlers:
        key = norm_key(b.get("name", ""))
        gadgets = b.get("gadgets") or []
        if not gadgets:
            continue
        if known and key not in known:
            # Brawlify carries collab/unreleased characters the bot's config
            # doesn't - worth naming rather than silently dropping.
            unmatched.append(key)
            continue

        entries = []
        for i, g in enumerate(gadgets, 1):
            url = g.get("imageUrl")
            if not url:
                continue
            fname = "%s_%d.png" % (key, i)
            dest = ICON_DIR / fname
            if dest.exists() and not args.force:
                skipped += 1
            else:
                try:
                    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    downloaded += 1
                    time.sleep(0.05)   # be polite to the CDN
                except Exception as exc:
                    print("  ! %s: %s" % (fname, exc))
                    failed += 1
                    continue
            entries.append({
                "name": g.get("name", ""),
                "id": g.get("id"),
                "file": fname,
                "description": re.sub(r"<[^>]*>", "", g.get("description", "") or "").strip(),
            })
        if entries:
            index[key] = entries

    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in index.values())
    print("\n%d gadgets across %d brawlers" % (total, len(index)))
    print("downloaded=%d  already-present=%d  failed=%d" % (downloaded, skipped, failed))
    if unmatched:
        print("not in brawlers_info (skipped): %s" % ", ".join(sorted(set(unmatched))))
    print("icons -> %s" % ICON_DIR)
    print("index -> %s" % INDEX_PATH)


if __name__ == "__main__":
    main()
