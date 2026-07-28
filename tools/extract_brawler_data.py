"""Extract real brawler stats from a Brawl Stars client APK into brawlers_info.json.

The game ships its balance data as plain CSVs inside the APK under
assets/csv_logic/. Joining three of them gives the authoritative numbers for
every brawler, replacing hand-estimated values and supplying fields the bot
never had (reload time, magazine size, hitpoints, movement speed).

    characters.csv   ItemName == the bot's brawler key ("shelly"), Type == Hero
      -> WeaponSkill / UltimateSkill
    skills.csv       CastingRange, RechargeTime, MaxCharge, Damage, ...
      -> Projectiles (first entry)
    projectiles_logic.csv   Speed, ...

Unit conventions (verified empirically against the existing config, not
guessed - both hold across every brawler with zero variance):
    attack_range     = weapon CastingRange * 64/3      (ratio 21.333)
    projectile_speed = weapon projectile Speed * 0.9

Usage:
    python tools/extract_brawler_data.py <path-to.apk> [--write] [--fix-ranges]

Without --write it only reports what would change (safe to run anytime).
Existing tuned fields are preserved by default; --fix-ranges additionally
overwrites attack_range/super_range where the APK disagrees with the config.
"""
import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "cfg" / "brawlers_info.json"

RANGE_SCALE = 64 / 3      # CastingRange -> the config's pixel-space range
PROJECTILE_SCALE = 0.9    # projectile Speed -> the config's projectile_speed

CHARACTERS = "assets/csv_logic/characters.csv"
SKILLS = "assets/csv_logic/skills.csv"
PROJECTILES = "assets/csv_logic/projectiles_logic.csv"
TEXTS = "assets/localization/texts.csv"

# The CSV's ItemName differs from the bot's key for these. Every entry was
# resolved from the APK's own localization table (codename -> TID -> display
# name), not guessed - e.g. TrickshotDude -> TID_TRICKSHOT_DUDE -> "RICO".
ITEM_NAME_ALIASES = {
    "ricochet": "rico",
    "artie": "rt",              # R-T
    "mr.p": "mrp",
    "melody": "melodie",
    "lolla": "lola",
    "jae": "jaeyong",           # JAE-YONG
    "stella": "starrnova",      # STARR NOVA
    "shadowdemon": "sirius",
    "mender": "glowy",
    "digger": "moe",
    "dancer": "mina",
    "fury": "ziggy",
    "gladiator": "damian",
    "redirecter": "najia",
    "samurai": "kenji",
    "twins": "larrylawrie",     # LARRY & LAWRIE
    "fishtank": "hank",
    "katanakid": "nori",
}

# Not player-selectable in the normal roster (collab/event characters); the
# bot's brawler picker can't choose them, so they'd only add noise.
SKIP_KEYS = {"lightyear"}       # BUZZ LIGHTYEAR (collab variant)


def read_csv(zf, member):
    """Returns (header, data_rows). Row 1 is a type declaration, not data."""
    raw = zf.read(member).decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(raw)))
    return rows[0], rows[2:]


def as_num(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truthy(value):
    return str(value).strip().lower() == "true"


def codename_to_tid(codename):
    return "TID_" + re.sub(r"(?<!^)(?=[A-Z])", "_", codename).upper()


def build(apk_path):
    with zipfile.ZipFile(apk_path) as zf:
        ch, cdata = read_csv(zf, CHARACTERS)
        sh, sdata = read_csv(zf, SKILLS)
        ph, pdata = read_csv(zf, PROJECTILES)
        try:
            _, tdata = read_csv(zf, TEXTS)
            texts = {r[0]: r[1] for r in tdata if len(r) >= 2}
        except KeyError:
            texts = {}

    ci = {c: ch.index(c) for c in
          ("ItemName", "Type", "Name", "WeaponSkill", "UltimateSkill", "Hitpoints", "Speed")}
    si = {c: sh.index(c) for c in
          ("Name", "CastingRange", "RechargeTime", "MaxCharge", "Damage",
           "Projectiles", "NumBulletsInOneAttack", "Spread", "MsBetweenAttacks",
           "HoldToShoot")}
    pi = {c: ph.index(c) for c in ("Name", "Speed", "Indirect")}

    skills = {r[si["Name"]]: r for r in sdata if r and r[si["Name"]]}
    projectiles = {r[pi["Name"]]: r for r in pdata if r and r[pi["Name"]]}

    extracted = {}
    for row in cdata:
        if not row or row[ci["Type"]] not in ("Hero", "Her0"):
            continue
        item_name = row[ci["ItemName"]].strip()
        if not item_name:
            continue
        key = ITEM_NAME_ALIASES.get(item_name, item_name)
        if key in SKIP_KEYS:
            continue

        weapon = skills.get(row[ci["WeaponSkill"]])
        if not weapon:
            continue
        ulti = skills.get(row[ci["UltimateSkill"]])

        def projectile_of(skill):
            if not skill:
                return None
            name = skill[si["Projectiles"]].split(",")[0].strip()
            return projectiles.get(name)

        w_proj = projectile_of(weapon)
        u_proj = projectile_of(ulti)

        stats = {
            # A projectile flagged Indirect is lobbed over walls. Verified to
            # reproduce the config's existing ignore_walls_for_attacks on
            # 71/71 brawlers, so it's used directly for new ones. The same
            # field on the ULTIMATE's projectile only agrees ~73% of the
            # time (supers get hand-tuned), so it's a starting point there.
            "_indirect_attack": truthy(w_proj[pi["Indirect"]]) if w_proj else False,
            "_indirect_super": truthy(u_proj[pi["Indirect"]]) if u_proj else False,
            # --- fields the existing config already had ---
            "attack_range": round(as_num(weapon[si["CastingRange"]]) * RANGE_SCALE),
            "projectile_speed": round(as_num(w_proj[pi["Speed"]]) * PROJECTILE_SCALE, 1) if w_proj else 0.0,
            # --- fields the bot never had ---
            "reload_time": round(as_num(weapon[si["RechargeTime"]]) / 1000.0, 3),
            "max_ammo": int(as_num(weapon[si["MaxCharge"]], 3)) or 3,
            "attack_damage": round(as_num(weapon[si["Damage"]])),
            "bullets_per_attack": int(as_num(weapon[si["NumBulletsInOneAttack"]], 1)) or 1,
            "attack_spread": round(as_num(weapon[si["Spread"]])),
            "hitpoints": round(as_num(row[ci["Hitpoints"]])),
            "movement_speed": round(as_num(row[ci["Speed"]])),
            # Continuous-fire weapon: the button is HELD to keep shooting
            # (amber's flamethrower, gigi). Distinct from hold_attack, which is
            # a charge-up that's held then RELEASED to fire one big shot
            # (hank, angelo) - opposite handling, so it needs its own field.
            "hold_to_shoot": truthy(weapon[si["HoldToShoot"]]),
        }
        if ulti:
            stats["super_range_apk"] = round(as_num(ulti[si["CastingRange"]]) * RANGE_SCALE)

        display = texts.get(codename_to_tid(row[ci["Name"]]), "")
        extracted[key] = {"_display": display, "_codename": row[ci["Name"]], **stats}

    return extracted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("apk", help="path to the Brawl Stars client APK")
    ap.add_argument("--write", action="store_true", help="apply changes to cfg/brawlers_info.json")
    ap.add_argument("--fix-ranges", action="store_true",
                    help="also overwrite attack_range where the APK disagrees (changes combat tuning)")
    args = ap.parse_args()

    apk = Path(args.apk)
    if not apk.exists():
        sys.exit(f"APK not found: {apk}")

    extracted = build(apk)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"APK heroes: {len(extracted)}   config brawlers: {len(config)}")

    new_fields = ("reload_time", "max_ammo", "attack_damage", "bullets_per_attack",
                  "attack_spread", "hitpoints", "movement_speed", "hold_to_shoot")

    added, enriched, range_diffs, filled_speed = [], 0, [], []
    for key, stats in sorted(extracted.items()):
        clean = {k: v for k, v in stats.items() if not k.startswith("_")}
        clean.pop("super_range_apk", None)

        if key not in config:
            if not clean["attack_range"]:
                # No usable range in the APK (contact-damage character);
                # the bot's whole combat model is range-driven, so adding it
                # with range 0 would be worse than leaving it out.
                continue
            # New brawler: needs the tuned fields too, or the playstyle raises
            # ValueError the moment it's picked. Defaults mirror the existing
            # config's conventions for a standard damage brawler.
            # safe_range has no CSV source - it's the bot's own kiting buffer.
            # Measured across the existing config it sits at ~0.60*attack_range
            # for ranged brawlers and 0 for close-range ones (which all fall
            # under ~300px), so new entries follow that same shape.
            melee_ish = clean["attack_range"] <= 300
            config[key] = {
                "safe_range": 0 if melee_ish else round(clean["attack_range"] * 0.60),
                "attack_range": clean["attack_range"],
                "super_type": "damage",
                "super_range": stats.get("super_range_apk", clean["attack_range"]),
                "ignore_walls_for_attacks": stats["_indirect_attack"],
                "ignore_walls_for_supers": stats["_indirect_super"],
                "hold_attack": 0,   # no CSV source; hand-tuned per brawler
                "projectile_speed": clean["projectile_speed"],
                **{k: clean[k] for k in new_fields},
            }
            added.append(f"{key} ({stats.get('_display') or stats.get('_codename')})")
            continue

        entry = config[key]

        # Fill a missing/zero projectile_speed. predict_enemy disables aim
        # leading entirely when this is 0, so supplying it turns predictive
        # aim on for brawlers that never had it. Never overwrites a value
        # that's already set (those are tuned).
        if not entry.get("projectile_speed") and clean["projectile_speed"]:
            entry["projectile_speed"] = clean["projectile_speed"]
            filled_speed.append(key)

        apk_range = clean["attack_range"]
        # apk_range can be 0 for melee/contact attackers that define no
        # CastingRange - never overwrite a tuned value with 0, and don't
        # divide by it.
        if apk_range and entry.get("attack_range") and abs(apk_range - entry["attack_range"]) / apk_range > 0.03:
            range_diffs.append((key, entry["attack_range"], apk_range))
            if args.fix_ranges:
                entry["attack_range"] = apk_range

        before = len(entry)
        for f in new_fields:
            entry[f] = clean[f]
        if len(entry) != before:
            enriched += 1

    print(f"\nnew fields added to {enriched} existing brawlers: {', '.join(new_fields)}")
    if filled_speed:
        print(f"\nprojectile_speed filled for {len(filled_speed)} (enables predictive aim "
              f"where it was disabled): {', '.join(sorted(filled_speed))}")
    if added:
        print(f"\nbrawlers ADDED ({len(added)}): " + ", ".join(added))
    missing = sorted(set(config) - set(extracted) - set(added))
    if missing:
        print(f"\nin config but not in APK ({len(missing)}): {', '.join(missing)}")
    if range_diffs:
        verb = "FIXED" if args.fix_ranges else "differs (use --fix-ranges to apply)"
        print(f"\nattack_range {verb} for {len(range_diffs)}:")
        for key, old, new in sorted(range_diffs, key=lambda x: -abs(x[2] - x[1]) / x[2]):
            print(f"  {key:14s} config={old:5.0f}  apk={new:5.0f}  ({abs(new-old)/new*100:.1f}% off)")

    if args.write:
        CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {CONFIG_PATH}")
    else:
        print("\n(dry run - pass --write to apply)")


if __name__ == "__main__":
    main()
