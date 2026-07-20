"""Thin client for Supercell's official Brawl Stars API (api.brawlstars.com).

Separate from and independent of the paid early_access module (early_access.py,
sold by the PylaAI devs) that main.py/stage_manager.py/utils.py already try to
import - this file is never imported by that fallback chain, so it can't
conflict if that module is installed later. It's wired in as its own opt-in
path, gated purely on cfg/general_config.toml's brawlstars_api_key being set.

Requires a free key from developer.brawlstars.com, bound to a whitelisted IP
(Supercell's API has no anonymous/keyless tier). For a dynamic-IP connection
(e.g. no static IP available, common with German ISPs that force a daily
reconnect), set use_royaleapi_proxy=true in cfg/general_config.toml and
whitelist RoyaleAPI's proxy IP (45.79.218.79) on the key ONCE instead - it
never changes even though your own IP does. Verified directly against
RoyaleAPI/cr-api-docs/docs/proxy.md: same auth header and endpoint paths,
only the domain differs (api.brawlstars.com -> bsproxy.royaleapi.dev).
https://docs.royaleapi.com/proxy.html

Every function degrades to None/{} on any failure (bad tag, IP not
whitelisted, rate limit, network error) so callers can unconditionally fall
back to the existing OCR/heuristic behavior.
"""
import time
import urllib.parse

import requests

DIRECT_API_BASE = "https://api.brawlstars.com/v1"
ROYALEAPI_PROXY_BASE = "https://bsproxy.royaleapi.dev/v1"
_TIMEOUT = 6.0

# Throttle error logging so a missing/invalid key or wrong tag prints once
# per cooldown instead of spamming every scroll tick / match end.
_ERROR_LOG_COOLDOWN = 120.0
_last_error_log = {"t": 0.0}


def _log_error(message):
    now = time.time()
    if now - _last_error_log["t"] >= _ERROR_LOG_COOLDOWN:
        _last_error_log["t"] = now
        print(f"[bs_api] {message}")


def _normalize_tag(tag):
    if not tag:
        return None
    tag = tag.strip().upper().replace("O", "0")  # common Supercell tag OCR/typo fix
    if not tag.startswith("#"):
        tag = "#" + tag
    return tag


def normalize_brawler_name(name):
    """Matches the squashed-lowercase convention used by cfg/names.json and
    lobby_automation.py's own OCR normalization (e.g. "El Primo" -> "elprimo")."""
    name = str(name).lower().strip()
    for symbol in (" ", "-", ".", "&"):
        name = name.replace(symbol, "")
    return name


def get_player_info(tag, api_key, use_royaleapi_proxy=False):
    """Raw player payload from Supercell's API, or None on any failure.

    use_royaleapi_proxy=True routes through bsproxy.royaleapi.dev instead of
    calling Supercell directly - use this when the requesting machine has no
    static IP, and whitelist 45.79.218.79 (RoyaleAPI's proxy IP, which never
    changes) on the key instead of your own address."""
    if not api_key:
        return None
    clean_tag = _normalize_tag(tag)
    if not clean_tag:
        return None

    base = ROYALEAPI_PROXY_BASE if use_royaleapi_proxy else DIRECT_API_BASE
    url = f"{base}/players/{urllib.parse.quote(clean_tag)}"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        _log_error(f"player lookup failed (network error): {exc}")
        return None

    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError:
            _log_error("player lookup failed: response was not valid JSON")
            return None
    if resp.status_code == 403:
        allowlist_hint = (
            "whitelist 45.79.218.79 (the RoyaleAPI proxy IP) on the key"
            if use_royaleapi_proxy else
            "the API key's IP allowlist likely doesn't include this machine's "
            "current public IP - whitelist it (or switch to "
            "use_royaleapi_proxy if it changes often)"
        )
        _log_error(f"player lookup failed: 403 Forbidden - {allowlist_hint} at developer.brawlstars.com.")
    elif resp.status_code == 404:
        _log_error(f"player lookup failed: 404 - tag '{clean_tag}' not found")
    elif resp.status_code == 429:
        _log_error("player lookup failed: 429 rate limited")
    else:
        _log_error(f"player lookup failed: HTTP {resp.status_code}")
    return None


def get_account_trophies(player_info):
    """Total account trophies across ALL brawlers combined, or None if
    unavailable. NOT the same thing as any single brawler's trophy count -
    do not use this to sync stage_manager's Trophy_observer.current_trophies,
    which tracks one brawler's progress toward its own push target
    (use get_brawler_trophies instead; conflating the two pops brawlers out
    of the push queue the moment the account total exceeds the target)."""
    if not isinstance(player_info, dict):
        return None
    trophies = player_info.get("trophies")
    return int(trophies) if isinstance(trophies, (int, float)) else None


def get_brawler_trophies_map(player_info):
    """{normalized_brawler_name: trophies} for every owned brawler. Supercell's
    API reports the true cumulative trophy count directly (no 1000-reset /
    prestige-badge math needed - that's purely a client-side display quirk),
    so this is a strictly more reliable source than the OCR prestige+offset
    reconstruction in lobby_automation._read_trophy_count."""
    if not isinstance(player_info, dict):
        return {}
    result = {}
    for b in player_info.get("brawlers") or []:
        name = b.get("name")
        trophies = b.get("trophies")
        if name and isinstance(trophies, (int, float)):
            result[normalize_brawler_name(name)] = int(trophies)
    return result


def get_brawler_trophies(player_info, brawler_name):
    """Trophies for one specific brawler, or None if unowned/unavailable.
    Use this (not get_account_trophies) anywhere that's comparing against a
    per-brawler push target - stage_manager's Trophy_observer.current_trophies
    tracks a single brawler's progress, not the account total."""
    return get_brawler_trophies_map(player_info).get(normalize_brawler_name(brawler_name))


def get_brawler_power(player_info, brawler_name):
    """Power level (1-11) of the given brawler, or None if unowned/unavailable."""
    if not isinstance(player_info, dict):
        return None
    target = normalize_brawler_name(brawler_name)
    for b in player_info.get("brawlers") or []:
        if normalize_brawler_name(b.get("name", "")) == target:
            power = b.get("power")
            return int(power) if isinstance(power, (int, float)) else None
    return None
