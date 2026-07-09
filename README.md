# lucid pyla

A fork of [PylaAI](https://pylaai.com/) — a Brawl Stars bot — with improved combat AI, perception, and a live web debug view.

## What's improved over stock Pyla

- **Predictive aim** — per-enemy velocity tracking with confidence scoring: tracks survive detection dropouts (bushes/occlusion), ID swaps are rejected, leads are confidence-scaled, deadbanded for stationary targets, and clamped so the bot never leads into a wall.
- **HP tracking** — reads the health bar down to ~5% HP via cached bar geometry + column scanning (stock contour detection went blind at low HP).
- **Ammo tracking** — segment-aware: learns the ammo bar layout per brawler and reports exact segments ready, plus a normalized 0–1 fill.
- **Apex playstyle** — decisive kite/hold/chase combat with predictive supers, cover seeking, gas avoidance, and per-brawler gadget modes (`playstyles/apex.pyla`).
- **Live debug overlay** — predicted aim points with confidence labels, HP/ammo readout, range circles, wall boxes.

## Setup (from source — recommended)

1. **Clone** this repo.
2. **Download `models.zip`** from [Releases](../../releases) and extract it into the repo root, so you have a `models/` folder containing `mainInGameModel.onnx`, `tileDetector.onnx`, `closeTileDetector.onnx`, and `easyocr/`.
3. Install **Python 3.11** (or have [uv](https://docs.astral.sh/uv/) on PATH — the launcher prefers it).
4. Double-click **`run.bat`** — the first run creates a virtual environment and installs the pinned dependencies, then starts the bot.

The web UI (settings + debug view) opens in your browser automatically.

## Setup (prebuilt exe)

Download `lucid-pyla-win64.zip` from [Releases](../../releases), extract anywhere, and run **`start.bat`**. No Python required.

## Requirements

- Windows 10/11
- An Android emulator with ADB (BlueStacks recommended) running Brawl Stars
- A GPU helps — ONNX inference uses DirectML

## Configuration

Everything lives in `cfg/*.toml` and is editable from the web UI. Playstyles are hot-reloaded from `playstyles/` — set `current_playstyle` in `cfg/bot_config.toml`.

## Credits

Based on [PylaAI](https://pylaai.com/) v0.8.14. This fork adds the combat/perception improvements listed above.
