import time
import numpy as np

import cv2
from utils import (
    EasyOCRInitializationError,
    count_hsv_pixels,
    extract_text_and_positions,
    load_toml_as_dict, load_all_brawlers_names, config_bool,
)


class LobbyAutomation:

    # Static grid the brawler-selection list snaps to, used both to stabilize
    # OCR bounding-box jitter and to locate the trophy/prestige crop regions.
    _TROPHY_GRID_COL_CENTERS = [580, 1110, 1630]
    _TROPHY_GRID_ROW_CENTERS = [440, 872]

    def __init__(self, window_controller):
        self.gray_pixels_treshold = load_toml_as_dict("./cfg/bot_config.toml").get('idle_pixels_minimum', 500)
        self.idle_reconnect_coords = load_toml_as_dict("cfg/buttons_config.toml")["idle_reconnect"]
        self.ocr_scale_down_factor = max(0.5, min(1, load_toml_as_dict("./cfg/general_config.toml").get('ocr_scale_down_factor', 1)))
        self.ocr_scale_up_factor = 1 / self.ocr_scale_down_factor
        self.all_brawlers_names = load_all_brawlers_names()
        self.window_controller = window_controller
        self.verbose_debug = config_bool(load_toml_as_dict("cfg/debug_settings.toml").get('verbose_debug'), False)

    def check_for_idle(self, frame):
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        x_start, x_end = int(460 * wr), int(1460 * wr)
        y_start, y_end = int(400 * hr), int(675 * hr)
        gray_pixels = count_hsv_pixels(frame[y_start:y_end, x_start:x_end], (0, 0, 10), (30, 60, 67))
        if self.verbose_debug: print(f"gray pixels (if > {self.gray_pixels_treshold} then bot will try to unidle) :", gray_pixels)
        if gray_pixels > self.gray_pixels_treshold:
            self.window_controller.click(self.idle_reconnect_coords[0], self.idle_reconnect_coords[1], already_include_ratio=False, blocking=True)
            time.sleep(2)
            print("Idle detected, clicking to unidle")

    @staticmethod
    def _should_interrupt(runtime_control=None, stop_event=None):
        if runtime_control and (runtime_control.should_stop() or runtime_control.should_pause()):
            return True
        return stop_event is not None and stop_event.is_set()

    @staticmethod
    def _sleep_interruptible(duration, runtime_control=None, stop_event=None, poll_interval=0.1):
        end_time = time.time() + duration
        while time.time() < end_time:
            if LobbyAutomation._should_interrupt(runtime_control, stop_event):
                return True
            time.sleep(min(poll_interval, max(end_time - time.time(), 0)))
        return False

    def _read_trophy_count(self, original_screenshot, orig_x, orig_y, wr=1.0, hr=1.0, retries=3, retry_delay=0.35):
        """OCR the trophy count + prestige badge near a brawler icon at the
        given (grid-snapped) screen coordinates. Returns total trophies.

        All offsets/sizes below are calibrated against native 1920x1080 and
        must be scaled by wr/hr (window_controller.width_ratio/height_ratio)
        - without this, any capture resolution other than 1920x1080 (e.g.
        scrcpy_max_width < 1920) crops the wrong region entirely and the OCR
        silently reads nothing, defaulting trophies to 0.

        The brawler-selection list is still animating in for a moment after
        it's opened, so the very first screenshot can catch the trophy badge
        before its digits have rendered. If no digit is found, retry with a
        fresh screenshot rather than silently trusting a blank crop as "0".
        """
        crop_w, crop_h = max(1, int(110 * wr)), max(1, int(50 * hr))
        trophy_offset = None
        for attempt in range(retries):
            xt = max(0, int(orig_x - 260 * wr))
            yt = max(0, int(orig_y - 280 * hr))
            trophy_crop = original_screenshot[yt:yt + crop_h, xt:xt + crop_w]
            # frames from scrcpy are RGB24 (see scrcpy/core.py), not BGR
            gray = cv2.cvtColor(trophy_crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            trophy_results = extract_text_and_positions(thresh)
            for t_text in trophy_results.keys():
                clean_t = ''.join(c for c in t_text if c.isdigit())
                if clean_t:
                    trophy_offset = int(clean_t)
                    break
            if trophy_offset is not None:
                break
            if attempt < retries - 1:
                time.sleep(retry_delay)
                original_screenshot = self.window_controller.screenshot()
        if trophy_offset is None:
            print(f"WARNING: Trophy OCR found no digits after {retries} attempts at ({orig_x},{orig_y}) - defaulting to 0")
            trophy_offset = 0

        xs = int(orig_x - 340 * wr)
        ys = int(orig_y - 255 * hr)
        mx, my = max(1, int(30 * wr)), max(1, int(30 * hr))
        x1, x2 = max(0, xs - mx), min(original_screenshot.shape[1], xs + mx)
        y1, y2 = max(0, ys - my), min(original_screenshot.shape[0], ys + my)
        shield_crop = original_screenshot[y1:y2, x1:x2]

        hsv = cv2.cvtColor(shield_crop, cv2.COLOR_RGB2HSV)
        # Purple badge pixels indicate a prestige (1000+) brawler. Pixel-count
        # threshold scales with crop area (wr*hr) same as the crop itself.
        purple_mask = cv2.inRange(hsv, np.array([130, 100, 50]), np.array([150, 255, 255]))
        purple_pixels = cv2.countNonZero(purple_mask)
        prestige_level = 1 if purple_pixels > 500 * wr * hr else 0

        return (prestige_level * 1000) + trophy_offset

    def select_brawler(self, brawler, get_latest_state, stop_event=None, runtime_control=None, **kwargs):
        self.window_controller.screenshot()
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        brawler = str(brawler).lower().strip()
        for symbol in [' ', '-', '.', "&"]:
            brawler = brawler.replace(symbol, "")

        x, y = load_toml_as_dict("cfg/buttons_config.toml")["brawlers_menu"]
        self.window_controller.click(x, y, already_include_ratio=False, blocking=True)
        time.sleep(0.5)
        c = 0
        print("Automatic brawler selection started for", brawler)
        shop_counter = 0
        
        target_trophies = kwargs.get("target_trophies", 1000)
        
        # Keep track of brawlers to avoid infinite loops in auto mode
        if brawler == "auto":
            if not hasattr(self, "processed_brawlers"):
                self.processed_brawlers = set()
            if not hasattr(self, "unowned_brawlers"):
                self.unowned_brawlers = set()
            # clear processed and unowned brawlers each time we start a new full scan
            self.processed_brawlers = set()
            self.unowned_brawlers = set()

        for i in range(100):
            if self._should_interrupt(runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"
            original_screenshot = self.window_controller.screenshot()
            screenshot = cv2.resize(original_screenshot, (int(original_screenshot.shape[1] * self.ocr_scale_down_factor), int(original_screenshot.shape[0] * self.ocr_scale_down_factor)), interpolation=cv2.INTER_AREA)

            print("Extracting text on current screen...")
            try:
                results = extract_text_and_positions(screenshot)
            except EasyOCRInitializationError as exc:
                raise RuntimeError(
                    f"Automatic brawler selection could not start OCR: {exc}"
                ) from exc
            except Exception as exc:
                print(f"WARNING: Automatic brawler selection could not read this screen with OCR: {exc}")
                print("The bot will continue without changing the currently selected brawler.")
                return "error"
            results = {k: v for k, v in results.items() if len(k) >= 2}
            clean_results = {}
            for key in results.keys():
                orig_key = key
                for symbol in [' ', '-', '.', "&"]:
                    key = key.replace(symbol, "")
                clean_results[key.lower()] = results[orig_key]

            current_state = get_latest_state()
            if "shop" in clean_results.keys():
                print("Latest screenshot is still of the lobby, waiting for the frame to update...")
                shop_counter += 1
                if shop_counter > 5:
                    print("WARNING: The bot has been waiting for the lobby screen to update for a long time. It's possible that the game is stuck or the OCR is having trouble reading the screen. The bot will continue without changing the currently selected brawler.")
                    return "stuck"
                continue
            elif current_state != "brawler_selection":
                print("Latest screenshot is no longer of the lobby, aborting brawler selection...")
                return "stuck"
            if brawler == "auto":
                # Find banner
                banner_y = float('inf')
                for text, details in clean_results.items():
                    if "to be unlocked" in text.lower() or "available on the" in text.lower():
                        banner_y = details['center'][1]
                        break
                        
                # Filter valid brawlers
                valid_brawlers = {}
                for detected_name, details in clean_results.items():
                    actual_name = None
                    if detected_name in self.all_brawlers_names:
                        actual_name = detected_name
                    else:
                        for official_name, aliases in self.all_brawlers_names.items():
                            if detected_name in aliases:
                                actual_name = official_name
                                break
                    if actual_name:
                        if details['center'][1] > banner_y:
                            self.unowned_brawlers.add(actual_name)
                            continue
                        if actual_name not in self.processed_brawlers and actual_name not in self.unowned_brawlers:
                            valid_brawlers[actual_name] = details
                
                if not valid_brawlers:
                    print("No unprocessed brawlers on screen. Scrolling...")
                else:
                    sorted_brawlers = sorted(valid_brawlers.items(), key=lambda item: (item[1]['center'][1] // 50, item[1]['center'][0]))
                    for actual_name, details in sorted_brawlers:
                        x, y = details['center']
                        orig_x = int(x * self.ocr_scale_up_factor)
                        orig_y = int(y * self.ocr_scale_up_factor)
                        
                        # Snap to static grid coordinates to prevent OCR bounding-box variance.
                        # Grid constants are calibrated at native 1920x1080, scale by wr/hr.
                        col_options = [c * wr for c in self._TROPHY_GRID_COL_CENTERS]
                        row_options = [c * hr for c in self._TROPHY_GRID_ROW_CENTERS]
                        snap_x = min(col_options, key=lambda cx: abs(cx - orig_x))
                        snap_y = min(row_options, key=lambda cy: abs(cy - orig_y))
                        print(f"[OCR] Brawler {actual_name} center ({orig_x}, {orig_y}) snapped to grid ({snap_x:.0f}, {snap_y:.0f})")
                        orig_x, orig_y = snap_x, snap_y

                        # Check for green power bar below name to verify ownership
                        bar_w, bar_h = max(1, int(120 * wr)), max(1, int(50 * hr))
                        xb = max(0, int(orig_x - 60 * wr))
                        yb = max(0, int(orig_y + 10 * hr))
                        bar_crop = original_screenshot[yb:yb+bar_h, xb:xb+bar_w]
                        hsv_bar = cv2.cvtColor(bar_crop, cv2.COLOR_RGB2HSV)
                        # "Not grey" check: Colorful power bars (green, pink, gold) have very high saturation (>110)
                        # Unowned brawlers have no bar, just the dark blue/grey background (saturation < 90)
                        mask = cv2.inRange(hsv_bar, np.array([0, 110, 50]), np.array([179, 255, 255]))
                        if cv2.countNonZero(mask) < 1500 * wr * hr:
                            print(f"Skipping {actual_name} - No colorful power bar found (likely unowned)")
                            self.unowned_brawlers.add(actual_name)
                            continue

                        total_trophies = self._read_trophy_count(original_screenshot, orig_x, orig_y, wr, hr)

                        if total_trophies < target_trophies:
                            y_click = y - (50 * self.ocr_scale_down_factor)
                            self.window_controller.click(int(x * self.ocr_scale_up_factor), int(y_click * self.ocr_scale_up_factor), blocking=True)
                            if self._sleep_interruptible(1.5, runtime_control, stop_event): return "aborted"
                            
                            # We already verified ownership via the power bar saturation check, so we can safely select
                            select_x, select_y = load_toml_as_dict("cfg/buttons_config.toml")["select_brawler"]
                            self.window_controller.click(select_x, select_y, already_include_ratio=False, blocking=True)
                            if self._sleep_interruptible(1.5, runtime_control, stop_event): return "aborted"
                            self.window_controller.screenshot()
                            print(f"Selected brawler {actual_name} with {total_trophies} trophies.")
                            return ("success", actual_name, total_trophies)
                        else:
                            print(f"Skipping {actual_name} - {total_trophies} >= {target_trophies}")
                            self.processed_brawlers.add(actual_name)
                            
                # If we processed brawlers and didn't select, continue to scroll
            else:
                if brawler in clean_results.keys():
                    matched_key = brawler
                else:
                    matched_key = None
                    for detected_name in clean_results.keys():
                        if detected_name in self.all_brawlers_names[brawler]:
                            matched_key = detected_name
                            print(f"Matched detected name '{detected_name}' to brawler '{brawler}' using alias list.")
                            break
    
                if self.verbose_debug:
                    print("OCR detected the following potential matches for the brawler name:")
                    import difflib
                    for detected_name in clean_results.keys():
                        match_ratio = difflib.SequenceMatcher(None, detected_name, brawler).ratio()
                        if match_ratio >= 0.25:
                            print(f" - '{detected_name}' with match ratio {match_ratio:.2f}")
                if matched_key:
                    x, y = clean_results[matched_key]['center']

                    # Read the actual trophy count off the card (grid-snapped,
                    # same as the "auto" path) before clicking - previously
                    # this branch always returned 0, which overwrote the real
                    # trophy count in the UI on every brawler selection.
                    orig_x = int(x * self.ocr_scale_up_factor)
                    orig_y = int(y * self.ocr_scale_up_factor)
                    col_options = [c * wr for c in self._TROPHY_GRID_COL_CENTERS]
                    row_options = [c * hr for c in self._TROPHY_GRID_ROW_CENTERS]
                    snap_x = min(col_options, key=lambda cx: abs(cx - orig_x))
                    snap_y = min(row_options, key=lambda cy: abs(cy - orig_y))
                    try:
                        total_trophies = self._read_trophy_count(original_screenshot, snap_x, snap_y, wr, hr)
                        print(f"[OCR] Brawler {brawler} center ({orig_x}, {orig_y}) snapped to grid ({snap_x:.0f}, {snap_y:.0f}), read {total_trophies} trophies")
                    except Exception as exc:
                        print(f"WARNING: Trophy OCR failed for {brawler}: {exc}")
                        total_trophies = None

                    y_offset = 50*self.ocr_scale_down_factor
                    y -= y_offset
                    self.window_controller.click(int(x * self.ocr_scale_up_factor), int(y * self.ocr_scale_up_factor), blocking=True)
                    print(f"Found brawler {brawler} ({matched_key}) clicking on its icon at {int(x * self.ocr_scale_up_factor)} {int(y * self.ocr_scale_up_factor)}")
                    if self._sleep_interruptible(1, runtime_control, stop_event):
                        print("Brawler selection aborted by user.")
                        return "aborted"
                    select_x, select_y = load_toml_as_dict("cfg/buttons_config.toml")["select_brawler"]
                    self.window_controller.click(select_x, select_y, already_include_ratio=False, blocking=True)
                    if self._sleep_interruptible(1.5, runtime_control, stop_event):
                        print("Brawler selection aborted by user.")
                        return "aborted"
                    self.window_controller.screenshot()
                    print("Selected brawler ", brawler)
                    return ("success", brawler, total_trophies)
                else:
                    print("Brawler name not found on screen, scrolling down to load more brawlers...")

            if c == 0:
                wr = self.window_controller.width_ratio
                hr = self.window_controller.height_ratio
                self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(850 * hr), duration=0.5)
                if self._sleep_interruptible(3, runtime_control, stop_event):
                    print("Brawler selection aborted by user.")
                    return "aborted"
                c += 1
                continue

            self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(650 * hr), duration=0.5)
            if self._sleep_interruptible(3, runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"

        print(f"WARNING: Brawler '{brawler}' was not found after 100 scroll attempts.")
        return "failed"
