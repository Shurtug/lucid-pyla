import sys
import time
import cv2

from state_finder import get_state
from trophy_observer import TrophyObserver, MatchResult
from utils import find_template_center, load_toml_as_dict, notify_user, save_brawler_data, config_bool, load_general_config
from bs_official_api import get_player_info as bs_get_player_info, get_brawler_trophies, get_brawler_power

try:
    from early_access.early_access import get_brawler_stats, get_player_info

    early_access = True
except (ImportError, ModuleNotFoundError):
    early_access = False


    def get_brawler_stats(_player_info, _brawler_name, _power_level=False):
        return None, None


    def get_player_info(_tag):
        return None


def load_image(image_path, scale_factor):
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]

    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


class StageManager:
    def __init__(self, brawlers_data, lobby_automator, window_controller, playstyle_info, state_getting, runtime_control=None):
        self.Lobby_automation = lobby_automator
        self.lobby_config = load_toml_as_dict("./cfg/lobby_config.toml")
        self.close_popup_icon = None
        self.brawlers_pick_data = brawlers_data
        self.Trophy_observer = TrophyObserver()
        self.time_since_last_stat_change = time.time()
        self.play_again_on_win = load_toml_as_dict("./cfg/bot_config.toml")["play_again_on_win"] == "yes"
        self.window_controller = window_controller
        self.states = {
            'shop': self.quit_shop,
            'brawler_selection': self.quit_shop,
            'popup': self.close_pop_up,
            'match': lambda: 0,
            'match_making': lambda: 0,
            'lobby': self.start_game,
            'star_drop_regular': lambda: self.click_star_drop("regular"),
            'star_drop_angelic': lambda: self.click_star_drop("angelic"),
            'star_drop_demonic': lambda: self.click_star_drop("demonic"),
            'star_drop_starr_nova': lambda: self.click_star_drop("starr_nova"),
            'trophy_reward': lambda: self.window_controller.press("proceed"),
            'prestige_milestone': lambda: self.window_controller.press("continue_or_equip"),
            'end_draw': self.end_game,
            'end_victory': self.end_game,
            'end_defeat': self.end_game,
            'end_trio_showdown_0': self.end_game,
            'end_trio_showdown_1': self.end_game,
            'end_trio_showdown_2': self.end_game,
            'end_trio_showdown_3': self.end_game,
            'nano_noodles': self.click_nano_noodles,
        }
        self.matches_since_last_webhook_ping = 0
        self.ping_every_x_match = load_toml_as_dict("cfg/webhook_config.toml")['ping_every_x_match']
        self.runtime_control = runtime_control
        general_cfg = load_general_config()
        # player_tag used to be loaded only for the paid early_access module;
        # the official-API path (bs_official_api.py) needs it too, so it's
        # now always loaded (empty string is a harmless default either way).
        self.player_tag = general_cfg.get('player_tag', '')
        self.brawlstars_api_key = general_cfg.get('brawlstars_api_key', '')
        self.use_royaleapi_proxy = config_bool(general_cfg.get('use_royaleapi_proxy'), False)
        self.ping_when_stuck = load_toml_as_dict("cfg/webhook_config.toml")["ping_when_stuck"]
        self.playstyle_info = playstyle_info
        self.get_latest_state = state_getting

    def _should_stop(self):
        return bool(self.runtime_control and self.runtime_control.should_stop())

    def _should_pause(self):
        return bool(self.runtime_control and self.runtime_control.should_pause())

    def _sleep_interruptible(self, duration, allow_pause=True, poll_interval=0.1):
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._should_stop():
                return True
            if allow_pause and self._should_pause():
                return True
            time.sleep(min(poll_interval, max(end_time - time.time(), 0)))
        return False

    @staticmethod
    def validate_trophies(trophies_string):
        trophies_string = trophies_string.lower()
        while "s" in trophies_string:
            trophies_string = trophies_string.replace("s", "5")
        numbers = ''.join(filter(str.isdigit, trophies_string))

        if not numbers:
            return False

        trophy_value = int(numbers)
        return trophy_value

    def start_game(self):
        if self._should_stop() or self._should_pause():
            return

        if early_access and self.player_tag:
            print("Waiting 3 seconds for API to update with latest data...")
            time.sleep(3)
            player_info = get_player_info(self.player_tag)
            if not player_info:
                print("Player tag is incorrect. Use your Brawl Stars player tag, not your Supercell ID. Skipping API stat refresh.")
            else:
                current_brawler = self.brawlers_pick_data[0]['brawler']
                trophies, win_streak = get_brawler_stats(player_info, current_brawler)
                if trophies is not None and win_streak is not None:
                    if trophies != self.Trophy_observer.current_trophies or win_streak != self.Trophy_observer.win_streak:
                        print(f"Warning: Trophies or win streak from API do not match current values. This may indicate a desync. API values: trophies={trophies}, win_streak={win_streak}. Current values: trophies={self.Trophy_observer.current_trophies}, win_streak={self.Trophy_observer.win_streak}")
                    self.Trophy_observer.current_trophies = trophies
                    self.Trophy_observer.win_streak = win_streak
        elif self.brawlstars_api_key and self.player_tag:
            # Same idea via Supercell's official API directly, independent of
            # the paid early_access module. current_trophies tracks ONE
            # brawler's progress toward its own push_until target (seeded
            # from brawlers_pick_data[0]['trophies'] in main.py), not the
            # account total - get_brawler_trophies mirrors the early_access
            # branch above (get_brawler_stats), just without win_streak,
            # which Supercell's API doesn't expose.
            print("Waiting 3 seconds for API to update with latest data...")
            time.sleep(3)
            player_info = bs_get_player_info(self.player_tag, self.brawlstars_api_key, self.use_royaleapi_proxy)
            if not player_info:
                print("Brawl Stars API lookup failed - check brawlstars_api_key and player_tag. Skipping API stat refresh.")
            else:
                current_brawler = self.brawlers_pick_data[0]['brawler']
                trophies = get_brawler_trophies(player_info, current_brawler)
                if trophies is not None:
                    if self.Trophy_observer.current_trophies is not None and trophies != self.Trophy_observer.current_trophies:
                        print(f"Warning: {current_brawler}'s trophies from API ({trophies}) do not match current tracked value ({self.Trophy_observer.current_trophies}). This may indicate a desync. Correcting to the API value.")
                    self.Trophy_observer.current_trophies = trophies
        print("state is lobby, starting game")
        values = {
            "trophies": self.Trophy_observer.current_trophies,
            "wins": self.Trophy_observer.current_wins
        }

        type_of_push = self.brawlers_pick_data[0]['type']
        value = values[type_of_push]
        push_current_brawler_till = self.brawlers_pick_data[0]['push_until']

        if value >= push_current_brawler_till:
            if len(self.brawlers_pick_data) <= 1:
                print("Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                      "Bot will now pause itself until closed.", value, push_current_brawler_till)
                screenshot = self.window_controller.screenshot()
                notify_user("completed", screenshot, self)
                print("Bot stopping: all targets completed with no more brawlers.")
                self.window_controller.release_movement()
                self.window_controller.close()
                sys.exit(0)
            ping_when_target_is_reached = load_toml_as_dict("cfg/webhook_config.toml")["ping_when_target_is_reached"]
            if ping_when_target_is_reached:
                screenshot = self.window_controller.screenshot()
                notify_user("brawler_goal", screenshot, self)
            print(f'Bot has reached the target trophies/wins for {self.brawlers_pick_data[0]["brawler"]}, moving on to the next one in the list.', value, push_current_brawler_till)
            self.brawlers_pick_data.pop(0)
            next_brawler_name = self.brawlers_pick_data[0]['brawler']
            if self.brawlers_pick_data[0]["automatically_pick"]:
                target_trophies = self.brawlers_pick_data[0].get("push_until", 0)
                select_brawler_res = self.Lobby_automation.select_brawler(next_brawler_name, self.get_latest_state, runtime_control=self.runtime_control, target_trophies=target_trophies)
                
                if isinstance(select_brawler_res, tuple):
                    status, act_brawler, act_trophies = select_brawler_res
                    if status == "completed":
                        print("Auto mode has completed all brawlers. Stopping.")
                        self.brawlers_pick_data.pop(0)
                        if not self.brawlers_pick_data:
                            self.window_controller.release_movement()
                            self.window_controller.close()
                            import sys
                            sys.exit(0)
                        return
                    if status == "success":
                        self.brawlers_pick_data[0]["brawler"] = act_brawler
                        self.brawlers_pick_data[0]["trophies"] = act_trophies if act_trophies is not None else self.brawlers_pick_data[0]["trophies"]
                        if next_brawler_name == "auto":
                            auto_entry = self.brawlers_pick_data[0].copy()
                            auto_entry["brawler"] = "auto"
                            self.brawlers_pick_data.insert(1, auto_entry)
                else:
                    status = select_brawler_res

                while status in ["failed", "error"]:
                    if self.ping_when_stuck:
                        screenshot = self.window_controller.screenshot()
                        notify_user("bot_failed_brawler_selection", screenshot, self)
                        print(f"Skipping {status}")
                    if self._should_stop() or self._should_pause():
                        return
                    
                    if next_brawler_name == "auto" and status == "failed":
                        print("Auto mode reached bottom of list, ending.")
                        self.brawlers_pick_data.pop(0)
                        if not self.brawlers_pick_data:
                            self.window_controller.release_movement()
                            self.window_controller.close()
                            import sys
                            sys.exit(0)
                        return

                    current_brawler = self.brawlers_pick_data.pop(0)
                    self.brawlers_pick_data.append(current_brawler)
                    next_brawler_name = self.brawlers_pick_data[0]['brawler']
                    self.quit_shop()
                    
                    target_trophies = self.brawlers_pick_data[0].get("push_until", 0)
                    select_brawler_res = self.Lobby_automation.select_brawler(next_brawler_name, self.get_latest_state, runtime_control=self.runtime_control, target_trophies=target_trophies)
                    if isinstance(select_brawler_res, tuple):
                        status, act_brawler, act_trophies = select_brawler_res
                        if status == "success":
                            self.brawlers_pick_data[0]["brawler"] = act_brawler
                            self.brawlers_pick_data[0]["trophies"] = act_trophies if act_trophies is not None else self.brawlers_pick_data[0]["trophies"]
                            if next_brawler_name == "auto":
                                auto_entry = self.brawlers_pick_data[0].copy()
                                auto_entry["brawler"] = "auto"
                                self.brawlers_pick_data.insert(1, auto_entry)
                    else:
                        status = select_brawler_res

                if status == "aborted" or status == "stuck":
                    return
                if status == "success":
                    self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
                    self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
                    self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
            else:
                self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
                self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
                self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
                print("Next brawler is in manual mode, waiting 10 seconds to let user switch.")
                if self._sleep_interruptible(10):
                    return
        save_brawler_data(self.brawlers_pick_data)
        self.matches_since_last_webhook_ping += 1
        if self.ping_every_x_match and self.matches_since_last_webhook_ping >= self.ping_every_x_match:
            screenshot = self.window_controller.screenshot()
            notify_user("regular_matches_ping", screenshot, self)
            self.matches_since_last_webhook_ping = 0

        if self._should_stop() or self._should_pause():
            return
        self.window_controller.release_movement()
        self.window_controller.press("proceed")
        print("Pressed to start a match")
        time.sleep(2)

    def click_star_drop(self, drop_type="regular"):
        if hasattr(self, '_star_drop_thread') and self._star_drop_thread.is_alive():
            return

        def _handle_drop():
            if drop_type in ["angelic", "demonic", "starr_nova"]:
                self.window_controller.press("proceed", 8)
            else:
                for _ in range(8):
                    self.window_controller.press("proceed", 0.05)
                    time.sleep(0.1)

        import threading
        self._star_drop_thread = threading.Thread(target=_handle_drop, daemon=True)
        self._star_drop_thread.start()

    def click_nano_noodles(self):
        noodle_x, noodle_y = 960, 740
        offset_x = 330 * self.window_controller.width_ratio
        self.window_controller.click(
            noodle_x,
            noodle_y,
            already_include_ratio=False
        )
        time.sleep(0.1)
        self.window_controller.click(
            noodle_x + offset_x,
            noodle_y,
            already_include_ratio=False
        )
        time.sleep(0.1)
        self.window_controller.click(
            noodle_x - offset_x,
            noodle_y,
            already_include_ratio=False
        )

    def end_game(self):
        screenshot = self.window_controller.screenshot()

        current_state = get_state(screenshot)
        end_screen_time = time.time()
        parsed_result = None
        while current_state.startswith("end") and time.time() - end_screen_time < 35:

            if parsed_result is None:
                raw_found_result = '_'.join(current_state.split("_")[1:])
                parsed_result = self.Trophy_observer.parse_game_result(raw_found_result)

            if time.time() - self.time_since_last_stat_change > 25:
                current_brawler = self.brawlers_pick_data[0]['brawler']
                if early_access:
                    power_level = get_brawler_stats(get_player_info(self.player_tag), current_brawler, power_level=True)[2]
                elif self.brawlstars_api_key and self.player_tag:
                    power_level = get_brawler_power(bs_get_player_info(self.player_tag, self.brawlstars_api_key, self.use_royaleapi_proxy), current_brawler)
                else:
                    power_level = None
                self.Trophy_observer.add_trophies(parsed_result, current_brawler, self.playstyle_info, power_level)
                self.Trophy_observer.add_win(parsed_result)
                self.time_since_last_stat_change = time.time()
                values = {
                    "trophies": self.Trophy_observer.current_trophies,
                    "wins": self.Trophy_observer.current_wins
                }
                type_to_push = self.brawlers_pick_data[0]['type']
                value = values[type_to_push]
                self.brawlers_pick_data[0][type_to_push] = value
                self.brawlers_pick_data[0]['win_streak'] = self.Trophy_observer.win_streak
                save_brawler_data(self.brawlers_pick_data)

            if self.play_again_on_win and parsed_result and parsed_result.result == MatchResult.VICTORY and not self._should_pause() and not self._should_stop():
                print("Pressing Play Again...")
                self.window_controller.press("play_again")
            else:
                print("Game has ended, proceeding")
                self.window_controller.press("proceed")

            time.sleep(3)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)

        if self.play_again_on_win and parsed_result and parsed_result.result == MatchResult.VICTORY and not self._should_pause():
            print("Waiting for match to start...")
            start_wait_time = time.time()
            while time.time() - start_wait_time < 25:
                if self._should_stop() or self._should_pause():
                    break
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                if current_state == "match":
                    print("Match started successfully!")
                    return
                if self._sleep_interruptible(0.5):
                    break

            print("Match did not start within 25s, restarting the game.")
            self.window_controller.restart_brawl_stars()
            time.sleep(2)
        elif time.time() - end_screen_time > 35:
            print("End screen timeout reached, restarting the game.")
            self.window_controller.restart_brawl_stars()
        print("Game has ended", current_state)

    def quit_shop(self):
        self.window_controller.click(100 * self.window_controller.width_ratio, 60 * self.window_controller.height_ratio)
        time.sleep(1)

    def close_pop_up(self):
        screenshot = self.window_controller.screenshot()
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image("images/states/close_popup.png", self.window_controller.scale_factor)
        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)

    def do_state(self, state, data=None):
        if data is not None:
            self.states[state](data)
            return
        self.states[state]()
