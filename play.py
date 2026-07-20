import math
import random
import time
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

from detect import Detect
try:
    from early_access.early_access import add_advanced_visuals
    early_access = True
except ImportError:
    early_access = False
    def add_advanced_visuals(a, b):
        return None
from state_finder import get_state
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info, interpret_pyla_code, \
    count_mask_pixels, JOYSTICK_RADIUS, clamp, config_bool, load_pyla_script, resolve_project_path


brawl_stars_width, brawl_stars_height = 1920, 1080
super_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['super']
gadget_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['gadget']
hypercharge_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['hypercharge']
POISON_LOW_HSV = np.array((30, 90, 221), dtype=np.uint8)
POISON_HIGH_HSV = np.array((57, 114, 235), dtype=np.uint8)
PLAYER_HIT_CIRCLE_RADIUS = 53

class GridPathfinder:
    def __init__(self, walls, width, height, cell_size=35):
        self.cell_size = cell_size
        self.cols = int(width // cell_size) + 1
        self.rows = int(height // cell_size) + 1
        self.blocked = set()
        
        # Mark cells inside walls as blocked
        for wall in walls:
            x1, y1, x2, y2 = wall[:4]
            # Safety margin so player hitbox doesn't scrape or get stuck on corners
            buffer = 20
            c1 = max(0, int((x1 - buffer) // cell_size))
            r1 = max(0, int((y1 - buffer) // cell_size))
            c2 = min(self.cols - 1, int((x2 + buffer) // cell_size))
            r2 = min(self.rows - 1, int((y2 + buffer) // cell_size))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    self.blocked.add((c, r))

    def find_path(self, start_pos, target_pos):
        sc = max(0, min(self.cols - 1, int(start_pos[0] // self.cell_size)))
        sr = max(0, min(self.rows - 1, int(start_pos[1] // self.cell_size)))
        tc = max(0, min(self.cols - 1, int(target_pos[0] // self.cell_size)))
        tr = max(0, min(self.rows - 1, int(target_pos[1] // self.cell_size)))
        
        if (sc, sr) == (tc, tr):
            return [(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1])]
            
        # BFS search to find shortest path
        queue = [(sc, sr)]
        came_from = {(sc, sr): None}
        found = False
        
        # Directions: 8-way movement
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        while queue:
            curr = queue.pop(0)
            if curr == (tc, tr):
                found = True
                break
                
            for dc, dr in dirs:
                nxt = (curr[0] + dc, curr[1] + dr)
                if 0 <= nxt[0] < self.cols and 0 <= nxt[1] < self.rows:
                    if nxt not in self.blocked and nxt not in came_from:
                        came_from[nxt] = curr
                        queue.append(nxt)
                        
        if not found:
            # Fallback: if target cell is blocked, find the closest unblocked cell to target
            best_cell = None
            best_dist = float('inf')
            for cell in came_from:
                d = (cell[0] - tc)**2 + (cell[1] - tr)**2
                if d < best_dist:
                    best_dist = d
                    best_cell = cell
            if best_cell:
                tc, tr = best_cell
            else:
                return [(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1])]
                
        # Reconstruct path
        curr = (tc, tr)
        path_cells = []
        while curr is not None:
            path_cells.append(curr)
            curr = came_from[curr]
        path_cells.reverse()
        
        # Convert path cells to screen coordinate steps (looking 3 steps ahead to smooth movement)
        steps = []
        for cell in path_cells[1:4]:
            cell_center_x = cell[0] * self.cell_size + self.cell_size / 2
            cell_center_y = cell[1] * self.cell_size + self.cell_size / 2
            steps.append((cell_center_x - start_pos[0], cell_center_y - start_pos[1]))
        if not steps:
            steps = [(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1])]
        return steps

class Play:

    def __init__(self, main_info_model, tile_detector_model, close_tile_detector_model, window_controller, pyla_code):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config["unstuck_movement_delay"],
            "duration": bot_config["unstuck_movement_hold_time"],
            "toggled": False,
            "started_at": time.time(),
            "fixed": (0, 0),
            "last_direction_key": None,
            "rotation_sign": 1,
            "rotation_angle_step": 1,
            "max_rotation_angle_step": 4,
        }
        self.super_treshold = time_config["super"]
        self.gadget_treshold = time_config["gadget"]
        self.hypercharge_treshold = time_config["hypercharge"]
        self.walls_treshold = time_config["wall_detection"]
        self.last_walls_data = []
        self.last_bushes_data = []
        self.keys_hold = []
        self.time_since_different_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.time_since_super_checked = time.time()
        self.is_super_ready = False
        self.window_controller = window_controller
        self.TILE_SIZE = bot_config.get("perceived_tile_size", 54)
        self.centered_wall_detection = config_bool(bot_config.get("centered_wall_detection"), False)
        self.centered_wall_crop_size = 640

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.verbose_debug = config_bool(load_toml_as_dict("cfg/debug_settings.toml").get('verbose_debug'), False)
        if self.verbose_debug:
            if not os.path.exists("debug_frames"):
                os.makedirs("debug_frames")
        self.Detect_main_info = Detect(main_info_model, classes=['enemy', 'teammate', 'player'])
        self.tile_detector_model_classes = bot_config["wall_model_classes"]
        self.Detect_tile_detector = None if self.centered_wall_detection else Detect(
            tile_detector_model,
            classes=self.tile_detector_model_classes
        )
        self.Detect_centered_tile_detector = Detect(
            close_tile_detector_model,
            classes=self.tile_detector_model_classes
        ) if self.centered_wall_detection else None

        self.time_since_walls_checked = 0
        self.time_since_player_last_found = time.time()
        self.current_brawler = None
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement = ''
        self.last_movement_change_time = time.time()
        self.minimum_movement_delay = bot_config["minimum_movement_delay"]
        self.no_detection_proceed_delay = time_config["no_detection_proceed"]
        self.gadget_pixels_minimum = bot_config["gadget_pixels_minimum"]
        self.hypercharge_pixels_minimum = bot_config["hypercharge_pixels_minimum"]
        self.super_pixels_minimum = bot_config["super_pixels_minimum"]
        self.wall_detection_confidence = bot_config["wall_detection_confidence"]
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self.seconds_to_hold_attack_after_reaching_max = load_toml_as_dict("cfg/bot_config.toml")["seconds_to_hold_attack_after_reaching_max"]
        self.persistent_data = {"time_since_holding_attack": None}
        self.pyla_code = pyla_code
        self.context = None
        self.frame = None
        self.aim_leading = config_bool(load_toml_as_dict("cfg/bot_config.toml").get("aim_leading"), True)
        self._enemy_tracks = []            # short-lived tracks: {'pos','vel','t'}
        self._enemy_velocities = []        # aligned to current enemy_data order, (vx,vy) px/s
        self._enemy_curr_centers = []      # aligned to current enemy_data order
        self._player_screen_center = None
        self._active_playstyle_file = load_toml_as_dict("cfg/bot_config.toml").get("current_playstyle", "default_up.pyla")
        self._active_playstyle_code = pyla_code
        self._last_reload_check = 0.0
        self._pending_gadget = False
        # compile once, not every tick: exec() on a raw string recompiles the
        # whole playstyle script each frame (~2ms for apex)
        self._pyla_compiled = None
        try:
            self._pyla_compiled = compile(pyla_code, self._active_playstyle_file, "exec")
        except Exception:
            pass
        # inference pipelining: run detection for the next frame in a worker
        # thread while the main thread does CV/playstyle work on this one.
        # ONNX releases the GIL during run(), so the overlap is real.
        self.pipeline_inference = config_bool(
            load_toml_as_dict("cfg/general_config.toml").get("pipeline_inference"), True)
        self._detect_executor = ThreadPoolExecutor(max_workers=1) if self.pipeline_inference else None
        self._pending_detection = None  # (frame, future) awaiting processing
        self._pending_tiles = None      # future for an in-flight tile/wall inference

    @staticmethod
    def get_entity_pos(entity):
        return (entity[0] + entity[2]) / 2, (entity[1] + entity[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    def attack(self, touch_up=True, touch_down=True, aim=None, distance_ratio=1.0):
        if (aim is not None and self.aim_leading and touch_up and touch_down
                and self._player_screen_center is not None):
            dx = aim[0] - self._player_screen_center[0]
            dy = aim[1] - self._player_screen_center[1]
            # Correct for tilted perspective vertical compression
            dy_corr = dy / 0.85
            self.window_controller.aim_swipe("attack", dx, dy_corr, distance_ratio=distance_ratio)
            return
        self.window_controller.press("attack", touch_up=touch_up, touch_down=touch_down)

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press("hypercharge")
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False

    def use_gadget(self):
        print("Queueing gadget")
        self._pending_gadget = True

    def use_super(self, aim=None, distance_ratio=1.0):
        print("Using super")
        if (aim is not None and self.aim_leading and self._player_screen_center is not None):
            dx = aim[0] - self._player_screen_center[0]
            dy = aim[1] - self._player_screen_center[1]
            # Correct for tilted perspective vertical compression
            dy_corr = dy / 0.85
            self.window_controller.aim_swipe("super", dx, dy_corr, duration=0.15, distance_ratio=distance_ratio)
        else:
            self.window_controller.press("super")
        self.time_since_super_checked = time.time()
        self.is_super_ready = False

    @staticmethod
    def get_random_movement():
        random_movement = random.randint(-75, 75), random.randint(-75, 75)
        return random_movement

    @staticmethod
    def movement_to_vector(movement):
        if not isinstance(movement, (tuple, list)) or len(movement) != 2:
            return None

        x, y = movement
        if x is None or y is None:
            return None

        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def rotate_movement(movement, angle_radians):
        x, y = movement
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        return (
            x * cos_angle - y * sin_angle,
            x * sin_angle + y * cos_angle,
        )

    @staticmethod
    def movement_direction_key(movement):
        x, y = movement
        magnitude = math.hypot(x, y)
        if magnitude < 1:
            return None

        angle = math.atan2(y, x)
        return round(angle / (math.pi / 8)) % 16

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()

        movement_vector = self.movement_to_vector(movement)
        if movement_vector is None:
            self.fix_movement_keys["toggled"] = False
            self.fix_movement_keys["last_direction_key"] = None
            self.fix_movement_keys["rotation_sign"] = 1
            self.fix_movement_keys["rotation_angle_step"] = 1
            self.time_since_different_movement = current_time
            return movement

        direction_key = self.movement_direction_key(movement_vector)
        if direction_key is None:
            self.fix_movement_keys["toggled"] = False
            self.fix_movement_keys["last_direction_key"] = None
            self.fix_movement_keys["rotation_sign"] = 1
            self.fix_movement_keys["rotation_angle_step"] = 1
            self.time_since_different_movement = current_time
            return movement_vector

        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False
                self.fix_movement_keys["last_direction_key"] = direction_key
                self.time_since_different_movement = current_time
                return movement_vector

            return self.fix_movement_keys['fixed']

        if self.fix_movement_keys["last_direction_key"] != direction_key:
            self.fix_movement_keys["last_direction_key"] = direction_key
            self.fix_movement_keys["rotation_sign"] = 1
            self.fix_movement_keys["rotation_angle_step"] = 1
            self.time_since_different_movement = current_time

        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            self.fix_movement_keys["rotation_sign"] *= -1
            angle_step = self.fix_movement_keys["rotation_angle_step"]
            rotated_movement = self.rotate_movement(
                movement_vector,
                self.fix_movement_keys["rotation_sign"] * angle_step * math.pi / 4
            )
            if self.fix_movement_keys["rotation_sign"] > 0:
                self.fix_movement_keys["rotation_angle_step"] += 1
                if self.fix_movement_keys["rotation_angle_step"] > self.fix_movement_keys["max_rotation_angle_step"]:
                    self.fix_movement_keys["rotation_angle_step"] = 1

            self.fix_movement_keys['fixed'] = rotated_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            return rotated_movement

        return movement_vector

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info['attack_range']
            safe_range = info['safe_range']
            super_range = info['super_range']
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info[brawler]['ignore_walls_for_attacks']
        elif skill_type == "super":
            return brawlers_info[brawler]['ignore_walls_for_supers']
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        return brawlers_info[brawler]['hold_attack'] > 0

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls):
        if not walls:
            return False

        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x, max_x = min(p1_t[0], p2_t[0]), max(p1_t[0], p2_t[0])
        min_y, max_y = min(p1_t[1], p2_t[1]), max(p1_t[1], p2_t[1])
        margin = 15
        for wall in walls:
            x1, y1, x2, y2 = wall
            # Inflate wall coordinates by margin to account for projectile width
            ix1, iy1, ix2, iy2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin

            if max_x < ix1 or min_x > ix2 or max_y < iy1 or min_y > iy2:
                continue

            rect = (int(ix1), int(iy1), int(ix2 - ix1), int(iy2 - iy1))
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                return True
        return False

    def get_player_hit_circle(self, player_box):
        radius = PLAYER_HIT_CIRCLE_RADIUS * (self.window_controller.scale_factor or 1)
        if player_box and len(player_box) >= 4:
            x1, y1, x2, y2 = player_box[:4]
            return ((x1 + x2) / 2, y2 - radius), radius

        return None, radius

    def get_actual_player_box(self, player_box):
        center, radius = self.get_player_hit_circle(player_box)
        if center is None:
            return None
        return [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]

    @staticmethod
    def point_rect_distance_sq(point, rect):
        x, y = point
        x1, y1, x2, y2 = rect
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return dx * dx + dy * dy

    @staticmethod
    def walls_block_swept_circle(p1, p2, radius, walls):
        if not walls:
            return False

        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x, max_x = min(p1_t[0], p2_t[0]), max(p1_t[0], p2_t[0])
        min_y, max_y = min(p1_t[1], p2_t[1]), max(p1_t[1], p2_t[1])
        radius = int(math.ceil(radius))

        for wall in walls:
            x1, y1, x2, y2 = wall[:4]
            wall_rect = (x1, y1, x2, y2)
            expanded_x1 = int(x1 - radius)
            expanded_y1 = int(y1 - radius)
            expanded_x2 = int(x2 + radius)
            expanded_y2 = int(y2 + radius)

            if max_x < expanded_x1 or min_x > expanded_x2 or max_y < expanded_y1 or min_y > expanded_y2:
                continue

            rect = (
                expanded_x1,
                expanded_y1,
                max(1, expanded_x2 - expanded_x1),
                max(1, expanded_y2 - expanded_y1),
            )
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                radius_sq = radius * radius
                start_distance_sq = Play.point_rect_distance_sq(p1, wall_rect)
                end_distance_sq = Play.point_rect_distance_sq(p2, wall_rect)
                if start_distance_sq <= radius_sq and end_distance_sq > start_distance_sq:
                    continue
                return True

        return False

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        if self.walls_block_line_of_sight(player_pos, enemy_pos, walls):
            return False
        return True

    def find_path(self, start_pos, target_pos, walls):
        pf = GridPathfinder(walls, self.window_controller.width, self.window_controller.height)
        return pf.find_path(start_pos, target_pos)

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_entity_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            return closest_hittable
        elif closest_unhittable:
            return closest_unhittable

        return None, None

    def update_camera_velocity(self, static_data, now):
        """Track static objects to determine camera screen velocity."""
        centers = [self.get_entity_pos(e) for e in static_data]
        prev = getattr(self, '_static_prev_centers', [])
        dt = None if getattr(self, '_static_prev_time', None) is None else (now - self._static_prev_time)
        
        vx, vy = 0.0, 0.0
        if prev and dt and 0 < dt <= 0.5:
            dxs, dys = [], []
            used = [False] * len(prev)
            for c in centers:
                best_i, best_d = -1, float('inf')
                for i, p in enumerate(prev):
                    if used[i]: continue
                    d = self.get_distance(c, p)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i >= 0 and best_d <= 150:
                    used[best_i] = True
                    p = prev[best_i]
                    dxs.append(c[0] - p[0])
                    dys.append(c[1] - p[1])
            if dxs:
                # median instead of mean: robust to a few mis-associated tiles
                vx = float(np.median(dxs)) / dt
                vy = float(np.median(dys)) / dt
        
        self._static_prev_centers = centers
        self._static_prev_time = now
        
        # EMA for camera velocity to smooth out detection jitter
        alpha = 0.3
        if hasattr(self, '_camera_velocity'):
            self._camera_velocity = (
                self._camera_velocity[0] * (1 - alpha) + vx * alpha,
                self._camera_velocity[1] * (1 - alpha) + vy * alpha
            )
        else:
            self._camera_velocity = (vx, vy)

    def update_enemy_tracks(self, enemy_data, now):
        """Estimate per-enemy world velocity (px/s) using short-lived tracks.
        Association is globally greedy (closest pairs first) against each track's
        coasted position, so tracks survive brief detection dropouts (bushes,
        occlusion) instead of resetting to zero velocity."""
        centers = [self.get_entity_pos(e) for e in enemy_data]
        scale = self.window_controller.scale_factor if self.window_controller and self.window_controller.scale_factor else 1.0
        tile = self.TILE_SIZE * scale
        cam_vx, cam_vy = getattr(self, '_camera_velocity', (0.0, 0.0))
        tracks = [t for t in getattr(self, '_enemy_tracks', []) if now - t['t'] <= 0.6]

        # Match detections to tracks globally: sort all candidate pairs by
        # distance and take the closest first. The old per-detection greedy
        # loop could steal another enemy's track when two were close together.
        gate = 6.0 * tile
        pairs = []
        for ci, c in enumerate(centers):
            for ti, tr in enumerate(tracks):
                dt = now - tr['t']
                coast = (tr['pos'][0] + (tr['vel'][0] + cam_vx) * dt,
                         tr['pos'][1] + (tr['vel'][1] + cam_vy) * dt)
                d = self.get_distance(c, coast)
                if d <= gate:
                    pairs.append((d, ci, ti))
        pairs.sort(key=lambda p: p[0])
        match = {}
        used = set()
        for d, ci, ti in pairs:
            if ci in match or ti in used:
                continue
            match[ci] = ti
            used.add(ti)

        vmax = 5.0 * tile   # fastest plausible sustained movement on screen
        spike = 9.0 * tile  # faster than any dash -> ID swap, not motion
        var0 = (2.0 * tile) ** 2  # starting velocity variance for new tracks
        velocities = []
        confidences = []
        new_tracks = []
        for ci, c in enumerate(centers):
            vx, vy = 0.0, 0.0
            hits, var = 0, var0
            ti = match.get(ci)
            if ti is not None:
                tr = tracks[ti]
                dt = now - tr['t']
                hits, var = tr['hits'], tr['var']
                if dt > 0:
                    # world velocity = screen velocity minus camera velocity
                    raw_vx = (c[0] - tr['pos'][0]) / dt - cam_vx
                    raw_vy = (c[1] - tr['pos'][1]) / dt - cam_vy
                    if math.hypot(raw_vx, raw_vy) > spike:
                        # likely an ID swap: keep a damped velocity, distrust it
                        vx, vy = tr['vel'][0] * 0.5, tr['vel'][1] * 0.5
                        hits, var = 1, var0
                    else:
                        alpha = 0.35
                        vx = tr['vel'][0] * (1 - alpha) + raw_vx * alpha
                        vy = tr['vel'][1] * (1 - alpha) + raw_vy * alpha
                        hits += 1
                        # residual between raw and smoothed velocity measures how
                        # noisy/unpredictable this track currently is
                        res = math.hypot(raw_vx - vx, raw_vy - vy)
                        var = var * 0.7 + (res * res) * 0.3
                    speed = math.hypot(vx, vy)
                    if speed > vmax:
                        vx, vy = vx * vmax / speed, vy * vmax / speed
                else:
                    vx, vy = tr['vel']
            velocities.append((vx, vy))
            confidences.append(self._track_confidence(hits, var, tile))
            new_tracks.append({'pos': c, 'vel': (vx, vy), 't': now, 'hits': hits, 'var': var})

        # Keep unmatched tracks alive briefly so a flickering detection resumes
        # its old velocity instead of restarting from zero.
        for ti, tr in enumerate(tracks):
            if ti not in used:
                new_tracks.append(tr)

        self._enemy_tracks = new_tracks
        self._enemy_curr_centers = centers
        self._enemy_velocities = velocities
        self._enemy_confidences = confidences
        return velocities

    @staticmethod
    def _track_confidence(hits, var, tile):
        """0..1 trust in a track's velocity: needs a few consecutive matched
        updates to ramp up, and degrades when the velocity residual is noisy."""
        conf_hits = min(1.0, hits / 3.0)
        # var holds per-frame residual noise; the EMA (alpha=0.35) suppresses it
        # by alpha/(2-alpha) in the smoothed estimate, so judge that instead -
        # plain detection jitter shouldn't tank confidence, erratic dodging should
        est_var = var * 0.21
        sigma_ref = (2.0 * tile) ** 2
        conf_noise = sigma_ref / (sigma_ref + est_var)
        return conf_hits * conf_noise

    def get_enemy_track(self, pos):
        """((vx,vy), confidence) of the tracked enemy nearest to pos."""
        if not self._enemy_curr_centers:
            return (0.0, 0.0), 0.0
        best_i, best_d = -1, float('inf')
        for i, c in enumerate(self._enemy_curr_centers):
            d = self.get_distance(c, pos)
            if d < best_d:
                best_d, best_i = d, i
        if best_i < 0:
            return (0.0, 0.0), 0.0
        confs = getattr(self, '_enemy_confidences', [])
        conf = confs[best_i] if best_i < len(confs) else 0.0
        return self._enemy_velocities[best_i], conf

    def get_enemy_velocity(self, pos):
        """Velocity (vx,vy) of the tracked enemy nearest to pos; (0,0) if none."""
        return self.get_enemy_track(pos)[0]

    def predict_enemy(self, player_pos, enemy_pos, fixed_lead=None):
        """Predicted position of the enemy nearest to enemy_pos.

        The lead is confidence-scaled (an unreliable velocity estimate pulls the
        aim back toward the enemy's current position), deadbanded (near-zero
        leads are detection noise), and clamped so it never crosses a wall the
        enemy can't actually run through."""
        (vx, vy), conf = self.get_enemy_track(enemy_pos)
        scale_factor = self.window_controller.scale_factor if self.window_controller and self.window_controller.scale_factor else 1.0
        tile = self.TILE_SIZE * scale_factor
        vx, vy = vx * conf, vy * conf

        if fixed_lead is not None:
            lead_seconds = fixed_lead
        else:
            speed = 0
            if self.current_brawler and self.brawlers_info:
                info = self.brawlers_info.get(self.current_brawler, {})
                speed = info.get('projectile_speed', 0)

            if speed > 0:
                speed_on_screen = speed * (64.0 / 300.0) * scale_factor
                lead_seconds = self.get_distance(player_pos, enemy_pos) / speed_on_screen
                # The projectile has to reach where the enemy WILL be, not where
                # it is now, so iterate the time-of-flight toward the fixed point.
                for _ in range(2):
                    tgt = (enemy_pos[0] + vx * lead_seconds, enemy_pos[1] + vy * lead_seconds)
                    lead_seconds = self.get_distance(player_pos, tgt) / speed_on_screen
                lead_seconds = min(lead_seconds, 1.0)
            else:
                lead_seconds = 0

        # leads below a third of a tile are indistinguishable from detection
        # jitter - aim straight at the enemy instead of wobbling around it
        if math.hypot(vx, vy) * lead_seconds < 0.35 * tile:
            return (enemy_pos[0], enemy_pos[1])

        led = (enemy_pos[0] + vx * lead_seconds, enemy_pos[1] + vy * lead_seconds)

        # The enemy can't run through walls: clamp the lead to the longest
        # unblocked fraction of its projected path.
        walls = getattr(self, 'last_walls_data', None)
        if walls and self.walls_block_line_of_sight(enemy_pos, led, walls):
            lo, hi = 0.0, 1.0
            for _ in range(4):
                mid = (lo + hi) / 2
                pt = (enemy_pos[0] + (led[0] - enemy_pos[0]) * mid,
                      enemy_pos[1] + (led[1] - enemy_pos[1]) * mid)
                if self.walls_block_line_of_sight(enemy_pos, pt, walls):
                    hi = mid
                else:
                    lo = mid
            led = (enemy_pos[0] + (led[0] - enemy_pos[0]) * lo,
                   enemy_pos[1] + (led[1] - enemy_pos[1]) * lo)

        return led

    def find_closest_teammate(self, teammate_data, player_coords, walls):
        closest_distance = float('inf')
        closest_teammate = None
        for teammate in teammate_data:
            teammate_pos = self.get_entity_pos(teammate)
            distance = self.get_distance(teammate_pos, player_coords)
            if distance < closest_distance:
                closest_distance = distance
                closest_teammate = teammate_pos
        return closest_teammate, closest_distance

    def is_there_poison_gas(self, player_data, threshold=7000, area_from_player_checked=1.5):
        actual_player_box = self.get_actual_player_box(player_data) or player_data
        px1, py1, px2, py2 = actual_player_box
        player_width = max(px2 - px1, 1)
        player_height = max(py2 - py1, 1)
        min_x = int(max(px1 - player_width*area_from_player_checked, 0))
        max_x = int(min(px2 + player_width*area_from_player_checked, self.window_controller.width))
        min_y = int(max(py1 - player_height*area_from_player_checked, 0))
        max_y = int(min(py2 + player_height*area_from_player_checked, self.window_controller.height))

        if min_x >= max_x or min_y >= max_y:
            return {
                "up": 0,
                "down": 0,
                "left": 0,
                "right": 0,
            }

        roi = self.frame[min_y:max_y, min_x:max_x]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsv_roi, POISON_LOW_HSV, POISON_HIGH_HSV)
        x, y = self.get_entity_pos(actual_player_box)
        roi_w = int(max_x - min_x)
        roi_h = int(max_y - min_y)
        local_px = int(clamp(x - min_x, 0, roi_w))
        local_py = int(clamp(y - min_y, 0, roi_h))

        counts = {
            "up": count_mask_pixels(mask, 0, 0, roi_w, local_py),
            "down": count_mask_pixels(mask, 0, local_py, roi_w, roi_h),
            "left": count_mask_pixels(mask, 0, 0, local_px, roi_h),
            "right": count_mask_pixels(mask, local_px, 0, roi_w, roi_h),
        }

        result = {
            direction: count if count > threshold else 0
            for direction, count in counts.items()
        }

        if self.verbose_debug:
            print("Poison gas pixels:", counts)

            ts = int(time.time())

            debug_regions = {
                "up": roi[0:local_py, 0:roi_w],
                "down": roi[local_py:roi_h, 0:roi_w],
                "left": roi[0:roi_h, 0:local_px],
                "right": roi[0:roi_h, local_px:roi_w],
            }

            for direction, img in debug_regions.items():
                if img.size > 0:
                    cv2.imwrite(
                        f"debug_frames/poison_gas_{direction}_debug_{ts}.png",
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    )

        return result

    def get_main_data(self, frame):
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        return data

    def is_path_blocked(self, player_box, move_direction, walls, distance=None):
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        movement = self.movement_to_vector(move_direction)
        if movement is None:
            return False

        magnitude = math.hypot(movement[0], movement[1])
        if magnitude < 1:
            return False

        dx = movement[0] / magnitude * distance
        dy = movement[1] / magnitude * distance
        hit_circle_center, hit_circle_radius = self.get_player_hit_circle(player_box)
        if hit_circle_center is None:
            return False

        new_pos = (hit_circle_center[0] + dx, hit_circle_center[1] + dy)
        return self.walls_block_swept_circle(hit_circle_center, new_pos, hit_circle_radius, walls)

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if "player" not in data.keys() or not data["player"]:
            incomplete = True  # This is required so track_no_detections can also keep track if enemy is missing

        if "enemy" not in data.keys():
            data['enemy'] = []

        if "teammate" not in data.keys():
            data['teammate'] = []

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        if 'bush' not in data.keys() or not data['bush']:
            data['bush'] = []

        return False if incomplete else data

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    def do_movement(self, movement):
        movement_vector = self.movement_to_vector(movement)
        if movement_vector is None:
            self.window_controller.release_movement()
        else:
            self.window_controller.move(*movement_vector)
            
        if getattr(self, '_pending_gadget', False):
            print("Executing gadget press after movement")
            self.window_controller.press("gadget")
            self.time_since_gadget_checked = time.time()
            self.is_gadget_ready = False
            self._pending_gadget = False

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    def clamp_movement(self, movement):
        x, y = movement
        target_x = clamp(x, -JOYSTICK_RADIUS*self.window_controller.width_ratio, JOYSTICK_RADIUS*self.window_controller.width_ratio)
        target_y = clamp(y, -JOYSTICK_RADIUS*self.window_controller.height_ratio, JOYSTICK_RADIUS*self.window_controller.height_ratio)
        return target_x, target_y

    def _playstyle_file_mtime(self, filename):
        try:
            return resolve_project_path("playstyles", filename).stat().st_mtime
        except Exception:
            return 0.0

    def reload_playstyle_if_changed(self, now):
        # Hot-reload the active playstyle when the config points to a different file
        # or the current file's CONTENT changes on disk. Content comparison is used
        # (not mtime) so it is robust. Broken edits are ignored (keep last good).
        if now - self._last_reload_check < 1.0:
            return
        self._last_reload_check = now
        try:
            target = load_toml_as_dict("cfg/bot_config.toml", cache=False).get(
                "current_playstyle", self._active_playstyle_file)
            _meta, code = load_pyla_script(target)
            if not code or not code.strip():
                return
            if target == self._active_playstyle_file and code == self._active_playstyle_code:
                return
            compiled = compile(code, target, "exec")  # sanity-check before swapping in
            self.pyla_code = code
            self._pyla_compiled = compiled
            self._active_playstyle_file = target
            self._active_playstyle_code = code
            print(f"[reload] playstyle -> {target} ({len(code)} chars)")
        except Exception as e:
            print(f"[reload] keeping previous playstyle (reload failed): {e}")

    def loop(self, brawler, data, current_time):
        self.reload_playstyle_if_changed(current_time)
        self._player_screen_center = self.get_entity_pos(data['player'][0])
        static_objects = data.get('wall', []) + data.get('bush', [])
        self.update_camera_velocity(static_objects, current_time)
        self.update_enemy_tracks(data['enemy'], current_time)
        self.context = {
                'player_data': data['player'][0],
                'enemy_data': data['enemy'],
                'teammate_data': data['teammate'],
                'brawler': brawler,
                'walls': data['wall'],
                'bushes': data['bush'],
                'brawlers_info': self.brawlers_info,
                'must_brawler_hold_attack': self.must_brawler_hold_attack,
                'is_gadget_ready': self.is_gadget_ready,
                'is_hypercharge_ready': self.is_hypercharge_ready,
                'is_super_ready': self.is_super_ready,
                'TILE_SIZE': self.TILE_SIZE*self.window_controller.scale_factor,
                'get_entity_pos': self.get_entity_pos,
                'get_distance': self.get_distance,
                'get_actual_player_box': self.get_actual_player_box,
                'get_brawler_range': self.get_brawler_range,
                'is_there_enemy': self.is_there_enemy,
                'attack': self.attack,
                'use_hypercharge': self.use_hypercharge,
                'use_super': self.use_super,
                'use_gadget': self.use_gadget,
                'get_random_movement': self.get_random_movement,
                'current_brawler': self.current_brawler,
                'last_movement': self.last_movement,
                'last_movement_change_time': self.last_movement_change_time,
                'seconds_to_hold_attack_after_reaching_max': self.seconds_to_hold_attack_after_reaching_max,
                "width": brawl_stars_width,
                "height": brawl_stars_height,
                'find_closest_enemy': self.find_closest_enemy,
                'find_closest_teammate': self.find_closest_teammate,
                'is_there_poison_gas': self.is_there_poison_gas,
                'is_path_blocked': self.is_path_blocked,
                'is_enemy_hittable': self.is_enemy_hittable,
                'find_path': self.find_path,
                'time': time,
                'random': random,
                "persistent_data": self.persistent_data,
                'debug': self.verbose_debug,
                'JOYSTICK_RADIUS': JOYSTICK_RADIUS,
                'rotate_movement': self.rotate_movement,
                'enemy_velocities': self._enemy_velocities,
                'predict_enemy': self.predict_enemy,
                'get_enemy_velocity': self.get_enemy_velocity,
                'get_enemy_track': self.get_enemy_track,
                'walls_block_line_of_sight': self.walls_block_line_of_sight,
                'can_attack_through_walls': lambda st: self.can_attack_through_walls(self.current_brawler, st, self.brawlers_info)
            }
        movement = self.get_movement()
        if self.movement_to_vector(movement) is None:
            self.window_controller.release_movement()
            self.last_movement = ''
            return None
        movement = self.clamp_movement(movement)
        current_time = time.time()
        if movement != self.last_movement:
            if current_time - self.last_movement_change_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_change_time = current_time
            else:
                movement = self.last_movement
        else:
            self.last_movement_change_time = current_time
        movement = self.unstuck_movement_if_needed(movement, current_time)
        return movement

    def update_player_hp(self, frame, player_data):
        """Track player HP and ammo from the bars above the player.

        A confident sighting of the HP bar caches its geometry (offset from the
        player center + width). Measurement then scans column fill inside that
        fixed rect, which keeps working at low HP (tiny green remnant that no
        contour filter accepts) and lets ammo be read even on frames where the
        HP contour itself is noisy or occluded.
        """
        if not player_data:
            return

        if not hasattr(self, 'persistent_data'):
            self.persistent_data = {}
        pd = self.persistent_data
        now = time.time()

        player_box = player_data[0]
        x_center, y_center = self.get_entity_pos(player_box)
        scale_factor = self.window_controller.scale_factor if self.window_controller and self.window_controller.scale_factor else 1.0

        y_min = int(max(0, y_center - 150 * scale_factor))
        y_max = int(min(frame.shape[0], y_center))
        # bar is centered above the player; a narrow window keeps teammate bars
        # and grass out of the mask
        x_min = int(max(0, x_center - 90 * scale_factor))
        x_max = int(min(frame.shape[1], x_center + 90 * scale_factor))

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        lower_green = np.array([40, 100, 120])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_w = int(100 * scale_factor)
        max_h = 20 * scale_factor
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 3 or h > max_h or w < 4 or w > 1.15 * target_w:
                continue
            candidates.append((w, x, y, h))

        max_w = int(pd.get("max_hp_width", target_w))
        if max_w > 1.15 * target_w or max_w < 0.8 * target_w:
            max_w = target_w
            pd["max_hp_width"] = target_w

        geom = pd.get("hp_bar_geom")
        if geom and abs(geom.get("scale", scale_factor) - scale_factor) > 1e-6:
            geom = None  # window was rescaled, cached geometry is void
            pd.pop("hp_bar_geom", None)

        # Refresh cached geometry. Discovery needs a clearly bar-shaped contour;
        # once cached, any candidate near the cached spot re-anchors it against
        # detection-box jitter (the green fill is left-anchored, so the contour's
        # left edge is the bar's left edge at any HP).
        best = None
        for w, x, y, h in candidates:
            is_barlike = w >= 0.55 * target_w and w / float(h) > 2.5
            near_cached = geom is not None and \
                abs((x_min + x) - (x_center + geom["dx"])) <= 15 * scale_factor and \
                abs((y_min + y) - (y_center + geom["dy"])) <= 15 * scale_factor
            if not (is_barlike or near_cached):
                continue
            if best is None or w > best[0]:
                best = (w, x, y, h)
        if best is not None:
            w, x, y, h = best
            if w > max_w:
                pd["max_hp_width"] = w
                max_w = w
            geom = {
                "dx": (x_min + x) - x_center,
                "dy": (y_min + y) - y_center,
                "h": max(h, geom["h"]) if geom else h,
                "scale": scale_factor,
            }
            pd["hp_bar_geom"] = geom

        # --- HP: fraction of filled columns inside the cached bar rect ---
        hp_pct = None
        hp_conf = 0.0
        if geom:
            bx = int(x_center + geom["dx"] - x_min)
            by = int(y_center + geom["dy"] - y_min)
            bh = max(3, int(geom["h"]))
            b0 = max(0, by - 2)
            b1 = min(mask.shape[0], by + bh + 2)
            x0 = max(0, bx)
            x1 = min(mask.shape[1], bx + max_w)
            if b1 > b0 and x1 - x0 > max_w * 0.5:
                band = mask[b0:b1, x0:x1]
                col_filled = (band > 0).sum(axis=0) >= max(1, int(bh * 0.4))
                hp_pct = float(col_filled.sum()) / float(max_w)
                hp_conf = 0.9
        if hp_pct is None and best is not None:
            hp_pct = best[0] / float(max_w)
            hp_conf = 0.6

        if hp_pct is not None:
            hp_pct = min(1.0, max(0.0, hp_pct))
            prev_hp = pd.get("current_hp_pct")
            if prev_hp is not None and now - pd.get("hp_updated_at", 0) < 0.5:
                hp_pct = prev_hp * 0.4 + hp_pct * 0.6  # smooth single-frame flicker
            pd["current_hp_pct"] = hp_pct
            pd["hp_confidence"] = hp_conf
            pd["hp_updated_at"] = now

        # --- Ammo: orange fill directly below the HP bar ---
        layout = pd.get("ammo_layout")
        if layout and layout.get("brawler") != (self.current_brawler or ""):
            layout = None
            pd.pop("ammo_layout", None)
            pd.pop("ammo_seg_candidate", None)
            pd.pop("ammo_seg_hits", None)

        if geom:
            bx = int(x_center + geom["dx"] - x_min)
            ay1 = max(0, int(y_center + geom["dy"] + geom["h"] - y_min) + 1)
            ay2 = min(hsv.shape[0], ay1 + max(6, int(20 * scale_factor)))
            ax0 = max(0, bx)
            ax1 = min(hsv.shape[1], bx + max_w)
            unclipped = (ax0 == bx and ax1 == bx + max_w)
            if ay2 > ay1 and ax1 - ax0 > max_w * 0.5:
                ammo_roi = hsv[ay1:ay2, ax0:ax1]
                lower_orange = np.array([4, 120, 120])
                upper_orange = np.array([32, 255, 255])
                mask_ammo = cv2.inRange(ammo_roi, lower_orange, upper_orange)
                ah = mask_ammo.shape[0]
                col_filled = (mask_ammo > 0).sum(axis=0) >= max(1, int(ah * 0.25))
                raw_pct = float(col_filled.sum()) / float(max_w)

                # Learn the segment layout from an (essentially) full bar:
                # runs of filled columns of similar width are the segments.
                # Requires the same count on 3 sightings before trusting it.
                if layout is None and unclipped and raw_pct >= 0.88:
                    runs, start = [], None
                    for i, f in enumerate(col_filled):
                        if f and start is None:
                            start = i
                        elif not f and start is not None:
                            runs.append((start, i))
                            start = None
                    if start is not None:
                        runs.append((start, col_filled.size))
                    runs = [r for r in runs if r[1] - r[0] >= 0.08 * max_w]
                    if 2 <= len(runs) <= 6:
                        widths = [r[1] - r[0] for r in runs]
                        if max(widths) <= 2.2 * min(widths):
                            cand = len(runs)
                            if pd.get("ammo_seg_candidate") == cand:
                                pd["ammo_seg_hits"] = pd.get("ammo_seg_hits", 0) + 1
                            else:
                                pd["ammo_seg_candidate"] = cand
                                pd["ammo_seg_hits"] = 1
                            if pd["ammo_seg_hits"] >= 3:
                                layout = {
                                    "brawler": self.current_brawler or "",
                                    "segments": cand,
                                    "bounds": [(s / float(max_w), e / float(max_w)) for s, e in runs],
                                }
                                pd["ammo_layout"] = layout

                if layout and unclipped:
                    # per-segment fill -> exact segment count and a normalized
                    # pct where a truly full bar reads 1.0 (separators excluded)
                    fills = []
                    for s, e in layout["bounds"]:
                        a, b = int(s * max_w), int(e * max_w)
                        seg = col_filled[a:b]
                        fills.append(float(seg.mean()) if seg.size else 0.0)
                    ammo_pct = min(1.0, sum(min(1.0, f / 0.95) for f in fills) / len(fills))
                    pd["current_ammo_segments"] = sum(1 for f in fills if f >= 0.85)
                    pd["max_ammo_segments"] = layout["segments"]
                    ammo_conf = 0.9
                else:
                    ammo_pct = min(1.0, raw_pct)
                    pd["current_ammo_segments"] = None
                    ammo_conf = 0.6

                prev_ammo = pd.get("current_ammo_pct")
                if prev_ammo is not None and now - pd.get("ammo_updated_at", 0) < 0.5:
                    ammo_pct = prev_ammo * 0.4 + ammo_pct * 0.6
                pd["current_ammo_pct"] = ammo_pct
                pd["ammo_confidence"] = ammo_conf
                pd["ammo_updated_at"] = now

    def check_if_hypercharge_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(hypercharge_crop_area[0] * wr), int(hypercharge_crop_area[1] * hr)
        x2, y2 = int(hypercharge_crop_area[2] * wr), int(hypercharge_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        purple_pixels = count_hsv_pixels(screenshot, (137, 158, 159), (179, 255, 255))
        if self.verbose_debug:
            print("hypercharge purple pixels:", purple_pixels, "(if > ", self.hypercharge_pixels_minimum, " then hypercharge is ready)")
            cv2.imwrite(f"debug_frames/hypercharge_debug_{purple_pixels}_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        if purple_pixels > self.hypercharge_pixels_minimum:
            return True
        return False

    def check_if_gadget_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(gadget_crop_area[0] * wr), int(gadget_crop_area[1] * hr)
        x2, y2 = int(gadget_crop_area[2] * wr), int(gadget_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        green_pixels = count_hsv_pixels(screenshot, (57, 219, 165), (62, 255, 255))
        if self.verbose_debug:
            print("gadget green pixels:", green_pixels, "(if > ", self.gadget_pixels_minimum, " then gadget is ready)")
            cv2.imwrite(f"debug_frames/gadget_debug_{green_pixels}_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        if green_pixels > self.gadget_pixels_minimum:
            return True
        return False

    def check_if_super_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(super_crop_area[0] * wr), int(super_crop_area[1] * hr)
        x2, y2 = int(super_crop_area[2] * wr), int(super_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        yellow_pixels = count_hsv_pixels(screenshot, (17, 170, 200), (27, 255, 255))
        if self.verbose_debug:
            print("super yellow pixels:", yellow_pixels, "(if > ", self.super_pixels_minimum, " then super is ready)")
            cv2.imwrite(f"debug_frames/super_debug_{yellow_pixels}_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        if yellow_pixels > self.super_pixels_minimum:
            return True
        return False

    def get_centered_wall_crop(self, frame, player_data=None):
        frame_height, frame_width = frame.shape[:2]
        crop_size = self.centered_wall_crop_size

        if player_data:
            center_x, center_y = self.get_entity_pos(player_data[0])
        else:
            center_x, center_y = frame_width / 2, frame_height / 2

        crop_x1 = int(clamp(round(center_x - crop_size / 2), 0, frame_width - crop_size))
        crop_y1 = int(clamp(round(center_y - crop_size / 2), 0, frame_height - crop_size))
        crop_x2 = crop_x1 + crop_size
        crop_y2 = crop_y1 + crop_size

        return frame[crop_y1:crop_y2, crop_x1:crop_x2], crop_x1, crop_y1

    @staticmethod
    def offset_tile_data(tile_data, offset_x, offset_y):
        if not offset_x and not offset_y:
            return tile_data

        offset_data = {}
        for class_name, boxes in tile_data.items():
            offset_data[class_name] = [
                [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y]
                for box in boxes
            ]
        return offset_data

    def get_tile_data(self, frame, player_data=None):
        if self.centered_wall_detection and self.Detect_centered_tile_detector is not None:
            crop, offset_x, offset_y = self.get_centered_wall_crop(frame, player_data)
            tile_data = self.Detect_centered_tile_detector.detect_objects(
                crop,
                conf_tresh=self.wall_detection_confidence
            )
            return self.offset_tile_data(tile_data, offset_x, offset_y)

        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    def process_tile_data(self, tile_data):
        walls = []
        bushes = []
        for class_name, boxes in tile_data.items():
            if 'bush' not in class_name:
                walls.extend(boxes)
            else:
                bushes.extend(boxes)
        return walls, bushes

    def get_movement(self):
        movement, updated_globals = interpret_pyla_code(self._pyla_compiled or self.pyla_code, self.context)
        return movement

    def publish_debug_view(self, frame, data, state, movement=None):
        if not hasattr(self.window_controller, "debug_view"):
            return

        # The payload below is expensive to build (a second poison-gas CV pass,
        # predict_enemy per enemy, range lookups). publish() drops it anyway
        # when the view is disabled or fps-throttled, so bail out before doing
        # the work, not after.
        dv = self.window_controller.debug_view
        if not getattr(dv, "enabled", False):
            return
        if time.perf_counter() - getattr(dv, "last_publish", 0.0) < getattr(dv, "publish_delay", 0.0):
            return

        self.frame = frame
        advanced_visuals = bool(getattr(self.window_controller.debug_view, "advanced_visuals", False))
        debug_data = {
            "state": state,
            "player": [],
            "enemy": [],
            "teammate": [],
            "wall": [],
            "attack_range": 0,
            "super_range": 0,
            "poison_gas": {},
            "movement": None,
            "joystick": [self.window_controller.joystick_x, self.window_controller.joystick_y],
            "advanced_visuals": advanced_visuals,
            "joystick_radius": int(JOYSTICK_RADIUS * (self.window_controller.scale_factor or 1)),
            "joystick_directions": [],
            "enemy_los_lines": [],
            "teammate_los_lines": [],
            "player_hit_circle": None,
        }

        if data:
            for key in ["player", "enemy", "teammate", "wall"]:
                debug_data[key] = [[int(v) for v in box[:4]] for box in (data.get(key) or []) if len(box) >= 4]
            try:
                _, attack_range, super_range = self.get_brawler_range(self.current_brawler)
                debug_data["attack_range"] = int(attack_range)
                debug_data["super_range"] = int(super_range)
            except Exception:
                pass
            pd = getattr(self, 'persistent_data', {})
            debug_data["hp_pct"] = pd.get("current_hp_pct")
            debug_data["ammo_pct"] = pd.get("current_ammo_pct")
            debug_data["ammo_segments"] = pd.get("current_ammo_segments")
            debug_data["max_ammo_segments"] = pd.get("max_ammo_segments")
            if debug_data["player"] and debug_data["enemy"]:
                try:
                    ppos = self.get_entity_pos(debug_data["player"][0])
                    preds = []
                    for box in debug_data["enemy"]:
                        epos = self.get_entity_pos(box)
                        led = self.predict_enemy(ppos, epos)
                        _, conf = self.get_enemy_track(epos)
                        preds.append([int(led[0]), int(led[1]), round(conf, 2)])
                    debug_data["enemy_pred"] = preds
                except Exception:
                    pass
            if debug_data["player"]:
                try:
                    debug_data["poison_gas"] = self.is_there_poison_gas(debug_data["player"][0])
                except Exception:
                    pass
                if advanced_visuals and early_access:
                    add_advanced_visuals(self, debug_data)

        if movement is not None:
            debug_data["movement"] = [float(movement[0]), float(movement[1])]

        self.window_controller.debug_view.publish(frame, debug_data)

    def main(self, frame, brawler, main):
        if self._detect_executor is not None:
            # Pipelined: kick off detection for THIS frame in the worker, then
            # process the PREVIOUS frame with its (usually already finished)
            # detections. One frame of extra latency, ~2x the iteration rate.
            future = self._detect_executor.submit(self.get_main_data, frame)
            pending, self._pending_detection = self._pending_detection, (frame, future)
            if pending is None:
                return  # first frame just primes the pipeline
            frame, prev_future = pending
            data = prev_future.result()
        else:
            data = self.get_main_data(frame)
        current_time = time.time()
        state = main.get_latest_state()
        if self._detect_executor is not None:
            # Tile/wall inference in the pipeline worker: running the full ONNX
            # pass on the main thread stalled one tick in five (walls refresh
            # every 0.2s). Walls change slowly, so consuming the result a tick
            # or two later costs nothing.
            if self._pending_tiles is not None and self._pending_tiles.done():
                try:
                    walls, bushes = self.process_tile_data(self._pending_tiles.result())
                    self.last_walls_data = walls
                    self.last_bushes_data = bushes
                except Exception as exc:
                    print(f"Tile detection failed: {exc}")
                self._pending_tiles = None
            if self._pending_tiles is None and current_time - self.time_since_walls_checked > self.walls_treshold:
                self.time_since_walls_checked = current_time
                self._pending_tiles = self._detect_executor.submit(
                    self.get_tile_data, frame, data.get("player"))
            data['wall'] = self.last_walls_data
            data['bush'] = self.last_bushes_data
        elif current_time - self.time_since_walls_checked > self.walls_treshold:
            tile_data = self.get_tile_data(frame, data.get("player"))
            walls, bushes = self.process_tile_data(tile_data)
            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            data['wall'] = walls
            self.last_bushes_data = bushes
            data['bush'] = bushes
        else:
            data['wall'] = self.last_walls_data
            data['bush'] = self.last_bushes_data

        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            self.time_since_player_last_found = time.time()
            if state != "match":
                data = None
            else:
                self.update_player_hp(frame, data.get("player"))

        if not data:
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.release_movement()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame)
                if current_state != "match":
                    main.handle_detected_state(current_state)
                    state = current_state
                    self.time_since_last_proceeding = current_time
                else:
                    print("haven't detected the player in a while proceeding")
                    self.window_controller.press("proceed", blocking=True)
                    self.time_since_last_proceeding = time.time()
            self.publish_debug_view(frame, data, state)
            return
        self.time_since_last_proceeding = time.time()
        if current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
        if current_time - self.time_since_gadget_checked > self.gadget_treshold:
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
        if current_time - self.time_since_super_checked > self.super_treshold:
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time
        self.frame = frame
        movement = self.loop(brawler, data, current_time)
        self.publish_debug_view(frame, data, state, movement)
        if movement is not None:
            self.do_movement(movement)
