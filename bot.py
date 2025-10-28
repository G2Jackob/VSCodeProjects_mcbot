import cv2 as cv
import pyautogui
import pydirectinput
from time import sleep, time
from threading import Thread, Lock
import random
import pytesseract
import re

class BotState:
    INITIALIZING = 0
    SEARCHING = 1
    MOVING = 2
    MINING = 3

class McBot:

    INITIALIZING_TIME = 6
    MINING_TIME = 5
    MOVEMENT_STOPPED_THRESHOLD = 0.95
    TOOLTIP_MATCH_THRESHOLD = 0.75

    stopped = True
    lock = None
    state = None
    targets = []
    screenshot = None
    timestamp = None
    movement_screenshot = None
    window_offset = (0, 0)
    window_w = 0
    window_h = 0
    wood_tooltip = None
    player_coords = None
    target_block_coords = None
    current_target = None  # Persistent target tracking
    target_distance = None  # Distance to target

    def __init__(self, window_offset, window_size):
        self.lock = Lock()
        self.offset_x, self.offset_y = window_offset
        self.window_w, self.window_h = window_size
        self.state = BotState.INITIALIZING
        self.timestamp = time()
        self.player_coords = None
        self.target_block_coords = None
        self.current_target = None
        self.target_distance = None

        self.wood_tooltip = cv.imread('wood_tooltip.jpg', cv.IMREAD_UNCHANGED)
        
    def read_f3_coordinates(self, screenshot):
        """Read player and targeted block coordinates from F3 debug screen with improved OCR"""
        try:
            # Convert screenshot to grayscale for better OCR
            gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Check brightness and enhance if needed
            mean_brightness = cv.mean(gray)[0]
            if mean_brightness < 80:  # Dark conditions
                print(f"[DEBUG] Dark OCR conditions (brightness: {mean_brightness:.1f}), enhancing")
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
            
            # Apply stronger thresholding for Minecraft's white text
            _, thresh = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
            
            # Scale up the image for better OCR (Minecraft font is small)
            scale_factor = 2
            
            # Read from LEFT side for "Targeted Block:"
            roi_left = thresh[0:height//2, 0:width//3]
            roi_left_scaled = cv.resize(roi_left, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
            text_left = pytesseract.image_to_string(roi_left_scaled, config='--psm 6')
            
            # Read from RIGHT side for player "Block:" coordinates  
            roi_right = thresh[0:height//2, 2*width//3:width]
            roi_right_scaled = cv.resize(roi_right, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
            text_right = pytesseract.image_to_string(roi_right_scaled, config='--psm 6')
            
            # Parse player Block coordinates from RIGHT side (spaces only, no commas)
            # Look for "Block:" followed by 3 numbers separated by spaces
            block_pattern = r'Block:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)'
            block_match = re.search(block_pattern, text_right)
            if block_match:
                new_coords = (int(block_match.group(1)), 
                             int(block_match.group(2)), 
                             int(block_match.group(3)))
                # Only update if coordinates are reasonable (not wildly different)
                if self.player_coords is None or self._coords_are_reasonable(new_coords, self.player_coords):
                    self.player_coords = new_coords
                    print(f"[DEBUG] Player Block Coordinates: {self.player_coords}")
                else:
                    print(f"[DEBUG] Ignoring invalid player coords: {new_coords} (previous: {self.player_coords})")
            
            # Parse Targeted Block coordinates from LEFT side (with commas)
            target_pattern = r'Targeted Block:\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)'
            target_match = re.search(target_pattern, text_left)
            if target_match:
                new_coords = (int(target_match.group(1)), 
                             int(target_match.group(2)), 
                             int(target_match.group(3)))
                # Only update if coordinates are reasonable
                if self.target_block_coords is None or self._coords_are_reasonable(new_coords, self.target_block_coords):
                    self.target_block_coords = new_coords
                    print(f"[DEBUG] Targeted Block Coordinates: {self.target_block_coords}")
                else:
                    print(f"[DEBUG] Ignoring invalid target coords: {new_coords} (previous: {self.target_block_coords})")
            
            return self.player_coords is not None or self.target_block_coords is not None
            
        except Exception as e:
            print(f"[DEBUG] Error reading F3 coordinates: {str(e)}")
            return False
    
    def _coords_are_reasonable(self, new_coords, old_coords):
        """Check if new coordinates are reasonable compared to old ones"""
        # Allow up to 10 blocks difference per coordinate (player can't move that fast)
        max_diff = 10
        for i in range(3):
            if abs(new_coords[i] - old_coords[i]) > max_diff:
                return False
        return True
    
    def get_best_target(self):
        """Find the best target based on confidence and size combined"""
        if not self.targets:
            return None
        
        # Each target is now (x, y, confidence, size)
        # Calculate a score combining both confidence and size
        targets_with_score = []
        for target in self.targets:
            x, y, conf, size = target
            # Score = confidence * normalized_size
            # Normalize size to 0-1 range (assuming max size ~50000 pixels)
            normalized_size = min(size / 50000.0, 1.0)
            score = conf * (0.7 + 0.3 * normalized_size)  # 70% confidence, 30% size weight
            targets_with_score.append((target, score))
        
        # Sort by score in descending order
        targets_with_score.sort(key=lambda t: t[1], reverse=True)
        
        best = targets_with_score[0]
        best_target = best[0]
        print(f"[DEBUG] Selected target: pos=({best_target[0]}, {best_target[1]}), conf={best_target[2]:.2f}, size={best_target[3]}, score={best[1]:.2f}")
        print(f"[DEBUG] Total targets: {len(self.targets)}")
        
        # Return just (x, y) for compatibility
        return (best_target[0], best_target[1])



    def click_next_target(self):
        if self.move_crosshair_to_target():
            # Press F3 to show tooltip
            pydirectinput.press('F3')
            sleep(0.5)
            
            if self.confirm_tooltip(self.screenshot):
              
                sleep(0.2)
                
                return True
            
            pydirectinput.press('F3') 
            sleep(0.2)
        return False


    def have_stopped_moving(self):
        if self.movement_screenshot is None:
           self.movement_screenshot = self.screenshot.copy()
           return False
        
        result = cv.matchTemplate(self.screenshot, self.movement_screenshot, cv.TM_CCOEFF_NORMED)

        similarity = result[0][0]
        if similarity >= self.MOVEMENT_STOPPED_THRESHOLD:
            self.movement_screenshot = self.screenshot.copy()
            return True
        self.movement_screenshot = self.screenshot.copy()
        return False

    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

    def update_targets(self, targets):
       self.lock.acquire()
       self.targets = targets
       self.lock.release()

    def update_screenshot(self, screenshot):
        self.lock.acquire()
        self.screenshot = screenshot
        self.lock.release()

    def start(self):
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            if self.state == BotState.INITIALIZING:
                if time() > self.timestamp + self.INITIALIZING_TIME:
                    print("[DEBUG] Initializing complete")
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()

            elif self.state == BotState.SEARCHING:
                if self.move_crosshair_to_target():
                    print("[DEBUG] Crosshair centered, transitioning to MOVING")
                    sleep(0.2)    
                    self.lock.acquire()
                    self.state = BotState.MOVING
                    self.lock.release()
                    sleep(0.2)

            elif self.state == BotState.MINING:
                # Recenter cursor on target before mining
                print("[DEBUG] Recentering cursor on target before mining")
                if self.move_crosshair_to_target():
                    print("[DEBUG] Cursor centered, starting mining")
                    if self.mine_tree():
                        print("[DEBUG] Mining complete, clearing target")
                        self.lock.acquire()
                        self.state = BotState.SEARCHING
                        self.current_target = None  # Clear target after mining
                        self.target_block_coords = None
                        self.target_distance = None
                        self.lock.release()
                else:
                    print("[DEBUG] Lost target while recentering, going back to searching")
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.current_target = None
                    self.lock.release()
                    
            elif self.state == BotState.MOVING:
                # Press F3 to show debug info and read coordinates
                pydirectinput.press('F3')
                sleep(0.3)
                
                # Read coordinates from current screenshot
                if self.screenshot is not None:
                    self.read_f3_coordinates(self.screenshot)
                
                # Hide F3 debug info
                pydirectinput.press('F3')
                sleep(0.2)
                
                if self.move_to_target():
                    print("[DEBUG] Reached target, starting mining")
                    self.lock.acquire()
                    self.state = BotState.MINING
                    self.lock.release()
                elif self.have_stopped_moving():
                    print("[DEBUG] Movement stopped, re-evaluating")
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.current_target = None  # Clear target to find new one
                    self.target_block_coords = None
                    self.lock.release()
            sleep(0.1)

    def move_crosshair_to_target(self):
        """Move crosshair to target with best confidence"""
        # Always get the best target (recalculated each frame as detections update)
        best_target = self.get_best_target()
        if best_target is None:
            print("[DEBUG] No targets available")
            self.current_target = None
            return False
        
        # If we don't have a target, lock onto this one
        if self.current_target is None:
            self.current_target = best_target
            print(f"[DEBUG] Locked onto new target at: {self.current_target}")
        
        # Use the best target position (it updates as we move the camera)
        target_pos = best_target
        center_x = self.window_w // 2
        center_y = self.window_h // 2
        dx = target_pos[0] - center_x
        dy = target_pos[1] - center_y
        
        # Calculate distance to target
        distance_to_center = (dx**2 + dy**2)**0.5
        print(f"[DEBUG] Target at {target_pos}, center at ({center_x}, {center_y})")
        print(f"[DEBUG] Offset: dx={dx}, dy={dy}, distance={distance_to_center:.1f}")
        
        # Check if we're close enough to center
        if abs(dx) < 10 and abs(dy) < 10:
            print(f"[DEBUG] Crosshair centered on target (within 10 pixels)")
            return True
        
        # Move towards target
        move_x = int(dx)
        move_y = int(dy)
        
        if abs(move_x) > 0 or abs(move_y) > 0:
            print(f"[DEBUG] Moving mouse by: ({move_x}, {move_y})")
            pydirectinput.moveRel(move_x, move_y, relative=True)
            sleep(0.1)
        
        return False  # Not yet centered

    def targets_ordered_by_distance(self, targets):
            center_x = self.window_w // 2
            center_y = self.window_h // 2

            def distance_to_target(target):
                dx = target[0] - center_x
                dy = target[1] - center_y
                return (dx ** 2 + dy ** 2) ** 0.5

            if targets:
                targets = sorted(targets, key=distance_to_target)
            return targets


    def move_to_target(self):
        """Move forward to target using coordinate-based distance (X and Z only)"""
        # Calculate distance to target if we have coordinates
        if self.target_block_coords and self.player_coords:
            dx = self.target_block_coords[0] - self.player_coords[0]
            dy = self.target_block_coords[1] - self.player_coords[1]
            dz = self.target_block_coords[2] - self.player_coords[2]
            # Calculate horizontal distance only (ignore Y coordinate)
            self.target_distance = (dx**2 + dz**2)**0.5
            
            print(f"[DEBUG] Horizontal distance to target: {self.target_distance:.2f} blocks (X: {dx:.1f}, Z: {dz:.1f}, Y: {dy:.1f})")
            
            # If we're close enough, we've reached the target
            if self.target_distance <= 3.0:
                print(f"[DEBUG] Reached target (within 3 blocks horizontally)")
                return True
        
        # Move forward if we're not at the target yet
        print("[DEBUG] Walking forward")
        pydirectinput.keyDown('w')
        sleep(0.3)
        pydirectinput.keyUp('w')
        
        return False

    def check_if_stuck(self):
        """Check if we're stuck by comparing positions using coordinates"""
        if self.player_coords is None:
            return False
        
        initial_coords = self.player_coords
        sleep(0.5)
        
        # Read new position via F3
        pydirectinput.press('F3')
        sleep(0.3)
        if self.screenshot is not None:
            self.read_f3_coordinates(self.screenshot)
        pydirectinput.press('F3')
        sleep(0.2)
        
        if self.player_coords is None:
            return False
        
        # Calculate how far we moved (only X and Z, ignore Y)
        dx = self.player_coords[0] - initial_coords[0]
        dz = self.player_coords[2] - initial_coords[2]
        distance_moved = (dx**2 + dz**2)**0.5
        
        print(f"[DEBUG] Distance moved: {distance_moved:.2f} blocks")
        return distance_moved < 0.5  # Stuck if moved less than 0.5 blocks

    def mine_tree(self):
        print("[DEBUG] Starting mining sequence")
        
        # Single mining action
        pydirectinput.mouseDown()
        sleep(3.0)
        pydirectinput.mouseUp()
        sleep(0.5)
        
        
        
        return True