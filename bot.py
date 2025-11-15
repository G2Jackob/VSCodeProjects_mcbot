import cv2 as cv
import pyautogui
import pydirectinput
from time import sleep, time
from threading import Thread, Lock
from collections import Counter
import random
import pytesseract
import re
import numpy as np

# Configure tesseract path 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        """Read player and targeted block coordinates from F3 debug screen using multi-sample OCR"""
        try:
            # Add debug OCR processing call at the start
            self.debug_ocr_processing(screenshot, 'ocr_debug.png')
            
            height, width = screenshot.shape[:2]
            
            # Convert BGR to HSV for better color isolation
            hsv_image = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
            
            # Define color range for Minecraft's white/light gray text
            color_lower = np.array([0, 0, 150])
            color_upper = np.array([180, 80, 255])
            
            # Create mask to isolate the text
            mask = cv.inRange(hsv_image, color_lower, color_upper)
            
            # Apply mask to get only the text
            result = cv.bitwise_and(screenshot, screenshot, mask=mask)
            
            # Convert to grayscale
            gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            
            # Apply threshold to get black and white image
            _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            
            # Define crop regions - now only showing coordinate numbers
            # LEFT side for targeted block coordinates
            y_left = 0
            h_left = int(height * 0.03)
            x_left = 0
            w_left = int(width * 0.3)
            
            # RIGHT side for player block coordinates
            y_right = int(height * 0.025)
            h_right = int(height * 0.025)
            x_right = int(width * 0.7)
            w_right = int(width * 0.3)
            
            # Crop the regions
            crop_left = thresh[y_left:y_left+h_left, x_left:x_left+w_left]
            crop_right = thresh[y_right:y_right+h_right, x_right:x_right+w_right]
            
            # Enhance images for better OCR
            crop_left = cv.bitwise_not(crop_left)
            crop_right = cv.bitwise_not(crop_right)
            
            # Apply morphological operations to clean up text
            kernel = np.ones((2, 2), np.uint8)
            crop_left = cv.morphologyEx(crop_left, cv.MORPH_CLOSE, kernel)
            crop_right = cv.morphologyEx(crop_right, cv.MORPH_CLOSE, kernel)
            
            # Multi-sample OCR: Read 10 times and pick most common results
            def extract_numbers_only(text):
                """Extract only numbers (including negative) from OCR text"""
                # First try to find numbers with separators (spaces, commas, etc.)
                number_pattern = r'-?\d+'
                matches = re.findall(number_pattern, text)
                
                # If we got less than 3 numbers, text might be concatenated like "-5467-371" or "-5967-4"
                # Try to split it intelligently
                if len(matches) < 3 and len(text.strip()) > 0:
                    text_clean = text.strip().replace(' ', '')
                    
                    # For concatenated coordinates like "-5467-371" or "-5967-4"
                    # This is likely: X (negative, 2-3 digits), Y (positive, 2 digits), Z (negative, 1-3 digits)
                    # Pattern: -XX YY -Z or -XX YY -ZZ or -XXX YY -ZZZ
                    
                    # Try multiple patterns with varying Z length
                    patterns = [
                        r'^(-\d{2,3})(\d{2})(-\d{1,3})$',  # Most flexible
                        r'^(-\d{2})(\d{2})(-\d{1,3})$',    # 2-digit X
                        r'^(-\d{3})(\d{2})(-\d{1,3})$',    # 3-digit X
                    ]
                    
                    for pattern in patterns:
                        match = re.match(pattern, text_clean)
                        if match:
                            try:
                                x = int(match.group(1))
                                y = int(match.group(2))
                                z = int(match.group(3))
                                print(f"[DEBUG] Parsed concatenated coords: X={x}, Y={y}, Z={z}")
                                return [x, y, z]
                            except ValueError:
                                continue
                
                # Otherwise use the original matches
                coords = []
                for match in matches:
                    try:
                        coords.append(int(match))
                    except ValueError:
                        continue
                return coords
            
            def get_most_common_coords(crop_image, num_samples=5):
                """Read OCR multiple times and return most common coordinate values"""
                # Config to only read numbers
                custom_config = r'--oem 3 --psm 6 '
                
                all_readings = []
                for i in range(num_samples):
                    try:
                        text = pytesseract.image_to_string(crop_image, lang='mc', config=custom_config)
                        coords = extract_numbers_only(text)
                        if len(coords) >= 3:
                            all_readings.append(tuple(coords[:3]))
                    except Exception as e:
                        pass
                
                if not all_readings:
                    return None
                
                # Find most common value for each coordinate position
                x_values = [reading[0] for reading in all_readings]
                y_values = [reading[1] for reading in all_readings]
                z_values = [reading[2] for reading in all_readings]
                
                most_common_x = Counter(x_values).most_common(1)[0][0]
                most_common_y = Counter(y_values).most_common(1)[0][0]
                most_common_z = Counter(z_values).most_common(1)[0][0]
                
                return (most_common_x, most_common_y, most_common_z)
            
            # Parse coordinates - extract numbers from the text
            # Split text and find strings containing digits
            # Use multi-sample OCR to get player coordinates from RIGHT side
            player_coords_result = get_most_common_coords(crop_right, num_samples=5)
            if player_coords_result is not None:
                # Only update if coordinates are reasonable
                if self.player_coords is None or self._coords_are_reasonable(player_coords_result, self.player_coords):
                    self.player_coords = player_coords_result
                    print(f"[DEBUG] Player Block Coordinates: {self.player_coords}")
                else:
                    print(f"[DEBUG] Ignoring invalid player coords: {player_coords_result} (previous: {self.player_coords})")
            
            # Use multi-sample OCR to get targeted block coordinates from LEFT side
            target_coords_result = get_most_common_coords(crop_left, num_samples=5)
            if target_coords_result is not None:
                # Only update if coordinates are reasonable
                if self.target_block_coords is None or self._coords_are_reasonable(target_coords_result, self.target_block_coords):
                    self.target_block_coords = target_coords_result
                    print(f"[DEBUG] Targeted Block Coordinates: {self.target_block_coords}")
                else:
                    print(f"[DEBUG] Ignoring invalid target coords: {target_coords_result} (previous: {self.target_block_coords})")
                    print(f"[DEBUG] Clearing target and going back to searching due to invalid coordinates")
                    self.target_block_coords = None
            
            return self.player_coords is not None or self.target_block_coords is not None
            
        except Exception as e:
            print(f"[DEBUG] Error reading F3 coordinates: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _coords_are_reasonable(self, new_coords, old_coords):
        """Check if new coordinates are reasonable compared to old ones"""
        # Allow up to 20 blocks difference per coordinate (player can't move that fast)
        max_diff = 20
        for i in range(3):
            if abs(new_coords[i] - old_coords[i]) > max_diff:
                return False
        return True
    
    def debug_ocr_processing(self, screenshot, save_path='ocr_debug.png'):
        """Save a debug image showing what the OCR is processing (optional, for troubleshooting)"""
        try:
            height, width = screenshot.shape[:2]
            
            # Convert BGR to HSV
            hsv_image = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
            
            # Apply mask
            color_lower = np.array([0, 0, 200])
            color_upper = np.array([180, 50, 255])
            mask = cv.inRange(hsv_image, color_lower, color_upper)
            result = cv.bitwise_and(screenshot, screenshot, mask=mask)
            gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            
            # Draw rectangles on the original image to show crop regions
            debug_img = screenshot.copy()
            y_left = 0
            h_left = int(height * 0.03)
            x_left = 0
            w_left = int(width * 0.3)
            y_right = int(height * 0.025)
            h_right = int(height * 0.025)
            x_right = int(width * 0.7)
            w_right = int(width * 0.3)
            
            cv.rectangle(debug_img, (x_left, y_left), (x_left+w_left, y_left+h_left), (0, 255, 0), 2)
            cv.rectangle(debug_img, (x_right, y_right), (x_right+w_right, y_right+h_right), (0, 0, 255), 2)
            
            # Create a side-by-side comparison
            combined = np.hstack([debug_img, cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)])
            cv.imwrite(save_path, combined)
            print(f"[DEBUG] OCR debug image saved to {save_path}")
            
        except Exception as e:
            print(f"[DEBUG] Error creating debug image: {str(e)}")
    
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
                # Store the original target block coordinates
                original_target_coords = self.target_block_coords
                
                # Show F3 debug info and wait for fresh screenshot
                pydirectinput.press('F3')
                sleep(0.5)  # Increased delay to ensure F3 is displayed and captured
                
                # Read coordinates from current screenshot
                if self.screenshot is not None:
                    self.read_f3_coordinates(self.screenshot)
                
                # Hide F3 debug info
                pydirectinput.press('F3')
                sleep(0.3)  # Increased delay
                
                # Check if target block changed (OCR read wrong coordinates)
                if original_target_coords is not None and self.target_block_coords is not None:
                    if original_target_coords != self.target_block_coords:
                        print(f"[DEBUG] Target block changed from {original_target_coords} to {self.target_block_coords}, going back to searching")
                        self.lock.acquire()
                        self.state = BotState.SEARCHING
                        self.current_target = None
                        self.target_block_coords = None
                        self.lock.release()
                        sleep(0.1)
                        continue
                
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
        sleep(0.5)  # Increased delay to ensure F3 is displayed and captured
        if self.screenshot is not None:
            self.read_f3_coordinates(self.screenshot)
        pydirectinput.press('F3')
        sleep(0.3)  # Increased delay
        
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
        
        # Mine 4 times, moving view up slightly between each mining action
        for i in range(4):
            print(f"[DEBUG] Mining iteration {i+1}/4")
            pydirectinput.mouseDown()
            sleep(3.2)
            pydirectinput.mouseUp()
            sleep(0.3)
            
            # Move view up slightly by dragging mouse down for a short time (except after the last iteration)
            if i < 3:
                pyautogui.drag(0, 5, duration=0.05)  # Small drag to adjust view upward
                sleep(0.2)
        
        # Hold W for 1 second to move forward
        print("[DEBUG] Moving forward")
        pydirectinput.keyDown('w')
        sleep(0.7)
        pydirectinput.keyUp('w')
        sleep(0.3)
        
        print("[DEBUG] Mining sequence complete")
        return True