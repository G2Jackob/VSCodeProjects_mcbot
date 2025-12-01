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

# Configure tesseract path, add mc3.traineddata to tesseract OCR\tessdata folder@
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
        self.searching_start_time = None  # Track how long we've been searching

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
            # Higher saturation threshold to exclude sky (white text has low saturation)
            color_lower = np.array([0, 0, 200])
            color_upper = np.array([180, 30, 255])
            
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
            y_left = int(height * 0.001)
            h_left = int(height * 0.037)
            x_left = int(width * 0.001)
            w_left = int(width * 0.4)
            
            # RIGHT side for player block coordinates
            y_right = int(height * 0.034)
            h_right = int(height * 0.035)
            x_right = int(width * 0.80)
            w_right = int(width * 0.195)
            
            # Crop the regions
            crop_left = thresh[y_left:y_left+h_left, x_left:x_left+w_left]
            crop_right = thresh[y_right:y_right+h_right, x_right:x_right+w_right]
            
            # Save crops to file for debugging
            cv.imwrite('crop_left.png', crop_left)
            cv.imwrite('crop_right.png', crop_right)
            
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
                # Find all minus signs and their positions to determine which numbers should be negative
                # Look for patterns like "- S52" or "-S52" and extract just the number with minus
                
                # Pattern to find minus followed by optional letters/space then digits
                minus_pattern = r'-\s*[A-Za-z]*\s*(\d+)'
                minus_matches = re.finditer(minus_pattern, text)
                
                # Extract negative numbers (with minus signs) with their positions
                coords_with_pos = []
                seen_positions = set()
                
                for match in minus_matches:
                    number = -int(match.group(1))  # Make it negative
                    coords_with_pos.append((match.start(), number))
                    # Mark these character positions as used
                    seen_positions.update(range(match.start(), match.end()))
                
                # Now find remaining positive numbers that weren't part of minus patterns
                positive_pattern = r'\d+'
                for match in re.finditer(positive_pattern, text):
                    # Only add if this position wasn't already captured by minus pattern
                    if match.start() not in seen_positions:
                        coords_with_pos.append((match.start(), int(match.group())))
                        seen_positions.update(range(match.start(), match.end()))
                
                # Sort by position in original text to maintain order
                coords_with_pos.sort(key=lambda x: x[0])
                return [num for pos, num in coords_with_pos]
            
            def get_coords_from_ocr(crop_image):
                """Read OCR once and return coordinate values"""
                custom_config = r'--oem 1 --psm 7 '
                
                try:
                    text = pytesseract.image_to_string(crop_image, lang='mc3', config=custom_config)
                    
                    # Remove commas and replace common OCR mistakes
                    text = text.replace(',', '')
                    text = text.replace('S', '5').replace('s', '5')  # Replace S with 5
                    text = text.replace('Q', '0')  # Replace Q with 0
                    
                    print(f"[DEBUG] OCR Text: {text.strip()}")
                    coords = extract_numbers_only(text)
                    
                    # Always use the last 3 numbers as coordinates
                    if len(coords) >= 3:
                        coords = coords[-3:]  # Take the last 3 numbers
                        return tuple(coords)
                except Exception as e:
                    print(f"[DEBUG] OCR error: {e}")
                
                return None
            
            # Parse coordinates - extract numbers from the text
            # Use single OCR reading for both sides
            player_coords_result = get_coords_from_ocr(crop_right)
            if player_coords_result is not None:
                # Only update if coordinates are reasonable
                if self.player_coords is None or self._coords_are_reasonable(player_coords_result, self.player_coords):
                    self.player_coords = player_coords_result
                    print(f"[DEBUG] Player Block Coordinates: {self.player_coords}")
                else:
                    print(f"[DEBUG] Ignoring invalid player coords: {player_coords_result} (previous: {self.player_coords})")
                    pydirectinput.keyDown('w')
                    sleep(0.2)
                    pydirectinput.keyUp('w')
            
            # Use single OCR reading to get targeted block coordinates from LEFT side
            target_coords_result = get_coords_from_ocr(crop_left)
            if target_coords_result is not None:
                # Only update if coordinates are reasonable
                if self.target_block_coords is None or self._coords_are_reasonable(target_coords_result, self.target_block_coords):
                    self.target_block_coords = target_coords_result
                    print(f"[DEBUG] Targeted Block Coordinates: {self.target_block_coords}")
                else:
                    print(f"[DEBUG] Ignoring invalid target coords: {target_coords_result} (previous: {self.target_block_coords})")
                    print(f"[DEBUG] Clearing target and going back to searching due to invalid coordinates")
                    self.target_block_coords = None
                    pydirectinput.keyDown('w')
                    sleep(0.2)
                    pydirectinput.keyUp('w')
            
            return self.player_coords is not None or self.target_block_coords is not None
            
        except Exception as e:
            print(f"[DEBUG] Error reading F3 coordinates: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _coords_are_reasonable(self, new_coords, old_coords):
        """Check if new coordinates are reasonable compared to old ones"""
        # Allow up to 20 blocks difference per coordinate (player can't move that fast)
        max_diff = 10000
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

            y_left = int(height * 0.001)
            h_left = int(height * 0.037)
            x_left = int(width * 0.001)
            w_left = int(width * 0.4)
            
            y_right = int(height * 0.034)
            h_right = int(height * 0.035)
            x_right = int(width * 0.80)
            w_right = int(width * 0.195)
            
            cv.rectangle(debug_img, (x_left, y_left), (x_left+w_left, y_left+h_left), (0, 255, 0), 2)
            cv.rectangle(debug_img, (x_right, y_right), (x_right+w_right, y_right+h_right), (0, 0, 255), 2)
            
            # Create a side-by-side comparison
            combined = np.hstack([debug_img, cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)])
            cv.imwrite(save_path, combined)
            print(f"[DEBUG] OCR debug image saved to {save_path}")
            
        except Exception as e:
            print(f"[DEBUG] Error creating debug image: {str(e)}")
    
    def get_best_target(self, include_distance=False):
        """Find the best target based on confidence and size combined"""
        if not self.targets:
            return None
        
        center_x = self.window_w // 2
        center_y = self.window_h // 2
        
        # Each target is now (x, y, confidence, size)
        # Calculate a score combining confidence (70%) and raw size (30%)
        targets_with_score = []
        for target in self.targets:
            x, y, conf, size = target
            # Use raw size directly - bigger trees get higher scores
            base_score = (conf) * (size * 0.01)
            
            if include_distance:
                # Add distance component (25% weight, inverted so closer is better)
                dx = x - center_x
                dy = y - center_y
                distance = (dx**2 + dy**2)**0.5
                distance_score = max(0, 500 - (distance))
                score = (base_score * 0.4) + (distance_score * 0.6)
            else:
                score = base_score
            
            targets_with_score.append((target, score))
        
        # Sort by score in descending order
        targets_with_score.sort(key=lambda t: t[1], reverse=True)
        
        best = targets_with_score[0]
        best_target = best[0]
        print(f"[DEBUG] Selected target: pos=({best_target[0]}, {best_target[1]}), conf={best_target[2]:.2f}, size={best_target[3]}, distance={distance:.2f}, score={best[1]:.2f}")
        print(f"[DEBUG] Total targets: {len(self.targets)}")
        
        # Return just (x, y) for compatibility
        return (best_target[0], best_target[1])


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
                # Start tracking time when entering SEARCHING state
                if self.searching_start_time is None:
                    self.searching_start_time = time()
                
                # Check if we've been searching for more than 5 seconds
                if time() - self.searching_start_time > 5.0:
                    print("[DEBUG] Been searching for 5s, moving randomly")
                    direction = random.choice(['w', 'a', 's', 'd'])
                    pydirectinput.keyDown(direction)
                    sleep(0.7)
                    pydirectinput.keyUp(direction)
                    self.searching_start_time = time()  # Reset timer
                
                if self.move_crosshair_to_target():
                    print("[DEBUG] Crosshair centered, transitioning to MOVING")
                    sleep(0.2)    
                    self.lock.acquire()
                    self.state = BotState.MOVING
                    self.searching_start_time = None  # Reset search timer
                    self.lock.release()
                    sleep(0.2)

            elif self.state == BotState.MINING:
                # Recenter cursor on target before mining
                #print("[DEBUG] Recentering cursor on target before mining")
                if self.move_crosshair_to_target():
                    #print("[DEBUG] Cursor centered, starting mining")
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
                original_player_coords = self.player_coords
                
                # Show F3 debug info and wait for fresh screenshot
                pydirectinput.press('F3')
                sleep(1.0)  # Longer delay to ensure F3 is displayed and captured
                
                # Read coordinates from current screenshot
                if self.screenshot is not None:
                    self.read_f3_coordinates(self.screenshot)
                
                # Hide F3 debug info
                pydirectinput.press('F3')
                
                # Check if OCR failed to read coordinates (returned None)
                if self.target_block_coords is None or self.player_coords is None:
                    print(f"[DEBUG] OCR failed to read coordinates (Target: {self.target_block_coords}, Player: {self.player_coords}), going back to searching")
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.current_target = None
                    self.target_block_coords = None
                    self.lock.release()
                    sleep(0.1)
                    continue
                
                # Calculate distance to check if OCR result is reasonable
                dx = self.target_block_coords[0] - self.player_coords[0]
                dz = self.target_block_coords[2] - self.player_coords[2]
                distance = (dx**2 + dz**2)**0.5
                
                # If distance is over 100, discard and retry OCR
                if distance > 100:
                    print(f"[DEBUG] Distance too large ({distance:.2f} blocks), retrying OCR")
                    # Move slightly to change screen view
                    random_direction = random.choice(['a', 'd'])
                    pydirectinput.keyDown(random_direction)
                    sleep(0.15)
                    pydirectinput.keyUp(random_direction)
                    self.lock.acquire()
                    self.target_block_coords = None
                    self.player_coords = None
                    self.current_target = None
                    self.state = BotState.SEARCHING
                    self.lock.release()
                    continue
                
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
        """Move crosshair to target using combined score (75% best, 25% distance)"""
        # Always use combined score with distance weighting
        if self.targets:
            target_pos = self.get_best_target(include_distance=True)
            if target_pos:
                self.current_target = target_pos
            else:
                print("[DEBUG] No valid target found")
                self.current_target = None
                return False
        else:
            print("[DEBUG] No targets found, scanning left")
            pydirectinput.moveRel(-400, 0, relative=True)
            sleep(0.3)
            self.current_target = None
            return False
        
        center_x = self.window_w // 2
        center_y = self.window_h // 2
        dx = target_pos[0] - center_x
        dy = target_pos[1] - center_y
        
        # Calculate distance to target
        distance_to_center = (dx**2 + dy**2)**0.5
       # print(f"[DEBUG] Target at {target_pos}, center at ({center_x}, {center_y})")
        #print(f"[DEBUG] Offset: dx={dx}, dy={dy}, distance={distance_to_center:.1f}")
        
        # Check if we're close enough to center
        if abs(dx) < 10 and abs(dy) < 10:
            #print(f"[DEBUG] Crosshair centered on target (within 10 pixels)")
            return True
        
        # Move towards target
        move_x = int(dx)
        move_y = int(dy)
        
        if abs(move_x) > 0 or abs(move_y) > 0:
            print(f"[DEBUG] Moving mouse by: ({move_x}, {move_y})")
            pydirectinput.moveRel(move_x, move_y, relative=True)
            sleep(0.1)
        
        return False  # Not yet centered



    def move_to_target(self):
        """Move forward to target using coordinate-based distance (X and Z only)"""
        # Calculate distance to target if we have coordinates
        if self.target_block_coords and self.player_coords:
            dx = self.target_block_coords[0] - self.player_coords[0]
            dy = self.target_block_coords[1] - self.player_coords[1]
            dz = self.target_block_coords[2] - self.player_coords[2]
            # Calculate horizontal distance only (ignore Y coordinate)
            self.target_distance = (dx**2 + dz**2)**0.5
            if self.target_distance > 1000:
                print(f"[DEBUG] Unreasonable target distance ({self.target_distance:.2f} blocks), likely OCR error. Aborting move.")
                return False
            
            print(f"[DEBUG] Horizontal distance to target: {self.target_distance:.2f} blocks (X: {dx:.1f}, Z: {dz:.1f}, Y: {dy:.1f})")
            
            # If we're close enough, we've reached the target
            if self.target_distance <= 3.5:
                print(f"[DEBUG] Reached target (within 3.5 blocks horizontally)")
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

    def detect_wood_tooltip(self, screenshot):
        """Detect wood tooltip in screenshot using OCR"""
        try:
            h, w = screenshot.shape[:2]
            crop_x1 = int(w * 0.01)
            crop_x2 = int(w * 0.3)
            crop_y1 = int(h * 0.29)
            crop_y2 = int(h * 0.325)
            tooltip_area = screenshot[crop_y1:crop_y2, crop_x1:crop_x2]

            #cv.rectangle(screenshot, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)

            #cv.imwrite('tooltip_area.png', screenshot)
            # Apply HSV filtering to isolate white text on dark background
            hsv = cv.cvtColor(tooltip_area, cv.COLOR_BGR2HSV)
            # White text: high value, low saturation
            color_lower = np.array([0, 0, 200], dtype=np.uint8)
            color_upper = np.array([180, 30, 255], dtype=np.uint8)
            mask = cv.inRange(hsv, color_lower, color_upper)
            
            #cv.imwrite('tooltip_mask.png', mask)
            # OCR on the filtered area
            custom_config = r'--oem 1 --psm 7 -l mc3'
            text = pytesseract.image_to_string(mask, config=custom_config)
            text = text.strip().lower()
            
            # Look for wood-related keywords
            wood_keywords = ['birch', 'oak', 'spruce', 'jungle', 'acacia', 'dark', 'log', 'wood']
            detected = any(keyword in text for keyword in wood_keywords)
            
            if detected:
                print(f"[DEBUG] Wood tooltip DETECTED! Text: '{text}'")
            else:
                print(f"[DEBUG] Wood tooltip not detected. Text: '{text}'")
                pydirectinput.moveRel(0, -5, relative=True)
            
            return detected
            
        except Exception as e:
            print(f"[DEBUG] Error detecting wood tooltip: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def mine_tree(self):
        print("[DEBUG] Starting mining sequence")
        
        # Press F3 to show debug info
        pydirectinput.press('F3')
        sleep(1.0)
        
        blocks_mined = 0
        max_blocks = 20  # Safety limit
        last_tooltip_time = time()
        tooltip_timeout = 2.0  # 2 seconds without tooltip = done
        previous_target_coords = None
        
        # Read initial target block coordinates
        if self.screenshot is not None:
            self.read_f3_coordinates(self.screenshot)
            if self.target_block_coords:
                previous_target_coords = self.target_block_coords
                print(f"[DEBUG] Initial target coords: {previous_target_coords}")
            
        while blocks_mined < max_blocks:
            if self.screenshot is not None:
                # FIRST BLOCK: Just look for tooltip with 5px movements
                if blocks_mined == 0:
                    tooltip_found = self.detect_wood_tooltip(self.screenshot)
                    
                    if tooltip_found:
                        print(f"[DEBUG] First block tooltip found, mining!")
                        
                        # Save coords before mining
                        self.read_f3_coordinates(self.screenshot)
                        coords_before_move = self.target_block_coords
                        
                        # Mine the block
                        pydirectinput.mouseDown()
                        sleep(3.4)
                        pydirectinput.mouseUp()
                        last_tooltip_time = time()
                        blocks_mined += 1
                        
                        # Move up for next block
                        movement = max(25, 150 - (blocks_mined * 25))
                        print(f"[DEBUG] Moving up {movement}px to find next block")
                        pydirectinput.moveRel(0, -movement, relative=True)
                        
                        # Save coords for comparison
                        previous_target_coords = coords_before_move
                    else:
                        # No tooltip - move 5px up
                        print(f"[DEBUG] No tooltip on first block, moving up 5px")
                        pydirectinput.moveRel(0, -5, relative=True)
                
                # SUBSEQUENT BLOCKS: Move with calculated amount, check tooltip, then check coords
                else:
                    tooltip_found = self.detect_wood_tooltip(self.screenshot)
                    
                    if tooltip_found:
                        print(f"[DEBUG] Tooltip found, checking coordinates")
                        
                        # Check if coordinates are correct
                        self.read_f3_coordinates(self.screenshot)
                        if self.target_block_coords and previous_target_coords is not None:
                            prev_x, prev_y, prev_z = previous_target_coords
                            curr_x, curr_y, curr_z = self.target_block_coords
                            
                            x_diff = curr_x - prev_x
                            y_diff = curr_y - prev_y
                            z_diff = curr_z - prev_z
                            
                            print(f"[DEBUG] Target coords X:{prev_x}→{curr_x} ({x_diff:+d}), Y:{prev_y}→{curr_y} ({y_diff:+d}), Z:{prev_z}→{curr_z} ({z_diff:+d})")
                            
                            # Check if only Y increased by 1 and X,Z stayed the same
                            if x_diff == 0 and y_diff == 1 and z_diff == 0:
                                print(f"[DEBUG] Coordinates correct! Mining block {blocks_mined + 1}")
                                
                                # Save coords before mining
                                coords_before_move = self.target_block_coords
                                
                                # Mine the block
                                pydirectinput.mouseDown()
                                sleep(3.4)
                                pydirectinput.mouseUp()
                                last_tooltip_time = time()
                                blocks_mined += 1
                                
                                # Move up for next block
                                if blocks_mined < max_blocks:
                                    movement = max(25, 150 - (blocks_mined * 25))
                                    print(f"[DEBUG] Moving up {movement}px to find next block")
                                    pydirectinput.moveRel(0, -movement, relative=True)
                                    
                                    # Save coords for next comparison
                                    previous_target_coords = coords_before_move
                            else:
                                # Coords not correct - move again with same amount
                                movement = max(25, 150 - (blocks_mined * 25))
                                print(f"[DEBUG] Coords not correct, moving up {movement}px again")
                                pydirectinput.moveRel(0, -movement, relative=True)
                        else:
                            print(f"[DEBUG] Could not read coordinates")
                    else:
                        # No tooltip - move again with calculated amount
                        movement = max(25, 150 - (blocks_mined * 25))
                        print(f"[DEBUG] No tooltip, moving up {movement}px")
                        pydirectinput.moveRel(0, -movement, relative=True)
                        
                        # Check timeout
                        if time() - last_tooltip_time > tooltip_timeout:
                            print(f"[DEBUG] No tooltip for {tooltip_timeout}s, mining complete")
                            break
            else:
                print("[DEBUG] No screenshot available")
                sleep(0.1)
        
        # Hide F3
        pydirectinput.press('F3')
        sleep(0.3)
        
        # Move mouse back down - sum up all the movements we made going up
        print("[DEBUG] Moving mouse back down")
        total_movement = 0
        for i in range(1, blocks_mined + 1):
            total_movement += max(25, 150 - (i * 25))
        
        if total_movement > 0:
            pydirectinput.moveRel(0, total_movement, relative=True)
            sleep(0.3)
        
        # Move forward after mining
        print("[DEBUG] Moving forward")
        pydirectinput.keyDown('w')
        sleep(0.7)
        pydirectinput.keyUp('w')
        
        print(f"[DEBUG] Mining sequence complete, mined {blocks_mined} blocks")
        return True