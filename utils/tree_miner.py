import cv2 as cv
import pytesseract
import numpy as np
import pydirectinput
from time import sleep, time

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TreeMiner:
    """Handles the tree mining sequence with tooltip detection and coordinate validation"""
    
    def __init__(self):
        self.tooltip_timeout = 3.5
        self.max_blocks = 20
    
    def detect_wood_tooltip(self, screenshot):
        """Detect wood tooltip in screenshot using OCR"""
        try:
            # Crop the left side where tooltips appear
            h, w = screenshot.shape[:2]
            crop_x1 = int(w * 0.001)
            crop_x2 = int(w * 0.3)
            crop_y1 = int(h * 0.001)
            crop_y2 = int(h * 0.15)
            
            tooltip_area = screenshot[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Apply HSV filtering to isolate white text on dark background
            hsv = cv.cvtColor(tooltip_area, cv.COLOR_BGR2HSV)
            color_lower = np.array([0, 0, 200], dtype=np.uint8)
            color_upper = np.array([180, 30, 255], dtype=np.uint8)
            mask = cv.inRange(hsv, color_lower, color_upper)
            
            # Save debug images
            cv.imwrite('other/tooltip_area.png', tooltip_area)
            cv.imwrite('other/tooltip_mask.png', mask)
            
            # OCR on the filtered area
            custom_config = r'--oem 1 --psm 6 -l mc3'
            text = pytesseract.image_to_string(mask, config=custom_config)
            text = text.strip().lower()
            
            # Look for wood-related keywords
            wood_keywords = ['birch', 'oak', 'spruce', 'jungle', 'acacia', 'dark', 'log', 'wood']
            detected = any(keyword in text for keyword in wood_keywords)
            
            if detected:
                print(f"[DEBUG] Wood tooltip DETECTED! Text: '{text}'")
            else:
                print(f"[DEBUG] Wood tooltip not detected. Text: '{text}'")
            
            return detected
            
        except Exception as e:
            print(f"[DEBUG] Error detecting wood tooltip: {str(e)}")
            return False
    
    def check_coords_correct(self, current_coords, previous_coords):
        """Check if coordinates didnt change """
        if current_coords is None or previous_coords is None:
            return False
        
        prev_x, prev_y, prev_z = previous_coords
        curr_x, curr_y, curr_z = current_coords
        
        x_diff = curr_x - prev_x
        y_diff = curr_y - prev_y
        z_diff = curr_z - prev_z
        
        print(f"[DEBUG] Target coords X:{prev_x}→{curr_x} ({x_diff:+d}), Y:{prev_y}→{curr_y} ({y_diff:+d}), Z:{prev_z}→{curr_z} ({z_diff:+d})")
        
        return (x_diff == 0 and z_diff == 0)
    
    def mine_tree(self, get_screenshot_func, read_coords_func):
        """
        Mine a tree using tooltip-based detection
        
        Args:
            get_screenshot_func: Function that returns current screenshot
            read_coords_func: Function that returns (player_coords, target_coords)
        """
        print("[DEBUG] Starting mining sequence")
        
        # Press F3 to show debug info
        pydirectinput.press('F3')
        sleep(1.0)
        
        blocks_mined = 0
        last_tooltip_time = time()
        previous_target_coords = None
        total_upward_movement = 0  # Track total mouse movement to return later
        
        # Read initial target block coordinates
        screenshot = get_screenshot_func()
        if screenshot is not None:
            _, target_coords = read_coords_func(screenshot)
            if target_coords:
                previous_target_coords = target_coords
                print(f"[DEBUG] Initial target coords: {previous_target_coords}")
        
        while blocks_mined < self.max_blocks:
            # Check timeout at the start of each iteration
            if time() - last_tooltip_time > self.tooltip_timeout:
                print(f"[DEBUG] No tooltip for {self.tooltip_timeout}s, mining complete")
                break
            
            screenshot = get_screenshot_func()
            if screenshot is not None:
                # FIRST BLOCK: Just look for tooltip with 25px movements
                if blocks_mined == 0:
                    tooltip_found = self.detect_wood_tooltip(screenshot)

                    if tooltip_found:
                        last_tooltip_time = time()
                        print(f"[DEBUG] First block tooltip found, mining!")
                        
                        # Save coords before mining
                        _, coords_before_move = read_coords_func(screenshot)
                        
                        # Mine the block
                        pydirectinput.mouseDown()
                        sleep(3.4)
                        pydirectinput.mouseUp()
                        last_tooltip_time = time()
                        blocks_mined += 1
                        
                        # Move up for next block
                        movement = max(20, 150 - (blocks_mined * 20))
                        print(f"[DEBUG] Moving up {movement}px to find next block")
                        pydirectinput.moveRel(0, -movement, relative=True)
                        total_upward_movement += movement
                        
                        # Save coords for comparison
                        previous_target_coords = coords_before_move
                    else:
                        # No tooltip - move 25px up
                        print(f"[DEBUG] No tooltip on first block, moving up 25px")
                        pydirectinput.moveRel(0, -25, relative=True)
                        total_upward_movement += 25
                
                # SUBSEQUENT BLOCKS: Move with calculated amount, check tooltip, then check coords
                else:
                    tooltip_found = self.detect_wood_tooltip(screenshot)
                    
                    if tooltip_found:
                        print(f"[DEBUG] Tooltip found, checking coordinates")
                        
                        # Check if coordinates are correct
                        _, current_coords = read_coords_func(screenshot)
                        
                        if self.check_coords_correct(current_coords, previous_target_coords):
                            print(f"[DEBUG] Coordinates correct! Mining block {blocks_mined + 1}")
                            
                            # Save coords before mining
                            coords_before_move = current_coords
                            
                            # Mine the block
                            pydirectinput.mouseDown()
                            sleep(3.4)
                            pydirectinput.mouseUp()
                            last_tooltip_time = time()
                            blocks_mined += 1
                            
                            # Move up for next block
                            if blocks_mined < self.max_blocks:
                                movement = max(20, 150 - (blocks_mined * 20))
                                print(f"[DEBUG] Moving up {movement}px to find next block")
                                pydirectinput.moveRel(0, -movement, relative=True)
                                total_upward_movement += movement
                                
                                # Save coords for next comparison
                                previous_target_coords = coords_before_move
                        else:
                            # Coords not correct - move again with same amount
                            movement = max(20, 150 - (blocks_mined * 20))
                            print(f"[DEBUG] Coords not correct, moving up {movement}px again")
                            pydirectinput.moveRel(0, -movement, relative=True)
                            total_upward_movement += movement
                    else:
                        # No tooltip - move again with calculated amount
                        movement = max(20, 150 - (blocks_mined * 20))
                        print(f"[DEBUG] No tooltip, moving up {movement}px")
                        pydirectinput.moveRel(0, -movement, relative=True)
                        total_upward_movement += movement
            else:
                print("[DEBUG] No screenshot available")
                sleep(0.1)
        
        # Hide F3
        pydirectinput.press('F3')
        sleep(0.3)
        
        # Move mouse back down to starting position
        print(f"[DEBUG] Moving mouse back down {total_upward_movement}px to starting position")
        if total_upward_movement > 0:
            pydirectinput.moveRel(0, total_upward_movement - 100, relative=True)
            sleep(0.3)
        
        # Move forward after mining
        print("[DEBUG] Moving forward")
        pydirectinput.keyDown('w')
        sleep(0.7)
        pydirectinput.keyUp('w')
        
        print(f"[DEBUG] Mining sequence complete, mined {blocks_mined} blocks")
        return True
