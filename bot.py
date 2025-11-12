import cv2 as cv
import pyautogui
import pydirectinput
from time import sleep, time
from threading import Thread, Lock
import random
import pytesseract
import numpy as np
import re

# Use user's Tesseract install path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class BotState:
    INITIALIZING = 0
    SEARCHING = 1
    MOVING = 2
    MINING = 3
    CRAFTING = 4

class McBot:

    INITIALIZING_TIME = 6
    MINING_TIME = 5
    MOVEMENT_STOPPED_THRESHOLD = 0.95
    TOOLTIP_MATCH_THRESHOLD = 0.75
    WOOD_LOG_THRESHOLD = 0.7  # Threshold for wood log pattern matching

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
    wood_log_icon = None  # Template for wood log icon
    wood_count = 0  # Tracked wood logs

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
        self.wood_count = 0

        self.wood_tooltip = cv.imread('wood_tooltip.jpg', cv.IMREAD_UNCHANGED)
        
        # Load wood log icon template for pattern matching
        try:
            self.wood_log_icon = cv.imread('wood_log_icon.png', cv.IMREAD_UNCHANGED)
            if self.wood_log_icon is not None:
                print("[DEBUG] Wood log icon template loaded")
            else:
                print("[DEBUG] Warning: wood_log_icon.png not found, crafting disabled")
        except:
            print("[DEBUG] Warning: Could not load wood_log_icon.png, crafting disabled")
        
    def read_f3_coordinates(self, screenshot):
        """Read player and targeted block coordinates from F3 debug screen using proportional positioning"""
        try:
            # Convert screenshot to grayscale for better OCR
            gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Scale up for better OCR quality
            scale_factor = 2
            gray_scaled = cv.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv.INTER_CUBIC)
            scaled_height, scaled_width = gray_scaled.shape
            
            # Use proportions of the TOP HALF of the screen for ROI calculations
            top_portion = gray_scaled[0:scaled_height//2, :]
            top_h, top_w = top_portion.shape

            # "Block:" coordinates (player) are in the top-right area (use proportions of top half)
            block_x_start = int(top_w * 0.8280)
            block_x_end = top_w
            # Y proportions relative to top half (derived from previous full-screen props divided by 0.5)
            block_y_start = int(top_h * 0.195)   # ~0.0975/0.5
            block_y_end = int(top_h * 0.235)     # ~0.1175/0.5

            roi_block = top_portion[block_y_start:block_y_end, block_x_start:block_x_end]

            # "Targeted Block:" coordinates are in top-left area of the top half
            target_x_start = int(top_w * 0.0029)
            target_x_end = int(top_w * 0.2313)
            target_y_start = int(top_h * 0.1876)  # ~0.0938/0.5
            target_y_end = int(top_h * 0.2374)    # ~0.1187/0.5

            roi_target = top_portion[target_y_start:target_y_end, target_x_start:target_x_end]
            
            # Use the blog's HSV masking approach to isolate Minecraft debug text
            # Scale the color screenshot and take the top half for ROI calculations
            color_scaled = cv.resize(screenshot, (scaled_width, scaled_height), interpolation=cv.INTER_CUBIC)
            color_top = color_scaled[0:scaled_height//2, :]

            # Follow the blog's pipeline exactly: convert to RGB -> HSV, mask, bitwise -> gray -> Otsu
            # Convert BGR->RGB to match the blog's RGB pipeline
            color_top_rgb = cv.cvtColor(color_top, cv.COLOR_BGR2RGB)
            hsv_top = cv.cvtColor(color_top_rgb, cv.COLOR_RGB2HSV)
            # Blog suggested values for isolating the white-gray debug text
            color_lower = np.array([100, 255, 220])
            color_upper = np.array([200, 255, 230])
            mask_top = cv.inRange(hsv_top, color_lower, color_upper)
            # Fallback to broader white range if strict mask yields almost nothing
            if cv.countNonZero(mask_top) < 20:
                mask_top = cv.inRange(hsv_top, np.array([0, 0, 200]), np.array([179, 90, 255]))

            # Apply mask to the RGB top image
            result_rgb = cv.bitwise_and(color_top_rgb, color_top_rgb, mask=mask_top)
            # Save debug images
            try:
                cv.imwrite("debug_top_mask.png", mask_top)
                cv.imwrite("debug_top_isolated_color.png", cv.cvtColor(result_rgb, cv.COLOR_RGB2BGR))
            except Exception:
                pass

            # Convert the masked RGB result to grayscale like the blog and threshold with Otsu
            top_gray_from_result = cv.cvtColor(result_rgb, cv.COLOR_RGB2GRAY)
            top_gray_from_result = cv.medianBlur(top_gray_from_result, 3)
            # Threshold full top area (blog does this) to produce a binary image for cropping
            top_thresh_full = cv.threshold(top_gray_from_result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

            # Prepare a focused top-right crop for XYZ OCR: upscale and threshold using Otsu
            # User configured XYZ to appear in the top-right; crop that area to reduce noise
            # Expand the crop upward so it doesn't cut off the top line
            xyz_x_start = int(top_w * 0.65)
            xyz_x_end = top_w
            xyz_y_start = 0
            xyz_y_end = int(top_h * 0.20)

            # Ensure bounds are valid
            xyz_x_start = max(0, min(xyz_x_start, top_w - 1))
            xyz_x_end = max(1, min(xyz_x_end, top_w))
            xyz_y_start = max(0, min(xyz_y_start, top_h - 1))
            xyz_y_end = max(1, min(xyz_y_end, top_h))

            xyz_roi = top_thresh_full[xyz_y_start:xyz_y_end, xyz_x_start:xyz_x_end]
            if xyz_roi.size == 0:
                # Fallback to whole top if crop failed for any reason
                xyz_roi = top_thresh_full

            # Save xyz crop for debugging
            try:
                cv.imwrite("debug_xyz_crop.png", xyz_roi)
            except Exception:
                pass

            xyz_large = cv.resize(xyz_roi, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
            _, xyz_thresh = cv.threshold(xyz_large, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # Morphological opening to remove small noise
            kernel = np.ones((2, 2), np.uint8)
            xyz_thresh = cv.morphologyEx(xyz_thresh, cv.MORPH_OPEN, kernel)

            # Use a conservative PSM and the mc traineddata; don't attempt heavy post-corrections here
            text_top = pytesseract.image_to_string(xyz_thresh, config='--psm 6', lang='mc')
            print(f"[DEBUG] Top-right (XYZ) OCR: {repr(text_top[:300])}")

            # For ROIs, crop from the blog-style thresholded top image and upscale
            roi_block = top_thresh_full[block_y_start:block_y_end, block_x_start:block_x_end]
            roi_target = top_thresh_full[target_y_start:target_y_end, target_x_start:target_x_end]

            roi_block_large = cv.resize(roi_block, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
            roi_target_large = cv.resize(roi_target, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)

            _, roi_block_thresh = cv.threshold(roi_block_large, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            _, roi_target_thresh = cv.threshold(roi_target_large, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            
            # Save debug images showing the ROI regions
            try:
                # Save processed ROIs
                cv.imwrite("debug_block_roi.png", roi_block_thresh)
                cv.imwrite("debug_target_roi.png", roi_target_thresh)
                
                # Save full screenshot with rectangles showing ROI locations
                debug_full = cv.cvtColor(gray_scaled, cv.COLOR_GRAY2BGR)
                # Draw Block ROI in RED
                cv.rectangle(debug_full, 
                           (block_x_start, block_y_start), 
                           (scaled_width, block_y_end), 
                           (0, 0, 255), 3)
                cv.putText(debug_full, "Block ROI", 
                          (block_x_start + 5, block_y_start - 5),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw Target ROI in GREEN
                cv.rectangle(debug_full, 
                           (target_x_start, target_y_start), 
                           (target_x_end, target_y_end), 
                           (0, 255, 0), 3)
                cv.putText(debug_full, "Target ROI", 
                          (target_x_start + 5, target_y_start - 5),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv.imwrite("debug_full_with_rois.png", debug_full)
                print("[DEBUG] Saved debug images with ROI markers")
            except Exception as e:
                print(f"[DEBUG] Could not save debug images: {e}")
            
            # OCR config optimized for single line of text
            ocr_config = '--psm 7'

            # (Already performed a simple top-area OCR above using grayscale.)

            # Read player Block coordinates (prefer XYZ line if present)
            player_coords_found = False
            xyz_pattern = r'XYZ[:\s]*([-]?\d+\.?\d*)\s*/\s*([-]?\d+\.?\d*)\s*/\s*([-]?\d+\.?\d*)'
            xyz_match = re.search(xyz_pattern, text_top)
            if xyz_match:
                try:
                    x = int(round(float(xyz_match.group(1))))
                    y = int(round(float(xyz_match.group(2))))
                    z = int(round(float(xyz_match.group(3))))
                    new_coords = (x, y, z)
                    if self.player_coords is None or self._coords_are_reasonable(new_coords, self.player_coords):
                        self.player_coords = new_coords
                        player_coords_found = True
                        print(f"[DEBUG] Player XYZ Coordinates: {self.player_coords}")
                except Exception:
                    pass
            else:
                # Fallback: look for a line containing slashes ("/"), which is typical for XYZ
                found = False
                for line in text_top.splitlines():
                    lower = line.lower()
                    if 'chunk' in lower or 'facing' in lower:
                        continue
                    if '/' in line:
                        nums = re.findall(r'-?\d+\.?\d*', line)
                        if len(nums) >= 3:
                            try:
                                x = int(round(float(nums[0])))
                                y = int(round(float(nums[1])))
                                z = int(round(float(nums[2])))
                                new_coords = (x, y, z)
                                if self.player_coords is None or self._coords_are_reasonable(new_coords, self.player_coords):
                                    self.player_coords = new_coords
                                    player_coords_found = True
                                    print(f"[DEBUG] Player XYZ Coordinates (line-with-slash fallback): {self.player_coords}")
                                    found = True
                                    break
                            except Exception:
                                continue
                if not found:
                    # Final fallback: only use raw numbers if the OCR output doesn't look like a chunk/facing line
                    lower_all = text_top.lower()
                    if 'chunk' not in lower_all and 'facing' not in lower_all:
                        nums = re.findall(r'-?\d+\.?\d*', text_top)
                        if len(nums) >= 3:
                            try:
                                x = int(round(float(nums[0])))
                                y = int(round(float(nums[1])))
                                z = int(round(float(nums[2])))
                                new_coords = (x, y, z)
                                if self.player_coords is None or self._coords_are_reasonable(new_coords, self.player_coords):
                                    self.player_coords = new_coords
                                    player_coords_found = True
                                    print(f"[DEBUG] Player XYZ Coordinates (numeric fallback): {self.player_coords}")
                            except Exception:
                                pass

            # Fallback: OCR the block ROI for 'Block:' if XYZ not found
            text_block = pytesseract.image_to_string(roi_block_thresh, config=ocr_config, lang='mc')
            print(f"[DEBUG] Block ROI text: {repr(text_block[:100])}")

            # Read Targeted Block coordinates using preprocessed image
            text_target = pytesseract.image_to_string(roi_target_thresh, config=ocr_config, lang='mc')
            print(f"[DEBUG] Target ROI text: {repr(text_target[:100])}")
            
            # Parse player Block coordinates (spaces only, no commas)
            # Look for "Block:" followed by 3 numbers separated by spaces
            block_pattern = r'Block:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)'
            block_match = re.search(block_pattern, text_block)
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
            else:
                # Fallback: try to find a line mentioning 'Block' or take the first numeric line
                used = False
                for line in text_block.splitlines():
                    lower = line.lower()
                    if 'chunk' in lower or 'facing' in lower:
                        continue
                    if 'block' in lower:
                        nums = re.findall(r'-?\d+', line)
                        if len(nums) >= 3:
                            try:
                                new_coords = (int(nums[0]), int(nums[1]), int(nums[2]))
                                if self.player_coords is None or self._coords_are_reasonable(new_coords, self.player_coords):
                                    self.player_coords = new_coords
                                    print(f"[DEBUG] Player Block Coordinates (block-line fallback): {self.player_coords}")
                                else:
                                    print(f"[DEBUG] Ignoring invalid player coords: {new_coords} (previous: {self.player_coords})")
                                used = True
                                break
                            except Exception:
                                continue
                if not used:
                    # Use first numeric line that doesn't look like chunk/facing
                    for line in text_block.splitlines():
                        lower = line.lower()
                        if 'chunk' in lower or 'facing' in lower:
                            continue
                        nums = re.findall(r'-?\d+', line)
                        if len(nums) >= 3:
                            try:
                                new_coords = (int(nums[0]), int(nums[1]), int(nums[2]))
                                if self.player_coords is None or self._coords_are_reasonable(new_coords, self.player_coords):
                                    self.player_coords = new_coords
                                    print(f"[DEBUG] Player Block Coordinates (numeric-line fallback): {self.player_coords}")
                                else:
                                    print(f"[DEBUG] Ignoring invalid player coords: {new_coords} (previous: {self.player_coords})")
                                used = True
                                break
                            except Exception:
                                continue
                    if not used:
                        print(f"[DEBUG] No player Block: match found in text")
            
            # Parse Targeted Block coordinates (with commas)
            target_pattern = r'Targeted Block:\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)'
            target_match = re.search(target_pattern, text_target)
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
            else:
                # Fallback: try to find a numeric line that isn't a chunk/facing line
                used = False
                for line in text_target.splitlines():
                    lower = line.lower()
                    if 'chunk' in lower or 'facing' in lower:
                        continue
                    nums = re.findall(r'-?\d+', line)
                    if len(nums) >= 3:
                        try:
                            new_coords = (int(nums[0]), int(nums[1]), int(nums[2]))
                            if self.target_block_coords is None or self._coords_are_reasonable(new_coords, self.target_block_coords):
                                self.target_block_coords = new_coords
                                print(f"[DEBUG] Targeted Block Coordinates (numeric-line fallback): {self.target_block_coords}")
                            else:
                                print(f"[DEBUG] Ignoring invalid target coords: {new_coords} (previous: {self.target_block_coords})")
                            used = True
                            break
                        except Exception:
                            continue
                if not used:
                    print(f"[DEBUG] No Targeted Block: match found in text")
            
            return self.player_coords is not None or self.target_block_coords is not None
            
        except Exception as e:
            print(f"[DEBUG] Error reading F3 coordinates: {str(e)}")
            return False
    
    def _fix_ocr_errors(self, text):
        """Fix common OCR mistakes with Minecraft font"""
        fixed = text
        
        # Fix common word misreads
        fixed = fixed.replace('Glock:', 'Block:')
        fixed = fixed.replace('Targeted Glock:', 'Targeted Block:')
        
        # Fix missing "Block:" or "Targeted Block:" prefix
        # If line starts with ":" it likely missed the word before
        if fixed.startswith(':-') or fixed.startswith(': -') or fixed.startswith(':'):
            # Could be either "Block:" or "Targeted Block:"
            # Check if it has commas (Targeted Block format)
            if ',' in fixed:
                fixed = 'Targeted Block' + fixed
            else:
                fixed = 'Block' + fixed
        
        # Fix missing spaces in coordinate strings
        # ":-396329" should be "Block: -39 63 29"
        # ":-4,64,29" should be "Targeted Block: -4, 64, 29"
        # Look for patterns like ": -<digits><digits><digits>" without spaces
        block_no_space = re.search(r'(Block:)\s*(-?\d{2})(\d{2})(\d{2})\s*$', fixed)
        if block_no_space:
            fixed = f"{block_no_space.group(1)} {block_no_space.group(2)} {block_no_space.group(3)} {block_no_space.group(4)}"
        
        # Fix tilde to minus before numbers (for negative coordinates)
        fixed = re.sub(r'(Block:)\s*~', r'\1 -', fixed)
        fixed = re.sub(r'(Targeted Block:)\s*~', r'\1 -', fixed)
        
        # Fix S to 5 when it appears as a digit
        fixed = re.sub(r'\bS(?=\d)', '5', fixed)  # S before digit
        fixed = re.sub(r'(?<=\d)S\b', '5', fixed)  # S after digit
        fixed = re.sub(r'(?<=\s)S(?=\s)', '5', fixed)  # S between spaces
        
        # Fix O to 0 in number contexts
        fixed = re.sub(r'(?<=\d)O(?=[\d,\s-])', '0', fixed)
        fixed = re.sub(r'(?<=[\s,:-])O(?=\d)', '0', fixed)
        
        # Fix I or l to 1 in number contexts
        fixed = re.sub(r'(?<=\d)[Il](?=[\d,\s-])', '1', fixed)
        fixed = re.sub(r'(?<=[\s,:-])[Il](?=\d)', '1', fixed)
        
        return fixed
    
    def _coords_are_reasonable(self, new_coords, old_coords):
        """Check if new coordinates are reasonable compared to old ones"""
        # Allow up to 20 blocks difference per coordinate (increased from 10)
        # Player can move faster when teleporting or in vehicles
        max_diff = 20
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
                        print("[DEBUG] Mining complete, counting wood")
                        self.count_wood_logs()
                        
                        # Check if we have enough wood to craft
                        if self.wood_count >= 10:
                            print(f"[DEBUG] Collected {self.wood_count} wood logs, transitioning to CRAFTING")
                            self.lock.acquire()
                            self.state = BotState.CRAFTING
                            self.lock.release()
                        else:
                            print(f"[DEBUG] Wood count: {self.wood_count}/10, clearing target")
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
                    
            elif self.state == BotState.CRAFTING:
                print("[DEBUG] Starting crafting sequence")
                if self.craft_planks():
                    print("[DEBUG] Crafting complete, resetting wood count")
                    self.wood_count = 0
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()
                    
            elif self.state == BotState.MOVING:
                # Press F3 to show debug info and read coordinates
                pydirectinput.press('F3')
                sleep(0.5)  # Increased wait time for F3 to fully display
                
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
        sleep(3.5)
        pydirectinput.mouseUp()
        sleep(0.5)
        
        
        
        return True
    
    def count_wood_logs(self):
        """Count wood logs in inventory using pattern matching"""
        if self.wood_log_icon is None or self.screenshot is None:
            print("[DEBUG] Cannot count wood - template or screenshot missing")
            return
        
        try:
            # Open inventory
            pydirectinput.press('e')
            sleep(0.5)
            
            # Get current screenshot (inventory open)
            # Note: You'll need to capture a new screenshot here
            # For now, using the existing screenshot
            
            # Define hotbar region (bottom of screen, proportional)
            hotbar_y_start = int(self.window_h * 0.85)  # Bottom 15% of screen
            hotbar_region = self.screenshot[hotbar_y_start:self.window_h, 0:self.window_w]
            
            # Match wood log icon in hotbar
            result = cv.matchTemplate(hotbar_region, self.wood_log_icon, cv.TM_CCOEFF_NORMED)
            
            # Find all matches above threshold
            locations = cv.findNonZero((result >= self.WOOD_LOG_THRESHOLD).astype('uint8'))
            
            # Count unique matches (avoiding duplicates from overlapping)
            if locations is not None:
                # Simple approximation: divide by expected match area
                self.wood_count = len(locations) // 100  # Rough estimate
            else:
                self.wood_count = 0
            
            print(f"[DEBUG] Found approximately {self.wood_count} wood log stacks in hotbar")
            
            # Close inventory
            pydirectinput.press('e')
            sleep(0.5)
            
        except Exception as e:
            print(f"[DEBUG] Error counting wood logs: {str(e)}")
            # Make sure inventory is closed
            pydirectinput.press('e')
            sleep(0.3)
    
    def craft_planks(self):
        """Craft planks from wood logs using proportional positioning"""
        try:
            print("[DEBUG] Opening inventory for crafting")
            pydirectinput.press('e')
            sleep(0.7)
            
            # Calculate positions based on proportions from the screenshots
            # The inventory window is centered, crafting grid is in upper right
            
            # Wood in hotbar (bottom left of inventory, first slot after off-hand)
            # Off-hand is at ~0.405, first hotbar slot is at ~0.43
            hotbar_slot1_x = int(self.window_w * 0.43) + self.offset_x
            hotbar_slot1_y = int(self.window_h * 0.67) + self.offset_y  # Bottom section of inventory
            
            # 2x2 Crafting grid (upper right of inventory)
            # Grid starts around 0.55 width, 0.39 height
            craft_grid_x = int(self.window_w * 0.56) + self.offset_x  # Top-left cell of 2x2 grid
            craft_grid_y = int(self.window_h * 0.395) + self.offset_y
            
            # Crafting output slot (right of the arrow)
            # Output is around 0.61 width, same height as grid
            craft_output_x = int(self.window_w * 0.61) + self.offset_x
            craft_output_y = int(self.window_h * 0.41) + self.offset_y
            
            print("[DEBUG] Clicking on wood logs in hotbar")
            # Click on wood in hotbar
            pydirectinput.click(x=hotbar_slot1_x, y=hotbar_slot1_y)
            sleep(0.3)
            
            print("[DEBUG] Placing wood in crafting grid")
            # Place one log in crafting grid (top-left slot)
            pydirectinput.click(x=craft_grid_x, y=craft_grid_y)
            sleep(0.3)
            
            print("[DEBUG] Collecting crafted planks")
            # Shift-click output to collect all planks
            pydirectinput.keyDown('shift')
            for _ in range(3):  # Click multiple times to collect all
                pydirectinput.click(x=craft_output_x, y=craft_output_y)
                sleep(0.2)
            pydirectinput.keyUp('shift')
            sleep(0.3)
            
            print("[DEBUG] Closing inventory")
            pydirectinput.press('e')
            sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"[DEBUG] Error during crafting: {str(e)}")
            # Make sure inventory is closed
            pydirectinput.press('e')
            sleep(0.3)
            return False