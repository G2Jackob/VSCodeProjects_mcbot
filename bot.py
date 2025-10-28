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

    def __init__(self, window_offset, window_size):
        self.lock = Lock()
        self.offset_x, self.offset_y = window_offset
        self.window_w, self.window_h = window_size
        self.state = BotState.INITIALIZING
        self.timestamp = time()
        self.player_coords = None
        self.target_block_coords = None

        self.wood_tooltip = cv.imread('wood_tooltip.jpg', cv.IMREAD_UNCHANGED)
        
    def read_f3_coordinates(self, screenshot):
        """Read player and targeted block coordinates from F3 debug screen"""
        try:
            # Convert screenshot to grayscale for better OCR
            gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Apply thresholding to improve OCR accuracy
            _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
            
            # Read from LEFT side for "Targeted Block:"
            roi_left = thresh[0:height//2, 0:width//3]
            text_left = pytesseract.image_to_string(roi_left, config='--psm 6')
            
            # Read from RIGHT side for player "Block:" coordinates
            roi_right = thresh[0:height//2, 2*width//3:width]
            text_right = pytesseract.image_to_string(roi_right, config='--psm 6')
            
            # Parse player Block coordinates from RIGHT side (spaces only, no commas)
            # Look for "Block:" followed by 3 numbers separated by spaces
            block_pattern = r'Block:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)'
            block_match = re.search(block_pattern, text_right)
            if block_match:
                self.player_coords = (int(block_match.group(1)), 
                                     int(block_match.group(2)), 
                                     int(block_match.group(3)))
                print(f"[DEBUG] Player Block Coordinates: {self.player_coords}")
            
            # Parse Targeted Block coordinates from LEFT side (with commas)
            target_pattern = r'Targeted Block:\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)'
            target_match = re.search(target_pattern, text_left)
            if target_match:
                self.target_block_coords = (int(target_match.group(1)), 
                                           int(target_match.group(2)), 
                                           int(target_match.group(3)))
                print(f"[DEBUG] Targeted Block Coordinates: {self.target_block_coords}")
            
            return self.player_coords is not None or self.target_block_coords is not None
            
        except Exception as e:
            print(f"[DEBUG] Error reading F3 coordinates: {str(e)}")
            return False



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
                    sleep(0.2)    
                    self.lock.acquire()
                    self.state = BotState.MOVING
                    self.lock.release()
                    sleep(0.2)

            elif self.state == BotState.MINING:
                if self.mine_tree():
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()
                    
            elif self.state == BotState.MOVING:
                # Press F3 to show debug info
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
                    self.lock.release()
            sleep(0.1)

    def move_crosshair_to_target(self):
        targets = self.targets
        print(f"[DEBUG] Moving crosshair. Targets found: {len(targets)}")
        
        if not targets:
            return False
            
        target_pos = targets[0]
        center_x = self.window_w // 2
        center_y = self.window_h // 2
        dx = target_pos[0] - center_x
        dy = target_pos[1] - center_y
        
        # Increase sensitivity for more noticeable movement
        sensitivity = 1.0
        
        move_x = int(dx * sensitivity)
        move_y = int(dy * sensitivity)
        
        if abs(move_x) > 0 or abs(move_y) > 0:
            print(f"[DEBUG] Moving mouse by: ({move_x}, {move_y})")
            pydirectinput.moveRel(move_x, move_y, relative=True)
            sleep(0.1)
        
        return abs(dx) < 10 and abs(dy) < 10

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
        """Move forward to target"""
        targets = self.targets
        if not targets:
            return False
        
        print("[DEBUG] Moving to target")
        initial_pos = self.targets
        
        print("[DEBUG] Walking forward")
        pydirectinput.keyDown('w')
        sleep(0.5)
        if self.check_if_stuck():
            print("[DEBUG] Stuck, attempting to jump")
            pydirectinput.press('space')
            sleep(1.0)
        
        pydirectinput.keyUp('w')
        
        return self.targets != initial_pos

    def check_if_stuck(self):
        """Check if we're stuck by comparing positions"""
        initial_targets = self.targets
        sleep(0.5)
        current_targets = self.targets
        return initial_targets == current_targets

    def mine_tree(self):
        print("[DEBUG] Starting mining sequence")
        
        # Single mining action
        pydirectinput.mouseDown()
        sleep(3.0)
        pydirectinput.mouseUp()
        sleep(0.5)
        
        
        
        return True