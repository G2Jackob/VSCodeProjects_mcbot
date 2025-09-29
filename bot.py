import cv2 as cv
import pyautogui
import pydirectinput
from time import sleep, time
from threading import Thread, Lock
import random

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

    def __init__(self, window_offset, window_size):
        self.lock = Lock()
        self.offset_x, self.offset_y = window_offset
        self.window_w, self.window_h = window_size
        self.state = BotState.INITIALIZING
        self.timestamp = time()

        self.wood_tooltip = cv.imread('wood_tooltip.jpg', cv.IMREAD_UNCHANGED)
        
    

    def confirm_tooltip(self, target_position):
        result = cv.matchTemplate(self.screenshot, self.wood_tooltip, cv.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if max_val >= self.TOOLTIP_MATCH_THRESHOLD:
            return True
        return False


    def click_next_target(self):
        if self.move_crosshair_to_target():
            # Press F3 to show tooltip
            pydirectinput.press('F3')
            sleep(0.5)
            
            if self.confirm_tooltip(self.screenshot):
              
                sleep(0.2)
                
                return True
            
            pydirectinput.press('F3')  # Hide tooltip if not confirmed
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


    def targets_ordered_by_distance(self, targets):
        center_x = self.window_w // 2
        center_y = self.window_h // 2

        def distance_to_target(target):
            dx = target[0] - center_x
            dy = target[1] - center_y
            return (dx ** 2 + dy ** 2) ** 0.5

        # Only sort if targets is not empty
        if targets:
            targets = sorted(targets, key=distance_to_target)
        return targets

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
                    print("[DEBUG] Aligned with target, checking with F3")
                    pydirectinput.press('F3')
                    sleep(0.5)
                    
                    if self.confirm_tooltip((self.window_w // 2, self.window_h // 2)):
                        print("[DEBUG] Wood confirmed, starting to mine")
                        pydirectinput.press('F3')
                        sleep(0.2)
                        self.lock.acquire()
                        self.state = BotState.MINING
                        self.lock.release()
                    else:
                        print("[DEBUG] Not wood, continuing search")
                        pydirectinput.press('F3')
                        sleep(0.2)

            elif self.state == BotState.MINING:
                if self.mine_tree():
                    print("[DEBUG] Mining complete, searching for next target")
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()

            sleep(0.1)

    def move_crosshair_to_target(self):
        """Align crosshair with target using controlled mouse movement"""
        targets = self.targets_ordered_by_distance(self.targets)
        print(f"[DEBUG] Moving crosshair. Targets found: {len(targets)}")
        
        if not targets:
            return False
            
        target_pos = targets[0]
        center_x = self.window_w // 2
        center_y = self.window_h // 2
        dx = target_pos[0] - center_x
        dy = target_pos[1] - center_y
        
        # Much higher sensitivity for faster movement
        sensitivity = 0.5
        print(f"[DEBUG] Moving mouse by: dx={dx}, dy={dy}")
        
        # Move X axis (larger movement)
        move_x = int(dx * sensitivity)
        if abs(move_x) > 0:
            pydirectinput.moveRel(move_x, 0, relative=True)
            sleep(0.1)
        
        # Move Y axis (larger movement)
        move_y = int(dy * sensitivity)
        if abs(move_y) > 0:
            pydirectinput.moveRel(0, move_y, relative=True)
            sleep(0.1)
        
        # Return True if we're close enough to target
        return abs(dx) < 10 and abs(dy) < 10

    def move_to_target(self):
        """Move forward to target"""
        targets = self.targets_ordered_by_distance(self.targets)
        if not targets:
            return False
        
        print("[DEBUG] Moving to target")
        initial_pos = self.targets
        
        # Walk forward for a longer duration
        print("[DEBUG] Walking forward")
        pydirectinput.keyDown('w')
        sleep(1.0)
        # Check if we're stuck
        if self.check_if_stuck():
            print("[DEBUG] Stuck, attempting to jump")
            pydirectinput.press('space')
            sleep(1.0)
        
        pydirectinput.keyUp('w')
        
        # Ensure we've actually moved
        return self.targets != initial_pos

    def check_if_stuck(self):
        """Check if we're stuck by comparing positions"""
        initial_targets = self.targets
        sleep(0.5)
        current_targets = self.targets
        return initial_targets == current_targets

    def mine_tree(self):
        """Simple mining sequence - just mine once"""
        print("[DEBUG] Starting mining sequence")
        
        # Single mining action
        pydirectinput.mouseDown()
        sleep(2.0)
        pydirectinput.mouseUp()
        sleep(0.5)
        
        # Random strafe after mining to avoid getting stuck
        if random.randint(0, 1):
            pydirectinput.keyDown('d')
        else:
            pydirectinput.keyDown('a')
        sleep(0.5)
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')
        
        return True