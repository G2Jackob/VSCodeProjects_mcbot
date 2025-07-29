import cv2 as cv
import pyautogui
import pydirectinput
from time import sleep, time
from threading import Thread, Lock


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
       
        self.window_offset = window_offset
        self.window_w = window_size[0]
        self.window_h = window_size[1]

        self.wood_tooltip = cv.imread('wood_tooltip.jpg', cv.IMREAD_UNCHANGED)
        
        self.state = BotState.INITIALIZING
        self.timestamp = time()

    

    def confirm_tooltip(self, target_position):
        result = cv.matchTemplate(self.screenshot, self.wood_tooltip, cv.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if max_val >= self.TOOLTIP_MATCH_THRESHOLD:
            return True
        return False


    def click_next_target(self):
        targets = self.targets_ordered_by_distance(self.targets)
        target_i = 0
        
        found_wood = False
        while not found_wood and target_i < len(targets):
            if self.stopped:
                break
            target_pos = targets[target_i]
            screen_x, screen_y = self.get_screen_position(target_pos)

            pydirectinput.moveTo(x=screen_x, y=screen_y)
            sleep(1.2)
            pydirectinput.press('F3')
            if self.confirm_tooltip(target_pos):
                found_wood = True
                pydirectinput.click()
                sleep(0.5)
            pydirectinput.press('F3')
            target_i += 1
        return found_wood


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
        my_pos= (self.window_w/2, self.window_h/2)

        def distance_to_target(pos):
            return cv.sqrt((pos[0] - my_pos[0]) ** 2 + (pos[1] - my_pos[1]) ** 2)
        targets.sort(key=distance_to_target)

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
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()
            elif self.state == BotState.SEARCHING:
                    success = self.click_next_target()
                    if success:
                        self.lock.acquire()
                        self.state = BotState.MOVING
                        self.lock.release()
                    else:
                        pass
            elif self.state == BotState.MOVING:
                if not self.have_stopped_moving():
                    sleep(0.5)
                else:
                    self.lock.acquire()
                    self.timestamp = time()
                    self.state = BotState.MINING
                    self.lock.release()
            elif self.state == BotState.MINING:
                if time() > self.timestamp + self.MINING_TIME:
                    self.lock.acquire()
                    self.state = BotState.SEARCHING
                    self.lock.release()