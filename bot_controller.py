import pydirectinput
from time import sleep, time
from threading import Thread, Lock
from utils.coordinate_reader import CoordinateReader
from utils.tree_miner import TreeMiner
from utils.navigation import TargetSelector, NavigationController

class BotState:
    INITIALIZING = 0
    SEARCHING = 1
    MOVING = 2
    MINING = 3
    
    NAMES = {
        0: "INITIALIZING",
        1: "SEARCHING",
        2: "MOVING",
        3: "MINING"
    }

class McBot:
    """Main bot controller that coordinates all bot activities"""
    
    INITIALIZING_TIME = 6
    
    def __init__(self, window_offset, window_size):
        self.lock = Lock()
        self.offset_x, self.offset_y = window_offset
        self.window_w, self.window_h = window_size
        
        # Bot state
        self.stopped = True
        self.state = BotState.INITIALIZING
        self.timestamp = time()
        
        # Screenshot and targets
        self.screenshot = None
        self.targets = []
        self.current_target = None
        
        # Coordinates
        self.player_coords = None
        self.target_block_coords = None
        self.target_distance = None
        self.expected_target_coords = None  # Expected coords when entering MOVING state
        
        # Timers
        self.searching_start_time = None
        
        # OCR failure tracking
        self.ocr_fail_count = 0
        
        # Distance tracking to detect passing target
        self.previous_distance = None
        
        # Track initial coords for stuck detection
        self.initial_move_coords = None
        self.stuck_count = 0
        
        # Screenshot-based stuck detection
        self.previous_screenshot = None
        self.screenshot_stuck_count = 0
        
        # Initialize components
        self.coord_reader = CoordinateReader()
        self.tree_miner = TreeMiner()
        self.target_selector = TargetSelector(window_size[0], window_size[1])
        self.nav_controller = NavigationController()
    
    def update_targets(self, targets):
        """Update the list of detected targets"""
        self.lock.acquire()
        self.targets = targets
        self.lock.release()
    
    def update_screenshot(self, screenshot):
        """Update the current screenshot"""
        self.lock.acquire()
        self.screenshot = screenshot
        self.lock.release()
    
    def get_screenshot(self):
        """Get current screenshot"""
        self.lock.acquire()
        screenshot = self.screenshot
        self.lock.release()
        return screenshot
    
    def read_f3_coordinates(self, screenshot):
        """Read coordinates from F3 screen and update bot state"""
        player_coords, target_coords = self.coord_reader.read_coordinates(screenshot)
        
        # Update player coordinates if valid
        if player_coords is not None:
            if self.player_coords is None or self.coord_reader.coords_are_reasonable(player_coords, self.player_coords):
                self.player_coords = player_coords
                print(f"[DEBUG] Player Block Coordinates: {self.player_coords}")
            else:
                print(f"[DEBUG] Ignoring invalid player coords: {player_coords} (previous: {self.player_coords})")
                pydirectinput.keyDown('w')
                sleep(0.2)
                pydirectinput.keyUp('w')
        
        # Update target coordinates if valid
        if target_coords is not None:
            if self.target_block_coords is None or self.coord_reader.coords_are_reasonable(target_coords, self.target_block_coords):
                self.target_block_coords = target_coords
                print(f"[DEBUG] Targeted Block Coordinates: {self.target_block_coords}")
            else:
                print(f"[DEBUG] Ignoring invalid target coords: {target_coords} (previous: {self.target_block_coords})")
                print(f"[DEBUG] Clearing target and going back to searching due to invalid coordinates")
                self.target_block_coords = None
                pydirectinput.keyDown('w')
                sleep(0.2)
                pydirectinput.keyUp('w')
        
        return self.player_coords is not None or self.target_block_coords is not None
    
    def mine_tree_wrapper(self):
        """Wrapper for tree mining that provides required callbacks"""
        return self.tree_miner.mine_tree(
            get_screenshot_func=self.get_screenshot,
            read_coords_func=self.coord_reader.read_coordinates
        )
    
    def change_state(self, new_state, **kwargs):
        """Helper method to change bot state and optionally clear variables
        
        Args:
            new_state: The state to transition to
            **kwargs: Optional variables, value to set/(None to clear):
                - current_target
                - target_block_coords
                - target_distance
                - expected_target_coords
                - previous_distance
                - searching_start_time
                - initial_move_coords
        """
        self.lock.acquire()
        self.state = new_state
        
        # Update any specified variables
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.lock.release()
    
    def start(self):
        """Start the bot thread"""
        self.stopped = False
        t = Thread(target=self.run)
        t.start()
    
    def stop(self):
        """Stop the bot"""
        self.stopped = True
    
    def run(self):
        """Main bot loop"""
        while not self.stopped:
            if self.state == BotState.INITIALIZING:
                if time() - self.timestamp > self.INITIALIZING_TIME:
                    print("[DEBUG] Initialization complete, transitioning to SEARCHING")
                    self.change_state(BotState.SEARCHING)
            
            elif self.state == BotState.SEARCHING:
                # Start tracking time when entering SEARCHING state
                if self.searching_start_time is None:
                    self.searching_start_time = time()
                
                # Check if we've been searching for more than 5 seconds
                if time() - self.searching_start_time > 5.0:
                    print("[DEBUG] Been searching for 5s, moving randomly")
                    pydirectinput.moveRel(500, 0, relative=True) 
                    self.nav_controller.move_randomly()
                    self.searching_start_time = time()
                
                if self.target_selector.move_crosshair_to_target(
                    self.targets, 
                    self.current_target
                ):
                    print("[DEBUG] Crosshair centered, transitioning to MOVING")
                    sleep(0.2)
                    # Save the target we're locking onto
                    if self.current_target is None and self.targets:
                        self.current_target = self.target_selector.get_best_target(self.targets)
                    self.change_state(BotState.MOVING, searching_start_time=None, expected_target_coords=self.target_block_coords, previous_distance=None, initial_move_coords=None)
                    sleep(0.2)
            
            elif self.state == BotState.MINING:
                # Recenter cursor on target before mining
                if self.target_selector.move_crosshair_to_target(
                    self.targets,
                    self.current_target
                ):
                    if self.mine_tree_wrapper():
                        print("[DEBUG] Mining complete, clearing target")
                        self.change_state(BotState.SEARCHING, current_target=None, target_block_coords=None, target_distance=None)
                else:
                    print("[DEBUG] Lost target while recentering, going back to searching")
                    self.change_state(BotState.SEARCHING, current_target=None, target_block_coords=None)
            
            elif self.state == BotState.MOVING:

                saved_screenshot = self.screenshot
                # Show F3 debug info
                pydirectinput.press('F3')
                sleep(0.3)  # Reduced from 1.0s to 0.3s for faster checks
                  # Store the original coordinates
                original_target_coords = self.target_block_coords
                original_player_coords = self.player_coords
                # Read coordinates from current screenshot
                if self.screenshot is not None:
                    self.read_f3_coordinates(self.screenshot)
                
                # Hide F3 debug info
                pydirectinput.press('F3')
                sleep(0.1)  # Small delay after hiding F3
                
                # Check if we're still looking at the same target
                if self.target_block_coords is not None and self.expected_target_coords is not None:
                    # Debug output
                    print(f"[DEBUG] Checking target coords - Expected: {self.expected_target_coords}, Current: {self.target_block_coords}")
                    
                    if self.target_block_coords != self.expected_target_coords:
                        print(f"[DEBUG] Target coords changed! Expected {self.expected_target_coords}, got {self.target_block_coords}. Lost target, going back to searching")
                       
                        self.change_state(BotState.SEARCHING, current_target=None, target_block_coords=None, expected_target_coords=None)
                        sleep(0.1)
                        continue
                else:
                    print(f"[DEBUG] Cannot check target coords - target_block_coords: {self.target_block_coords}, expected: {self.expected_target_coords}")
                
                # Check if OCR failed or distance is unreasonable
                if self.target_block_coords is None or self.player_coords is None:
                    self.ocr_fail_count += 1
                    print(f"[DEBUG] OCR failed to read coordinates (Target: {self.target_block_coords}, Player: {self.player_coords}), fail count: {self.ocr_fail_count}")
                else:
                    # Calculate distance to check if OCR result is reasonable
                    dx = self.target_block_coords[0] - self.player_coords[0]
                    dz = self.target_block_coords[2] - self.player_coords[2]
                    distance = (dx**2 + dz**2)**0.5
                    self.target_distance = distance
                    
                    if distance > 100:
                        self.ocr_fail_count += 1
                        print(f"[DEBUG] Distance too large ({distance:.2f} blocks), fail count: {self.ocr_fail_count}")
                
                # Handle OCR failures
                if self.ocr_fail_count > 0:
                    # After consecutive failures, move randomly
                    if self.ocr_fail_count >= 3:
                        print(f"[DEBUG] {self.ocr_fail_count} OCR failures in a row, moving randomly to change view")
                        self.nav_controller.move_randomly()
                        self.ocr_fail_count = 0
                                    
                # OCR succeeded, reset fail counter
                self.ocr_fail_count = 0
                
                # Check if we passed the target (distance increased)
                if self.previous_distance is not None:
                    if distance > self.previous_distance:
                        print(f"[DEBUG] Distance increased! Was {self.previous_distance:.2f}, now {distance:.2f}. Passed target, going back to searching")
                       
                        self.change_state(BotState.SEARCHING, current_target=None, target_block_coords=None, expected_target_coords=None, previous_distance=None)
                        sleep(0.1)
                        continue
                
                # Update previous distance
                self.previous_distance = distance
                
                if self.nav_controller.check_if_stuck(original_player_coords, self.player_coords, saved_screenshot, self.screenshot):
                    self.stuck_count += 1
                    print(f"[DEBUG] Stuck detected {self.stuck_count}/5")
                    
                    if self.stuck_count >= 5:
                        print(f"[DEBUG] Player stuck! Moving randomly to unstuck")
                        self.nav_controller.move_randomly()
                        self.original_player_coords = self.player_coords  # Reset after unstucking
                        self.stuck_count = 0
                else:
                    self.stuck_count = 0

                # Move towards target
                if self.nav_controller.move_to_target(self.player_coords, self.target_block_coords):
                    print("[DEBUG] Reached target, transitioning to MINING")
                    self.change_state(BotState.MINING, initial_move_coords=None)
                    continue
                
                
            
            # Small sleep to prevent busy-waiting
            sleep(0.01)
