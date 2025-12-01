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
        
        # Timers
        self.searching_start_time = None
        
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
        """Get current screenshot (thread-safe)"""
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
                    self.nav_controller.move_randomly()
                    self.searching_start_time = time()  # Reset timer
                
                if self.target_selector.move_crosshair_to_target(
                    self.targets, 
                    self.player_coords, 
                    self.target_block_coords, 
                    self.current_target
                ):
                    print("[DEBUG] Crosshair centered, transitioning to MOVING")
                    sleep(0.2)
                    self.lock.acquire()
                    self.state = BotState.MOVING
                    self.searching_start_time = None  # Reset search timer
                    self.lock.release()
                    sleep(0.2)
            
            elif self.state == BotState.MINING:
                # Recenter cursor on target before mining
                if self.target_selector.move_crosshair_to_target(
                    self.targets,
                    self.player_coords,
                    self.target_block_coords,
                    self.current_target
                ):
                    if self.mine_tree_wrapper():
                        print("[DEBUG] Mining complete, clearing target")
                        self.lock.acquire()
                        self.state = BotState.SEARCHING
                        self.current_target = None
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
                # Store the original coordinates
                original_target_coords = self.target_block_coords
                original_player_coords = self.player_coords
                
                # Show F3 debug info
                pydirectinput.press('F3')
                sleep(1.0)
                
                # Read coordinates from current screenshot
                if self.screenshot is not None:
                    self.read_f3_coordinates(self.screenshot)
                
                # Hide F3 debug info
                pydirectinput.press('F3')
                
                # Check if OCR failed to read coordinates
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
                
                # Update target_distance for display
                self.target_distance = distance
                
                # If distance is over 100, discard and retry OCR
                if distance > 100:
                    print(f"[DEBUG] Distance too large ({distance:.2f} blocks), retrying OCR")
                    # Move slightly to change screen view
                    import random
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
                    sleep(0.1)
                    continue
                
                # Move towards target
                if self.nav_controller.move_to_target(self.player_coords, self.target_block_coords):
                    print("[DEBUG] Reached target, transitioning to MINING")
                    self.lock.acquire()
                    self.state = BotState.MINING
                    self.lock.release()
                    continue
            
            # Small sleep to prevent busy-waiting
            sleep(0.01)
