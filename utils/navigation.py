import pydirectinput
from time import sleep
import random

class TargetSelector:
    """Handles target selection and crosshair movement"""
    
    def __init__(self, window_w, window_h):
        self.window_w = window_w
        self.window_h = window_h
        self.center_x = window_w // 2
        self.center_y = window_h // 2
    
    def get_best_target(self, targets):
        """
        Find the best target based on confidence, size, and distance
        
        Args:
            targets: List of (x, y, confidence, size) tuples
            player_coords: Optional player coordinates for distance calculation
            target_block_coords: Optional target block coordinates for distance calculation
        
        Returns:
            (x, y) tuple of best target position, or None if no targets
        """
        if not targets:
            return None
        
        # Calculate score combining confidence, size, and distance from screen center
        targets_with_score = []
        for target in targets:
            x, y, conf, size = target
            
            # Base score from confidence and size
            base_score = conf * size * 0.01
            
            # Distance from screen center (in pixels)
            dx = x - self.center_x
            dy = y - self.center_y
            distance_from_center = (dx**2 + dy**2)**0.5
            
            # Distance score (prefer targets closer to center, max distance ~800px)
            distance_score = max(0, 500 - distance_from_center)
            
            # Combine scores: 40% base + 60% distance
            final_score = (base_score * 0.4) + (distance_score * 0.6)
            
            targets_with_score.append(((x, y, conf, size), final_score, distance_from_center))
        
        # Sort by score in descending order
        targets_with_score.sort(key=lambda t: t[1], reverse=True)
        
        best = targets_with_score[0]
        best_target = best[0]
        distance = best[2]
        
        print(f"[DEBUG] Selected target: pos=({best_target[0]}, {best_target[1]}), conf={best_target[2]:.2f}, size={best_target[3]}, distance={distance:.2f}, score={best[1]:.2f}")
        print(f"[DEBUG] Total targets: {len(targets)}")
        
        # Return just (x, y) for compatibility
        return (best_target[0], best_target[1])
    
    def move_crosshair_to_target(self, targets, player_coords=None, target_block_coords=None, current_target=None):
        """
        Move crosshair to target position
        
        Returns:
            True if centered on target, False otherwise
        """
        # Get target position
        if targets:
            if current_target is None:
                target_pos = self.get_best_target(targets)
            else:
                target_pos = current_target
        else:
            print("[DEBUG] No targets available")
            return False
        
        if target_pos is None:
            return False
        
        dx = target_pos[0] - self.center_x
        dy = target_pos[1] - self.center_y
        
        # Check if we're close enough to center
        if abs(dx) < 10 and abs(dy) < 10:
            return True
        
        # Move towards target
        move_x = int(dx)
        move_y = int(dy)
        
        if abs(move_x) > 0 or abs(move_y) > 0:
            print(f"[DEBUG] Moving mouse by: ({move_x}, {move_y})")
            pydirectinput.moveRel(move_x, move_y, relative=True)
            sleep(0.1)
        
        return False

class NavigationController:
    """Handles movement and navigation"""
    
    @staticmethod
    def move_to_target(player_coords, target_block_coords):
        """
        Move forward to target using coordinate-based distance
        
        Returns:
            True if reached target, False otherwise
        """
        # Calculate distance to target if we have coordinates
        if target_block_coords and player_coords:
            dx = target_block_coords[0] - player_coords[0]
            dy = target_block_coords[1] - player_coords[1]
            dz = target_block_coords[2] - player_coords[2]
            
            # Calculate horizontal distance only (ignore Y coordinate)
            distance = (dx**2 + dz**2)**0.5
            
            if distance > 1000:
                print(f"[DEBUG] Unreasonable target distance ({distance:.2f} blocks), likely OCR error. Aborting move.")
                return False
            
            print(f"[DEBUG] Horizontal distance to target: {distance:.2f} blocks (X: {dx:.1f}, Z: {dz:.1f}, Y: {dy:.1f})")
            
            # If we're close enough, we've reached the target
            if distance <= 3.75:
                print(f"[DEBUG] Reached target (within 3.75 blocks horizontally)")
                return True
        
        # Move forward if we're not at the target yet
        print("[DEBUG] Walking forward")
        pydirectinput.keyDown('w')
        sleep(0.3)
        pydirectinput.keyUp('w')
        
        return False
    
    @staticmethod
    def check_if_stuck(initial_coords, current_coords):
        """
        Check if player is stuck by comparing positions
        
        Returns:
            True if stuck, False if moving
        """
        if initial_coords is None or current_coords is None:
            return False
        
        # Calculate how far we moved (only X and Z, ignore Y)
        dx = current_coords[0] - initial_coords[0]
        dz = current_coords[2] - initial_coords[2]
        distance_moved = (dx**2 + dz**2)**0.5
        
        print(f"[DEBUG] Distance moved: {distance_moved:.2f} blocks")
        return distance_moved < 0.5  # Stuck if moved less than 0.5 blocks
    
    @staticmethod
    def move_randomly():
        """Move in a random direction for exploration"""
        direction = random.choice([ 'a', 's', 'd'])
        pydirectinput.keyDown(direction)
        sleep(0.7)
        pydirectinput.keyUp(direction)
