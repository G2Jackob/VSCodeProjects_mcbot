import cv2 as cv
import numpy as np

class Vision:
    def __init__(self, wood_img_path=None, method=cv.TM_CCOEFF_NORMED):
        self.method = method
        self.screen_center = None

    def set_screen_center(self, width, height):
        """Set the screen center coordinates"""
        self.screen_center = (width // 2, height // 2)


    def targets_ordered_by_size(self, targets):
        def size_of_target(target):

            size = (target[0] - target[2]) * (target[1] - target[3])
            return size

        if targets:
            targets = sorted(targets, key=size_of_target, reverse=True)
        return targets

   
    def get_click_points(self, results):
        """Convert YOLO results to click points with distances"""
        if not results or not self.screen_center:
            return []

        try:
            # Get detection data
            boxes = results.boxes.xyxy.tolist()
            classes = results.boxes.cls.tolist()
            names = results.names
            
            # Process each detection and return list of center points
            click_points = []
            for box, cls in zip(boxes, classes):
                self.targets_ordered_by_size([box])
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Add center point to list
                click_points.append((center_x, center_y))
            
            return click_points
        except Exception as e:
            print(f"[DEBUG] Error getting click points: {str(e)}")
            return []