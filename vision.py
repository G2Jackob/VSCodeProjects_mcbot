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
        """Convert YOLO results to click points with confidence and size info"""
        if not results or not self.screen_center:
            return []

        try:
            # Get detection data
            boxes = results.boxes.xyxy.tolist()
            classes = results.boxes.cls.tolist()
            confidences = results.boxes.conf.tolist()  # Get confidence scores
            names = results.names
            
            # Process each detection and return list of (x, y, confidence, size) tuples
            click_points = []
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Calculate size of bounding box
                width = x2 - x1
                height = y2 - y1
                size = width * height
                
                # Calculate bottom-center point (center X, 10 pixels above bottom Y)
                # This will make the bot aim slightly above the bottom of the tree
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2) - 10  # 10 pixels above the bottom of the bounding box
                
                # Add bottom-center point with confidence and size to list
                click_points.append((center_x, bottom_y, conf, size))
            
            return click_points
        except Exception as e:
            print(f"[DEBUG] Error getting click points: {str(e)}")
            return []