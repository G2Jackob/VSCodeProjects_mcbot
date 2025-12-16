from threading import Thread, Lock
from ultralytics import YOLO

class Detection:
    stopped = True
    lock = None
    rectangles = []
    model = None
    screenshot = None
    debug_image = None
    results = None

    def __init__(self, model_file_path):
        self.lock = Lock()
        self.model = YOLO(model_file_path)
        self.screen_center = None
        print(f"[DEBUG] YOLO model loaded: {model_file_path}")

    def update(self, screenshot):
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
            if self.screenshot is not None:
                try:
                   
                    # Get results from YOLO model
                    results = self.model(self.screenshot, show=False, conf=0.6, line_width=1, classes=[1,3,5,7,9,11])[0]
                    
                    self.lock.acquire()
                    self.debug_image = results.plot()
                    self.results = results
                    self.lock.release()
                    
                except Exception as e:
                    print(f"[DEBUG] Detection error: {str(e)}")
                    if self.lock.locked():
                        self.lock.release()
                    continue
                
    def is_running(self):
        return not self.stopped
    
    def set_screen_center(self, width, height):
        """Set the screen center coordinates"""
        self.screen_center = (width // 2, height // 2)
    
    def get_click_points(self, results):
        """Convert YOLO results to click points with confidence and size info"""
        if not results or not self.screen_center:
            return []

        try:
            # Get detection data
            boxes = results.boxes.xyxy.tolist()
            classes = results.boxes.cls.tolist()
            confidences = results.boxes.conf.tolist()  # Get confidence scores
            
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