import cv2 as cv
import numpy as np
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
                    # Check if image is too dark and brighten it for better detection
                    screenshot_to_process = self.screenshot
                    mean_brightness = cv.mean(cv.cvtColor(self.screenshot, cv.COLOR_BGR2GRAY))[0]
                    
                    if mean_brightness < 50:  # Image is very dark (nighttime only)
                        print(f"[DEBUG] Dark conditions detected (brightness: {mean_brightness:.1f}), enhancing image")
                        # Strong brightness increase for nighttime
                        screenshot_to_process = cv.convertScaleAbs(self.screenshot, alpha=2.5, beta=50)
                    
                    # Get results from YOLO model
                    results = self.model(screenshot_to_process, show=False, conf=0.6, line_width=1, classes=[1,3,5,7,9,11])[0]
                    
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