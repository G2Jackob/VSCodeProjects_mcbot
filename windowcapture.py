import numpy as np
import win32gui, win32ui, win32con
import cv2 as cv
from threading import Thread, Lock
import ctypes

# Set DPI awareness to prevent black screen issues
ctypes.windll.shcore.SetProcessDpiAwareness(2)

class WindowCapture:

    stopped = True
    lock = None
    screenshot = None

    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    def __init__(self, window_name=None):

        self.lock = Lock()

        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception(f"Window '{window_name}' not found!")
        
        # Get window rect
        window_rect = win32gui.GetWindowRect(self.hwnd)
        client_rect = win32gui.GetClientRect(self.hwnd)
        
        self.w = client_rect[2]
        self.h = client_rect[3]
        
        # Calculate client area offset within window
        border_width = ((window_rect[2] - window_rect[0]) - self.w) // 2
        title_height = (window_rect[3] - window_rect[1]) - self.h - border_width
        
        self.cropped_x = border_width
        self.cropped_y = title_height
        
        self.offset_x = window_rect[0] + border_width
        self.offset_y = window_rect[1] + title_height

    def get_screenshot(self):
        # Fast BitBlt capture
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # Convert to numpy array
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # Cleanup
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # Convert color format
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        
        return img
    
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
    
    def start(self):
        self.stopped = False
        t = Thread(target=self.run)
        t.start()
    
    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
           screenshot = self.get_screenshot()
           self.lock.acquire()
           self.screenshot = screenshot
           self.lock.release()
    
    def is_running(self):
        return not self.stopped