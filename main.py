import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
from detection import Detection
from threading import Thread
from bot import McBot, BotState



os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap = WindowCapture('Minecraft 1.21.8 - Singleplayer')

DEBUG = True
#vision_wood.init_control_gui()

#hsv_filter = HsvFilter(14,181,0,24,227,255, 78, 0, 0, 0)

detector = Detection('cascade/cascade.xml')

vision_wood = Vision(None)


bot = McBot((wincap.offset_x, wincap.offset_y), (wincap.w, wincap.h))

wincap.start()
detector.start()
bot.start()

loop_time = time()
while True:

    if wincap.screenshot is None:
        continue
    

   # processed_image = vision_wood.apply_hsv_filter(screenshot, hsv_filter)

    detector.update(wincap.screenshot)

    if DEBUG:
        output_image = vision_wood.draw_rectangles(wincap.screenshot, detector.rectangles)
        cv.imshow('Matches', output_image)

    if bot.state == BotState.INITIALIZING:
        targets = vision_wood.get_click_points(detector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)
    elif bot.state == BotState.SEARCHING:
        targets = vision_wood.get_click_points(detector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)
    elif bot.state == BotState.MOVING:
        bot.update_screenshot(wincap.screenshot)
    elif bot.state == BotState.MINING:
        pass

    #cv.imshow('Unprocessed', screenshot)
    print('FPS: {}'.format(1/(time() - loop_time + 0.0001)))
    loop_time = time()

    key = cv.waitKey(1)
    if key == ord('q'):
        wincap.stop()
        detector.stop()
        bot.stop()
        cv.destroyAllWindows()
        break
    elif key == ord('a'):
       cv.imwrite('positive/{}.jpg'.format(loop_time), wincap.screenshot)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), wincap.screenshot)

print("Done")