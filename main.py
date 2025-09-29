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
wincap = WindowCapture('Minecraft 1.21.8 - Singleplayer') #Name of the window to capture

DEBUG = False
#vision_wood.init_control_gui()

#hsv_filter = HsvFilter(14,181,0,24,227,255, 78, 0, 0, 0)

detector = Detection('cascade/cascade.xml')
vision_wood = Vision(None)
bot = McBot((wincap.offset_x, wincap.offset_y), (wincap.w, wincap.h))

wincap.start()
detector.start()
#bot.start()

BOT_STATE_NAMES = {
    0: "INITIALIZING",
    1: "SEARCHING",
    2: "MOVING",
    3: "MINING"
}

loop_time = time()
while True:

    if wincap.screenshot is None:
        continue
    

   # processed_image = vision_wood.apply_hsv_filter(screenshot, hsv_filter)

    detector.update(wincap.screenshot)

    if DEBUG:
        # Draw rectangles
        output_image = vision_wood.draw_rectangles(wincap.screenshot, detector.rectangles)
        state_text = f"Bot State: {BOT_STATE_NAMES.get(bot.state, str(bot.state))}"
        cv.putText(
            output_image,
            state_text,
            (10, 30),  # Position (x, y)
            cv.FONT_HERSHEY_SIMPLEX,
            1,         # Font scale
            (0, 255, 255),  # Color (BGR): Yellow
            2,         # Thickness
            cv.LINE_AA
        )
        #cv.imshow('Matches', output_image)
    cv.imshow('Unprocessed', wincap.screenshot)


    if bot.state == BotState.INITIALIZING:
        targets = vision_wood.get_click_points(detector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)

    elif bot.state == BotState.MOVING:
        targets = vision_wood.get_click_points(detector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)

    elif bot.state == BotState.SEARCHING:
        targets = vision_wood.get_click_points(detector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)

    elif bot.state == BotState.MINING:
        bot.update_screenshot(wincap.screenshot)



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