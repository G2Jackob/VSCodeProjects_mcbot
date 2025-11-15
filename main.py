import cv2 as cv
import numpy as np
import os
from time import time, sleep
from windowcapture import WindowCapture
from vision import Vision
from detection import Detection
from threading import Thread
from bot import McBot, BotState



os.chdir(os.path.dirname(os.path.abspath(__file__)))
wincap = WindowCapture('Minecraft 1.21.10 - Singleplayer') #Name of the window to capture

DEBUG = True

detector = Detection('tree_detect_yolo.pt')
vision_wood = Vision(None)
vision_wood.set_screen_center(wincap.w // 2, wincap.h // 2)
bot = McBot((wincap.offset_x, wincap.offset_y), (wincap.w, wincap.h))

wincap.start()
detector.start()
bot.start()

BOT_STATE_NAMES = {
    0: "INITIALIZING",
    1: "SEARCHING",
    2: "MOVING",
    3: "MINING"
}

loop_time = time()
frame_count = 0

while True:
    if not wincap.is_running():
        break

    if not detector.is_running():
        break
       
    screenshot = None
    for _ in range(10):
        if wincap.screenshot is not None:
            screenshot = wincap.screenshot.copy()
            break
        sleep(0.1)
            
    if screenshot is None:
        continue
    detector.update(screenshot)
        

    if DEBUG:
        if detector.debug_image is not None and detector.results is not None:
            try:
                debug_image = detector.debug_image.copy()
                state_text = f"Bot State: {BOT_STATE_NAMES.get(bot.state, str(bot.state))}"
                cv.putText(
                    debug_image,
                    state_text,
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv.LINE_AA
                )
                
                # Display player coordinates
                player_coords_text = f"Player Block: {bot.player_coords if bot.player_coords else 'N/A'}"
                cv.putText(
                    debug_image,
                    player_coords_text,
                    (10, 70),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA
                )
                
                # Display targeted block coordinates
                target_coords_text = f"Targeted Block: {bot.target_block_coords if bot.target_block_coords else 'N/A'}"
                cv.putText(
                    debug_image,
                    target_coords_text,
                    (10, 110),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                    cv.LINE_AA
                )
                
                # Display distance to target
                if bot.target_distance is not None:
                    distance_text = f"Distance: {bot.target_distance:.2f} blocks"
                    cv.putText(
                        debug_image,
                        distance_text,
                        (10, 150),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                        cv.LINE_AA
                    )
                
                targets = vision_wood.get_click_points(detector.results)
                bot.update_targets(targets)
                bot.update_screenshot(screenshot)
                
                # Show debug info
                cv.imshow('Minecraft Bot', debug_image)
                #print(f"[DEBUG] Found targets: {len(targets)}")
            except Exception as e:
                print(f"[DEBUG] Display error: {str(e)}")
    #print('FPS: {}'.format(1/(time() - loop_time + 0.0001)))
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

    

wincap.stop()
detector.stop()
bot.stop()
cv.destroyAllWindows()

print("Done")