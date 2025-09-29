import cv2 as cv
import numpy as np
from hsvfilter import HsvFilter

class Vision:

    TRACKBAR_WINDOW = "Trackbars"
    wood_img = None
    wood_width = 0
    wood_height = 0
    method = None

    def __init__(self, wood_img_path ,method=cv.TM_CCOEFF_NORMED):
        if wood_img_path:
            self.wood_img =cv.cvtColor(cv.imread(wood_img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2BGR)
            self.wood_width, self.wood_height = self.wood_img.shape[1], self.wood_img.shape[0]
        self.method = method

    def find(self, forest_img, threshold=0.5, max_matches=10):

        result = cv.matchTemplate(forest_img, self.wood_img, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1])) 
        #print(locations)

        if not locations:
            return np.array([], dtype=np.int32).reshape(0, 4)

        rectangres = []

        for loc in locations:
            rect= [int(loc[0]), int(loc[1]), self.wood_width, self.wood_height]
            rectangres.append(rect)
            rectangres.append(rect)



        rectangres, weights =cv.groupRectangles(rectangres, groupThreshold=1, eps=0.5)
        #print(rectangres)

        if len(rectangres) > max_matches:
            rectangres = rectangres[:max_matches]

        return rectangres

    def get_click_points(self, rectangres):
        points = []
        if len(rectangres):

            for (x,y,w,h) in rectangres:

                center_x = x + int(w / 2)
                center_y = y + int(h / 2)
                points.append((center_x, center_y))

        return points
    
    def draw_rectangles(self, forest_img, rectangres):

        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangres:

            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv.rectangle(forest_img, top_left, bottom_right, line_color, line_type)

        return forest_img
    def draw_crosshairs(self, forest_img, points):

        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        for (center_x, center_y) in points:

            cv.drawMarker(forest_img, (center_x, center_y), marker_color, marker_type)

        return forest_img

    def init_control_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

       
        def nothing(position):
            pass

        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('Vmin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)

        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing) 
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

    def get_hsv_filter_from_controls(self):
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('Vmin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)   
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)

        return hsv_filter
    
    def apply_hsv_filter(self, orginal_img, hsv_filter=None):
        hsv = cv.cvtColor(orginal_img, cv.COLOR_BGR2HSV)

        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge((h, s, v))

        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])

        mask = cv.inRange(hsv, lower, upper)

        result = cv.bitwise_and(hsv, hsv, mask=mask)

        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)
        return img
    
    def shift_channel(self, c, ammount):
        if ammount > 0:
            lim = 255 - ammount
            c[c >= lim] = 255
            c[c < lim] += ammount
        elif ammount < 0:
            ammount = -ammount
            lim = ammount
            c[c <= lim] = 0
            c[c > lim] -= ammount
        return c

    def find_trees_hsv(self, screenshot):
        """Find trees using HSV color filtering"""
        hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
        
        # Tree bark color range
        lower = np.array([16, 20, 20])
        upper = np.array([20, 255, 200])
        
        mask = cv.inRange(hsv, lower, upper)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        tree_positions = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 30000:
                x, y, w, h = cv.boundingRect(contour)
                tree_positions.append((x + w//2, y + h//2))
        
        return tree_positions