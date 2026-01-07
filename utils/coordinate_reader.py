import cv2 as cv
import pytesseract
import numpy as np
import re
import traceback

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class CoordinateReader:
    """Handles reading coordinates from F3 debug screen using OCR"""
    
    
    def extract_numbers_only(self, text):
        """Extract numbers from OCR text, handling minus signs"""
        
        coords_with_pos = []
        seen_positions = set()
        
        # Pattern for negative numbers
        minus_pattern = r'-\s*[A-Za-z]*\s*(\d+)'
        for match in re.finditer(minus_pattern, text):
            if match.start() not in seen_positions:
                coords_with_pos.append((match.start(), -int(match.group(1))))
                seen_positions.update(range(match.start(), match.end()))
        
        # Pattern for positive numbers
        positive_pattern = r'(?<![A-Za-z\-])\d+'
        for match in re.finditer(positive_pattern, text):
            if match.start() not in seen_positions:
                coords_with_pos.append((match.start(), int(match.group())))
                seen_positions.update(range(match.start(), match.end()))
        
        # Sort by position in original text to maintain order
        coords_with_pos.sort(key=lambda x: x[0])
        return [num for pos, num in coords_with_pos]
    
    def get_coords_from_ocr(self, crop_image):
        """Read OCR once and return coordinate values"""
        custom_config = r'--oem 1 --psm 7 '
        
        try:
            text = pytesseract.image_to_string(crop_image, lang='mc3', config=custom_config)
            
            # Remove commas and replace common OCR mistakes
            text = text.replace(',', '').replace('.', '')
            text = text.replace('S', '5').replace('s', '5')
            text = text.replace('Q', '0').replace('O', '0')
            
            print(f"[DEBUG] OCR Text: {text.strip()}")
            coords = self.extract_numbers_only(text)
            
            # Always use the last 3 numbers as coordinates
            if len(coords) >= 3:
                coords = coords[-3:]
                return tuple(coords)
        except Exception as e:
            print(f"[DEBUG] OCR error: {e}")
        
        return None
    
    def read_coordinates(self, screenshot):
        """Read player and targeted block coordinates from F3 debug screen"""
        try:
            height, width = screenshot.shape[:2]
            
            # Convert BGR to HSV for better color isolation
            hsv_image = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
            
            # Define color range for white/light gray text
            color_lower = np.array([0, 0, 200])
            color_upper = np.array([180, 30, 255])
            
            # Create mask to isolate the text
            mask = cv.inRange(hsv_image, color_lower, color_upper)
            
            # Apply mask to get only the text
            result = cv.bitwise_and(screenshot, screenshot, mask=mask)
            
            # Convert to grayscale
            gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            
            # Define crop regions
            # LEFT side for targeted block coordinates
            y_left = int(height * 0.001)
            h_left = int(height * 0.037)
            x_left = int(width * 0.001)
            w_left = int(width * 0.4)
            
            # RIGHT side for player block coordinates
            y_right = int(height * 0.034)
            h_right = int(height * 0.035)
            x_right = int(width * 0.6)
            w_right = int(width * 0.399)
            
            # Crop the regions
            crop_left = thresh[y_left:y_left+h_left, x_left:x_left+w_left]
            crop_right = thresh[y_right:y_right+h_right, x_right:x_right+w_right]
            
            # Save debug images
            cv.imwrite('other/crop_left.png', crop_left)
            cv.imwrite('other/crop_right.png', crop_right)
            
            # Enhance images for better OCR
            crop_left = cv.bitwise_not(crop_left)
            crop_right = cv.bitwise_not(crop_right)
            
            # Apply morphological operations
            kernel = np.ones((2, 2), np.uint8)
            crop_left = cv.morphologyEx(crop_left, cv.MORPH_CLOSE, kernel)
            crop_right = cv.morphologyEx(crop_right, cv.MORPH_CLOSE, kernel)
            
            # Read coordinates
            player_coords = self.get_coords_from_ocr(crop_right)
            target_coords = self.get_coords_from_ocr(crop_left)
            
            return player_coords, target_coords
            
        except Exception as e:
            print(f"[DEBUG] Error reading F3 coordinates: {str(e)}")
            traceback.print_exc()
            return None, None
    
    def coords_are_reasonable(self, current_coords, previous_coords, max_diff=10000):
        """Check if the change in coordinates is within reasonable limits"""
        if current_coords is None or previous_coords is None:
            return False
        
        for i in range(3):
            if abs(current_coords[i] - previous_coords[i]) > max_diff:
                return False
        return True
