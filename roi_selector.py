import cv2 as cv
import numpy as np

# Global variables
drawing = False
ix, iy = -1, -1
rectangles = []
current_rect = None
img = None
img_display = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_rect, img_display
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect = [ix, iy, x, y]
    
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img.copy()
            # Draw all saved rectangles
            for i, rect in enumerate(rectangles):
                color = (0, 255, 0) if i == 0 else (0, 0, 255)
                label = "Target" if i == 0 else "Block"
                cv.rectangle(img_display, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
                cv.putText(img_display, label, (rect[0], rect[1]-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw current rectangle being drawn
            color = (0, 255, 0) if len(rectangles) == 0 else (0, 0, 255)
            cv.rectangle(img_display, (ix, iy), (x, y), color, 2)
            current_rect = [ix, iy, x, y]
    
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        img_display = img.copy()
        rectangles.append([ix, iy, x, y])
        
        # Draw all rectangles
        for i, rect in enumerate(rectangles):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            label = "Target" if i == 0 else "Block"
            cv.rectangle(img_display, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
            cv.putText(img_display, label, (rect[0], rect[1]-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"\n{'='*60}")
        if len(rectangles) == 1:
            print("Rectangle 1 (GREEN - Target Block) saved!")
            print("Now draw the second rectangle for Block coordinates (RED)")
        elif len(rectangles) == 2:
            print("Rectangle 2 (RED - Block) saved!")
            print("\nCalculating proportions...")
            calculate_proportions()

def calculate_proportions():
    global rectangles, img
    
    height, width = img.shape[:2]
    
    print(f"\nImage dimensions: {width} x {height}")
    print(f"{'='*60}")
    
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        label = "Targeted Block (GREEN)" if i == 0 else "Block (RED)"
        
        # Calculate proportions
        x_start_prop = x1 / width
        x_end_prop = x2 / width
        y_start_prop = y1 / height
        y_end_prop = y2 / height
        
        print(f"\n{label}:")
        print(f"  Pixel coordinates: x({x1}, {x2}), y({y1}, {y2})")
        print(f"  X proportions: {x_start_prop:.4f} to {x_end_prop:.4f}")
        print(f"  Y proportions: {y_start_prop:.4f} to {y_end_prop:.4f}")
        print(f"\n  Code for bot.py:")
        if i == 0:  # Target
            print(f"  target_x_start = int(scaled_width * {x_start_prop:.4f})")
            print(f"  target_x_end = int(scaled_width * {x_end_prop:.4f})")
            print(f"  target_y_start = int(scaled_height * {y_start_prop:.4f})")
            print(f"  target_y_end = int(scaled_height * {y_end_prop:.4f})")
        else:  # Block
            print(f"  block_x_start = int(scaled_width * {x_start_prop:.4f})")
            print(f"  block_x_end = int(scaled_width * {x_end_prop:.4f})")
            print(f"  block_y_start = int(scaled_height * {y_start_prop:.4f})")
            print(f"  block_y_end = int(scaled_height * {y_end_prop:.4f})")
    
    print(f"\n{'='*60}")
    print("Press any key to close and copy the code above")

# Load the debug image
img = cv.imread('debug_full_with_rois.png')
if img is None:
    print("Error: Could not load debug_full_with_rois.png")
    print("Make sure the file exists in the current directory")
    exit()

# Scale down if image is too large
max_height = 800
height, width = img.shape[:2]
if height > max_height:
    scale = max_height / height
    new_width = int(width * scale)
    img = cv.resize(img, (new_width, max_height))
    print(f"Image scaled down for display: {new_width} x {max_height}")
    print("(Proportions will still be correct)")

img_display = img.copy()

cv.namedWindow('ROI Selector')
cv.setMouseCallback('ROI Selector', draw_rectangle)

print("="*60)
print("ROI Selector Tool")
print("="*60)
print("\nInstructions:")
print("1. Draw a rectangle around 'Targeted Block: X, Y, Z' (will be GREEN)")
print("2. Draw a rectangle around 'Block: X Y Z' (will be RED)")
print("3. Proportions will be calculated automatically")
print("4. Press 'r' to reset and start over")
print("5. Press 'q' to quit")
print("\nClick and drag to draw rectangles...")

while True:
    cv.imshow('ROI Selector', img_display)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        rectangles = []
        img_display = img.copy()
        print("\nReset! Draw rectangles again...")

cv.destroyAllWindows()
