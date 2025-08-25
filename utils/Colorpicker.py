import cv2
import numpy as np

def empty(a):
    pass

# Load the image
img = cv2.imread("/Users/wangzhan/Documents/KIBO_2025_AI/img/FindPaperEdge2.png")  # Replace with your actual image path
img = cv2.resize(img, (640, 480))
# Convert your image into HSV image
imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create Trackbars for HSV range
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

while True:
    # Get current trackbar positions
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    # Define HSV range and create mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask_inv)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask_bgr, result])
    cv2.imshow('Horizontal Stacking', hStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
