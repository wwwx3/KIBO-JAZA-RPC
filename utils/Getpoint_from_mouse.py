-----------------------------------------------------------------------
"""This util is for helping users get the exact (x,y) point from the image by just clicking on the image , helps with debugging and 
assuming the size of the area."""
-----------------------------------------------------------------------
import cv2
import numpy as np 
import cvzone

img1 = cv2.imread('img/FindPaperEdge4.png')

points = []

def marking_point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))

        cv2.circle(img1,(x,y),10,(189,126,155),-1)
        cv2.imshow("Edge Marking",img1)
        print(f"Points added: {x},{y}")

cv2.imshow("Edge Marking",img1)
cv2.setMouseCallback("Edge Marking",marking_point)
cv2.waitKey(0)
cv2.destroyAllWindows()
