--------------------------------------------------------------
"""In order for AI to recognize the items, we need to send the cropped part of the whole image containing only the item img, along with
The ARUco markers, we have to dissect only the A4 paper out of the Astrobee's navigation cam. And sendd the cropped part (paper) to AI for the 
AI to later recognize the item and count the items. This code would later be converted to JAVA."""
--------------------------------------------------------------
import cv2
from datetime import datetime
from random import randint
import numpy as np
import os
import glob

def order_points(pts):
    # Sort based on sum (top-left and bottom-right) and diff (top-right and bottom-left)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

# This is only training, so, I'm testing the images from the captured pic from Astrobee's navigation camera.
# We could change the image path later on. (.mp4 training is also highly recommended)
image_paths = glob.glob("img/*.png")
for image_path in image_paths :
    print(f"Processing(image_path)")
    img1 = cv2.imread(image_path)
    #Converting the image into HSV
    hsv_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

    # White paper color img 1 
    paper_lower1 = np.array([0,0,0])
    paper_upper1 = np.array([118,0,240])
    #Find more range
    #White paper color img 2,3
    paper_lower2 = np.array([0,0,0])
    paper_upper2 = np.array([75,158,186])
    """#Red color
    red_upper = np.array([10, 255, 255])
    red_lower = np.array([0, 50, 50])"""

    """#Purple color
    purple_upper = np.array([155, 255, 255])
    purple_lower = np.array([125, 50, 50])"""


    # Mask area with upper & lower for both of the ranges
    # Get the mask inverse because it's hard to find the real value for the upper and lower
    # kernel = np.ones((5, 5), np.uint8) ; slows down the procedure , doesnt help mask better 
    mask_paper1 = cv2.inRange(hsv_img1, paper_lower1, paper_upper1)
    mask_inv1 = cv2.bitwise_not(mask_paper1)
    mask_paper2 = cv2.inRange(hsv_img1, paper_lower2, paper_upper2)
    mask_inv2 = cv2.bitwise_not(mask_paper2)
    #mask_red = cv2.inRange(hsv_img1,red_lower,red_upper)
    #mask_purple = cv2.inRange(hsv_img1,purple_lower,purple_upper)

    # Needed to combine the masks to make a universal mask for different environments in the spaceship(different lightings)
    #combine_mask = cv2.bitwise_or(mask_paper)
    #combine_mask = cv2.bitwise_or(combine_mask,mask_purple)
    #Draw contour 
    contour1,_ = cv2.findContours(mask_inv1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour2,_ = cv2.findContours(mask_inv2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Draw contour for debugging and ensuring we got the paper correctly.
    contour_img1 = img1.copy()
    contour_img2 = img1.copy()
    # Filter out noise by contouring only rectangular parts
    """Creating snapshot by cropping 
    snapshot_count = 0
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                cropped = img1[y:y+h, x:x+w]
                
                # Save snapshot
                date = datetime.now().strftime("%Y-%m-%d")
                randnum=randint(1000,9999)
                filename = "IMG_{}_{}".format(date,randnum)
                cv2.imwrite("output/{}.jpg".format(filename),cropped)
                snapshot_count += 1
                
                # Draw contour on original image
                cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 5)"""
    #Creating snapshot by extracting from contour seperating the 2 masks since we want both of the masks to get contoured .
    snapshot_count = 0

    for cnt in contour1:
        area = cv2.contourArea(cnt)
        if area > 6000:
            print (area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
        # Get corner points from the contour
                pts = approx.reshape(4, 2)

                # Order points: top-left, top-right, bottom-right, bottom-left
                rect = order_points(pts)
                (tl, tr, br, bl) = rect

                # Compute the width and height of the new image
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))
                """
                for point in approx: doesnt help 
                    cv2.circle(contour_img1, tuple(point[0]), 5, (255, 0, 0), -1)"""

                # Destination points for perspective transform
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                # Compute the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img1, M, (maxWidth, maxHeight))

                # Save the warped result
                date = datetime.now().strftime("%Y-%m-%d")
                randnum = randint(1000,9999)
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                filename = f"{base_filename}_warp_{snapshot_count}_{date}_{randnum}"
                cv2.imwrite(f"output/{filename}.png",warped)

                print(f"Snapped from: {image_path}")
                print(f"Warp points: {rect.tolist()}")
                snapshot_count += 1

                # Optional: draw contour on original for reference
                cv2.drawContours(contour_img1, [approx], -1, (0, 255, 0), 5)
    for cnt in contour2:
        area = cv2.contourArea(cnt)
        if area > 7000:
            print (area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
        # Get corner points from the contour
                pts = approx.reshape(4, 2)

                # Order points: top-left, top-right, bottom-right, bottom-left
                rect = order_points(pts)
                (tl, tr, br, bl) = rect

                # Compute the width and height of the new image
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))
                """
                for point in approx:
                    cv2.circle(contour_img1, tuple(point[0]), 5, (255, 0, 0), -1)"""

                # Destination points for perspective transform
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                # Compute the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img1, M, (maxWidth, maxHeight))

                # Save the warped result
                date = datetime.now().strftime("%Y-%m-%d")
                randnum = randint(1000,9999)
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                filename = f"{base_filename}_warp_{snapshot_count}_{date}_{randnum}"
                cv2.imwrite(f"output/{filename}.png",warped)

                print(f"Snapped from: {image_path}")
                print(f"Warp points: {rect.tolist()}")
                snapshot_count += 1


                # Optional: draw contour on original for reference
                cv2.drawContours(contour_img2, [approx], -1, (0, 255, 0), 5)


    #This version of the code results in all contours on the mask 2 without landing on mask1 
    # Somehow i was able to combine the 2 masks

    cv2.imshow("Original",img1)
    cv2.imshow("Masked Area 1",mask_paper1)
    cv2.imshow("Masked Area 2",mask_paper2)
    cv2.imshow("Contour1",contour_img1)
    cv2.imshow("Contour2",contour_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




------------------------------------------------------------------------------------------------------
#Vent:  

#The problem right now is the Img 1 that uses the contour 1
#Contour 1 couldn't be read for wrapping image, so what we'll do is , we'll check the x,y points for image wrapping.
#We'll use the Get point from mouse function to check the X, Y for supposed contour edge.
#Problem Hyporthesis , it uses the largest area contoured to get snapped and wrap so the largest area and the larghest x , y , z got snapped 
#Checked the points the points are useless 
#Not a problem with range when we checked the coordinates some cooridnates just doont add up , they link into 2 same dots for delta x and y 
#Maybe we have to check the mmax min algorithm and rethink it . and maybe we have to do the cannyedge detection again 
#To ensure that its a code that could be worked on still doent know the problem so we help out by generating ai imgs .
