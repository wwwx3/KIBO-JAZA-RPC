#contour all and show and save all the contour area
#still have to do it in Java language
import cv2
from datetime import datetime
from random import randint
import numpy as np
import os
import glob 

image_paths = glob.glob("img/*.png")

for image_path in image_paths :
    print(f"Processing {image_path}")
    img1 = cv2.imread(image_path)
    hsv_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

    #White paper color img 1 
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


    #Mask area with upper & lower for both of the ranges
    #Get the mask inverse because its hard to find the real value for the upper and lower
    mask_paper1 = cv2.inRange(hsv_img1, paper_lower1, paper_upper1)
    mask_inv1 = cv2.bitwise_not(mask_paper1)
    mask_paper2 = cv2.inRange(hsv_img1, paper_lower2, paper_upper2)
    mask_inv2 = cv2.bitwise_not(mask_paper2)
    #mask_red = cv2.inRange(hsv_img1,red_lower,red_upper)
    #mask_purple = cv2.inRange(hsv_img1,purple_lower,purple_upper)

    #combine_mask = cv2.bitwise_or(mask_paper)
    #combine_mask = cv2.bitwise_or(combine_mask,mask_purple)
    #Draw contour 
    contour1,_ = cv2.findContours(mask_inv1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour2,_ = cv2.findContours(mask_inv2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #Draw contour
    contour_img1 = img1.copy()
    contour_img2 = img1.copy()
    #Filter out noise by contouring only rectangular parts
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
        if area > 7000:
            print (area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Create an empty mask same size as image
                mask = np.zeros_like(img1)

                # Fill the contour on the mask (white where the shape is)
                cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

                # Apply mask to image
                masked = cv2.bitwise_and(img1, mask)

                # Get bounding box for cropping
                x, y, w, h = cv2.boundingRect(approx)
                cropped = masked[y:y+h, x:x+w]

                # Optional: add transparency using alpha channel
                cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY)
                rgba = cv2.merge([cropped, alpha])

                # Save with transparency (PNG)
                date = datetime.now().strftime("%Y-%m-%d")
                randnum=randint(1000,9999)
                filename = "IMG_{}_{}".format(date,randnum)
                cv2.imwrite("output/{}.png".format(filename),cropped)
                snapshot_count += 1

                # Draw contour on main image
                cv2.drawContours(contour_img1, [approx], -1, (0, 255, 0), 5)
    for cnt in contour2:
        area = cv2.contourArea(cnt)
        if area > 7000:
            print (area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Create an empty mask same size as image
                mask = np.zeros_like(img1)

                # Fill the contour on the mask (white where the shape is)
                cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

                # Apply mask to image
                masked = cv2.bitwise_and(img1, mask)

                # Get bounding box for cropping 
                x, y, w, h = cv2.boundingRect(approx)
                cropped = masked[y:y+h, x:x+w]

                # Optional: add transparency using alpha channel
                cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY)
                rgba = cv2.merge([cropped, alpha])

                # Save with transparency (PNG)
                date = datetime.now().strftime("%Y-%m-%d")
                randnum=randint(1000,9999)
                filename = "IMG_{}_{}".format(date,randnum)
                cv2.imwrite("output/{}.png".format(filename),cropped)
                snapshot_count += 1

                # Draw contour on main image
                cv2.drawContours(contour_img2, [approx], -1, (0, 255, 0), 5)




    cv2.imshow("Original",img1)
    cv2.imshow("Masked Area 1",mask_paper1)
    cv2.imshow("Masked Area 2",mask_paper2)
    cv2.imshow("Contour1",contour_img1)
    cv2.imshow("Contour2",contour_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
