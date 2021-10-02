import cv2
import numpy as np
import os
import csv
import random as rng



def extract_ex(image):
    img_NEW = cv2.resize(image, (1280, 1192))
    img_NEW = 255 - img_NEW
    blur = cv2.GaussianBlur(img_NEW, (99, 99), 0)
    kernel = np.ones((27, 27), np.uint8)
    img_erosion = cv2.erode(blur, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    retValue, threshImg = cv2.threshold(img_dilation, 169, 169, cv2.THRESH_BINARY_INV)
    gray = cv2.cvtColor(threshImg, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    inter = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out2 = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(out2, [cnt], -1, 128, cv2.FILLED)
    out2 = cv2.bitwise_and(gray, out2)

    inter = cv2.morphologyEx(out2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out1 = np.zeros(out2.shape, np.uint8)
    cv2.drawContours(out2, [cnt], -1, 0, cv2.FILLED)
    out1 = cv2.bitwise_and(out2, out1)

    contours, hierarchy = cv2.findContours(out2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For each contour, find the convex hull and draw it
    # on the original image.
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    img_NEW = 255 - img_NEW
    for i in range(len(contours)):
        color = (255, 255, 255)
        # cv2.drawContours(image, contours_poly, i, color,1)
        # t[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv2.circle(img_NEW, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)

    return img_NEW

if __name__ == "__main__":
    pathFolder = "./DR"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
    destinationFolder = "./DR/CottonWool/"
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder + '/' + file_name)
        exudates = extract_ex(fundus)
        cv2.imwrite(destinationFolder + file_name_no_extension + "_cottonwool.png", exudates)