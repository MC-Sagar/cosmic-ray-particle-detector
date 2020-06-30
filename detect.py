import cv2
from datetime import datetime
import numpy as np

import imutils

def hit_detection(img,brightest_pixel,non_zero,count):
    flag = False
    crop_size = 10
    hit_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    pos = np.unravel_index(img.argmax(),img.shape)
    bilateral = cv2.bilateralFilter(img, 3, 3, 3)
    edges = cv2.Canny(bilateral, 35, 60)
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    xy_coordinates = set()
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            xy_coordinates.add((cX,cY))
    multi_detection = 0
    for coordinates in xy_coordinates:
        img_crop = img[coordinates[1]-crop_size:coordinates[1]+crop_size,coordinates[0]-crop_size:coordinates[0]+crop_size]
        if img_crop.shape[0] == crop_size * 2 and img_crop.shape[1] == crop_size * 2:
            flag = True
            multi_detection += 1
            cv2.imwrite("/home/pi/i/PiCamera/OpenCV/test/"+str(count)+"."+str(multi_detection)+".png",img_crop)
    if flag:
        count += 1
    print(hit_time,pos,brightest_pixel,xy_coordinates,sep=',', file=open("/home/pi/i/report.txt","a"))
    return count
    

