import cv2
import os
import sys

files = os.listdir(sys.argv[1])
for i in range(len(files)):
    img = cv2.imread(sys.argv[1]+files[i],0)
    if img.max() == 0:
        img[0,0] = 1
        cv2.imwrite(sys.argv[1]+files[i], img)
