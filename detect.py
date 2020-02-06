# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import re
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# loop over the image paths
# load the image and resize it to (1) reduce detection time
# and (2) improve detection accuracy

#.DSstoreなどの隠しDir削除する

parentDirPath = 'sample_img0131_100'
childDirs = os.listdir(path=parentDirPath)

resultsDirPath = "results"
os.makedirs(resultsDirPath, exist_ok=True)

for dir_name in childDirs:
    okDir = resultsDirPath+'/'+dir_name+'_ok'
    os.makedirs(okDir, exist_ok=True)
    ngDir = resultsDirPath+'/'+dir_name+'_ng'
    os.makedirs(ngDir, exist_ok=True)
    pattern = ".*\.(jpg)"
    imgs = [f for f in os.listdir(parentDirPath+'/'+dir_name) if re.search(pattern, f, re.IGNORECASE)] # 大小文字無視
    for img in imgs:
        print(parentDirPath+'/'+dir_name+'/'+img)

        imagePath = parentDirPath+'/'+dir_name+'/'+img
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.05)
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # show some information on the number of bounding boxes
        filename = imagePath[imagePath.rfind("/") + 1:]
        # print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
        print("{}: {} boxes, {} suppression".format(filename, len(rects), len(pick)))

        image = cv2.imread(imagePath)
        if len(rects)>0 and len(pick) >0:
            print('ok')
            cv2.imwrite(okDir+"/"+img, image)
        else:
            print('ng')
            cv2.imwrite(ngDir+"/"+img, image)

# # cv2.imwrite('before.jpg', orig)
# # cv2.imwrite('after.jpg', image)
