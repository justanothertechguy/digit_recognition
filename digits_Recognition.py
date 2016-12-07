#!/usr/bin/python

# Import the modules
import cv2
import numpy as np
import argparse as ap
from sklearn.externals import joblib
from skimage.feature import hog

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-c", "--classifer_path", help="Classifier File Path", required="True")
parser.add_argument("-i", "--image_path", help="Image Path", required="True")
args = vars(parser.parse_args())

# Load the classifier
classifier, pre_process = joblib.load(args["classifer_path"])

# Read the input image 
read_inp_image = cv2.imread(args["image_path"])

# Convert to grayscale and apply Gaussian filtering
cnv_gray = cv2.cvtColor(read_inp_image, cv2.COLOR_BGR2GRAY)
cnv_gray = cv2.GaussianBlur(cnv_gray, (5, 5), 0)

# Threshold the image

ret, img_thresh = cv2.threshold(cnv_gray, 90,255,cv2.THRESH_BINARY_INV)

# Find contours in the image
_,ctrs, hier = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(read_inp_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = img_thresh[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_feat = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_hog_feat = pre_process.transform(np.array([roi_hog_feat], 'float64'))
    nbr = classifier.predict(roi_hog_feat)
    cv2.putText(read_inp_image, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.namedWindow("Resulting Image with Rectangular Region of Interest(ROIs)", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular Region of Interest(ROIs)", read_inp_image)
cv2.waitKey()
