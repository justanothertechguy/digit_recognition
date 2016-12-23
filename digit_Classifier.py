#!/usr/bin/python

# Import the modules

import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from collections import Counter

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

# Extract the histogram of oriented gradient(HOG) features
lst_hog_feat = []
for feature in features:
    feat = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    lst_hog_feat.append(feat)//storing the hog features in a 							list
hog_features = np.array(lst_hog_feat, 'float64')

# Normalizing the features
pre_process= preprocessing.StandardScaler().fit(hog_features)
hog_features = pre_process.transform(hog_features)

print "Total no of digits in dataset", Counter(labels)

# Create an linear SVM object
classifier = LinearSVC()

# Perform the training
classifier.fit(hog_features, labels)

# Save the classifier for using it in digit prediction later
joblib.dump((classifier, pre_process), "digits_classifier.pkl", compress=3)
