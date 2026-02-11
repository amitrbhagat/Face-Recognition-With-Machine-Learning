import pandas as pd
import numpy as np
import sklearn
import pickle

import matplotlib.pyplot as plt
import cv2



#Load all models

haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # cascade classifier
model_svm = pickle.load(open('./model/model_svm.pickle',mode='rb')) # machine learning model svm
pca_models = pickle.load(open('./model/pca_dict.pickle',mode='rb')) # pca dictionary

model_pca = pca_models['pca'] # PCA model
mean_face_arr = pca_models['mean_face'] # Mean face



#### Create pipeline --- 11 Steps

# step - 01  -- read image
img = cv2.imread('./test_images/getty_test.jpg') #BGR

# step - 02  -- convert into gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# step - 03  -- crop the face (haar cascad classifier)
faces = haar.detectMultiScale(gray, 1.5, 3)
for x,y,w,h in faces:
    roi = gray[y:y+h, x:x+w]
    plt.imshow(roi, cmap='gray')
    plt.show()