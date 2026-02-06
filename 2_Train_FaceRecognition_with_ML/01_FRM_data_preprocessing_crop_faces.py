#import all the modules

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt 


#Get the path of all images

fpath = glob('./data/female/*.jpg')
mpath = glob('./data/male/*.jpg')


print('number of images in fpath = ',len(fpath))
print('number of images in mpath is = ',len(mpath))


#Data Preprocess

# # - 1 - Read image and convert to RGB ---------------------------------

# img = cv2.imread(fpath[0])  # Reads the image inn BGR format
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #convert image from GBR to RGB format


# # - 2 - Apply Haar cascade classifier ----------------------------------

# haar = cv2.CascadeClassifier('.\model\haarcascade_frontalface_default.xml')
# gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# faces_list = haar.detectMultiScale(gray, 1.5, 5)
# for x,y,w,h in faces_list:
#     cv2.rectangle(img_rgb, (x,y), (x+w, y+h), (0, 255, 0), 2)

# # - 3 - Crop face------------------------

# roi = img_rgb[y:y+h, x:x+w]
# plt.imshow(roi)
# plt.axis('off')
# plt.show()

# # - 4 - Save image -------------------

# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()


### Crop all images

#Female

for i in range (len(fpath)):
    try:

        # # - 1 - Read image and convert to RGB 
        img = cv2.imread(fpath[i])  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # # - 2 - Apply Haar cascade classifier ----------------------------------
        haar = cv2.CascadeClassifier('.\model\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces_list = haar.detectMultiScale(gray, 1.5, 5)
    
        for x,y,w,h in faces_list:
            # # - 3 - Crop face
            roi = img_rgb[y:y+h, x:x+w]

            # # - 4 - Save image
            cv2.imwrite(f'./crop_data/female/female_{i}.jpg', roi)

            print('Image successfully processed')

    except:
        print('Unable to Process the image')


#Male

for i in range (len(mpath)):
    try:

        # # - 1 - Read image and convert to RGB 
        img = cv2.imread(mpath[i])  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # # - 2 - Apply Haar cascade classifier ----------------------------------
        haar = cv2.CascadeClassifier('.\model\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces_list = haar.detectMultiScale(gray, 1.5, 5)
    
        for x,y,w,h in faces_list:
            # # - 3 - Crop face
            roi = img_rgb[y:y+h, x:x+w]

            # # - 4 - Save image
            cv2.imwrite(f'./crop_data/male/male_{i}.jpg', roi)

            print('Image successfully processed')

    except:
        print('Unable to Process the image')        