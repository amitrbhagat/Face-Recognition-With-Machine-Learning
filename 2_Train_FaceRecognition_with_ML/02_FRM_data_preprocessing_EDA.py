### Exploratory Data Analysis

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# computer vision library
import cv2

# glob
from glob import glob

#extract paths of male and female in crop_data folder and put them in a list
fpath = glob('./crop_data/female/*.jpg')
mpath = glob('./crop_data/male/*.jpg')

df_female = pd.DataFrame(fpath, columns=['filepath'])
df_female['gender'] = 'female'

df_male = pd.DataFrame(mpath, columns=['filepath'])
df_male['gender'] = 'male'

df = pd.concat((df_female, df_male), axis = 0)

# it will take each image path and then return the  width of the image

def get_size(path):
    img  = cv2.imread(path)
    return img.shape[0]

df['dimension'] = df['filepath'].apply(get_size)   # store dimensions of image in this column(dimension) 

print(df.head())

dist_gender = df['gender'].value_counts()
print(dist_gender)


# Distrribution of male and female
# Plot the bar graph to visualize the distribution of female and male

# fig, ax = plt.subplots(nrows=1, ncols=2)
# dist_gender.plot(kind='bar', ax=ax[0])
# dist_gender.plot(kind='pie', ax=ax[1], autopct = '%0.0f%%')
# plt.show()


# Distribution of size of all images
#Histogram
#Box plot
#Split by "Gender"

# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# sns.histplot(df['dimension'])
# plt.subplot(2, 1, 2)
# sns.boxplot(df['dimension'])
# plt.show()



# So based on  the above graph the conclusions are:
# 1. We almost have the equal distribution of images
# 2. Most of the images are having dimensions more than 60
# 3. Most of the female images are HD compare to male images

# Consider the image with dimension more than 60
# Resize all the image into 100 x 100

df_filter = df.query('dimension > 60').copy()
df_filter.shape

print(df_filter['gender'].value_counts(normalize=True))




# Resize the images
# 100 x 100

def structuring(path):
    try:

        #step 1: read the image
        img = cv2.imread(path)
        #step 2: convert into greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # step 3:resize into 100 x 100 array

        size = gray.shape[0]
        if size >= 100:
            gray_resize = cv2.resize(gray, (100, 100), cv2.INTER_AREA)
        else:
            gray_resize = cv2.resize(gray, (100, 100), cv2.INTER_CUBIC)

        # Flatten image (1 x 10,000)
        flatten_image = gray_resize.flatten()
        return flatten_image
    
    except:
        return None
    

df_filter.loc[:, 'data'] = df_filter['filepath'].apply(structuring)  # convert all images into 100 x 100

# data = df_filter['data'].apply(pd.Series)
df_filter = df_filter.dropna(subset=['data'])
df_filter = df_filter.reset_index(drop=True)

# print(data.head())


data = df_filter['data'].apply(pd.Series)


# Data Normalization

data = data/255.0
data['gender'] = df_filter['gender']
# print(data.head())


#Remove the null data
# data.dropna(inplace=True)




###  Save the data for future study

import pickle

pickle.dump(data, open('./data/data_images_100_100.pickle', mode='wb'))