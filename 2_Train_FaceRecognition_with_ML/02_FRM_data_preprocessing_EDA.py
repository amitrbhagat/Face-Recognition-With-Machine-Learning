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