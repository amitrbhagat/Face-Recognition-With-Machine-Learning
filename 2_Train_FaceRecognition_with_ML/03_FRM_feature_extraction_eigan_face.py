import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

#Eigan face
from sklearn.decomposition import PCA

import pickle



#Load the data
data = pickle.load(open('./data/data_images_100_100.pickle', mode='rb'))  #Load the data

# print(data.head())



#### Eigan Face

# Mean face
X = data.drop('gender', axis=1).values.astype(np.float32)  #all images

mean_face = X.mean(axis = 0)
# print(mean_face.shape)



#Visualize the mean face

# plt.imshow(mean_face.reshape((100, 100)), cmap='gray')
# plt.axis('off')
# plt.show()





###  Subtract data with mean face
X_t = X - mean_face  # transformed data

###  Apply X_t data to PCA
# 1 - Find the right number opf component -- Elbow method
# 2 - With right number of component conpute principal components

pca = PCA(n_components=50, whiten=True, svd_solver='auto',copy=False)
pca.fit(X_t)


exp_var_df = pd.DataFrame()
exp_var_df['explained_var'] = pca.explained_variance_ratio_
exp_var_df['cum_explained_var'] = exp_var_df['explained_var'].cumsum()
exp_var_df['principle components'] = np.arange(1, len(exp_var_df)+1)

print(exp_var_df.head())


exp_var_df.set_index('principle components', inplace=True)

#Visualize explained variance
fix, ax = plt.subplots(nrows=2, figsize=(15, 12))

exp_var_df['explained_var'].plot(kind='line', marker='o', ax=ax[0])
exp_var_df['cum_explained_var'].plot(kind='line', marker='o', ax=ax[1])



pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto',copy=False)
pca_data = pca_50.fit_transform(X_t)