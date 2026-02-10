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

# fix, ax = plt.subplots(nrows=2, figsize=(15, 12))
# exp_var_df['explained_var'].plot(kind='line', marker='o', ax=ax[0])
# exp_var_df['cum_explained_var'].plot(kind='line', marker='o', ax=ax[1])
# plt.show()



pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto',copy=False)
pca_data = pca_50.fit_transform(X_t)
# print(pca_data.shape)


#Saving data and model

y = data['gender'].values
np.savez('./data/data_pca_50_target', pca_data, y)


#saving the model

pca_dict = {'pca': pca_50, 'mean_face': mean_face}
pickle.dump(pca_dict, open('model/pca_dict.pickle', 'wb'))



#Visualize Eigan image

pca_data_inv = pca_50.inverse_transform(pca_data)
print(pca_data_inv.shape)

eig_img = pca_data_inv[0,:].reshape((100, 100))
print(eig_img.shape)

# plt.imshow(eig_img, cmap='gray')
# plt.axis('off')
# plt.show()


np.random.seed(1001)
pics = np.random.randint(0,4319,40)
plt.figure(figsize=(15,8))
for i,pic in enumerate(pics):
    plt.subplot(4,10,i+1)
    img = X[pic:pic+1].reshape(100,100)
    plt.imshow(img,cmap='gray')
    plt.title('{}'.format(y[pic]))
    plt.xticks([])
    plt.yticks([])
plt.show()

print("="*20+'Eigen Images'+"="*20)
plt.figure(figsize=(15,8))
for i,pic in enumerate(pics):
    plt.subplot(4,10,i+1)
    img = pca_data_inv[pic:pic+1].reshape(100,100)
    plt.imshow(img,cmap='gray')
    plt.title('{}'.format(y[pic]))
    plt.xticks([])
    plt.yticks([])
    
plt.show()