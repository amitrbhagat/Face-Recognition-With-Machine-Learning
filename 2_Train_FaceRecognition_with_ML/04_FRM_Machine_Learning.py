import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


#Loadf the numy array
data = np.load('./data/data_pca_50_target.npz')
print(data.files)


data.allow_pickle = True

X = data['arr_0'] # pca data with 50 components
y = data['arr_1'] # target or dependent variable



# Split the data into train-test dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)



#### Training machine learning model

model_svc = SVC(probability=True)

param_grid = {'C':[0.5,1,10,20,30,40,50],
              'kernel':['rbf','poly'],
              'gamma':[0.1, 0.05,0.01,0.001,0.002,0.005],
              'coef0':[0,1]
            }

model_grid = GridSearchCV(model_svc, 
                          param_grid=param_grid, 
                          scoring='accuracy', 
                          cv=3, 
                          verbose=2)


model_grid.fit(x_train, y_train)

print(model_grid.best_params_)

model_final = model_grid.best_estimator_

print(model_final.get_params())




#### Model Evaluation

y_pred = model_final.predict(x_test)
print(y_pred)


# Classification Report

cr = metrics.classification_report(y_test, y_pred, output_dict=True)
print(pd.DataFrame(cr).T)


# kappa score

score = metrics.cohen_kappa_score(y_test, y_pred)
print(score)


# Area under curve (AUC)

auc = metrics.roc_auc_score(np.where(y_test=='male', 1, 0),
                      np.where(y_pred=='male', 1, 0))
print(auc)




# Save face recognition model

import pickle

pickle.dump(model_final, open('./model/model_svm.pickle', mode='wb'))