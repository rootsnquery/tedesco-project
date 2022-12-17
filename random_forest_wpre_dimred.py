# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:51:11 2022

@author: Rubyxu
"""

import datetime
import numpy as np
from matplotlib import pyplot as plt, dates
import seaborn as sns 
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

import time

data = pd.read_csv('/rigel/home/yx2693/albedo_df_updated.csv')
data = data.drop('AL2', axis=1)

data = data.sample(frac = 0.1, random_state=42) # for testing


y = np.array(data['albedo']) 
X = data.drop(['albedo'], axis = 1)  #try dropping some variable 1125
X_list = list(X.columns)
#X = np.array(X)


# log transformation
X.loc[X.ME<1e-10,'ME'] = 1e-10
X.loc[X.RF<1e-10,'RF'] = 1e-10
X.loc[X.SF<1e-10,'SF'] = 1e-10
X.loc[X.CD<1e-10,'CD'] = 1e-10
X.loc[X.CM<1e-10,'CM'] = 1e-10
X.ME = np.log(X.ME)
X.RF = np.log(X.RF)
X.SF = np.log(X.SF)
X.CD = np.log(X.CD)
X.CM = np.log(X.CM)

'''
X_train = X[:int(X.shape[0]*0.8)]
y_train = y[:int(y.shape[0]*0.8)]
X_test = X[int(X.shape[0]*0.8):]
y_test = y[int(y.shape[0]*0.8):]
'''

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 42)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
#X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

minmax = MinMaxScaler()
y_train = pd.Series(minmax.fit_transform(pd.DataFrame(y_train)).reshape(1,-1)[0])
y_test = pd.Series(minmax.transform(pd.DataFrame(y_test)).reshape(1,-1)[0])

'''
#truncated svd
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=14, algorithm='randomized',
                   random_state=42)

svd.fit(X_train)
X_train = svd.transform(X_train)
X_test = svd.transform(X_test)
'''

'''
#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=15)#X_train.shape[1])
pca.fit(X_train)
print(pca.explained_variance_)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''


# hyperparameter tuning
n_estimators = [200,500]##[10, 50, 200]#[200, 300, 500, 1000]
#max_features = ['auto', 'sqrt']
max_depth = [20, 100]##[5,20,100]#[int(x) for x in np.linspace(5, 100, num = 5)]
#max_depth.append(None)
#min_samples_split = [2,10]##[2, 10]#[2, 5, 10]
#min_samples_leaf = [1, 2, 4]
#bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               #'max_features': max_features,
               'max_depth': max_depth}
               #'min_samples_split': min_samples_split}
               #'min_samples_leaf': min_samples_leaf,
               #'bootstrap': bootstrap}
print('Hyperparameter options: ', random_grid)
start = time.time()
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
end = time.time()
print('Time for random search: ', end-start)
bp = rf_random.best_params_
print('Best params: ', bp)



rf_best = RandomForestRegressor(n_estimators = bp['n_estimators'], 
                                #min_samples_split = bp['min_samples_split'],
                                #min_samples_leaf = bp['min_samples_leaf'],
                                #max_features = bp['max_features'],
                                max_depth = bp['max_depth'],
                                #bootstrap = bp['bootstrap'],
                                random_state = 42)

'''
s = time.time()
rf_best = RandomForestRegressor(n_estimators = 500, 
                                min_samples_split = 10,
                                #min_samples_leaf = 2,
                                #max_features = 'sqrt',
                                max_depth = 100,
                                #bootstrap = False,
                                random_state = 42)
'''

rf_best.fit(X_train, y_train)
#e = time.time()
#print('time for one fit:', e-s)

y_pred_train = rf_best.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)
print("training R^2 : % f" %(r2_train))
y_pred = rf_best.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("test R^2 : % f" %(r2))

filename = '/rigel/home/yx2693/albedo_df_updated/rf_wpre_model_01.sav'
pickle.dump(rf_best, open(filename, 'wb'))


#<UNI>@habanero.rcs.columbia.edu







