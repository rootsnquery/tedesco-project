import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import scipy.io
from scipy.io import savemat
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBRegressor
warnings.filterwarnings('ignore')

# load df
df = pd.read_csv('/rigel/home/yx2693/no_na_df.csv')
# take subsample
#df = df.sample(frac = 0.05)
# split into X and y
df_new_X = df.drop(columns=['albedo','y','x','year','day'])
df_new_Y = df['albedo']
X_dev, X_test, y_dev, y_test = train_test_split(df_new_X, df_new_Y, test_size=0.2, random_state=0)
print(X_dev.columns)

# standardize and split
scaler = StandardScaler()
X_dev = scaler.fit_transform(X_dev)
X_test = scaler.transform(X_test)
X_dev = np.hstack([np.ones((X_dev.shape[0], 1)), X_dev])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# elastic net, random search + cv = 10
def elastic_net(X_dev,X_test,y_dev,y_test):
    start = time.time()
    elastic_hyperpara_dict = dict(l1_ratio = uniform(),alpha=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0])
    print('Start training')
    elastic_rdcv = RandomizedSearchCV(ElasticNet(random_state=0),elastic_hyperpara_dict,cv=10,random_state=42)
    elastic_rd_search = elastic_rdcv.fit(X_dev,y_dev)
    stop = time.time()
    print(f"Elastic Training time: {stop - start}s")
    print('elastic best param',elastic_rd_search.best_params_)
    # get best params and build with all dev data
    elastic = ElasticNet(l1_ratio = elastic_rd_search.best_params_['l1_ratio'],\
                         alpha = elastic_rd_search.best_params_['alpha'],random_state = 0)
    elastic.fit(X_dev,y_dev)
    # get predictions and save
    elastic_y_dev_pred = elastic.predict(X_dev)
    elastic_y_test_pred = elastic.predict(X_test)
    np.savetxt('/rigel/home/yx2693/all/elastic_y_dev_pred.txt',elastic_y_dev_pred)
    np.savetxt('/rigel/home/yx2693/all/elastic_y_test_pred.txt',elastic_y_test_pred)
    print('elastic train r2',elastic.score(X_dev,y_dev))
    print('elastic test r2',elastic.score(X_test,y_test))
    # save the model to disk
    filename = '/rigel/home/yx2693/all/elastic_model.sav'
    pickle.dump(elastic, open(filename, 'wb'))

# logistic ridge + cv = 5
# not in use now. 
# For logistic, the problem is treated by default as a multi-classification problem
# That is why the training takes so much time and the result is not reasonable. (about 100 separating lines)

def logistic(X_dev,X_test,y_dev,y_test):
    start = time.time()
    log_hyperpara_dict = dict(C=[10, 1.0, 0.1,0.01])
    log_gdcv = GridSearchCV(LogisticRegression(random_state=0),log_hyperpara_dict,cv=10)
    log_gd_search = log_gdcv.fit(X_dev,y_dev)
    stop = time.time()
    print(f"Logistic Training time: {stop - start}s")
    print('logistic best params ',log_gd_search.best_params_)
    # get best params and build with all dev data
    logistic = LogisticRegression(C = log_gd_search.best_params_['C'], random_state = 0)
    logistic.fit(X_dev,y_dev)
    # get predictions and save
    logistic_y_dev_pred = logistic.predict(X_dev)
    logistic_y_test_pred = logistic.predict(X_test)
    np.savetxt('/rigel/home/yx2693/all/logistic_y_dev_pred.txt',logistic_y_dev_pred)
    np.savetxt('/rigel/home/yx2693/all/logistic_y_test_pred.txt',logistic_y_test_pred)
    print('logistic train r2',logistic.score(X_dev,y_dev))
    print('logistic test r2',logistic.score(X_test,y_test))
    # save the model to disk
    filename = '/rigel/home/yx2693/all/logistic_model.sav'
    pickle.dump(logistic, open(filename, 'wb'))

def xgboost(X_dev, X_test, y_dev,y_test):
    start = time.time()
    xgb_hyperpara_dict = {'max_depth': [2, 8, 32],
        'n_estimators': [10, 30, 100, 300, 1000],
        'learning_rate': [0.01, 0.1, 1, 10],
        "gamma": uniform()}
    #xgb_gdcv = GridSearchCV(estimator=XGBRegressor(objective = 'reg:squarederror',scoring='ls',seed=27), param_grid=xgb_hyperpara_dict, cv=10)
    xgb_gdcv = RandomizedSearchCV(estimator=XGBRegressor(objective = 'reg:squarederror',scoring='r2',seed=27), 
                              param_distributions = xgb_hyperpara_dict, cv=10,random_state=42, n_iter=10)
    xgb_gdcv.fit(X_dev,y_dev)
    finish = time.time()
    print('Random Search fitting time: ', finish-start)
    print("Best paramters and score:", xgb_gdcv.best_params_,xgb_gdcv.best_score_)
    # get best params and build with all dev data
    xgb = XGBRegressor(gamma = xgb_gdcv.best_params_['gamma'], learning_rate = xgb_gdcv.best_params_['learning_rate'],
          max_depth = xgb_gdcv.best_params_['max_depth'], n_estimators = xgb_gdcv.best_params_['n_estimators'],
          objective = 'reg:squarederror',scoring='r2',seed=27)
    xgb.fit(X_dev,y_dev)
    # get predictions and save
    xgb_y_dev_pred = xgb.predict(X_dev)
    xgb_y_test_pred = xgb.predict(X_test)
    np.savetxt('/rigel/home/yx2693/all/xgboost_y_dev_pred.txt',xgb_y_dev_pred)
    np.savetxt('/rigel/home/yx2693/all/xgboost_y_test_pred.txt',xgb_y_test_pred)
    print('xgb r2',xgb.score(X_dev,y_dev))
    print('xgb test r2',xgb.score(X_test,y_test))
    # save the model to disk
    filename = '/rigel/home/yx2693/all/xgboost_model.sav'
    pickle.dump(xgb, open(filename, 'wb'))

#<UNI>@habanero.rcs.columbia.edu


#elastic_net(X_dev,X_test,y_dev,y_test)
#logistic(X_dev,X_test,y_dev,y_test)
xgboost(X_dev,X_test,y_dev,y_test)