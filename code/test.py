#coding:utf-8
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
 
#1.以CSV形式导入数据集
df_train = pd.read_csv(r"C:\Users\ld\Desktop\train.csv")
df_test = pd.read_csv(r"C:\Users\ld\Desktop\test.csv")
 
X_train = df_train[[i for i in df_train.columns.tolist() if i not in ["label"]]] #训练集
y_train = df_train["label"]                                                      #训练标签
X_test =df_test[[i for i in df_test.columns.tolist() if i not in ["label"]]]     #测试集
y_test = df_test["label"]                                                        #测试标签
 
#2.参数集定义
param_grid = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'n_estimators': [30, 50, 100, 300, 500, 1000,2000],
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.5],
            "gamma":[0.0, 0.1, 0.2, 0.3, 0.4],
            "reg_alpha":[0.0001,0.001, 0.01, 0.1, 1, 100],
            "reg_lambda":[0.0001,0.001, 0.01, 0.1, 1, 100],
            "min_child_weight": [2,3,4,5,6,7,8],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "subsample":[0.6, 0.7, 0.8, 0.9]}
#3.网格搜索并打印最佳参数
gsearch1 = GridSearchCV(estimator=XGBRegressor(scoring='ls',seed=27), param_grid=param_grid, cv=5)
gsearch1.fit(X_train, y_train)
print("best_score_:",gsearch1.best_params_,gsearch1.best_score_)
 
#4.用最佳参数进行预测
y_test_pre= gsearch1.predict(X_test)
 
#5.打印测试集RMSE
rmse = sqrt(mean_squared_error(np.array(list(y_test)), np.array(list(y_test_pre))))
print("rmse:",rmse)