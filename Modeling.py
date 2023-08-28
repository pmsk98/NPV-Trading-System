#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:19:51 2023

@author: user
"""


#x_train,y_train,x_test,y_test

x_train =[]
y_train =[]
x_test=[]
# y_test=[]


#######7216
for i in range(len(nasdaq100_price_copy)):
    x_train.append(train_data[i].drop(['Label'],axis=1))
    y_train.append(train_data[i]['Label'])
    
    x_test.append(test_data[i])
    # y_test.append(test_data[i]['Label']) 

    
#모델링
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pred_gbm_threshold=[]
pred_xgb_threshold=[]
pred_ada = []
pred_lgb = []
pred_cat = []


# pred_gbm =[]
# pred_xgb =[]

random_state = 0

threshold_0 = 1
threshold_1 = 1
for i in range(len(x_train)):        
    # gradient boosting
    gbm = GradientBoostingClassifier(random_state=random_state)
    gbm.fit(x_train[i], y_train[i])
    pred_gbm = gbm.predict(x_test[i])
    pred_gbm_threshold.append(pred_gbm)

    # XGBoost
    xgb = XGBClassifier(random_state=random_state)
    xgb.fit(x_train[i], y_train[i].astype(int))
    pred_xgb = xgb.predict(x_test[i])
    pred_xgb_threshold.append(pred_xgb)
    
    #AdaBoost
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=random_state)
    adaboost.fit(x_train[i], y_train[i])
    pred_ada.append(adaboost.predict(x_test[i]))
    
    #LightGBM
    lgb = LGBMClassifier(random_state=random_state)
    lgb.fit(x_train[i], y_train[i])
    pred_lgb.append(lgb.predict(x_test[i]))
    
    #catboost
    cat_model = CatBoostClassifier(iterations=1000, learning_rate=0.05, loss_function='Logloss',random_state=random_state)
    cat_model.fit(x_train[i],y_train[i])
    pred_cat.append(cat_model.predict(x_test[i]))
