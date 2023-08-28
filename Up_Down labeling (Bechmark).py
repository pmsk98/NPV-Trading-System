#%%
import glob
import os
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


files=glob.glob('C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/*.csv')

path = "C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/"

file_list =os.listdir(path)

len(file_list)

df=[]

for file in file_list:
    path = "C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/"
    data=pd.read_csv(path+"/"+file)
    # data=data.drop(['Unnamed: 0'],axis=1)
    df.append(data)


for i in range(len(df)):
    df[i] = df[i].drop(['Close'],axis=1)
    df[i] = df[i].reset_index(drop=True)
    
    
for i in range(len(df)):
    df[i].columns = ['date','open','high','low','close','volume']


for  i in df:
    ADX=talib.ADX(i.high,i.low,i.close,timeperiod=14)

    ADXR=talib.ADXR(i.high,i.low,i.close,timeperiod=14)
    
    APO=talib.APO(i.close,fastperiod=12,slowperiod=26,matype=0)
    
    aroondown,aroonup =talib.AROON(i.high, i.low, timeperiod=14)
    
    AROONOSC=talib.AROONOSC(i.high,i.low,timeperiod=14)
    
    BOP=talib.BOP(i.open,i.high,i.low,i.close)
    
    CCI=talib.CCI(i.high,i.low,i.close,timeperiod=14)
    
    CMO=talib.CMO(i.close,timeperiod=14)
    
    DX=talib.DX(i.high,i.low,i.close,timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(i.close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    ma_macd, ma_macdsignal, ma_macdhist = talib.MACDEXT(i.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    
    fix_macd,fix_macdsignal, fix_macdhist = talib.MACDFIX(i.close, signalperiod=9)
    
    MFI=talib.MFI(i.high, i.low,i.close, i.volume, timeperiod=14)
    
    MINUS_DI=talib.MINUS_DI(i.high, i.low, i.close, timeperiod=14)
    
    MINUS_DM=talib. MINUS_DM(i.high, i.low, timeperiod=14)
    
    MOM=talib.MOM(i.close,timeperiod=10)
    
    PLUS_DM=talib.PLUS_DM(i.high,i.low,timeperiod=14)
    
    PPO=talib.PPO(i.close, fastperiod=12, slowperiod=26, matype=0)
    
    ROC=talib.ROC(i.close,timeperiod=10)
    
    ROCP=talib.ROCP(i.close,timeperiod=10)
    
    ROCR=talib.ROCR(i.close,timeperiod=10)
    
    ROCR100=talib.ROCR100(i.close,timeperiod=10)
    
    RSI=talib.RSI(i.close,timeperiod=14)
    
    slowk, slowd = talib.STOCH(i.high, i.low, i.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    fastk, fastd = talib.STOCHF(i.high, i.low, i.close, fastk_period=5, fastd_period=3, fastd_matype=0)
    
    TRIX=talib.TRIX(i.close,timeperiod=72)
    
    ULTOSC=talib.ULTOSC(i.high,i.low,i.close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    
    WILLR=talib.WILLR(i.high,i.low,i.close,timeperiod=14)
    
    
    i['ADX']=ADX
    i['ADXR']=ADXR
    i['APO']=APO
    i['aroondown']=aroondown
    i['aroonup']=aroonup
    i['AROONOSC']=AROONOSC
    i['BOP']=BOP
    i['CCI']=CCI
    i['CMO']=CMO
    i['DX']=DX
    i['MACD']=macd
    i['macdsignal']=macdsignal
    i['macdhist']=macdhist
    i['ma_macd']=ma_macd
    i['ma_macdsignal']=ma_macdsignal
    i['ma_macdhist']=ma_macdhist
    i['fix_macd']=fix_macd
    i['fix_macdsignal']=fix_macdsignal
    i['fix_macdhist']=fix_macdhist
    i['MFI']=MFI
    i['MINUS_DI']=MINUS_DI
    i['MINUS_DM']=MINUS_DM
    i['MOM']=MOM
    i['PLUS_DM']=PLUS_DM
    i['PPO']=PPO
    i['ROC']=ROC
    i['ROCP']=ROCP
    i['ROCR']=ROCR
    i['ROCR100']=ROCR100
    i['RSI']=RSI
    i['slowk']=slowk
    i['slowd']=slowd
    i['fastk']=fastk
    i['fastd']=fastd
    i['TRIX']=TRIX
    i['ULTOSC']=ULTOSC
    i['WILLR']=WILLR
    

#########7209~7272년만 뽑기




#####
for i in df:
    i['diff']=i.close.diff().shift(-1).fillna(0)
    i['Label'] = None
    

#label 생성

for i in range(len(df)):
    for e in df[i].index:
        if df[i]['diff'][e] > 0:
            df[i]['Label'][e] = '1'
        elif df[i]['diff'][e]==0:
            df[i]['Label'][e] ='0'
        else:        
            df[i]['Label'][e] = '0'



#인덱스 번호 한번더 초기화
for i in range(len(df)):
    df[i]=df[i].reset_index(drop=True)
    df[i]=df[i].fillna(0)


###########modeling

#model train/test set 생성  
train_data=[]
test_data=[]


############train/test 분리 ###
for i in range(len(df)):
    train=None
    train=df[i]['date'].str.contains('2015|2016|2017|2018|2019|2020')
    train_data.append(df[i][train])
for i in range(len(df)):
    test=None    
    test=df[i]['date'].str.contains('2021|2022')
    test_data.append(df[i][test])
    

for i in range(len(df)):
    train_data[i]=train_data[i].drop(['date','open','high','low','close','volume','diff'],axis=1)
    test_data[i]=test_data[i].drop(['date','open','high','low','close','volume','diff'],axis=1)



for i in range(len(df)):
    print(df[i].isnull().sum())

#x_train,y_train,x_test,y_test

x_train =[]
y_train =[]
x_test=[]
# y_test=[]

#######7216
for i in range(len(df)):
    x_train.append(train_data[i].drop(['Label'],axis=1))
    y_train.append(train_data[i]['Label'])
    
    x_test.append(test_data[i].drop(['Label'],axis=1))
    # y_test.append(test_data[i]['Label']) 
    

###
######LSTM 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(20, x_train[0].shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#%%

######Modeling     
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np


pred_gbm_threshold=[]
pred_xgb_threshold=[]
pred_ada = []
pred_lgb = []
pred_cat = []
pred_ng = []
pred_lstm= []


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
    


    
    print('{}_finish'.format(i))
    




#%%

#LSTM

for e in range(len(x_train)):
    x_train[e] = x_train[e].reset_index(drop=True)
    y_train[e] = y_train[e].reset_index(drop=True)
    x_test[e] = x_test[e].reset_index(drop=True)
    window_size = 20
    
    
    
    # 각 시계열 시퀀스를 window_size 크기의 윈도우로 나눔
    x_train_reshaped = []
    y_train_reshaped = []
    for i in range(window_size, len(x_train[e])):
        x_train_reshaped.append(x_train[e].iloc[i-window_size:i, :].values)
        y_train_reshaped.append(y_train[e][i])
    
    # numpy 배열로 변환
    x_train_reshaped = np.array(x_train_reshaped)
    y_train_reshaped = np.array(y_train_reshaped)
    
    # 모델 학습
    model.fit(x_train_reshaped, y_train_reshaped, epochs=30, batch_size=32, verbose=1)
    
    # 각 시계열 시퀀스를 window_size 크기의 윈도우로 나눔
    x_test_reshaped = []
    for i in range(window_size, len(x_test[e])):
        if i + window_size <= len(x_test[e]):
            x_test_reshaped.append(x_test[e].iloc[i-window_size:i, :].values)
        else:
            # window_size보다 작은 부분도 예측 결과 생성
            x_test_reshaped.append(x_test[e].iloc[-window_size:, :].values)
    
    # numpy 배열로 변환하되, 차원 추가
    x_test_reshaped = np.array(x_test_reshaped).reshape(-1, window_size, x_test[e].shape[1])
    
    # 모델 예측
    y_pred = model.predict(x_test_reshaped)
    
    train_y_pred = model.predict(x_train_reshaped)
    
    
    
    y_pred_all =np.concatenate ((train_y_pred[-20:],y_pred))
    
    y_pred_all =np.where(y_pred_all > 0.55,1,0)
    
    new_list=[]
    
    new_list = [array[0] for array in y_pred_all]
    
    pred_lstm.append(new_list)
