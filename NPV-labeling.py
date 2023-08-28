import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

nasdaq100_price_copy = nasdaq100_price.copy()

for i in range(len(nasdaq100_price_copy)):
    nasdaq100_price_copy[i] = nasdaq100_price_copy[i].drop(['Close'],axis=1)
    nasdaq100_price_copy[i] = nasdaq100_price_copy[i].reset_index(drop=False)
    
    
for i in range(len(nasdaq100_price_copy)):
    nasdaq100_price_copy[i].columns = ['date','open','high','low','close','volume']

#Technical Indicators 추가 
for  i in nasdaq100_price_copy:
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



# minmax=MinMaxScaler()   

# for i in nasdaq100_price_copy:
#     x=i[['ADX','aroondown','aroonup','BOP','CCI','CMO','DX','MFI','PPO','ROC','RSI','slowk','slowd','fastk','fastd','ULTOSC','WILLR']]
#     x=minmax.fit_transform(x)
#     i[['ADX','aroondown','aroonup','BOP','CCI','CMO','DX','MFI','PPO','ROC','RSI','slowk','slowd','fastk','fastd','ULTOSC','WILLR']]=x


#%%
for i in range(len(nasdaq100_price_copy)):
    nasdaq100_price_copy[i]['date'] = nasdaq100_price_copy[i]['date'].astype('str')

#%%
    
#train 기준 수익률 표준편차 구하기
    
nasdaq_price =[]

for i in range(len(nasdaq100_price_copy)):
    train=None
    train=nasdaq100_price_copy[i]['date'].str.contains('2015|2016|2017|2018|2019|2020')
    nasdaq_price.append(nasdaq100_price_copy[i][train])



# 기간 설정
n = 20

for i in range(len(nasdaq100_price_copy)):
    nasdaq_price[i]['log_returns'] = np.log(nasdaq_price[i]['close']).diff()
    nasdaq_price[i]['n_period_log_returns'] = nasdaq_price[i]['log_returns'].rolling(window=n).sum().shift(-n)

# n-기간 로그 수익률의 표준편차 계산하기
std_log_return = []

for i in range(len(nasdaq100_price_copy)):
    std_log_return.append(nasdaq_price[i]['n_period_log_returns'].std())



for i in range(len(nasdaq100_price_copy)):
    nasdaq_price[i].loc[nasdaq_price[i]['n_period_log_returns'] >= 0.75*std_log_return[i], 'Label'] = '1'
    nasdaq_price[i].loc[nasdaq_price[i]['n_period_log_returns'] <= -0.75*std_log_return[i], 'Label'] = '0'


#결측치 처리
for i in range(len(nasdaq_price)):
    nasdaq_price[i]=nasdaq_price[i].dropna(axis=0)
    nasdaq_price[i] =nasdaq_price[i].reset_index(drop=True)


for i in range(len(nasdaq_price)):
    temp = nasdaq_price[i]
    deleted = []
    for year in range(2015, 2021):
        deleted.append(temp[('%d-01-01' %year <= temp['date']) & (temp['date'] < '%d-01-01' %(year+1))].iloc[:-20])
    nasdaq_price[i] = pd.concat(deleted)    



# # 수익률 분포 확인
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(20,20))
# plt.rcParams['figure.dpi'] =350
# plt.rcParams['font.size'] = 8
# fig,ax = plt.subplots()

# plt.style.use("ggplot")


# for i in range(len(nasdaq_price)):
#     plt.hist(nasdaq_price[i]['log_returns'], bins=50,color='green')
#     plt.xlabel('Returns')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Returns')
#     plt.show()
    



#필요없는 변수 삭제
for i in range(len(nasdaq_price)):
    nasdaq_price[i] = nasdaq_price[i].drop(['log_returns','n_period_log_returns'],axis=1)
    
    
    
#결측치 처리
for i in range(len(nasdaq_price)):
    nasdaq_price[i]=nasdaq_price[i].dropna(axis=0)
    nasdaq_price[i] =nasdaq_price[i].reset_index(drop=True)


#%%      

#Train / Test Split           
            
# for i in range(len(nasdaq100_price_copy)):
#     nasdaq100_price_copy[i]['date'] = nasdaq100_price_copy[i]['date'].astype('str')


# train_data = []

# for i in range(len(nasdaq100_price_copy)):
#     train=None
#     train=nasdaq100_price_copy[i]['date'].str.contains('2014|2015|2016|2017|2018|2019|2020')
#     train_data.append(nasdaq100_price_copy[i][train])
    
    
# test_data = []

# for i in range(len(nasdaq100_price_copy)):
#     test=None    
#     test=nasdaq100_price_copy[i]['date'].str.contains('2021|2022')
#     test_data.append(nasdaq100_price_copy[i][test])

# # 필요없는 변수 삭제
# for i in range(len(nasdaq100_price_copy)):
#     train_data[i]=train_data[i].drop(['date','open','high','low','close','volume'],axis=1)
#     test_data[i]=test_data[i].drop(['date','open','high','low','close','volume'],axis=1)

#%%
    

for i in range(len(nasdaq_price)):
    nasdaq_price[i]['date'] = nasdaq_price[i]['date'].astype('str')


train_data = []

for i in range(len(nasdaq_price)):
    train=None
    train=nasdaq_price[i]['date'].str.contains('2015|2016|2017|2018|2019|2020')
    train_data.append(nasdaq_price[i][train])

    
test_data = []

for i in range(len(nasdaq100_price_copy)):
    test=None    
    test=nasdaq100_price_copy[i]['date'].str.contains('2021|2022')
    test_data.append(nasdaq100_price_copy[i][test])


# 필요없는 변수 삭제
for i in range(len(nasdaq_price)):
    train_data[i]=train_data[i].drop(['date','open','high','low','close','volume'],axis=1)
    test_data[i]=test_data[i].drop(['date','open','high','low','close','volume'],axis=1)
