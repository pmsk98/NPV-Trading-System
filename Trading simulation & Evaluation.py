#%%
    
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:18:53 2023

@author: user
"""


test_7219=[]


for i in range(len(nasdaq100_price_copy)):
    test=None    
    test=nasdaq100_price_copy[i]['date'].str.contains('2021|2022')
    test_7219.append(nasdaq100_price_copy[i][test])

for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['pred_gbm']=pred_gbm_threshold[i]
    test_7219[i]['pred_xgb']=pred_xgb_threshold[i]
    test_7219[i]['pred_ada']=pred_ada[i]
    test_7219[i]['pred_lgb']=pred_lgb[i]
    test_7219[i]['pred_cat']=pred_cat[i]

#pred 자료형 변경
for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['pred_gbm']=test_7219[i]['pred_gbm'].astype('float')
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    test_7219[i]['pred_ada']=test_7219[i]['pred_ada'].astype('float')
    test_7219[i]['pred_lgb']=test_7219[i]['pred_lgb'].astype('float')
    test_7219[i]['pred_cat']=test_7219[i]['pred_cat'].astype('float')

    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(len(nasdaq100_price_copy)):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_gbm'][e]+test_7219[i]['pred_gbm'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_gbm'][e]+test_7219[i]['pred_gbm'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_gbm'][e] > test_7219[i]['pred_gbm'][e+1]:
                test_7219[i]['position'][e+1]='sell'
            else:
                test_7219[i]['position'][e+1]='buy'
        except:
            pass

#첫날 position이 holding일 경우 buy로 변경
for i in range(len(nasdaq100_price_copy)):
    if test_7219[i]['position'][test_7219[i].index[0]]=='holding':
        test_7219[i]['position'][test_7219[i].index[0]]='buy'
    elif test_7219[i]['position'][test_7219[i].index[0]]=='sell':
        test_7219[i]['position'][test_7219[i].index[0]]='buy'
    else:
        test_7219[i]['position'][test_7219[i].index[0]]='buy'

# for i in range(len(test_7219)):
#     print(test_7219[i]['position'][1472])
#     print(test_7219[i]['position'][0])
#     print(test_7219[i]['position'].value_counts())

#강제 청산
for i in range(len(nasdaq100_price_copy)):
    for e in test_7219[i].index[-1:]:
        if test_7219[i]['position'][e]=='holding':
            test_7219[i]['position'][e]='sell'
        elif test_7219[i]['position'][e]=='buy':
            test_7219[i]['position'][e]='sell'
        elif test_7219[i]['position'][e]=='no action':
            test_7219[i]['position'][e]='sell'
        else:
            print(i)



for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['profit']=None
    
#다음날 시가를 가져오게 생성
for i in range(len(nasdaq100_price_copy)):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['position'][e]=='buy':
                test_7219[i]['profit'][e]=test_7219[i]['open'][e+1]
            elif test_7219[i]['position'][e]=='sell':
                test_7219[i]['profit'][e]=test_7219[i]['open'][e+1]
            else:
                print(i)
        except:
            pass



for i in range(len(nasdaq100_price_copy)):
    for e in test_7219[i].index[-1:]:
        if test_7219[i]['position'][e]=='sell':
            test_7219[i]['profit'][e]=test_7219[i]['open'][e]
        
####

buy_label=[]
for i in range(len(nasdaq100_price_copy)):
    buy_position=test_7219[i]['position']=='buy'
    buy_label.append(test_7219[i][buy_position])
    
sell_label=[]
for i in range(len(nasdaq100_price_copy)):
    sell_position=test_7219[i]['position']=='sell'
    sell_label.append(test_7219[i][sell_position])    


buy=[]
sell=[]
for i in range(len(nasdaq100_price_copy)):
    buy.append(buy_label[i]['open'].reset_index(drop=True))
    sell.append(sell_label[i]['open'].reset_index(drop=True))
    
  
profit_2=[]    
for i in range(len(nasdaq100_price_copy)):
    profit_2.append((sell[i]-(0.0015*sell[i]))-buy[i])
  

for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['profit_2']=None
    

#profit 결측치 처리
for i in range(len(nasdaq100_price_copy)):
    profit_2[i]=profit_2[i].dropna()
    
    
#profit_2 sell에 해당하는 행에 값 넣기
for tb, pf in zip(test_7219, profit_2):
    total_idx = tb[tb['position'] == 'sell'].index
    total_pf_idx = pf.index
    for idx, pf_idx in zip(total_idx, total_pf_idx):
        tb.loc[idx, 'profit_2'] = pf[pf_idx]




for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['profit_cumsum']=None
    
    
    

#profit 누적 합 
for i in range(len(nasdaq100_price_copy)):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['position'][e]=='holding':
                test_7219[i]['profit_2'][e]=0
            elif test_7219[i]['position'][e]=='no action':
                test_7219[i]['profit_2'][e]=0
            elif test_7219[i]['position'][e]=='buy':
                test_7219[i]['profit_2'][e]=0
            else:
                print(i)
        except:
            pass


#새로운 청산 기준 누적합

for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['profit_cumsum2']=None    
    
    
for i in range(len(nasdaq100_price_copy)):
    test_7219[i]['profit_cumsum']=test_7219[i]['profit_2'].cumsum()



################# ratio 작성

#ratio 작성
for i in range(len(nasdaq100_price_copy)):
    profit_2[i]=pd.DataFrame(profit_2[i])

#거래횟수
trade= []

for i in range(len(nasdaq100_price_copy)):
    trade.append(len(profit_2[i]))
    
#승률


for i in range(len(nasdaq100_price_copy)):
    profit_2[i]['average']=None

   
for i in range(len(nasdaq100_price_copy)):
    for e in range(len(profit_2[i])):      
        if profit_2[i]['open'][e] > 0:
            profit_2[i]['average'][e]='gain'
        else:
            profit_2[i]['average'][e]='loss'
            
for i in range(len(nasdaq100_price_copy)):
    for e in range(len(profit_2[i])):
        if profit_2[i]['open'][e] < 0:
            profit_2[i]['open'][e]=profit_2[i]['open'][e] * -1
        else:
            print(i)

win=[]
for i in range(len(nasdaq100_price_copy)):
    try:
        win.append(profit_2[i].groupby('average').size()[0]/len(profit_2[i]))
    except:
        win.append('0')
    
#평균 수익

gain=[]

for i in range(len(nasdaq100_price_copy)):
    gain.append(profit_2[i].groupby('average').mean())
    

real_gain=[]

for i in range(len(nasdaq100_price_copy)):
    try:
        real_gain.append(gain[i]['open'][0])
    except:
        real_gain.append('0')



#평균 손실
loss=[]

for i in range(len(nasdaq100_price_copy)):
    try:
        loss.append(gain[i]['open'][1])
    except:
        loss.append('0')

    
loss
#payoff ratio
payoff=[]

for i in range(len(nasdaq100_price_copy)):
    try:
        payoff.append(gain[i]['open'][0]/gain[i]['open'][1])
    except:
        payoff.append('inf')
    
#profit factor

factor_sum=[]

len(factor_sum)
for i in range(len(nasdaq100_price_copy)):
    factor_sum.append(profit_2[i].groupby('average').sum())

factor=[]

for i in range(len(nasdaq100_price_copy)):
    try:
        factor.append(factor_sum[i]['open'][0]/factor_sum[i]['open'][1])
    except:
        factor.append('0')

#year
year=[]

for i in range(len(nasdaq100_price_copy)):
    year.append('2021~2022')

#최종 결과물 파일 작성
file_name = symbol_true

stock_name=pd.DataFrame({'stock_name':file_name})

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)
