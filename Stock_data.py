#%%
#Nasdaq 100 list


import pandas as pd
import FinanceDataReader as fdr

symbol = ["MSFT",
"AAPL",
"AMZN",
"NVDA",
"GOOGL",
"GOOG",
"META",
"TSLA",
"AVGO",
"PEP",
"COST",
"CSCO",
"TMUS",
"ADBE",
"TXN",
"CMCSA",
"NFLX",
"AMD",
"QCOM",
"AMGN",
"INTC",
"HON",
"INTU",
"SBUX",
"GILD",
"BKNG",
"AMAT",
"ADI",
"MDLZ",
"ISRG",
"ADP",
"REGN",
"PYPL",
"VRTX",
"FISV",
"MU",
"LRCX",
"ATVI",
"MELI",
"CSX",
"MRNA",
"PANW",
"CDNS",
"ASML",
"SNPS",
"ORLY",
"MNST",
"FTNT",
"CHTR",
"KLAC",
"MAR",
"KDP",
"KHC",
"AEP",
"ABNB",
"CTAS",
"LULU",
"DXCM",
"NXPI",
"AZN",
"MCHP",
"ADSK",
"EXC",
"BIIB",
"PDD",
"IDXX",
"WDAY",
"PAYX",
"XEL",
"SGEN",
"PCAR",
"ODFL",
"CPRT",
"ILMN",
"ROST",
"GFS",
"EA",
"MRVL",
"WBD",
"DLTR",
"CTSH",
"WBA",
"FAST",
"VRSK",
"CRWD",
"BKR",
"ENPH",
"CSGP",
"ANSS",
"FANG",
"ALGN",
"CEG",
"TEAM",
"EBAY",
"DDOG",
"ZM",
"JD",
"SIRI",
"ZS",
"LCID",
"RIVN"]
    

# nasdaq100_price = []

# for code in symbol:
#     try:
#         prices = fdr.DataReader(code, '2014-12-01', '2022-12-31')
#         nasdaq100_price.append(prices)
#     except:
#         pass


nasdaq100_price = []

symbol_true =[]

symbol_false =[]

for code in symbol:
    try:
        prices = fdr.DataReader(code, '2014-01-01', '2022-12-31')
        if pd.Timestamp('2015-01-02') not in prices.index:
            continue 
        nasdaq100_price.append(prices)
        symbol_true.append(code)
        print(code)
    except:
        symbol_false.append(code)
        
#2014년 주가 정보가 있는 Symbol
symbol_true

# for i in range(len(nasdaq100_price)):
#     nasdaq100_price[i].to_csv('C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/{}_price.csv'.format(symbol_true[i]))
