import pandas as pd
import yfinance as yf
import requests
import numpy as np

tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
data = pd.DataFrame(columns=tickers_list)

for ticker in tickers_list:
    data[ticker] = yf.download(ticker, '2014-9-1', '2018-9-1')['Adj Close']

daily_return = data.pct_change(1)

df1 = pd.DataFrame(columns=tickers_list)
for ticker in tickers_list:
    df1[ticker] = daily_return[ticker]

df1_var = pd.DataFrame(columns=['ticker', 'variance'])
df1_var.variance = df1_var.variance.astype(float)
for ticker in tickers_list:
    value = df1[ticker].var()
    new_row = {'ticker': ticker, 'variance': value}
    df1_var = df1_var.append(new_row, ignore_index=True)

df1_var = df1_var.nsmallest(3, 'variance')
print(df1_var)




