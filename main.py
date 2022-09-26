import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt import plotting
from pypfopt import risk_models
from pypfopt.expected_returns import mean_historical_return

# parsing data

table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]

df = df[~(df['Date first added'] > '2014-09-01')]
df = df.dropna()
mask = df['Symbol'].isin(['BF.B', 'FOXA', 'BRK.B', 'NKE'])
df = df[~mask]

print(df)

# list of tickers

df = df['Symbol']
df = df.to_numpy()

tickers_list = df
print(df)

# initial data

data = pd.DataFrame(columns=tickers_list)

for ticker in tickers_list:
    data[ticker] = yf.download(ticker, '2014-9-1', '2018-9-1')['Adj Close']
print(data)

# initial data up to 2022

data_22 = pd.DataFrame(columns=tickers_list)

for ticker in tickers_list:
    data_22[ticker] = yf.download(ticker, '2018-9-1', '2022-9-1')['Adj Close']

print(data_22)

# daily returns for both periods

daily_return = data.pct_change(1)

df1 = pd.DataFrame(columns=tickers_list)
for ticker in tickers_list:
    df1[ticker] = daily_return[ticker]

daily_return_22 = data_22.pct_change(1)

df2 = pd.DataFrame(columns=tickers_list)
for ticker in tickers_list:
    df2[ticker] = daily_return_22[ticker]

# conservative portfolio weights

df1_var = pd.DataFrame(columns=['ticker', 'variance'])
df1_var.variance = df1_var.variance.astype(float)

for ticker in tickers_list:
    value = df1[ticker].var()
    new_row = {'ticker': ticker, 'variance': value}
    df1_var = df1_var.append(new_row, ignore_index=True)

df2_var = df1_var.nsmallest(20, 'variance')
mask = df2_var['ticker']
portfolio_cons = data[mask]

portfolio_cons_22 = pd.DataFrame(columns=mask)
for i in mask:
    portfolio_cons_22[i] = data_22[i].copy()

# portfolio_cons_22 = data_22[mask]
print(portfolio_cons_22)

mu_cons = expected_returns.mean_historical_return(portfolio_cons)
S_cons = risk_models.sample_cov(portfolio_cons)

ef = EfficientFrontier(mu_cons, S_cons)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
weightsCons = ef.min_volatility()

print(weightsCons)
ef.portfolio_performance(verbose=True)

# conservative portfolio for 2018-2022

weightsCons_df = pd.DataFrame.from_dict(weightsCons, orient='index')
weightsCons_df.columns = ['weights']

portfolio_cons_22['Optimized Conservative Portfolio'] = 0

for i in range(len(weightsCons_df)):
    portfolio_cons_22['Optimized Conservative Portfolio'] += portfolio_cons_22.iloc[:, i] * weightsCons_df['weights'][i]

fig_cum_returns_optimized = plot_cum_returns(portfolio_cons_22['Optimized Conservative Portfolio'],
                                             'Cumulative Returns of Optimized Conservative Portfolio '
                                             'Starting with $100')
fig_cum_returns_optimized.show()

fig1, ax1 = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))
tickers_listed = (df2_var['ticker']).to_numpy()
data = (weightsCons_df['weights']).to_numpy()

colors = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94', '#8AAD9D', '#778387', '#787070', '#C0B8B7', '#DEDBDE',
          '#EDCEE5', '#BBD5F2', '#DEC3E1', '#FFCCCB', '#C7EDDC', '#DBF4F0', '#FFCCCB', '#EEEBE2', '#C7EDDC', '#7FB9BC']
plt.rcParams.update({'font.size': 15})

# ax1.pie(data, colors = colors, labels=tickers_listed, autopct='%1.1f%%', startangle=90)
ax1.pie(data, colors=colors, labels=tickers_listed, autopct='%1.1f%%', pctdistance=1.2, labeldistance=None,
        startangle=90)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.legend(fontsize=15, title="Company", title_fontsize=15, loc='center right', bbox_to_anchor=(1, 0, 0.5, 1))

"""

def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
wedges, texts, autotexts = ax1.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax1.legend(wedges, tickers_listed,


          )   
          #labelcolor=['#A8E6CF','#DCEDC1','#FFD3B6','#FFAAA5','#FF8B94','#8AAD9D','#778387','#787070','#C0B8B7','#DEDBDE','#EDCEE5','#BBD5F2','#DEC3E1','#FFCCCB','#C7EDDC','#DBF4F0','#FFCCCB','#EEEBE2','#C7EDDC','#7FB9BC']

"""

plt.show()

# neutral portfolio weights

mu = mean_historical_return(data)
S = risk_models.sample_cov(data)

df1_sharpe = pd.DataFrame(columns=['ticker', 'calculation'])
for i in range(0, len(tickers_list)):
    value = (mu[i] - 0.0255) / df1_var['variance'][i]
    new_row = {'ticker': tickers_list[i], 'calculation': value}
    df1_sharpe = df1_sharpe.append(new_row, ignore_index=True)

df2_sharpe = df1_sharpe.nlargest(20, 'calculation')
mask = df2_sharpe['ticker']
portfolio_neu = data[mask]

portfolio_neu_22 = pd.DataFrame(columns=mask)
for i in mask:
    portfolio_neu_22[i] = data_22[i].copy()

portfolio_agg_22 = portfolio_neu_22

# portfolio_agg_22 = pd.DataFrame(columns=mask)
# for i in mask:
#     portfolio_agg_22[i] = data_22[i].copy()

mu_sharpe = expected_returns.mean_historical_return(portfolio_neu)
S_sharpe = risk_models.sample_cov(portfolio_neu)

ef = EfficientFrontier(mu_sharpe, S_sharpe)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
weightsNeu = ef.max_sharpe(risk_free_rate=0.0255)

ef.portfolio_performance(verbose=True)

# neutral portfolio for 2018-2022

weightsNeu_df = pd.DataFrame.from_dict(weightsNeu, orient='index')
weightsNeu_df.columns = ['weights']

portfolio_neu_22['Optimized Neutral Portfolio'] = 0

for i in range(len(weightsNeu_df)):
    portfolio_neu_22['Optimized Neutral Portfolio'] += portfolio_neu_22.iloc[:, i] * weightsNeu_df['weights'][i]

print(portfolio_neu_22)
fig1, ax1 = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))
tickers_listed = (df2_sharpe['ticker']).to_numpy()
data = (weightsCons_df['weights']).to_numpy()

colors = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94', '#8AAD9D', '#778387', '#787070', '#C0B8B7', '#DEDBDE',
          '#EDCEE5', '#BBD5F2', '#DEC3E1', '#FFCCCB', '#C7EDDC', '#DBF4F0', '#FFCCCB', '#EEEBE2', '#C7EDDC', '#7FB9BC']
plt.rcParams.update({'font.size': 15})

# ax1.pie(data, colors = colors, labels=tickers_listed, autopct='%1.1f%%', startangle=90)
ax1.pie(data, colors=colors, labels=tickers_listed, autopct='%1.1f%%', pctdistance=1.2, labeldistance=None,
        startangle=90)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.legend(fontsize=15, title="Company", title_fontsize=15, loc='center right', bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()

# drawing graphs

ef = EfficientFrontier(mu_sharpe, S_sharpe)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
fig, ax = plt.subplots(figsize=(8, 6))
ef_max_sharpe = copy.deepcopy(ef)
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_max_sharpe.max_sharpe(risk_free_rate=0.0255)
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

# aggressive portfolio weights

# ef = EfficientFrontier(mu_sharpe, S_sharpe, weight_bounds=(0.03,1))
# ef.add_objective(objective_functions.L2_reg, gamma=0.1)
# weightsAggReturn = ef.efficient_return(target_return=.5, market_neutral=True)

# print(weightsAggReturn)
# ef.portfolio_performance(verbose=True)

ef = EfficientFrontier(mu_sharpe, S_sharpe, weight_bounds=(0.02, 1))
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
weightsAggRisk = ef.efficient_risk(target_volatility=0.9, market_neutral=False)
print(weightsAggRisk)
ef.portfolio_performance(verbose=True)

df2_sharpe = df1_sharpe.nlargest(10, 'calculation')
mask = df2_sharpe['ticker']
portfolio_neu = data[mask]

portfolio_neu_22 = pd.DataFrame(columns=mask)
for i in mask:
    portfolio_neu_22[i] = data_22[i].copy()

portfolio_agg_22 = portfolio_neu_22

# portfolio_agg_22 = pd.DataFrame(columns=mask)
# for i in mask:
#     portfolio_agg_22[i] = data_22[i].copy()

mu_sharpe = expected_returns.mean_historical_return(portfolio_neu)
S_sharpe = risk_models.sample_cov(portfolio_neu)

ef = EfficientFrontier(mu_sharpe, S_sharpe)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
weightsNeu = ef.max_sharpe(risk_free_rate=0.0255)

ef.portfolio_performance(verbose=True)

# aggressive portfolio 2018-2022

weightsAgg_df = pd.DataFrame.from_dict(weightsAggRisk, orient='index')
weightsAgg_df.columns = ['weights']

portfolio_agg_22['Optimized Aggressive Portfolio'] = 0

for i in range(len(weightsAgg_df)):
    portfolio_agg_22['Optimized Aggressive Portfolio'] += portfolio_agg_22.iloc[:, i] * weightsAgg_df['weights'][i]

fig_cum_returns_optimized = plot_cum_returns(portfolio_agg_22['Optimized Aggressive Portfolio'],
                                             'Cumulative Returns of Optimized Aggressive Portfolio Starting with $100')
fig_cum_returns_optimized.show()

fig1, ax1 = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))
tickers_listed = (df2_sharpe['ticker']).to_numpy()
data = (weightsAgg_df['weights']).to_numpy()
colors = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94', '#8AAD9D', '#778387', '#787070', '#C0B8B7', '#DEDBDE',
          '#EDCEE5', '#BBD5F2', '#DEC3E1', '#FFCCCB', '#C7EDDC', '#DBF4F0', '#FFCCCB', '#EEEBE2', '#C7EDDC', '#7FB9BC']
ax1.pie(data, colors=colors, labels=tickers_listed, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.show()


# conservative portfolio behaviour

def plot_cum_returns(data, title):
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod() * 100
    fig = px.line(daily_cum_returns, title=title)
    return fig


fig_cum_returns = plot_cum_returns(portfolio_cons, 'Cumulative Returns of Individual Stocks Starting with $100')
fig_cum_returns.show()
