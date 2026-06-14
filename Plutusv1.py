import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import reduce
import statsmodels.api as sm

st.sidebar.title("PlutusV1")

ticker = st.text_input("Enter a Stock Ticker:")

if ticker:

    st.write("You Entered:", ticker)

    df = yf.Ticker(ticker).history(
        period="max",
        interval="1mo",
        auto_adjust=True
    )

#creating a new dataset for price, return, volume and dividends only
df_stock = df[["Close","Volume","Dividends"]].copy()
df_stock.rename(columns={"Close":"Price"},inplace=True)
df_stock["log_returns"] = (np.log(df_stock["Price"])-np.log(df_stock["Price"]).shift(1))

#Importing shiller data
df_shiller = pd.read_csv("shiller_data_06.csv",index_col="Date")
df_shiller['sp500_log_returns'] = (np.log(df_shiller["Price"])-np.log(df_shiller["Price"]).shift(1))

#plotting historical price chart
fig1, ax1 = plt.subplots(figsize=(15,6))
ax1.plot(df_stock["Price"])
ax1.set_title("historical Stock Price")
ax1.set_ylabel("Price")
ax1.set_xlabel("Date")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(15,6))
ax2.plot(df_stock["log_returns"])
ax2.set_title("Historical Stock Returns (Log)")
ax2.set_ylabel("Returns")
ax2.set_xlabel("Date")
st.pyplot(fig2)

#Comapany information
#company summary
ceo = yf.Ticker(ticker).info['companyOfficers'][0]['name']
industry = yf.Ticker(ticker).info['industry']
sector = yf.Ticker(ticker).info['sector']
country = yf.Ticker(ticker).info['country']

#key Financial Metrics
eps = yf.Ticker(ticker).info['trailingEps']
pe_ratio = yf.Ticker(ticker).info['trailingPE']
market_cap = yf.Ticker(ticker).info['marketCap']
current_ratio = yf.Ticker(ticker).info['currentRatio']
peg_ratio = yf.Ticker(ticker).info['pegRatio']

#analyst targets
current_price = yf.Ticker(ticker).analyst_price_targets['current']
low_estimate = yf.Ticker(ticker).analyst_price_targets['low']
high_estimate = yf.Ticker(ticker).analyst_price_targets['high']

table2 = {'Company Summary':['CEO','Industry','Sector','Country'],
         'Info':[ceo,industry,sector,country]}

company_sum = pd.DataFrame(table2)

st.dataframe(company_sum, hide_index=True)

table3 = {'Key Financial Metrics':['EPS','PE Ratio','Market Cap','Current Ratio','PEG Ratio'],
         'Value':[eps,pe_ratio,market_cap,current_ratio,peg_ratio]}

fin_metrics = pd.DataFrame(table3)

st.dataframe(fin_metrics, hide_index=True)

table4 = {'Analyst Targets':['Current Price','Low Estimate','High Estimate'],
         'Value':[current_price,low_estimate,high_estimate]}

analyst_targets = pd.DataFrame(table4)

st.dataframe(analyst_targets, hide_index=True)

#creating summary statistics table
table1 = {'Monthly Return Comparison to Market':[ticker,'SP500'],
          'Mean':[df_stock["log_returns"].mean(), df_shiller['sp500_log_returns'].mean()],
          'Median':[df_stock["log_returns"].median(), df_shiller['sp500_log_returns'].median()],
          'Max':[df_stock["log_returns"].max(), df_shiller['sp500_log_returns'].max()],
          'Min':[df_stock["log_returns"].min(), df_shiller['sp500_log_returns'].min()],
          'Std. Dev':[df_stock["log_returns"].std(), df_shiller['sp500_log_returns'].std()],
          'Skewness':[df_stock["log_returns"].skew(), df_shiller['sp500_log_returns'].skew()],
          'Kurtosis':[df_stock["log_returns"].kurt(), df_shiller['sp500_log_returns'].kurt()]}

desc_stat = pd.DataFrame(table1)
st.table(desc_stat)

#stock returns distribution
fig, ax = plt.subplots(figsize=(10,6))

count, bins, ignored = ax.hist(
    df_stock['log_returns'],
    bins=50,
    density=True,
    alpha=0.6,
    color='skyblue',
    edgecolor='black',
    label='Returns Distribution'
)

# Normal distribution
mu = df_stock['log_returns'].mean()
std = df_stock['log_returns'].std()

x = np.linspace(bins[0], bins[-1], 1000)
pdf = norm.pdf(x, mu, std)

ax.plot(x, pdf, 'r--', linewidth=2, label='Normal Distribution')

# Labels
ax.set_title('UNH return distribution vs Normal Distribution')
ax.set_ylabel('Density')
ax.set_xlabel('Returns')
ax.legend()
ax.grid(True)
plt.tight_layout()

st.pyplot(fig)

#Distribution summary
kurt = df_stock['log_returns'].kurt()
skew = df_stock['log_returns'].skew()

st.subheader("Statistical Summary")

if kurt > 0.5:
    st.write("This stock exhibits leptokurtic distribution (common for stock returns), meaning it has fatter tails and higher probability of extreme returns.")
elif kurt < -0.5:
    st.write("This stock exhibits a platykurtic distributions indicating thinner tails and fewer extreme returns compared to the normal distribution")
else:
    st.write("This stock exhibits a mesokurtic distribution, similar to a normal distribution")

if skew > 0:
    st.write("This stock is positively skewed (right skewed) which means extreme positive returns occur more than negative returns")
elif skew < 0:
    st.write("This stock is negatively skewed (left skewed) which means extreme negative returns occur more than positive returns")
else:
    st.write("Returns are fairly symmetrical")

#CAPM Model

df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)


df_shiller.index = pd.to_datetime(df_shiller.index)

combined_df = pd.merge(
    df_shiller,
    df_stock,
    left_index=True,
    right_index=True,
    how='inner'  # only dates in both datasets
)

combined_df = combined_df.dropna()

combined_df['excess_return'] = combined_df['log_returns'] - combined_df['RF']

y = combined_df['excess_return']
X = combined_df['Mkt-RF']

X = sm.add_constant(X)

capm_model = sm.OLS(y,X).fit()

st.subheader("CAPM Model")

alpha = capm_model.params['const']
beta = capm_model.params['Mkt-RF']
r2 = capm_model.rsquared

table5 = {'CAPM Model Results':['Alpha','Beta','RSquared'],
         'Value':[alpha,beta,r2]}

capm_model = pd.DataFrame(table5)

st.dataframe(capm_model, hide_index=True)

#CAPM visualization
fig2, ax = plt.subplots(figsize=(15,6))
ax.scatter(combined_df['Mkt-RF'],combined_df['excess_return'],alpha=0.5,label='Monthly Returns')

#CAPM line
x = np.linspace(combined_df['Mkt-RF'].min(),combined_df['Mkt-RF'].max(),100)

y_hat = alpha + beta * x

ax.plot(x,y_hat,linewidth=2,label='CAPM Line')

ax.set_title('CAPM Regression Line')
ax.set_xlabel('Market Excess Return')
ax.set_ylabel('Stock Exces Return')
ax.legend()
ax.grid(True)

st.pyplot(fig2)

#Fama French Model
st.subheader('Fama French Model')
y1 = combined_df['excess_return']
X1 = combined_df[['Mkt-RF','SMB','HML']]
X1 = sm.add_constant(X1)

fama_french = sm.OLS(y1,X1).fit()

alphaff = fama_french.params['const']

beta_market = fama_french.params['Mkt-RF']
beta_size = fama_french.params['SMB']
beta_value = fama_french.params['HML']

r2ff = fama_french.rsquared

table5 = {'Fama French Model':['Alpha','Beta-Market','Beta-Size','Beta-Value','RSquared'],
         'Value':[alphaff,beta_market,beta_size,beta_value,r2ff]}

fama_french_model = pd.DataFrame(table5)

st.dataframe(fama_french_model, hide_index=True)