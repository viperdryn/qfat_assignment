# 1.3 Loser Portfolio: 
############################################################################
import pandas as pd
import numpy as np

#Load the average ranks:
industry = pd.read_csv('/Users/konstantin/QFAT - Assignment 1 /qfat_assignment/data/industry.csv') ##change these to your directory where the data is stored

industry.rename(columns = {'mdate' : 'Date', 'MktRf' : 'Market Excess'}, inplace = True)

industry['Date'] = pd.to_datetime(industry['Date'], format = '%Y%m')

cols = industry.columns.difference(['Date'])

for c in cols:
    industry[c] = (industry[c].astype(str).str.replace('%', '').astype(float) / 100)

industries = [c for c in industry.columns if c not in ['Date', 'Rf', 'Market Excess']]

# past 12-month average return, excluding current month (shift by 1)

past12 = industry[industries].shift(1).rolling(12).mean()

rankings = past12.rank(axis = 1, method = "first")

loosers = rankings <= 15

print(rankings)


# loser portfolio total return each month (equal-weight across loosers)
loosers_ret = industry[industries].where(loosers).mean(axis=1)

# loser portfolio excess return
loosers_excess = loosers_ret - industry['Rf']

# stats 
mean_excess = loosers_excess.mean()
std_excess  = loosers_excess.std(ddof=1)
sharpe_m    = mean_excess / std_excess
sharpe_a    = sharpe_m * np.sqrt(12)

print("Mean monthly excess:", mean_excess)
print("Std monthly excess :", std_excess)
print("Monthly Sharpe     :", sharpe_m)
print("Annual Sharpe      :", sharpe_a)

