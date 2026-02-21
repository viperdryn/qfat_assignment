# 1.3 Loser Portfolio: 
############################################################################
import pandas as pd
import numpy as np

#Load the average ranks:
industry_ranks = pd.read_csv('/Users/konstantin/QFAT - Assignment 1 /qfat_assignment/data/industry_avg_ranks.csv') ##change these to your directory where the data is stored
industry_ranks.rename(columns={'Unnamed: 0': 'Industry', '0': 'Average Rank'}, inplace=True)
industry_ranks = industry_ranks.sort_values(by='Average Rank', ascending=True, ignore_index=True)

#Load the average returns:
average_return = pd.read_csv('/Users/konstantin/QFAT - Assignment 1 /qfat_assignment/data/industry_avg_return.csv') ##change these to your directory where the data is stored

#Get the excess returns for each month:
average_ex_return = average_return.copy()
for i in range(len(average_ex_return.columns)-2):
    average_ex_return.iloc[:, i+2] = average_ex_return.iloc[:, i+2].sub(average_ex_return.iloc[:, 1], axis=0)

#Get the loser protfolio: 
loser_portfolio = industry_ranks.loc[2:16, 'Industry'].to_list()    #the 15 industries with the worst average rank
print(f"The industries in the loser portfolio are: {loser_portfolio}")

############################################################################
#Answer for the questions in 1.3:
############################################################################

#only keep the average monthly return for the loser portfolio: 
average_ex_return = average_ex_return.drop(columns=[col for col in average_ex_return.columns if col not in loser_portfolio])
average_ex_monthly_returns_monthly = average_ex_return.mean(axis=1)
average_ex_monthly_return = average_ex_monthly_returns_monthly.mean()
print(f"The average monthly return of the loser portfolio is: {average_ex_monthly_return:.5f}")


#What is the standard deviation of the monthly returns of the loser portfolio?
average_ex_monthly_return_std = average_ex_monthly_returns_monthly.std()
print(f"The standard deviation of the monthly returns of the loser portfolio is: {average_ex_monthly_return_std:.5f}")

#What is the monthly Sharpe ratio of the loser portfolio?
loser_sharpe_ratio = average_ex_monthly_return / average_ex_monthly_return_std
print(f"The monthly Sharpe ratio of the loser portfolio is: {loser_sharpe_ratio:.2f}")

#What is the annual Sharpe ratio of the loser portfolio?
loser_annual_sharpe_ratio = loser_sharpe_ratio * np.sqrt(12)
print(f"The annual Sharpe ratio of the loser portfolio is: {loser_annual_sharpe_ratio:.2f}")

#What is the annualized sharpe ratio of the overal market index? 
market_ex_monthly_return = average_ex_return.iloc[:, 1].mean()
market_ex_monthly_return_std = average_ex_return.iloc[:, 1].std()
market_sharpe_ratio = market_ex_monthly_return / market_ex_monthly_return_std
market_annual_sharpe_ratio = market_sharpe_ratio * np.sqrt(12)
print(f"The annual Sharpe ratio of the market index is: {market_annual_sharpe_ratio:.2f}")
