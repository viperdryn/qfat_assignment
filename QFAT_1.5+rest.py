# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:24:13 2026

@author: Crenguta Irimie
"""
#TOT

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

project_root = Path().resolve()
data_path = project_root / "data"
figure_path = project_root / "figures"
print(project_root)
print(data_path)
print(data_path.exists())

# prepare data table
industry_returns = pd.read_csv(data_path / "Industry.csv")
print(industry_returns.columns)

industry_returns['mdate'] = pd.to_datetime(industry_returns['mdate'].astype(str), format="%Y%m")
industry_returns = industry_returns.set_index('mdate')
industry_returns = industry_returns.sort_index()

industry_returns = industry_returns.replace("%", "", regex=True).astype(float) / 100
industry_returns.head()

industry_avg_returns = industry_returns.shift(1).rolling(window=12).mean().dropna()
industry_avg_returns.to_csv(data_path / "industry_avg_return.csv")

industry_ranks = industry_avg_returns.rank(axis=1, method="min", ascending=True).astype(int)
industry_ranks.to_csv(data_path / "industry_ranks.csv")
head = industry_ranks.head()
print(head)

industry_avg_ranks = industry_ranks.mean(axis=0)
industry_avg_ranks.to_csv(data_path / "industry_avg_ranks.csv")
show = industry_avg_ranks.sort_values()
print(show)

industry_long_short = (industry_ranks > len(industry_ranks.columns) // 2)    #monthly winners vs losers
industry_avg_long_short = industry_long_short.mean(axis=0)  #average winner frequency per industry

worst_industries = industry_avg_long_short.sort_values().index.to_list()[0:int(len(industry_avg_long_short) / 2)]    #15 worst industries in the losers portofolio
best_industries = industry_avg_long_short.sort_values(ascending=False).index.to_list()[0:int(len(industry_avg_long_short) / 2)]  #15 best industries in the winners portofolio

# General

plt.figure(figsize=(12, 6))
for col in industry_avg_returns:
    plt.plot(industry_avg_returns.index, industry_avg_returns[col], label=col)

plt.xlabel("Month")
plt.ylabel("Average Return (12M Ex-Ante)")
plt.title("Average Industry Returns Over Time")
plt.legend(loc="best", fontsize=8)
plt.grid(True)

plt.savefig(figure_path / "avg_ind_return.png")

worst_avg_rank_industries = industry_avg_ranks.sort_values().index.to_list()[0:int(len(industry_avg_ranks) / 2)]     #Find the lowest average rank industries
best_avg_rank_industries = industry_avg_ranks.sort_values(ascending=False).index.to_list()[0:int(len(industry_avg_ranks) / 2)]    #Find the highest average rank industries

print(f"Industries with the worst average rank: {worst_avg_rank_industries}")
print(f"Industries with the best average rank: {best_avg_rank_industries}")

# Calculate the average rank of the 15 worst and 15 best industries (from 1.1a)

mean_worst_rank = industry_avg_ranks[worst_avg_rank_industries].mean()
mean_best_rank = industry_avg_ranks[best_avg_rank_industries].mean()

print(f"Average rank of the 15 worst industries: {mean_worst_rank:.2f}")
print(f"Average rank of the 15 best industries: {mean_best_rank:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(industry_ranks['Autos'])
plt.title("Rank of the 'Autos' Industry Over Time")
plt.xlabel("Time")
plt.ylabel("Rank")
plt.axhline(y=17, color='red', linestyle='--', linewidth=3)
plt.savefig(figure_path / "autos_industry_rank.png")

# Calculate standard deviation of ranks for each industry (measure of variability)
rank_std = industry_ranks.std(axis=0).sort_values(ascending=False)

print("Standard deviation of ranks (higher = more volatile):")
print(rank_std)
print(f"\nAverage standard deviation across all industries: {rank_std.mean():.2f}")

# Calculate turnover: How many industries change position from top to bottom half or vice versa
industry_long_short_prev = industry_long_short.shift(1)
turnover = (industry_long_short != industry_long_short_prev).sum(axis=1)
avg_turnover = turnover.mean()

print(f"\nAverage number of industries switching between top/bottom half each month: {avg_turnover:.2f} out of 30")
print(f"Percentage turnover per month: {(avg_turnover/30)*100:.1f}%")

# Plot turnover over time
plt.figure(figsize=(12, 6))
plt.plot(turnover.index, turnover)
plt.title("Monthly Turnover: Number of Industries Switching Between Top/Bottom Half")
plt.xlabel("Time")
plt.ylabel("Number of Industries Switching")
plt.axhline(y=avg_turnover, color='red', linestyle='--', label=f'Average: {avg_turnover:.2f}')
plt.legend()
plt.grid(True)
plt.savefig(figure_path / "industry_turnover.png")
plt.show()


# =====================================================
# PART 1.2 (Original Code 2 – minimally adapted path)
# =====================================================

import numpy as np
import statsmodels.api as sm    

# Reuse the same data file location as in 1.1
industry = pd.read_csv(data_path / "Industry.csv")

industry.rename(columns = {'mdate' : 'Date', 'MktRf' : 'Market Excess'}, inplace = True)

industry['Date'] = pd.to_datetime(industry['Date'], format = '%Y%m')

cols = industry.columns.difference(['Date'])

for c in cols:
    industry[c] = (industry[c].astype(str).str.replace('%', '').astype(float) / 100)

print(cols)

# industry return columns (exclude Date, Rf, Market Excess)
industries = [c for c in industry.columns if c not in ['Date', 'Rf', 'Market Excess']]

# past 12-month average return, excluding current month
past12 = industry[industries].shift(1).rolling(12).mean()

rankings = past12.rank(axis = 1, method = "first")

winners = rankings >= 16

print(rankings)

# winner portfolio total return each month (equal-weight across winners)
winner_ret = industry[industries].where(winners).mean(axis=1)

# winner portfolio excess return
winner_excess = winner_ret - industry['Rf']

# stats 
mean_excess = winner_excess.mean()
std_excess  = winner_excess.std(ddof=1)
sharpe_m    = mean_excess / std_excess
sharpe_a    = sharpe_m * np.sqrt(12)

print("\n===== 1.2 Results =====")
print("Mean monthly excess:", mean_excess)
print("Std monthly excess :", std_excess)
print("Monthly Sharpe     :", sharpe_m)
print("Annual Sharpe      :", sharpe_a)

############################################################################
# 1.3 Loser Portfolio:
############################################################################

import pandas as pd
import numpy as np

# Load the average ranks:
industry_ranks = pd.read_csv(data_path / "industry_avg_ranks.csv")
industry_ranks.rename(columns={'Unnamed: 0': 'Industry', '0': 'Average Rank'}, inplace=True)
industry_ranks = industry_ranks.sort_values(by='Average Rank', ascending=True, ignore_index=True)

# Load the average returns:
average_return = pd.read_csv(data_path / "industry_avg_return.csv")

# Get the excess returns for each month:
average_ex_return = average_return.copy()
for i in range(len(average_ex_return.columns)-2):
    average_ex_return.iloc[:, i+2] = average_ex_return.iloc[:, i+2].sub(average_ex_return.iloc[:, 1], axis=0)

# Get the loser portfolio:
loser_portfolio = industry_ranks.loc[2:16, 'Industry'].to_list()    # the 15 industries with the worst average rank
print(f"The industries in the loser portfolio are: {loser_portfolio}")

############################################################################
# Answer for the questions in 1.3:
############################################################################

# only keep the average monthly return for the loser portfolio:
average_ex_return = average_ex_return.drop(columns=[col for col in average_ex_return.columns if col not in loser_portfolio])
average_ex_monthly_returns_monthly = average_ex_return.mean(axis=1)
average_ex_monthly_return = average_ex_monthly_returns_monthly.mean()
print(f"The average monthly return of the loser portfolio is: {average_ex_monthly_return:.5f}")

# What is the standard deviation of the monthly returns of the loser portfolio?
average_ex_monthly_return_std = average_ex_monthly_returns_monthly.std()
print(f"The standard deviation of the monthly returns of the loser portfolio is: {average_ex_monthly_return_std:.5f}")

# What is the monthly Sharpe ratio of the loser portfolio?
loser_sharpe_ratio = average_ex_monthly_return / average_ex_monthly_return_std
print(f"The monthly Sharpe ratio of the loser portfolio is: {loser_sharpe_ratio:.2f}")

# What is the annual Sharpe ratio of the loser portfolio?
loser_annual_sharpe_ratio = loser_sharpe_ratio * np.sqrt(12)
print(f"The annual Sharpe ratio of the loser portfolio is: {loser_annual_sharpe_ratio:.2f}")

# What is the annualized Sharpe ratio of the overall market index?
market_ex_monthly_return = average_ex_return.iloc[:, 1].mean()
market_ex_monthly_return_std = average_ex_return.iloc[:, 1].std()
market_sharpe_ratio = market_ex_monthly_return / market_ex_monthly_return_std
market_annual_sharpe_ratio = market_sharpe_ratio * np.sqrt(12)
print(f"The annual Sharpe ratio of the market index is: {market_annual_sharpe_ratio:.2f}")


# =====================================================
# PART 1.4 (Original code – minimally adapted to your setup)
# =====================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data (changed ONLY the path to match your project structure)
df = pd.read_csv(data_path / "Industry.csv")

# mdate is YYYYMM (e.g., 192607). convert to month-end timestamps
df["mdate"] = df["mdate"].astype(int)
df["date"] = pd.to_datetime(df["mdate"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
df = df.sort_values("date").set_index("date")

# convert percentage to decimals
def to_decimal(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.endswith("%"):
            return float(x[:-1]) / 100.0
        return float(x)
    return float(x)

# apply conversion
df = df.apply(lambda col: col.map(to_decimal))

# label/identify columns
rf_col = "Rf"
mktrf_col = "MktRf"
industry_cols = [c for c in df.columns if c not in {rf_col, mktrf_col, "mdate"}]

# compute past 12-month average returns for each industry (excluding month t)
past12_mean = df[industry_cols].rolling(window=12, min_periods=12).mean().shift(1)

# winner/loser portfolios starting 1927/07
def winner_loser_returns(row_returns, row_signal, k=15):
    s = row_signal.dropna()
    if len(s) < 2 * k:
        return np.nan, np.nan
    winners = s.nlargest(k).index
    losers = s.nsmallest(k).index
    w_ret = row_returns[winners].mean()
    l_ret = row_returns[losers].mean()
    return w_ret, l_ret

# build time-series
winner = []
loser = []
for dt in df.index:
    w, l = winner_loser_returns(df.loc[dt, industry_cols], past12_mean.loc[dt, industry_cols], k=15)
    winner.append(w)
    loser.append(l)

winner = pd.Series(winner, index=df.index, name="Winner")
loser = pd.Series(loser, index=df.index, name="Loser")

# keep only months where portfolios are defined
mask_defined = winner.notna() & loser.notna()
winner = winner[mask_defined]
loser = loser[mask_defined]

# (a) long-short ind-mom and annualised Sharpe ratio
ind_mom = (winner - loser).rename("IndMom")

mean_m = ind_mom.mean()
std_m = ind_mom.std(ddof=1)
sharpe_monthly = mean_m / std_m
sharpe_annualised = sharpe_monthly * np.sqrt(12)

first_date = ind_mom.first_valid_index()
print("First IndMom date:", first_date)
print("Last date:", ind_mom.index.max())

n_obs = ind_mom.dropna().shape[0]
print("Number of observations:", n_obs)

print(f"Ind-Mom monthly mean: {mean_m:.6f}")
print(f"Ind-Mom monthly std:  {std_m:.6f}")
print(f"Ind-Mom monthly Sharpe: {sharpe_monthly:.4f}")
print(f"Ind-Mom annualised Sharpe (1.4a): {sharpe_annualised:.4f}")

# (b) regression of IndMom on MktRf
reg_df = pd.concat([ind_mom, df["MktRf"]], axis=1).dropna()
reg_df.columns = ["IndMom", "MktRf"]

X = sm.add_constant(reg_df["MktRf"])
y = reg_df["IndMom"]

model = sm.OLS(y, X).fit()

print(model.summary())

beta = model.params["MktRf"]
beta_t_stat = model.tvalues["MktRf"]
print(f"Ind-Mom beta on MktRf: {beta:.4f}")
print(f"Ind-Mom beta t-stat: {beta_t_stat:.4f}")

# (c) alpha and t-stat
alpha = model.params["const"]
alpha_t_stat = model.tvalues["const"]

print(f"Monthly alpha (const): {alpha:.6f} or {alpha*100:.2f}% per month")
print(f"Alpha t-stat: {alpha_t_stat:.4f}")

# (d) annualised alpha
annualised_alpha = 12 * alpha
print(f"Annualised alpha: {annualised_alpha:.4f} or {annualised_alpha*100:.2f}% per year")

# robustness check: Newey-West standard errors (lag=6)
model_hac6 = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(model_hac6.summary())

# =====================================================
# PART 1.5 — Cumulative returns (Winner, Loser, Long/Short, Market) + log-scale plot
# =====================================================

import numpy as np
import matplotlib.pyplot as plt


# Market TOTAL return (MktRf is excess, so add Rf)
market_total = (df["MktRf"] + df["Rf"]).reindex(ind_mom.index)

# Align series to the same dates 
common_idx = winner.index.intersection(loser.index).intersection(ind_mom.index).intersection(market_total.index)

winner_15 = winner.reindex(common_idx)
loser_15 = loser.reindex(common_idx)
indmom_15 = ind_mom.reindex(common_idx)
market_15 = market_total.reindex(common_idx)


# Cumulative wealth using log-returns (numerically stable) 
cum_winner = np.exp(np.log1p(winner_15).cumsum())
cum_loser  = np.exp(np.log1p(loser_15).cumsum())
cum_indmom = np.exp(np.log1p(indmom_15).cumsum())
cum_market = np.exp(np.log1p(market_15).cumsum())

# Normalize to start at 1 ($1 invested initially)
cum_winner = cum_winner / cum_winner.iloc[0]
cum_loser  = cum_loser / cum_loser.iloc[0]
cum_indmom = cum_indmom / cum_indmom.iloc[0]
cum_market = cum_market / cum_market.iloc[0]

# Plot cumulative returns on a log-scale 
plt.figure(figsize=(12, 6))
plt.plot(cum_winner.index, cum_winner, label="Winner (Top 15)")
plt.plot(cum_loser.index,  cum_loser,  label="Loser (Bottom 15)")
plt.plot(cum_indmom.index, cum_indmom, label="Long/Short Ind-Mom")
plt.plot(cum_market.index, cum_market, label="Market (Total)")

plt.yscale("log")
plt.title("Cumulative Returns (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Cumulative Wealth (log scale)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(figure_path / "cumulative_returns_logscale.png", dpi=200)
plt.show()
