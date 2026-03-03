# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:37:25 2026

@author: Crenguta Irimie
"""


import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

project_root = Path().resolve()
data_path = project_root / "data"
figure_path = project_root / "figures"
print(project_root)
print(data_path)
print(data_path.exists())


# =====================================================
# PART 1.4 
# =====================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

data_path = project_root / "data"
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

#Long/Short TOTAL return
ind_mom_total = (ind_mom + df['Rf'].reindex(ind_mom.index))

# Align series to the same dates 
common_idx = winner.index.intersection(loser.index).intersection(ind_mom_total.index).intersection(market_total.index)

winner_15 = winner.reindex(common_idx)
loser_15 = loser.reindex(common_idx)
indmom_15 = ind_mom_total.reindex(common_idx)
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
