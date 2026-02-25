import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = r"C:\Users\Alexandru\Desktop\QFAT Assignment 1.4\Industry.csv"

df = pd.read_csv(path)

# mdate is YYYYMM (e.g., 192607). convert to month date timestamps by parsing as int, then to string, then to datetime with format, then add month end offset.
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
# for each month t, we want mean over t-12 to t-1, so use rolling(12).mean() then shift(1).
past12_mean = df[industry_cols].rolling(window=12, min_periods=12).mean().shift(1)

# winner/loser portfolios starting 1927/07 (first month where past12_mean is defined)
# For each timestamp t, pick top 15 and bottom 15 industries by past12_mean, then equal-weight their month-t returns.
def winner_loser_returns(row_returns, row_signal, k=15):
    # row_signal: past 12m mean signal values for that date across industries
    s = row_signal.dropna() # just in case so that ranking is clean
    if len(s) < 2 * k:
        return np.nan, np.nan # also double check that there are enough industries with defined signal for both portfolios
    # sort and pick
    winners = s.nlargest(k).index
    losers = s.nsmallest(k).index
    # portfolio returns (equal weight!)
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

# keep only months where portfolios are defined (starts at 1927/07 so earlier months will be NaN)
mask_defined = winner.notna() & loser.notna()
winner = winner[mask_defined]
loser = loser[mask_defined]

#(a)
# long-short ind-mom excess return and annualised Sharpe ratio 
ind_mom = (winner - loser).rename("IndMom")  # already excess return since both winner and loser are raw returns, their difference is also excess return (Rf cancels out)

mean_m = ind_mom.mean()
std_m = ind_mom.std(ddof=1)
sharpe_monthly = mean_m / std_m
sharpe_annualised = sharpe_monthly * np.sqrt(12)

# double check the first and last dates used and nr. of months used

first_date = ind_mom.first_valid_index()
print("First IndMom date:", first_date)
print("Last date:", ind_mom.index.max())

# nr of obs
n_obs = ind_mom.dropna().shape[0]
print("Number of observations:", n_obs)

print(f"Ind-Mom monthly mean: {mean_m:.6f}")
print(f"Ind-Mom monthly std:  {std_m:.6f}")
print(f"Ind-Mom monthly Sharpe: {sharpe_monthly:.4f}")
print(f"Ind-Mom annualised Sharpe (1.4a): {sharpe_annualised:.4f}")

#(b)
# align the ind_mom and MktRf series for the regression and drop NaN values(rows)
reg_df = pd.concat([ind_mom, df["MktRf"]], axis=1).dropna()
reg_df.columns = ["IndMom", "MktRf"]

# fit OLS; IndMom - dependent, MktRf - independent
X = sm.add_constant(reg_df["MktRf"])  # add intercept (alpha)
#print(X.head()) this was to check that constant and MktRf were correctly set up
y = reg_df["IndMom"]

model = sm.OLS(y, X).fit()

print(model.summary())

# extract beta and its t-stat
beta = model.params["MktRf"]
beta_t_stat = model.tvalues["MktRf"]
print(f"Ind-Mom beta on MktRf: {beta:.4f}")
print(f"Ind-Mom beta t-stat: {beta_t_stat:.4f}")

#(c)
# extract alpha and its t-stat
# alpha is also an excess return since IndMom is and excess return
# alpha is part of IndMom that is not explained by MktRf 
alpha = model.params["const"]
alpha_t_stat = model.tvalues["const"]

print(f"Monthly alpha (const): {alpha:.6f} or {alpha*100:.2f}% per month")
print(f"Alpha t-stat: {alpha_t_stat:.4f}")

#(d)
annualised_alpha = 12 * alpha # scales linearly with time
print(f"Annualised alpha: {annualised_alpha:.4f} or {annualised_alpha*100:.2f}% per year")

#robustness check: run regression with Newey-West standard errors (lag=6 for monthly data)
model_hac6 = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(model_hac6.summary())
# when calculating HAC standard errors with 6 lags, beta becomes insignificant, alpha remains significant
