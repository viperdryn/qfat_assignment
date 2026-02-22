import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = r"C:\Users\cg_ir\Downloads\30_Industry_Portfolios.csv"  

raw = pd.read_csv(
    file_path,
    skiprows=11,
    engine="python",
    na_values=[-99.99, -999, "-99.99", "-999"]
)

raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)


raw["Date"] = pd.to_numeric(raw["Date"], errors="coerce")
raw = raw.dropna(subset=["Date"])
raw["Date"] = raw["Date"].astype(int)

raw = raw[
    (raw["Date"] >= 190001) & (raw["Date"] <= 210012) &
    (raw["Date"] % 100 >= 1) & (raw["Date"] % 100 <= 12)
].copy()


raw["Date"] = pd.to_datetime(raw["Date"].astype(str), format="%Y%m", errors="coerce")
raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()


for c in raw.columns:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")


raw = raw / 100.0


raw = raw[~raw.index.duplicated(keep="first")].copy()


if not {"Mkt-RF", "RF"}.issubset(raw.columns):
    raise ValueError(f"Need columns 'Mkt-RF' and 'RF'. Found: {list(raw.columns)}")

factor_cols = ["Mkt-RF", "SMB", "HML", "RF"]
industry_cols = [c for c in raw.columns if c not in factor_cols]

R = raw[industry_cols].copy()


signal = R.shift(1).rolling(12, min_periods=12).mean()


signal = signal.loc["1927-07":]
raw = raw.loc[signal.index]
R = R.loc[signal.index]
signal = signal.loc[signal.index]

ranks = signal.rank(axis=1, ascending=True, method="average")


top_k = 15
n = len(industry_cols)

winner_mask = (ranks >= (n - top_k + 1)).to_numpy(dtype=float)
loser_mask  = (ranks <= top_k).to_numpy(dtype=float)

R_vals = R.to_numpy(dtype=float)

winner_ret = np.nansum(R_vals * winner_mask, axis=1) / np.nansum(winner_mask, axis=1)
loser_ret  = np.nansum(R_vals * loser_mask,  axis=1) / np.nansum(loser_mask,  axis=1)

winner_ret = pd.Series(winner_ret, index=R.index, name="Winner")
loser_ret  = pd.Series(loser_ret,  index=R.index, name="Loser")

indmom_ret = winner_ret - loser_ret
market_total = raw["Mkt-RF"] + raw["RF"]  


ok = (1 + winner_ret > 0) & (1 + loser_ret > 0) & (1 + indmom_ret > 0) & (1 + market_total > 0)
winner_ret, loser_ret, indmom_ret, market_total = winner_ret[ok], loser_ret[ok], indmom_ret[ok], market_total[ok]


log_cum_winner = np.log1p(winner_ret).cumsum()
log_cum_loser  = np.log1p(loser_ret).cumsum()
log_cum_indmom = np.log1p(indmom_ret).cumsum()
log_cum_market = np.log1p(market_total).cumsum()

plt.figure(figsize=(10, 6))
plt.plot(log_cum_winner, label="Winner")
plt.plot(log_cum_loser,  label="Loser")
plt.plot(log_cum_indmom, label="Long/Short ind-mom")
plt.plot(log_cum_market, label="Market")

plt.title("Cumulative Returns (Log Scale) — plotted as log(wealth)")
plt.xlabel("Date")
plt.ylabel("log(Cumulative Wealth)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
