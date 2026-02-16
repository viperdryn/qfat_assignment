import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data"
figure_path = project_root / "figures"

# Prepare data table
industry_returns = pd.read_csv(data_path / "Industry.csv")
print(industry_returns.columns)


industry_returns['mdate'] = pd.to_datetime(industry_returns['mdate'].astype(str), format="%Y%m")
industry_returns = industry_returns.set_index('mdate')
industry_returns = industry_returns.sort_index()

industry_returns = industry_returns.replace("%", "", regex=True).astype(float) / 100

# Calculate the results for the 1.1 problem
industry_avg_returns = industry_returns.shift(1).rolling(window=12).mean().dropna()
industry_avg_returns.to_csv(data_path / "industry_avg_return.csv")

industry_ranks = industry_avg_returns.rank(axis=1, method="min", ascending=True).astype(int)
industry_ranks.to_csv(data_path / "industry_ranks.csv")

industry_avg_ranks = industry_ranks.mean(axis=0)
industry_avg_ranks.to_csv(data_path / "industry_avg_ranks.csv")

industry_long_short = (industry_ranks > len(industry_ranks.columns) // 2)
industry_avg_long_short = industry_long_short.mean(axis=0)

worst_industries = industry_avg_long_short.sort_values().index.to_list()[0:int(len(industry_avg_long_short) / 2)]
best_industries = industry_avg_long_short.sort_values(ascending=False).index.to_list()[0:int(len(industry_avg_long_short) / 2)]

####################################################################################
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

####################################################################################
# Answer 1.1.a

worst_avg_rank_industries = industry_avg_ranks.sort_values().index.to_list()[0:int(len(industry_avg_ranks) / 2)]
best_avg_rank_industries = industry_avg_ranks.sort_values(ascending=False).index.to_list()[0:int(len(industry_avg_ranks) / 2)]

print(f"Industries with the worst average rank: {worst_avg_rank_industries}")
print(f"Industries with the best average rank: {best_avg_rank_industries}")

####################################################################################
# Answer 1.1.b

worst_avg_ranks = industry_avg_ranks[worst_avg_rank_industries]
best_avg_ranks = industry_avg_ranks[best_avg_rank_industries].sort_values()

print(f"Worst ranks: \n{worst_avg_ranks}")
print(f"Best average ranks: \n{best_avg_ranks}")

####################################################################################
# Answer 1.1.c

plt.figure(figsize=(12, 6))
plt.plot(industry_ranks['Autos'])
plt.title("Rank of the 'Autos' Industry Over Time")
plt.xlabel("Time")
plt.ylabel("Rank")
plt.axhline(y=17, color='red', linestyle='--', linewidth=3)
plt.savefig(figure_path / "autos_industry_rank.png")

####################################################################################
# Answer 1.1.d

plt.figure(figsize=(12, 6))
colors = ["blue" if col in best_industries else "red" for col in industry_avg_long_short.index.to_list()]
plt.bar(industry_avg_long_short.index, industry_avg_long_short.values, color=colors)
plt.xticks(rotation=45)
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
plt.ylabel("Long Ratio")
plt.title("Industries Long / Short Ratios")

plt.savefig(figure_path / "industries_avg_long_short.png")

####################################################################################
# Answer 1.1.d

industries_in_order = industry_avg_long_short.sort_values(ascending=False).index.to_list()

fig, axes = plt.subplots(8, 2, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(0, len(industries_in_order), 2):
    ax_index = i // 2  # subplot index
    ax = axes[ax_index]

    # Select 4 industries for this subplot
    cols = industries_in_order[i:i+2]
    for col in cols:
        ax.plot(industry_ranks.index, industry_ranks[col], label=col)

    ax.axhline(y=17, color='red', linestyle='--', linewidth=1)
    ax.set_title(f"Most longed industries from {i+1} to {i+2}")
    ax.grid(True)
    ax.yaxis.grid(False)
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()

####################################################################################

# Answer 1.3: Loser Portfolio