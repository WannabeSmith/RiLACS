import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocess

sys.path.append("../../../confseq_wor/cswor")
sys.path.append("../../../SHANGRLA/Code")
sys.path.append("../../../Betting/Code")

# for visual studio because debugging is a nightmare...
sys.path.append("/Users/i/Documents/GitProjects/confseq_wor/cswor")
sys.path.append("/Users/i/Documents/GitProjects/SHANGRLA/Code")
sys.path.append("/Users/i/Documents/GitProjects/Betting/Code")

sys.path.append("../../")

test = sys.path
import cswor as cswor
import assertion_audit_utils as aau
import betting_eb as beb
from vacsine import *


tnnm = aau.TestNonnegMean()
kmart_N = lambda x: tnnm.kaplan_martingale(x, N=N, t=m_null)[1]

# MC-Kmart
mckmart_N = lambda x, m: mckmart(x, m=m, N=N, reps=100)

mckmart2_N = lambda x: mckmart2(x, m=m_null, N=N, reps=50)

# Hedged
hedged_N = lambda x: beb.betting_mart(
    x, m=m_null, WoR=True, N=N, theta=1, alpha=0.05 * 2, trunc_scale=0.8
)

# aSOS
aSOS_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda x, m: np.maximum(0, beb.lambda_aSOS(x, m)),
    trunc_scale=0.8,
)

# LBOW
LBOW_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda x, m: np.maximum(0, beb.lambda_LBOW(x, m)),
)

ONSm_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda x, m: np.maximum(0, beb.lambda_COLT18_ONS(x, m)),
    trunc_scale=0.8,
)


apriori_Kelly_N = lambda x, m: beb.betting_mart(
    x,
    m=m,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda y, m: get_apriori_Kelly_bet(x),
    trunc_scale=1,
)

TriKelly_N = lambda x, m: distKelly(x, m=m, N=N, dist=linear_gamma_dist)

SqKelly_N = lambda x, m: distKelly(x, m=m, N=N, dist=square_gamma_dist, reps=50)

apriori_BRAVO = lambda x, m: bravo(x, mu_alt=np.mean(x), num_samples=None)


martingale = SqKelly_N

N = 10000
alpha = 0.05
means = [0.5]
nuisances = [0]

data_dict = get_data_dict(N, means, nuisances)
x = data_dict[list(data_dict.keys())[0]]
t = np.arange(1, N + 1)

nulls = (0.45, 0.48, 0.5)
pval_list = [None] * len(nulls)
for i in np.arange(0, len(nulls)):
    mart_values = martingale(x, m=nulls[i])
    # Just a little hack for the plots.
    # this is taken care of rigorously in the cs code.
    mart_values[np.isnan(mart_values)] = math.inf if i != 2 else 1
    pval_list[i] = np.minimum.accumulate(np.minimum(1/mart_values, 1))

plt.style.use("seaborn-white")
plt.style.use("seaborn-colorblind")
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[0].spines['left'].set_edgecolor('lightgrey')
# ax[0].spines['bottom'].set_edgecolor('lightgrey')
# ax[1].spines['left'].set_edgecolor('lightgrey')
# ax[1].spines['bottom'].set_edgecolor('lightgrey')
for spine in ax[0].spines.values():
    spine.set_edgecolor('lightgrey')

for spine in ax[1].spines.values():
    spine.set_edgecolor('lightgrey')

l, u = beb.cs_from_martingale(
    x, mart_fn=martingale, WoR=True, N=N, parallel=True, breaks=500
)
l = np.maximum.accumulate(l)

colors = ["tab:pink", "mediumseagreen", "royalblue"]
linestyles = ("-.", "--", "-")
ax[0].plot(t, l, color="tab:grey")

stopping_times = [None] * 3
for i in np.arange(0, len(nulls)):
    ax[0].axhline(nulls[i], linestyle=linestyles[i], color=colors[i])
    ax[1].plot(
        pval_list[i],
        color=colors[i],
        linestyle=linestyles[i],
        label="$H_0: \mu^\star \leq " + str(nulls[i]) + "$",
    )
    # Add vertical line for when the sampling can stop
    if any(l > nulls[i]):
        stopping_time = np.where(l > nulls[i])[0][0]
        stopping_times[i] = stopping_time
        ax[0].vlines(
            x=stopping_time, color=colors[i], linestyle=":", ymin=0, ymax=nulls[i]
        )
        ax[0].text(
            x=stopping_time + 30,
            y=0.35,
            s=str(stopping_time) + "\nballots",
            color=colors[i],
        )
        ax[1].vlines(
            x=stopping_time, color=colors[i], linestyle=":", ymin=-1, ymax=alpha
        )

ax[0].set(xlabel="Number of ballots", ylabel="Lower confidence sequence")

yticks = np.append(np.arange(0, 1 + 0.1, step=0.1), [0.45, 0.48])
base_xticks = np.arange(0, 10000, step=2500)
ax[0].set_yticks(yticks)
ax[0].get_yticklabels()[5].set_color(colors[2])
ax[0].get_yticklabels()[-2].set_color(colors[0])
ax[0].get_yticklabels()[-1].set_color(colors[1])
ax[0].set_ylim(0.3, 0.55)
ax[0].set_xticks(base_xticks)

ax[1].set(xlabel="Number of ballots", ylabel="Anytime $p$-values")
ax[1].axhline(alpha, color="grey", linestyle=":")
xticks = np.append(base_xticks, [stopping_times[0], stopping_times[1]])
closest_tick_0 = min(base_xticks, key=lambda y: abs(y - stopping_times[0]))
closest_tick_1 = min(base_xticks, key=lambda y: abs(y - stopping_times[1]))
xticklabels = [
    str(t) if (t != closest_tick_0 and t != closest_tick_1) else "" for t in xticks
]

ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xticklabels)
ax[1].get_xticklabels()[-2].set_color(colors[0])
ax[1].get_xticklabels()[-1].set_color(colors[1])
ax[1].set_ylim(-0.1, 1.1)
ax[1].set_yticks(np.append(alpha, np.arange(0.5, 1.5, step=0.5)))

ax[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("figures/cs_testing.pdf", bbox_inches='tight')

plt.show()
