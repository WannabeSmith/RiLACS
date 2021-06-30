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

martingale_dict = {
    r"$\it{a\ priori}$ Kelly": apriori_Kelly_N,
    "SqKelly": SqKelly_N,
    # r"$\it{a\ priori}$ BRAVO": apriori_BRAVO,
    "dKelly/KMart": mckmart_N,
    # "TriKelly": TriKelly_N,
    # 'KMart' : kmart_N,
    # 'PM' : hedged_N,
    # 'aKelly' : aSOS_N,
    # 'LBOW' : LBOW_N,
    # 'ONS-m' : ONSm_N
}

colors = ["royalblue", "mediumseagreen", "tab:pink"]
linestyles = ["--", "-.", ":"]
assert len(colors) == len(martingale_dict) and len(linestyles) == len(colors)
colors_dict = {
    list(martingale_dict.keys())[i]: colors[i] for i in range(len(martingale_dict))
}
linestyles_dict = {
    list(martingale_dict.keys())[i]: linestyles[i] for i in range(len(martingale_dict))
}


N = 10000
alpha = 0.05
means = [0.54]
nuisances = [0]

data_dict = get_data_dict(N, means, nuisances)
x = data_dict[list(data_dict.keys())[0]]
t = np.arange(1, N + 1)

plt.figure(figsize=(6, 3))
plt.style.use("seaborn-white")
plt.style.use("seaborn-colorblind")
plt.rcParams["font.family"] = "serif"
for spine in plt.gca().spines.values():
    spine.set_edgecolor("lightgrey")

for mart_name in martingale_dict:
    l, u = beb.cs_from_martingale(
        x, mart_fn=martingale_dict[mart_name], WoR=True, N=N, parallel=True, breaks=500
    )
    stopping_idx = np.where(l > 0.5)[0][0]
    plt.plot(
        t,
        l,
        color=colors_dict[mart_name],
        alpha=0.6,
    )
    plt.vlines(
        stopping_idx + 1,
        ymin=0,
        ymax=l[stopping_idx],
        color=colors_dict[mart_name],
        linestyles=linestyles_dict[mart_name],
        label=mart_name + "\nstopping time: " + str(stopping_idx + 1),
    )

plt.axhline(means[0], color="grey", linestyle="-", label=r'true mean $\mu^\star$')
plt.ylim(0.2, 0.55)
plt.legend(loc="best")
plt.xlabel("Ballots sampled")
plt.ylabel("Lower confidence sequence")
plt.savefig("figures/example_confseqs.pdf")
