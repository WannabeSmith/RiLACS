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

N = 10000

SqKelly_twosided_N = lambda x, m: distKelly(x, m=m, N=N, dist=square_gamma_dist, reps=50, theta=1/2)
SqKelly_onesided_N = lambda x, m: distKelly(x, m=m, N=N, dist=square_gamma_dist, reps=50, theta=1)

dKelly_N = lambda x, m: distKelly(x, m=m, N=N, dist=lambda y: 1/50, reps=50, theta=1/2)

martingale_dict = {
    "SqKelly 2sided": SqKelly_twosided_N,
    "SqKelly 1sided": SqKelly_onesided_N,
    # r"$\it{a\ priori}$ BRAVO": apriori_BRAVO,
    # "dKelly/KMart": dKelly_N,
    # "TriKelly": TriKelly_N,
    # 'KMart' : kmart_N,
    # 'PM' : hedged_N,
    # 'aKelly' : aSOS_N,
    # 'LBOW' : LBOW_N,
    # 'ONS-m' : ONSm_N
}

colors = ["royalblue", "tomato"]#, "tab:pink"]
linestyles = ["-", "--"]#, ":"]
assert len(colors) == len(martingale_dict) and len(linestyles) == len(colors)
colors_dict = {
    list(martingale_dict.keys())[i]: colors[i] for i in range(len(martingale_dict))
}
linestyles_dict = {
    list(martingale_dict.keys())[i]: linestyles[i] for i in range(len(martingale_dict))
}

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
        linestyle=linestyles_dict[mart_name],
        alpha=0.6,
    )
    if not mart_name.endswith('1sided'):
        plt.plot(
            t,
            u,
            color=colors_dict[mart_name],
            linestyle=linestyles_dict[mart_name],
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

plt.ylim(0.2, 0.8)
plt.axhline(means[0], color="grey", linestyle="-", label=r'true mean $\mu^\star$')
plt.legend(loc="lower right")
plt.xlabel("Ballots sampled")
plt.ylabel("Confidence sequence")
plt.savefig("figures/example_tallies.pdf")
