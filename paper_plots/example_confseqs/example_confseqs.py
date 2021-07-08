from rilacs.confseqs import apriori_Kelly, dKelly, sqKelly
import numpy as np
import matplotlib.pyplot as plt
from rilacs.misc import get_data_dict

cs_dict = {
    r"$\it{a\ priori}$ Kelly": lambda x: apriori_Kelly(
        x,
        N=N,
        n_A=np.sum(x),
        n_B=N - np.sum(x),
        breaks=500,
        alpha=alpha,
        running_intersection=False,
        parallel=True,
    ),
    "SqKelly": lambda x: sqKelly(
        x,
        N=N,
        D=10,
        beta=1,
        breaks=500,
        alpha=alpha,
        running_intersection=False,
        parallel=True,
    ),
    "dKelly/KMart": lambda x: dKelly(
        x,
        N=N,
        D=10,
        beta=1,
        breaks=500,
        alpha=alpha,
        running_intersection=False,
        parallel=True,
    ),
}

colors = ["royalblue", "mediumseagreen", "tab:pink"]
linestyles = ["--", "-.", ":"]
assert len(colors) == len(cs_dict) and len(linestyles) == len(colors)
colors_dict = {list(cs_dict.keys())[i]: colors[i] for i in range(len(cs_dict))}
linestyles_dict = {list(cs_dict.keys())[i]: linestyles[i] for i in range(len(cs_dict))}


N = 10000
alpha = 0.05
margins = [0.04]
nuisances = [0]

data_dict = get_data_dict(N, margins, nuisances)
x = data_dict[list(data_dict.keys())[0]]
t = np.arange(1, N + 1)

plt.figure(figsize=(6, 2.5))
plt.style.use("seaborn-white")
plt.style.use("seaborn-colorblind")
plt.rcParams["font.family"] = "serif"
for spine in plt.gca().spines.values():
    spine.set_edgecolor("lightgrey")

for cs_name in cs_dict:
    l, u = cs_dict[cs_name](x)
    stopping_idx = np.where(l > 0.5)[0][0]
    plt.plot(
        t,
        l,
        color=colors_dict[cs_name],
        alpha=0.6,
    )
    plt.vlines(
        stopping_idx + 1,
        ymin=0,
        ymax=l[stopping_idx],
        color=colors_dict[cs_name],
        linestyles=linestyles_dict[cs_name],
        label=cs_name + "\nstopping time: " + str(stopping_idx + 1),
    )

plt.axhline(
    margins[0] + 0.5, color="grey", linestyle="-", label=r"true mean $\mu^\star$"
)
plt.ylim(0.2, 0.55)
plt.legend(loc="best")
plt.xlabel("Ballots sampled")
plt.ylabel("Lower confidence sequence")
plt.savefig("example_confseqs.pdf", bbox_inches="tight")
