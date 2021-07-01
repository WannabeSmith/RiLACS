from rilacs.misc import get_data_dict
from rilacs.confseqs import sqKelly
import sys
import numpy as np
import matplotlib.pyplot as plt

N = 10000

cs_dict = {
    "SqKelly 2sided": lambda x: sqKelly(
        x,
        N=N,
        D=20,
        beta=1 / 2,
        breaks=1000,
        alpha=0.05,
        running_intersection=True,
        parallel=True,
    ),
    "SqKelly 1sided": lambda x: sqKelly(
        x,
        N=N,
        D=20,
        beta=1,
        breaks=1000,
        alpha=0.05,
        running_intersection=True,
        parallel=True,
    ),
}

colors = ["royalblue", "tomato"]
linestyles = ["-", "--"]
assert len(colors) == len(cs_dict) and len(linestyles) == len(colors)
colors_dict = {list(cs_dict.keys())[i]: colors[i] for i in range(len(cs_dict))}
linestyles_dict = {list(cs_dict.keys())[i]: linestyles[i] for i in range(len(cs_dict))}

margins = [0.04]
nuisances = [0]

data_dict = get_data_dict(N, margins, nuisances)
x = data_dict[list(data_dict.keys())[0]]
t = np.arange(1, N + 1)

plt.figure(figsize=(6, 3))
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
        linestyle=linestyles_dict[cs_name],
        alpha=0.6,
    )
    if not cs_name.endswith("1sided"):
        plt.plot(
            t,
            u,
            color=colors_dict[cs_name],
            linestyle=linestyles_dict[cs_name],
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

plt.ylim(0.2, 0.8)
plt.axhline(
    0.5 + margins[0], color="grey", linestyle="-", label=r"true mean $\mu^\star$"
)
plt.legend(loc="lower right")
plt.xlabel("Ballots sampled")
plt.ylabel("Confidence sequence")
plt.savefig("example_tallies.pdf")
