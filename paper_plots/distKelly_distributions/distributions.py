import numpy as np
import matplotlib.pyplot as plt
from rilacs.strategies import linear_gamma_dist, square_gamma_dist

d = np.arange(0, 1 + 0.01, step=0.01)

theta_linear = linear_gamma_dist(d) / np.sum(linear_gamma_dist(d))
theta_square = square_gamma_dist(d) / np.sum(square_gamma_dist(d))
theta_constant = np.repeat(1 / len(d), len(d))

plt.rcParams["figure.figsize"] = (5, 2)
plt.style.use("seaborn-white")
plt.style.use("seaborn-colorblind")
for spine in plt.gca().spines.values():
    spine.set_edgecolor("lightgrey")

plt.rcParams["font.family"] = "serif"
plt.plot(d, theta_linear, label="linear", linestyle="-", color="tab:pink")
plt.plot(d, theta_square, label="square", linestyle="--", color="mediumseagreen")
plt.plot(d, theta_constant, label="constant (KMart)", linestyle=":", color="royalblue")
plt.xlabel(r"$d \in [D]$")
plt.ylabel(r"$\theta_d$")
plt.legend(loc="best")

locs, labs = plt.yticks()
plt.yticks(locs[1:], labels=labs[1:])

plt.savefig("distKelly_distributions.pdf", bbox_inches="tight")
