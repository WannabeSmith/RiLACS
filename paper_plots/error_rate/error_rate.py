import numpy as np
import matplotlib.pyplot as plt
from logging import info
import warnings
from scipy.stats import binomtest
from rilacs.martingales import distKelly_martingale
from rilacs.strategies import square_gamma_dist

N = 10000
mean = 0.54
alpha = 0.05
x = np.append(np.ones(int(mean * N)), np.zeros(N - int(mean * N)))
nsim = 1000
ever_reject = np.ones(nsim)

for i in range(nsim):
    if (i + 1) % 10 == 0:
        print("Starting simulation " + str(i + 1))
    np.random.shuffle(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mart = distKelly_martingale(
            x, m=0.54, N=N, dist=square_gamma_dist, beta=1, D=50
        )
    ever_reject[i] = 1 if any(mart >= 1 / alpha) else 0

binom = binomtest(int(np.sum(ever_reject)), n=nsim, alternative="two-sided")
print(binom.proportion_ci(confidence_level=0.95, method="exact"))
