from rilacs.strategies import linear_gamma_dist
import pytest
import numpy as np
from rilacs.martingales import (
    apriori_Kelly_martingale,
    distKelly_martingale,
    sqKelly_martingale,
    dKelly_martingale,
)


def test_apriori_Kelly_martingale():
    N = 100000
    alpha = 0.05

    # Martingale should have power against the alternative of m = 0.5.
    # This will fail with some small probability.
    n_A = 51000
    n_B = N - n_A
    x = np.append(np.ones(n_A), np.zeros(n_B))
    np.random.shuffle(x)
    mart = apriori_Kelly_martingale(x, m=0.5, N=N, n_A=n_A, n_B=n_B)
    assert any(mart >= 1 / alpha)

    # Martingale should not reject after N samples.
    # This will fail with some small probability.
    n_A = int(N / 2)
    n_B = N - n_A
    x = np.append(np.ones(n_A), np.zeros(n_B))
    np.random.shuffle(x)
    mart = apriori_Kelly_martingale(x, m=0.5, N=N, n_A=n_A, n_B=n_B)
    assert mart[-1] < 1 / alpha


def test_distKelly_martingale():
    N = 10000
    alpha = 0.05

    x = np.random.beta(1, 1, N)

    one_sided_martingales = [
        lambda x, m: distKelly_martingale(x, m, N=N, dist=linear_gamma_dist, beta=1),
        lambda x, m: sqKelly_martingale(x, m, N=N, beta=1),
        lambda x, m: dKelly_martingale(x, m, N=N, beta=1),
    ]

    m = np.mean(x)
    for martingale in one_sided_martingales:
        assert martingale(x, m)[-1] < 1 / alpha

    m = np.mean(x)
    for martingale in one_sided_martingales:
        assert martingale(x, m + 0.01)[-1] >= 1 / alpha
