from rilacs.strategies import linear_gamma_dist
import pytest
import numpy as np
from rilacs.martingales import (
    apriori_Kelly_martingale,
    distKelly_martingale,
    sqKelly_martingale,
    dKelly_martingale,
)
import itertools


@pytest.mark.parametrize("mu", [0.51, 0.52, 0.53, 0.9, 0.99])
def test_apriori_Kelly_martingale_power(mu):
    N = 100000
    alpha = 0.05

    # Martingale should have power against the alternative of m = 0.5.
    # This will fail with some small probability.
    n_A = int(N * mu)
    n_B = N - n_A
    x = np.append(np.ones(n_A), np.zeros(n_B))
    np.random.shuffle(x)
    mart = apriori_Kelly_martingale(x, m=0.5, N=N, n_A=n_A, n_B=n_B)
    assert any(mart >= 1 / alpha)


def test_apriori_Kelly_martingale_type1err():
    N = 100000
    alpha = 0.05
    # Martingale should not reject after N samples.
    # This will fail with some small probability.
    n_A = int(N / 2)
    n_B = N - n_A
    x = np.append(np.ones(n_A), np.zeros(n_B))
    np.random.shuffle(x)
    mart = apriori_Kelly_martingale(x, m=0.5, N=N, n_A=n_A, n_B=n_B)
    assert mart[-1] < 1 / alpha


@pytest.mark.parametrize(
    "data_fn, beta",
    itertools.product(
        [
            lambda: np.random.binomial(1, 0.5, 10000),
            lambda: np.random.binomial(1, 0.1, 10000),
            lambda: np.random.beta(1, 1, 10000),
            lambda: np.random.beta(10, 30, 10000),
        ],
        [0, 1 / 2, 1],
    ),
)
def test_distKelly_martingale(data_fn, beta):
    alpha = 0.05

    x = data_fn()
    print(x)
    N = len(x)

    martingales = [
        lambda x, m: distKelly_martingale(x, m, N=N, dist=linear_gamma_dist, beta=beta),
        lambda x, m: sqKelly_martingale(x, m, N=N, beta=beta),
        lambda x, m: dKelly_martingale(x, m, N=N, beta=beta),
    ]

    from confseq.betting import mu_t

    mu = np.mean(x)
    for martingale in martingales:
        # After roughly 5000 observations, should not reject under the null, but should
        # under the alternative.
        # These tests will fail with small probability.
        assert martingale(x, mu)[int(N / 2)] < 1 / alpha
        if beta == 1:
            assert martingale(x, mu - 0.05)[int(N / 2)] >= 1 / alpha
        if beta == 0:
            assert martingale(x, mu + 0.05)[int(N / 2)] >= 1 / alpha
        if beta == 1 / 2:
            assert martingale(x, mu - 0.05)[int(N / 2)] >= 1 / alpha
            assert martingale(x, mu + 0.05)[int(N / 2)] >= 1 / alpha
