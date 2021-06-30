import numpy as np
from scipy.optimize import root
from confseq.betting import (
    betting_mart,
    diversified_betting_mart,
    cs_from_martingale,
    mu_t,
)


def apriori_Kelly_bet_general(x_reported):
    objective = lambda l: np.sum((x_reported - 0.5) / (1 + l * (x_reported - 0.5)))

    sol = root(objective, x0=0.1)
    assert sol["success"]
    bet = sol["x"]
    return bet


def apriori_Kelly_bet(n_A, n_B):
    return 2 * (n_A - n_B) / (n_A + n_B)


def apriori_Kelly_martingale(x, m, N, n_A, n_B):
    apriori_Kelly_lambda = lambda x, m: apriori_Kelly_bet(n_A, n_B)
    return betting_mart(
        x,
        m,
        lambdas_fn_negative=apriori_Kelly_lambda,
        N=N,
        convex_comb=True,
        theta=1,
        trunc_scale=1,
        m_trunc=True,
    )


def distKelly_martingale(x, m, N, dist, D=10, beta=1):
    m_t = mu_t(x, m, N)

    lambdas_fns_positive = [
        lambda x, m, i=i: (i + 1) / (m_t * (D + 1)) for i in range(D)
    ]

    discrete_unnormalized_dist = [dist((i + 1) / (D + 1)) for i in range(D)]

    lambdas_weights = np.array(discrete_unnormalized_dist) / sum(
        discrete_unnormalized_dist
    )

    mart = diversified_betting_mart(
        x,
        m,
        lambdas_fns_positive=lambdas_fns_positive,
        lambdas_weights=lambdas_weights,
        N=N,
        theta=beta,
        trunc_scale=1,
        m_trunc=True,
        convex_comb=True,
    )
    return mart


def linear_gamma_dist(y):
    return np.maximum(1 - 4 * y, 0)


def square_gamma_dist(y):
    return (y <= 1 / 3) * (1 / 3 - y) ** 2
