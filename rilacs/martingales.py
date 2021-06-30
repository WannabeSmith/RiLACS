from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.optimize import root
from confseq.betting import (
    betting_mart,
    diversified_betting_mart,
    mu_t,
)


def apriori_Kelly_bet_general(x_reported: npt.ArrayLike[float]) -> float:
    objective = lambda l: np.sum((x_reported - 0.5) / (1 + l * (x_reported - 0.5)))

    sol = root(objective, x0=0.1)
    assert sol["success"]
    bet = sol["x"]
    return bet


def apriori_Kelly_bet(n_A: int, n_B: int) -> float:
    return 2 * (n_A - n_B) / (n_A + n_B)


def apriori_Kelly_martingale(
    x: npt.ArrayLike[float], m: float, N: int, n_A: int, n_B: int
) -> npt.ArrayLike[float]:
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


def distKelly_martingale(
    x: npt.ArrayLike[float],
    m: float,
    N: int,
    dist: Callable[[npt.ArrayLike[float]], npt.ArrayLike[float]],
    D: int = 10,
    beta: float = 1,
):
    lambdas_fns_positive = [
        lambda x, m, i=i: (i + 1) / (mu_t(x, m, N) * (D + 1)) for i in range(D)
    ]
    lambdas_fns_negative = [
        lambda x, m, i=i: (i + 1) / ((1 - mu_t(x, m, N)) * (D + 1)) for i in range(D)
    ]

    discrete_unnormalized_dist = [dist((i + 1) / (D + 1)) for i in range(D)]

    lambdas_weights = np.array(discrete_unnormalized_dist) / sum(
        discrete_unnormalized_dist
    )

    mart = diversified_betting_mart(
        x,
        m,
        lambdas_fns_positive=lambdas_fns_positive,
        lambdas_fns_negative=lambdas_fns_negative,
        lambdas_weights=lambdas_weights,
        N=N,
        theta=beta,
        trunc_scale=1,
        m_trunc=True,
        convex_comb=True,
    )
    return mart


def linear_gamma_dist(y: npt.ArrayLike[float]):
    return np.maximum(1 - 4 * y, 0)


def square_gamma_dist(y: npt.ArrayLike[float]):
    return (y <= 1 / 3) * (1 / 3 - y) ** 2
