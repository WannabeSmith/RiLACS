from typing import Callable
import numpy as np
from confseq.betting import (
    betting_mart,
    diversified_betting_mart,
    mu_t,
)
from rilacs.strategies import (
    apriori_Kelly_bet,
    square_gamma_dist,
    uniform_gamma_dist,
    get_conv_weights_from_dist,
)


def apriori_Kelly_martingale(
    x: np.ndarray, m: float, N: int, n_A: int, n_B: int
) -> np.ndarray:
    apriori_Kelly_lambda = lambda x, m: apriori_Kelly_bet(n_A, n_B)
    return betting_mart(
        x,
        m,
        lambdas_fn_positive=apriori_Kelly_lambda,
        N=N,
        convex_comb=True,
        theta=1,
        trunc_scale=1,
        m_trunc=True,
    )


def distKelly_martingale(
    x: np.ndarray,
    m: float,
    N: int,
    dist: Callable[[float], float],
    D: int = 10,
    beta: float = 1,
) -> np.ndarray:
    lambdas_fns_positive = [
        lambda x, m, i=i: (i + 1) / (mu_t(x, m, N) * (D + 1)) for i in range(D)
    ]
    lambdas_fns_negative = [
        lambda x, m, i=i: (i + 1) / ((1 - mu_t(x, m, N)) * (D + 1)) for i in range(D)
    ]

    lambdas_weights = get_conv_weights_from_dist(dist=dist, D=D)

    with np.errstate(divide="ignore"):
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


def sqKelly_martingale(
    x: np.ndarray,
    m: float,
    N: int,
    D: int = 10,
    beta: float = 1,
) -> np.ndarray:
    return distKelly_martingale(x=x, m=m, N=N, dist=square_gamma_dist, D=D, beta=beta)


def dKelly_martingale(
    x: np.ndarray,
    m: float,
    N: int,
    D: int = 10,
    beta: float = 1,
) -> np.ndarray:
    return distKelly_martingale(x=x, m=m, N=N, dist=uniform_gamma_dist, D=D, beta=beta)
