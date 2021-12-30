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
    x: np.ndarray,
    N: int,
    n_A: int,
    n_B: int,
    m: float = 1 / 2,
) -> np.ndarray:
    """The "a priori Kelly" martingale

    This function implements the "a priori Kelly" martingale from RiLACS
    (https://arxiv.org/pdf/2107.11323.pdf). This martingale attempts to
    construct an effective lambda value (a tuning parameter) based on the
    reported outcomes n_A and n_B for candidates A and B, respectively.

    Parameters
    ----------
    x : np.ndarray
        A numpy array containing a sequence of ballots with 1 denoting votes for
        candidate A, 0 for candidate B, and 1/2 for nuisance/invalid ballots.
    N : int
        The total number of ballots cast.
    n_A : int
        The reported number of ballots for candidate A
    n_B : int
        The reported number of ballots for candidate B.
    m : float
        The "null hypothesis" mean of x. For example, to test the null that
        candidate A ballots <= candidate B ballots, set this to 1/2.

    Returns
    -------
    np.ndarray
        The resulting nonnegative martingale
    """
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
    N: int,
    dist: Callable[[float], float],
    m: float = 1 / 2,
    D: int = 10,
    beta: float = 1,
) -> np.ndarray:
    """The distKelly martingale

    This function implements the distKelly martingale from RiLACS
    (https://arxiv.org/pdf/2107.11323.pdf). This method uses `D`
    lambda values, constructs `D` individual martingales from them,
    and then takes a convex combination of them according to the distribution
    specified in `dist`.

    Parameters
    ----------
    x : np.ndarray
        A numpy array containing a sequence of ballots.
    N : int
        The total number of ballots cast.
    dist : Callable[[float], float]
        The function from [0, 1] -> R describing the weights theta_1, ..., theta_D
    m : float
        The "null hypothesis" mean of x. In other words, distKelly forms a
        nonnegative martingale starting at one when mean(x) = m.
    D : int
        The size of the grid to derive lambda values from.
    beta : float
        Convex weights describing the alternatives to have power against.
        For example, beta = 1 corresponds to a one-sided test of mean(x) <= m,
        while beta = 0 corresponds to a one-sided test of mean(x) >= m, and
        beta = 1/2 yields a two-sided test.


    Returns
    -------
    np.ndarray
        The resulting nonnegative martingale
    """
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
    """The SqKelly martingale

    Instantiation of distKelly_martingale with dist=square_gamma_dist.
    """
    return distKelly_martingale(x=x, m=m, N=N, dist=square_gamma_dist, D=D, beta=beta)


def dKelly_martingale(
    x: np.ndarray,
    m: float,
    N: int,
    D: int = 10,
    beta: float = 1,
) -> np.ndarray:
    """The dKelly martingale

    Instantiation of distKelly_martingale with dist=uniform_gamma_dist.
    """
    return distKelly_martingale(x=x, m=m, N=N, dist=uniform_gamma_dist, D=D, beta=beta)
