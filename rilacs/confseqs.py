from typing import Callable, Tuple
import numpy as np
from rilacs.strategies import square_gamma_dist, uniform_gamma_dist
from rilacs.martingales import apriori_Kelly_martingale, distKelly_martingale
from confseq.betting import cs_from_martingale


def apriori_Kelly(
    x: np.ndarray,
    N: int,
    n_A: int,
    n_B: int,
    breaks: int = 1000,
    alpha: float = 0.05,
    running_intersection: bool = True,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    return cs_from_martingale(
        x,
        mart_fn=lambda x, m: apriori_Kelly_martingale(x, m=m, N=N, n_A=n_A, n_B=n_B),
        breaks=breaks,
        alpha=alpha,
        N=N,
        running_intersection=running_intersection,
        parallel=parallel,
    )


def distKelly(
    x: np.ndarray,
    N: int,
    dist: Callable[[float], float],
    D: int = 10,
    beta: float = 1,
    breaks: int = 1000,
    alpha: float = 0.05,
    running_intersection: bool = True,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    return cs_from_martingale(
        x,
        mart_fn=lambda x, m: distKelly_martingale(
            x, m=m, N=N, dist=dist, D=D, beta=beta
        ),
        breaks=breaks,
        alpha=alpha,
        N=N,
        running_intersection=running_intersection,
        parallel=parallel,
    )


def sqKelly(
    x: np.ndarray,
    N: int,
    D: int = 10,
    beta: float = 1,
    breaks: int = 1000,
    alpha: float = 0.05,
    running_intersection: bool = True,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    return distKelly(
        x,
        N=N,
        dist=square_gamma_dist,
        D=D,
        beta=beta,
        breaks=breaks,
        alpha=alpha,
        running_intersection=running_intersection,
        parallel=parallel,
    )


def dKelly(
    x: np.ndarray,
    N: int,
    D: int = 10,
    beta: float = 1,
    breaks: int = 1000,
    alpha: float = 0.05,
    running_intersection: bool = True,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    return distKelly(
        x,
        N=N,
        dist=uniform_gamma_dist,
        D=D,
        beta=beta,
        breaks=breaks,
        alpha=alpha,
        running_intersection=running_intersection,
        parallel=parallel,
    )
