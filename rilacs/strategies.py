import numpy as np
import numpy.typing as npt
from scipy.optimize import root


def apriori_Kelly_bet_general(x_reported: npt.ArrayLike[float]) -> float:
    objective = lambda l: np.sum((x_reported - 0.5) / (1 + l * (x_reported - 0.5)))

    sol = root(objective, x0=0.1)
    assert sol["success"]
    bet = sol["x"]
    return bet


def apriori_Kelly_bet(n_A: int, n_B: int) -> float:
    return 2 * (n_A - n_B) / (n_A + n_B)


def linear_gamma_dist(y: npt.ArrayLike[float]):
    return np.maximum(1 - 4 * y, 0)


def square_gamma_dist(y: npt.ArrayLike[float]):
    return (y <= 1 / 3) * (1 / 3 - y) ** 2
