import numpy as np
from rilacs.strategies import apriori_Kelly_bet, apriori_Kelly_bet_general
from confseq.misc import expand_grid
import pytest


@pytest.mark.parametrize(
    "N, p", expand_grid([10, 1000, 5821, 10000], [0.1, 0.5, 0.710293847, 0.9])
)
def test_apriori_Kelly_bet(N, p):
    # For binary x, computing the optimal bets via optimization or through
    # a closed-form expression should be identical. The only caveat being,
    # there must be at least one observation in each category, otherwise
    # the optimum is not unique.

    x = np.append(np.ones(int(N * p)), np.zeros(N - int(N * p)))
    np.random.shuffle(x)
    bets1 = apriori_Kelly_bet_general(x_reported=x)
    bets2 = apriori_Kelly_bet(
        n_A=np.count_nonzero(x == 1), n_B=np.count_nonzero(x == 0)
    )
    print(bets1 - bets2)
    assert abs(bets1 - bets2) < 10e-12

    assert apriori_Kelly_bet(n_A=5000, n_B=5000) == 0
