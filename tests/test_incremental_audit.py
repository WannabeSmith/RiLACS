import numpy as np
from rilacs.confseqs import sqKelly, apriori_Kelly
from rilacs.incremental_audit import (
    Betting_Audit,
    DistKelly_Bettor,
    Kelly_Bettor,
    lower_cs_from_audit,
)


def test_sqKelly():
    N = 10000
    alpha = 0.05
    breaks = 500
    x = np.random.binomial(1, 0.5, N)

    sqKelly_audit = Betting_Audit(
        N=N, bettor=DistKelly_Bettor(D=10), breaks=breaks, alpha=alpha
    )

    l_incremental = lower_cs_from_audit(x, audit=sqKelly_audit)

    l_confseqs, u = sqKelly(
        x,
        N=N,
        D=10,
        beta=1,
        breaks=breaks,
        alpha=alpha,
        running_intersection=False,
        parallel=True,
    )

    assert all(l_incremental == l_confseqs)


def test_Kelly():
    n_A = 3000
    n_B = 2980
    N = 10000
    x = np.hstack((np.ones(n_A), np.ones(n_B), np.repeat(1 / 2, N - n_A - n_B)))
    np.random.shuffle(x)

    kelly_audit = Betting_Audit(N=N, bettor=Kelly_Bettor(n_A = n_A, n_B = n_B), breaks=500, alpha=0.1)

    l_incremental = lower_cs_from_audit(x, audit=kelly_audit)
    l_confseqs, u = apriori_Kelly(
        x, N=N, n_A=n_A, n_B=n_B, breaks=500, alpha=0.1, running_intersection=False
    )

    assert all(l_incremental == l_confseqs)
    