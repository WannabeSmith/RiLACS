from rilacs.martingales import (
    apriori_Kelly_martingale,
    dKelly_martingale,
    sqKelly_martingale,
)
import numpy as np
import matplotlib.pyplot as plt
from rilacs.misc import get_bravo_workload, get_data_dict, get_workload_from_mart
from rilacs.plotting import plot_workload

N = 5000
alpha = 0.05
m_null = 1 / 2

margins = [0.05]
nuisances = [0, 1000, 2500, 4000]

data_dict = get_data_dict(N, margins, nuisances)

workload_dict = {
    r"$\it{a\ priori}$ Kelly": lambda x: get_workload_from_mart(
        x,
        mart_fn=lambda y: apriori_Kelly_martingale(
            y, m=m_null, N=N, n_A=np.count_nonzero(x == 1), n_B=np.count_nonzero(x == 0)
        ),
        alpha=alpha,
    ),
    "SqKelly": lambda x: get_workload_from_mart(
        x,
        mart_fn=lambda y: sqKelly_martingale(y, 1 / 2, N=N, D=20, beta=1),
        alpha=alpha,
    ),
    # "SqKelly": lambda x: np.where(
    #     sqKelly_martingale(x, m=m_null, N=N, D=20, beta=1)[0][0]
    # )[0][0],
    "BRAVO": lambda x: get_bravo_workload(x, mu_alt=np.mean(x), alpha=alpha),
    "dKelly/KMart": lambda x: get_workload_from_mart(
        x,
        mart_fn=lambda y: dKelly_martingale(y, m=m_null, N=N, D=20, beta=1),
        alpha=alpha,
    ),
}


plot_workload(
    workload_dict,
    data_dict,
    nsim=1000,
    alpha=alpha,
    num_proc=8,
    multiple_of=1,
    figsize=(3.5, 2.3),
    bbox_to_anchor=(0.505, -0.82),
    ncol=2,
)
