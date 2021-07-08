from rilacs.martingales import (
    apriori_Kelly_martingale,
    dKelly_martingale,
    sqKelly_martingale,
)
import numpy as np
import matplotlib.pyplot as plt
from rilacs.misc import get_data_dict, bravo
from rilacs.plotting import plot_stopping_times

N = 10000
alpha = 0.05
m_null = 1 / 2

margins = [0.05]
nuisances = [0, 3000, 6000, 8000]

data_dict = get_data_dict(N, margins, nuisances)

martingale_dict = {
    r"$\it{a\ priori}$ Kelly": lambda x: apriori_Kelly_martingale(
        x, m=m_null, N=N, n_A=np.count_nonzero(x == 1), n_B=np.count_nonzero(x == 0)
    ),
    "SqKelly": lambda x: sqKelly_martingale(x, m=m_null, N=N, D=20, beta=1),
    r"$\it{a\ priori}$ BRAVO": lambda x: bravo(x, mu_alt=np.mean(x), num_samples=None),
    "dKelly/KMart": lambda x: dKelly_martingale(x, m=m_null, N=N, D=20, beta=1),
}


plot_stopping_times(
    martingale_dict,
    data_dict,
    nsim=1000,
    alpha=alpha,
    num_proc=8,
    multiple_of=1,
    figsize=(3.5, 2.3),
    bbox_to_anchor=(0.505, -0.82),
    ncol=2,
)
