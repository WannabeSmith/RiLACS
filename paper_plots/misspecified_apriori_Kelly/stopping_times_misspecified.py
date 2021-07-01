from rilacs.martingales import apriori_Kelly_martingale
from rilacs.confseqs import apriori_Kelly
import numpy as np
from rilacs.misc import get_data_dict
from rilacs.plotting import plot_stopping_times


N = 10000
alpha = 0.05
m_null = 1 / 2

means = [0.53, 0.55]
nuisances = [0]

data_dict = get_data_dict(N, means, nuisances)

apriori_Kelly_martingale_fn = lambda x: apriori_Kelly_martingale(
    x, m=m_null, N=N, n_A=np.sum(x), n_B=N - np.sum(x)
)

diff1 = -0.01
mean_missp = means[0] + diff1
x_missp = np.append(np.ones(int(N * mean_missp)), np.zeros(N - int(N * mean_missp)))

apriori_Kelly_martingale_fn_missp = lambda x: apriori_Kelly_martingale(
    x, m=m_null, N=N, n_A=np.sum(x_missp), n_B=N - np.sum(x_missp)
)

diff2 = 0.03
mean_missp2 = means[0] + diff2
x_missp2 = np.append(np.ones(int(N * mean_missp2)), np.zeros(N - int(N * mean_missp2)))

apriori_Kelly_martingale_fn_missp2 = lambda x: apriori_Kelly_martingale(
    x, m=m_null, N=N, n_A=np.sum(x_missp2), n_B=N - np.sum(x_missp2)
)

diff3 = 0.08
mean_missp3 = means[0] + diff3
x_missp3 = np.append(np.ones(int(N * mean_missp3)), np.zeros(N - int(N * mean_missp3)))

apriori_Kelly_martingale_fn_missp3 = lambda x: apriori_Kelly_martingale(
    x, m=m_null, N=N, n_A=np.sum(x_missp3), n_B=N - np.sum(x_missp3)
)

martingale_dict = {
    r"$N_A' - N_A^\star = 0$": apriori_Kelly_martingale_fn,
    r"$N_A' - N_A^\star = " + str(diff1 * N) + "$": apriori_Kelly_martingale_fn_missp,
    r"$N_A' - N_A^\star = " + str(diff2 * N) + "$": apriori_Kelly_martingale_fn_missp2,
    r"$N_A' - N_A^\star = " + str(diff3 * N) + "$": apriori_Kelly_martingale_fn_missp3,
}

plot_stopping_times(
    martingale_dict,
    data_dict,
    nsim=1000,
    alpha=alpha,
    num_proc=8,
    multiple_of=1,
    bbox_to_anchor=(0.505, -0.7),
    filename_prefix="",
    ncol=2,
)
