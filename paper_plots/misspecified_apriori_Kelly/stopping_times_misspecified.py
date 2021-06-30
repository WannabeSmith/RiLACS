import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocess

sys.path.append("../../../confseq_wor/cswor")
sys.path.append("../../../SHANGRLA/Code")
sys.path.append("../../../Betting/Code")
sys.path.append("../../")

import cswor as cswor
import assertion_audit_utils as aau
import betting_eb as beb
from vacsine import *

N = 10000
alpha = 0.05
m_null = 1 / 2

means =  [0.53, 0.55]
nuisances = [0]

data_dict = get_data_dict(N, means, nuisances)
# data_dict["Beta(20.2, 20)"] = np.random.permutation(np.random.beta(20.2, 20, N))
# data_dict["Beta(20.08, 20)"] = np.random.permutation(np.random.beta(20.08, 20, N))

# Prepare the auditing algorithms
# KMart
tnnm = aau.TestNonnegMean()
kmart_N = lambda x: tnnm.kaplan_martingale(x, N=N, t=m_null)[1]

# MC-Kmart
mckmart_N = lambda x: mckmart(x, m=m_null, N=N, reps=100)

mckmart2_N = lambda x: mckmart2(x, m=m_null, N=N, reps=100)

# Hedged
hedged_N = lambda x: beb.betting_mart(
    x, m=m_null, WoR=True, N=N, theta=1, alpha=0.05 * 2, trunc_scale=0.8
)

# aSOS
aSOS_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda x, m: np.maximum(0, beb.lambda_aSOS(x, m)),
    trunc_scale=0.8,
)

# LBOW
LBOW_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda x, m: np.maximum(0, beb.lambda_LBOW(x, m)),
)

ONSm_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda x, m: np.maximum(0, beb.lambda_COLT18_ONS(x, m)),
    trunc_scale=0.8,
)


apriori_Kelly_N = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda y, m: get_apriori_Kelly_bet(x),
    trunc_scale=1,
)

diff1 = -0.01
mean_missp = means[0] + diff1
x_missp = np.append(np.ones(int(N*mean_missp)), np.zeros(N - int(N*mean_missp)))
                    
apriori_Kelly_N_missp = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda y, m: get_apriori_Kelly_bet(x_missp),
    trunc_scale=1,
)

diff2 = 0.03
mean_missp2 = means[0] + diff2
x_missp2 = np.append(np.ones(int(N*mean_missp2)), np.zeros(N - int(N*mean_missp2)))

apriori_Kelly_N_missp2 = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda y, m: get_apriori_Kelly_bet(x_missp2),
    trunc_scale=1,
)

diff3 = 0.08
mean_missp3 = means[0] + diff3
x_missp3 = np.append(np.ones(int(N*mean_missp3)), np.zeros(N - int(N*mean_missp3)))

apriori_Kelly_N_missp3 = lambda x: beb.betting_mart(
    x,
    m=m_null,
    WoR=True,
    N=N,
    theta=1,
    lambdas_fn_positive=lambda y, m: get_apriori_Kelly_bet(x_missp3),
    trunc_scale=1,
)


SqKelly_N = lambda x: distKelly(x, m=m_null, N=N, dist=square_gamma_dist)

martingale_dict = {
    r"$N_A' - N_A^\star = 0$": apriori_Kelly_N,
    r"$N_A' - N_A^\star = " + str(diff1*N) + "$": apriori_Kelly_N_missp,
    r"$N_A' - N_A^\star = " + str(diff2*N) + "$": apriori_Kelly_N_missp2,
    r"$N_A' - N_A^\star = " + str(diff3*N) + "$": apriori_Kelly_N_missp3,
    # r"$\it{a\ priori}$ BRAVO": apriori_BRAVO,
    # "dKelly/KMart": mckmart_N,
    #"TriKelly": TriKelly_N,
    # "SqKelly": SqKelly_N,
    #'KMart' : kmart_N,
    # 'PM' : hedged_N,
    # 'aKelly' : aSOS_N,
    # 'LBOW' : LBOW_N,
    # 'ONS-m' : ONSm_N
}

plot_stopping_times(
    martingale_dict,
    data_dict,
    nsim=1000,
    alpha=alpha,
    num_proc=8,
    multiple_of=1,
    bbox_to_anchor=(0.505, -0.7),
    filename_prefix="figures/",
    ncol=2
)
