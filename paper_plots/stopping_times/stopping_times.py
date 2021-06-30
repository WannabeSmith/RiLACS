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

means =  [0.55]
nuisances = [0, 3000, 6000, 8000]

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

TriKelly_N = lambda x: distKelly(x, m=m_null, N=N, dist=linear_gamma_dist)

SqKelly_N = lambda x: distKelly(x, m=m_null, N=N, dist=square_gamma_dist)

apriori_BRAVO = lambda x: bravo(x, mu_alt=np.mean(x), num_samples=None)

martingale_dict = {
    r"$\it{a\ priori}$ Kelly": apriori_Kelly_N,
    "SqKelly": SqKelly_N,
    r"$\it{a\ priori}$ BRAVO": apriori_BRAVO,
    "dKelly/KMart": mckmart_N,
    #"TriKelly": TriKelly_N,
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
    ncol=2,
)
