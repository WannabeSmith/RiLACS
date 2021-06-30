import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../../confseq_wor/cswor")
sys.path.append("../../")
sys.path.append("../../../SHANGRLA/Code")
sys.path.append("../../../Betting/Code")
from vacsine import *

N = 10000
alpha = 0.05
m_null = 1 / 2

margins = np.arange(0.501, 0.95, step=0.001)
nuisances = [0]

data_dict = get_data_dict(N, margins, nuisances)
keys = list(data_dict.keys())

lambda_opts = np.zeros(len(margins))

for i in range(len(keys)):
    data = data_dict[keys[i]]
    lambda_opts[i] = get_apriori_Kelly_bet(data)
  
  
plt.style.use("seaborn-whitegrid")
plt.rcParams["font.family"] = "serif"

plt.plot(margins, lambda_opts, color='tab:green')
