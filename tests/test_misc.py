from rilacs.misc import get_data_dict, bravo
import numpy as np
import pytest


def test_get_data_dict():
    N = 10
    # Output should be the same (modulo random permutations of the data)
    # if using margins of 0 or 0.01 with N=10 since 0.01*10 will round
    # down to 0.
    dict1 = get_data_dict(N, [0, 0.01], nuisances=[0])
    dict2 = get_data_dict(N, [0], nuisances=[0])
    assert len(dict1) == len(dict2)
    for name in dict1:
        assert all(np.sort(dict1[name]) == np.sort(dict2[name]))

    N = 10
    margins = [0, 0.3]
    nuisances = [0, 5]
    data_dict = get_data_dict(N, margins, nuisances)
    data_dict
    arr = data_dict["$(N_A^\\star, N_U^\\star, N_B^\\star) = (5, 0, 5)$"]
    assert np.count_nonzero(arr == 1) == 5
    assert np.count_nonzero(arr == 1 / 2) == 0
    assert np.count_nonzero(arr == 0) == 5
    arr = data_dict["$(N_A^\\star, N_U^\\star, N_B^\\star) = (2, 5, 3)$"]
    assert np.count_nonzero(arr == 1) == 2
    assert np.count_nonzero(arr == 1 / 2) == 5
    assert np.count_nonzero(arr == 0) == 3
    arr = data_dict["$(N_A^\\star, N_U^\\star, N_B^\\star) = (8, 0, 2)$"]
    assert np.count_nonzero(arr == 1) == 8
    assert np.count_nonzero(arr == 1 / 2) == 0
    assert np.count_nonzero(arr == 0) == 2
    arr = data_dict["$(N_A^\\star, N_U^\\star, N_B^\\star) = (4, 5, 1)$"]
    assert np.count_nonzero(arr == 1) == 4
    assert np.count_nonzero(arr == 1 / 2) == 5
    assert np.count_nonzero(arr == 0) == 1

    # Check to make sure margins/nuisances are correct for large N
    N = 100000
    data_dict = get_data_dict(N, [0.01], nuisances=[0, 50000])
    arr = data_dict["$(N_A^\\star, N_U^\\star, N_B^\\star) = (51000, 0, 49000)$"]
    assert np.count_nonzero(arr == 1) == 51000
    assert np.count_nonzero(arr == 1 / 2) == 0
    assert np.count_nonzero(arr == 0) == 49000
    arr = data_dict["$(N_A^\\star, N_U^\\star, N_B^\\star) = (25500, 50000, 24500)$"]


def test_bravo():
    N = 100000
    mu_alt = 0.51
    alpha = 0.05
    x = np.append(np.ones(int(N * 0.5)), np.zeros(int(N - N * 0.5)))
    np.random.shuffle(x)
    mart = bravo(x, mu_alt=mu_alt)
    # Martingale should not exceed 1/alpha at time N
    # This test will fail with very small probability.
    assert mart[-1] < 1 / alpha

    x = np.append(np.ones(int(N * 0.51)), np.zeros(int(N - N * 0.51)))
    np.random.shuffle(x)
    mart = bravo(x, mu_alt=mu_alt)
    # Martingale should exceed 1/alpha at time N
    # This test will fail with very small probability.
    assert mart[-1] >= 1 / alpha
