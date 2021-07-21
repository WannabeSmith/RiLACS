import numpy as np
import itertools
import math
import multiprocess
from collections import defaultdict


def bravo(x, mu_alt, num_samples=None):
    num_ones = np.count_nonzero(x == 1)
    num_zeros = np.count_nonzero(x == 0)
    num_halves = np.count_nonzero(x == 1 / 2)

    num_samples = len(x) if num_samples is None else num_samples

    x_contribution = np.ones(num_samples)

    n = len(x)

    for i in range(num_samples):
        random_sample = np.random.choice(
            (1, 0, 1 / 2), p=[num_ones / n, num_zeros / n, num_halves / n]
        )

        if random_sample == 1:
            x_contribution[i] = 2 * mu_alt
        elif random_sample == 0:
            x_contribution[i] = 2 * (1 - mu_alt)
        elif random_sample == 1 / 2:
            # do nothing to martingale, but remove that observation from
            # the population.
            num_halves -= 1
            n -= 1

    martingale = np.cumprod(x_contribution)

    return martingale


def get_bravo_workload(x, mu_alt, alpha, num_samples=None):
    num_samples = len(x) if num_samples is None else num_samples

    unique_ballots = set()
    ballot_numbers = list(range(len(x)))
    sample_num = 1
    martingale = 1
    while martingale < 1 / alpha and len(unique_ballots) < num_samples:
        random_ballot_number = np.random.choice(ballot_numbers)

        unique_ballots.add(random_ballot_number)

        ballot = x[random_ballot_number]
        martingale *= (
            2 * mu_alt if ballot == 1 else 2 * (1 - mu_alt) if ballot == 0 else 1
        )
        sample_num += 1

    workload = len(unique_ballots)

    return workload


def get_workload_from_mart(x, mart_fn, alpha):
    where_cross = np.where(mart_fn(x) >= 1 / alpha)[0]
    if len(where_cross) == 0:
        workload = len(x)
    else:
        workload = where_cross[0]

    return workload


def get_data_dict(N, margins, nuisances):
    election_outcomes_dicts = []

    for margin, nuisance in itertools.product(margins, nuisances):
        election_outcomes_dicts = election_outcomes_dicts + [
            {"margin": margin, "nuisances": nuisance}
        ]

    data_dict = {}
    for election_outcomes_dict in election_outcomes_dicts:
        halves = election_outcomes_dict["nuisances"]
        ones = int((N - halves) * (0.5 + election_outcomes_dict["margin"]))
        zeros = int(N - halves - ones)
        data = np.hstack(
            (
                np.repeat(0, zeros),
                np.repeat(1, ones),
                np.repeat(0.5, halves),
            )
        )

        name = (
            r"$(N_A^\star, N_U^\star, N_B^\star) = ("
            + str(ones)
            + ", "
            + str(halves)
            + ", "
            + str(zeros)
            + ")$"
        )
        data_dict[name] = np.random.permutation(data)

    return data_dict


def get_workloads(workload_dict, data, nsim=100, alpha=0.05, num_proc=1):
    """
    Get workloads for a collection of audits

    Parameters
    ----------
    workload_dict, dict of {string : function}
        dictionary of various functions where the key is a
        string indicating the name of the audit, and
        the function is a univariate function which takes
        an array-like of [0, 1]-bounded real numbers (the
        observations) and outputs the workload for the audit.

    data, array-like of [0, 1]-bounded reals
        The population of ballots

    nsim, positive integer
        number of simulations to perform

    alpha, (0, 1)-bounded real
        Significance level

    num_proc, positive integer
        Number of CPU processes to spawn. This should be less than or
        equal to the number of CPU cores on your machine.
    """
    N = np.sum(len(data))

    # Get things ready for the plot
    t = np.ones(N).cumsum()

    audit_names = list(workload_dict.keys())
    stopping_times_dict = {}

    def get_workload(i):
        np.random.seed()
        np.random.shuffle(data)

        result_array = [None] * len(audit_names)
        for i in range(len(audit_names)):
            audit_name = audit_names[i]
            workload_closure = workload_dict[audit_name]
            workload_value = workload_closure(data)
            workload_value = np.minimum(workload_value, N)

            result_array[i] = workload_value
        return result_array

    with multiprocess.Pool(processes=num_proc) as pool:
        results = np.array(pool.map(get_workload, range(nsim)))

    for j in range(len(audit_names)):
        audit_name = audit_names[j]
        stopping_times_dict[audit_name] = results[:, j]

    return stopping_times_dict
