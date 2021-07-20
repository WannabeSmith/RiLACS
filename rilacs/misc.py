import numpy as np
import itertools
import math
import multiprocess


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


def get_stopping_times(martingale_dict, data, nsim=100, alpha=0.05, num_proc=1):
    """
    Get stopping times for a collection of martingales

    Parameters
    ----------
    martingale_dict, dict of {string : function}
        dictionary of various martingales where the key is a
        string indicating the name of the martingale, and
        the function is a bivariate function which takes
        an array-like of [0, 1]-bounded real numbers (the
        observations) and a [0, 1]-bounded real number (the
        candidate mean), and outputs an array-like of
        nonnegative numbers (the martingale)

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

    mart_names = list(martingale_dict.keys())
    stopping_times_dict = {}

    def get_stopping_times(i):
        np.random.seed()
        np.random.shuffle(data)

        result_array = np.array([])
        for mart_name in mart_names:
            mart_closure = martingale_dict[mart_name]
            mart_value = mart_closure(data)
            mart_value[-1] = math.inf

            stopping_time = np.where(mart_value > 1 / alpha)[0][0] + 1
            result_array = np.append(result_array, stopping_time)
        return result_array

    with multiprocess.Pool(processes=num_proc) as pool:
        results = np.array(pool.map(get_stopping_times, range(nsim)))

    for j in range(len(mart_names)):
        mart_name = mart_names[j]
        stopping_times_dict[mart_name] = results[:, j]

    return stopping_times_dict
