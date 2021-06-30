import numpy as np
import itertools

def bravo(x, mu_alt, num_samples=None):
    num_samples = len(x) if num_samples is None else num_samples

    x_contribution = [
        2 * mu_alt if x_i == 1 else 2 * (1 - mu_alt) if x_i == 0 else 1 for x_i in x
    ]

    x_contrib_sample = np.random.choice(x_contribution, size=num_samples, replace=True)

    return np.cumprod(x_contrib_sample)


def get_data_dict(N, margins, nuisances):
    election_outcomes_dicts = []

    for margin, nuisance in itertools.product(margins, nuisances):
        election_outcomes_dicts = election_outcomes_dicts + [
            {"margin": margin, "nuisances": nuisance}
        ]

    data_dict = {}
    for election_outcomes_dict in election_outcomes_dicts:
        halves = election_outcomes_dict["nuisances"]
        ones = int((N - halves) * election_outcomes_dict["margin"])
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
