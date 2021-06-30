import matplotlib.pyplot as plt
import time
import numpy as np
from logging import info
from rilacs.misc import stopping_times


def plot_stopping_times(
    martingale_dict,
    data_dict,
    nsim=100,
    alpha=0.05,
    num_proc=1,
    multiple_of=1,
    bbox_to_anchor=None,
    ncol=None,
    filename_prefix="stopping_times_",
):
    for data_name in data_dict:
        start_time = time.time()

        plt.style.use("seaborn-white")
        plt.style.use("seaborn-colorblind")
        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.75))

        for spine in ax.spines.values():
            spine.set_edgecolor("lightgrey")

        data = data_dict[data_name]
        stopping_times_dict = stopping_times(
            martingale_dict=martingale_dict,
            data=data,
            nsim=nsim,
            alpha=alpha,
            num_proc=num_proc,
            multiple_of=multiple_of,
        )

        # get counts of each of 0, 0.5, and 1. To use barplot in matplotlib
        u, inv = np.unique(data, return_inverse=True)
        counts = np.bincount(inv)
        for mart_name in martingale_dict:
            stopping_times = stopping_times_dict[mart_name]
            avg_stopping_time_str = str(
                np.around(np.median(stopping_times), decimals=1)
            )

            ax.hist(
                stopping_times,
                alpha=0.7,
                label=mart_name + "\n(Median=" + avg_stopping_time_str + ")",
                histtype="step",
                density=True,
                linewidth=1.5,
            )

        ax.set_title(data_name)
        ax.set_xlim(0, len(data))
        ax.set_xlabel("Stopping time")
        ax.yaxis.set_ticklabels([])

        plt.tight_layout()

        if bbox_to_anchor is not None:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=bbox_to_anchor,
                ncol=ncol,
            )
        else:
            ax.legend(loc="best")
        # plt.legend(loc="best")
        plt.savefig(
            filename_prefix
            + data_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("^\star", "")
            .replace("$", "")
            + ".pdf",
            bbox_inches="tight",
        )
        end_time = time.time()
        info("Finished " + data_name + " in " + str(end_time - start_time) + " seconds")
