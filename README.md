# RiLACS

This package implements the martingales and confidence sequences from ["RiLACS: Risk-limiting audits via confidence sequences"](https://arxiv.org/pdf/2107.11323.pdf) and contains code for reproducing all of the plots therein.

## Installation

```zsh
pip install rilacs
```

## Compute a confidence sequence

```python
import numpy as np
from rilacs.confseqs import sqKelly

# Create some data
x = np.random.binomial(1, 0.5, 1000)

# Compute the confidence sequence
l, u = sqKelly(x, N = 1000)
```

## Run unit tests
```zsh
# Clone the repository 
git clone https://github.com/WannabeSmith/RiLACS.git

# Run unit tests via pytest
pytest RiLACS
```

## Reproduce the paper's plots

To generate all plots, we've created a shell script at `RiLACS/paper_plots/generate_plots.sh`:

```zsh
# Enter the plots directory
cd RiLACS/paper_plots

# Execute the script with your shell, e.g.:
zsh generate_plots.sh
```

Alternatively, you can produce each plot one-by-one:

```zsh
python RiLACS/paper_plots/distKelly_distributions/distributions.py
python RiLACS/paper_plots/relationship_to_testing/cs_testing_plots.py
# and so on.
```

## App to audit Canada's 2019 federal election

The `canada_audit` folder contains code to produce a web application which allows for interactive auditing of Canadian ridings in the 2019 federal election.

For more details, view the README in [`canada_audit`](./canada_audit)

![audit](https://ian.waudbysmith.com/audit_demo_quick.gif)

## Contributing

If you find a bug, please [open an issue](https://github.com/wannabesmith/RiLACS/issues) and we'll try to fix it as soon as possible.

## Citing

The algorithms in this codebase are based on "RiLACS: Risk-limiting audits via confidence sequences" by Ian Waudby-Smith, Philip B. Stark, and Aaditya Ramdas. The BibTeX entry is provided here.

@inproceedings{waudby2021rilacs,
  title={RiLACS: Risk Limiting Audits via Confidence Sequences},
  author={Waudby-Smith, Ian and Stark, Philip B and Ramdas, Aaditya},
  booktitle={International Joint Conference on Electronic Voting},
  pages={124--139},
  year={2021},
  organization={Springer}
}

