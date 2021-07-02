# RiLACS

This package implements the martingales and confidence sequences from ["RiLACS: Risk-limiting audits via confidence sequences"](https://ian.waudbysmith.com/audit.pdf) and contains code for reproducing all of the plots therein.


## Installation

```zsh
pip install git+ssh://git@github.com/WannabeSmith/RiLACS.git
```
_Note: we have plans to release this on pypi in the future_

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
git clone git@github.com:WannabeSmith/RiLACS.git

# Run unit tests via pytest
pytest RiLACS
```

## Produce the paper's plots

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

## Contributing
If you find a bug, please do [open an issue](https://github.com/wannabesmith/RiLACS/issues) and we'll try to fix it as soon as possible.

If you want to add a feature or have other suggestions, please feel free to reach out [via email](mailto:ianws@cmu.edu) or simply submit a [pull request](https://github.com/WannabeSmith/RiLACS/pulls)!

## Credits
The algorithms in this codebase were derived by [Ian Waudby-Smith](https://ian.waudbysmith.com), [Philip B. Stark](https://www.stat.berkeley.edu/~stark/), and [Aaditya Ramdas](http://stat.cmu.edu/~aramdas).
