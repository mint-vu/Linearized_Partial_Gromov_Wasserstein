# Linearized Partial Gromov Wasserstein

## Installation

Example code for linear partial Gromov Wasserstein solver. The code reproduces the all the numerical examples in the paper.

## Required packages

We suggest to install [Pytorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [PythonOT](https://pythonot.github.io/), [scipy](https://scipy.org/),
[numba](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html), [sk-learning](https://scikit-learn.org/stable/).

Can create necessary Conda environment with 
```bash
conda env create -f environment.yml
```

## Outline of repository

`lib/` contains code of partial GW solvers and GW-based methods for all experiments. See `lib/README.md` for references.

## Numerical experiments