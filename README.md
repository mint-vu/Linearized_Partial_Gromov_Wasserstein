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

`shape_retrieval_3d/` contains code for shape retrieval experiment on 3D dataset. See `shape_retrieval_3d/README.md` for details. 
`shape_retrieval_2d` contains code for shape retrieval experiment on 2D dataset. See `shape_retrieval_2d/README.md` for details
`mnist` contains code for classification experiment in MNIST dataset. See `mnist/READMe.md` for details


## citation
- In `lib\HK`, the code (LUOT/UOT solver) is imported from https://github.com/bernhard-schmitzer/UnbalancedLOT.
- In `lib\mpgw`, the code (mpgw sovler) is imported from  https://github.com/lchapel/partial-GW-for-PU
- In `ellipses\`, the data and code (for visulization) in `utils.py` is imported from https://github.com/Gorgotha/LGW
- In `shape_retrieval_2d`, we import and modify the code (for 2D data generation and SVM classification) in https://github.com/Gorgotha/LGW.
- In `shape_retrieval_mvp`, we import and modify the code (for SVM classification) in https://github.com/Gorgotha/LGW.


