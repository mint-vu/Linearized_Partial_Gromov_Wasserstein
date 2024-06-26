#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import os
import ot

from ot.utils import dist, list_to_array
from ot.backend import get_backend, NumpyBackend

import numba as nb

# from typing import Tuple #,List
from numba.typed import List
import matplotlib.pyplot as plt

from ot.lp.emd_wrap import emd_c, check_result, emd_1d_sorted

epsilon = 1e-10
import warnings


def check_number_threads(numThreads):
    """Checks whether or not the requested number of threads has a valid value.

    Parameters
    ----------
    numThreads : int or str
        The requested number of threads, should either be a strictly positive integer or "max" or None

    Returns
    -------
    numThreads : int
        Corrected number of threads
    """
    if (numThreads is None) or (
        isinstance(numThreads, str) and numThreads.lower() == "max"
    ):
        return -1
    if (not isinstance(numThreads, int)) or numThreads < 1:
        raise ValueError(
            'numThreads should either be "max" or a strictly positive integer'
        )
    return numThreads


# def emd(a, b, M, numItermax=100000, log=False, numThreads=1,**kwargs):
#     # ensure float64
#     a = np.asarray(a, dtype=np.float64)
#     b = np.asarray(b, dtype=np.float64)
#     M = np.asarray(M, dtype=np.float64, order='C')

#     # if empty array given then use uniform distributions
#     if len(a) == 0:
#         a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
#     if len(b) == 0:
#         b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

#     assert (a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]), \
#         "Dimension mismatch, check dimensions of M with a and b"

#     # ensure that same mass
#     np.testing.assert_almost_equal(a.sum(0),
#                                    b.sum(0), err_msg='a and b vector must have the same sum')
#     b = b * a.sum() / b.sum()

#     asel = a != 0
#     bsel = b != 0

#     numThreads = check_number_threads(numThreads)

#     G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)
#     result_code_string = check_result(result_code)

#     return G


def emd_lp(
    a,
    b,
    M,
    numItermax=100000,
    log=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
    
):
    r"""Solves the Earth Movers distance problem and returns the OT matrix


    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights

    .. warning:: Note that the :math:`\mathbf{M}` matrix in numpy needs to be a C-order
        numpy.array in float64 format. It will be converted if not in this
        format

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    .. note:: This function will cast the computed transport plan to the data type
        of the provided input with the following priority: :math:`\mathbf{a}`,
        then :math:`\mathbf{b}`, then :math:`\mathbf{M}` if marginals are not provided.
        Casting to an integer tensor might result in a loss of precision.
        If this behaviour is unwanted, please make sure to provide a
        floating point input.

    .. note:: An error will be raised if the vectors :math:`\mathbf{a}` and :math:`\mathbf{b}` do not sum to the same value.

    Uses the algorithm proposed in :ref:`[1] <references-emd>`.

    Parameters
    ----------
    a : (ns,) array-like, float
        Source histogram (uniform weight if empty list)
    b : (nt,) array-like, float
        Target histogram (uniform weight if empty list)
    M : (ns,nt) array-like, float
        Loss matrix (c-order array in numpy with type float64)
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: bool, optional (default=False)
        If True, returns a dictionary containing the cost and dual variables.
        Otherwise returns only the optimal transportation matrix.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :py:func:`ot.lp.center_ot_dual`.
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.


    Returns
    -------
    gamma: array-like, shape (ns, nt)
        Optimal transportation matrix for the given
        parameters
    log: dict, optional
        If input log is true, a dictionary containing the
        cost and dual variables and exit status


    Examples
    --------

    Simple example with obvious solution. The function emd accepts lists and
    perform automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd(a, b, M)
    array([[0.5, 0. ],
           [0. , 0.5]])


    .. _references-emd:
    References
    ----------
    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011,
        December).  Displacement interpolation using Lagrangian mass transport.
        In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    # convert to numpy if list
    # a, b, M = list_to_array(a, b, M)

    a0, b0, M0 = a, b, M
    if len(a0) != 0:
        type_as = a0
    elif len(b0) != 0:
        type_as = b0
    else:
        type_as = M0
    nx = get_backend(M0, a0, b0)

    # convert to numpy
    M, a, b = nx.to_numpy(M, a, b)

    # ensure float64
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64, order="C")

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    assert (
        a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]
    ), "Dimension mismatch, check dimensions of M with a and b"

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(
            a.sum(0),
            b.sum(0),
            err_msg="a and b vector must have the same sum",
            decimal=6,
        )
    b = b * a.sum() / b.sum()

    # asel = a != 0
    # bsel = b != 0

    numThreads = check_number_threads(numThreads)

    G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

    #     if center_dual:
    #         u, v = center_ot_dual(u, v, a, b)

    #     if np.any(~asel) or np.any(~bsel):
    #         u, v = estimate_dual_null_weights(u, v, a, b, M)

    result_code_string = check_result(result_code)
    # if not nx.is_floating_point(type_as):
    #     warnings.warn(
    #         "Input histogram consists of integer. The transport plan will be "
    #         "casted accordingly, possibly resulting in a loss of precision. "
    #         "If this behaviour is unwanted, please make sure your input "
    #         "histogram consists of floating point elements.",
    #         stacklevel=2
    #     )
    if log:
        log_dict = {}
        log_dict["cost"] = cost
        log_dict["u"] = nx.from_numpy(u, type_as=type_as)
        log_dict["v"] = nx.from_numpy(v, type_as=type_as)
        log_dict["warning"] = result_code_string
        log_dict["result_code"] = result_code
        return nx.from_numpy(G, type_as=type_as), log_dict
    return nx.from_numpy(G, type_as=type_as), None


def opt_lp(mu, nu, M, Lambda, nb_dummies=1, log=False, **kwargs):
    """
    Solves the partial optimal transport problem
    and returns the OT plan by linear programming in PythonOT

    Parameters
    ----------
    mu : np.ndarray (dim_mu,) float64
        Unnormalized histogram of dimension `dia_mu`
    nu : np.ndarray (dim_nu,) float64
        Unnormalized histograms of dimension `dia_nu`
    M : np.ndarray (dim_mu, dim_nu) float64
        cost matrix
    reg : float
        Regularization term > 0
    numItermax : int64, optional
        Max number of iterations


    Returns
    -------
    gamma : (dim_mu, dim_nu) ndarray
        Optimal transportation matrix for the given parameters
    cost : float64

    """
    eps = 1e-20
    M = np.asarray(M, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)

    n, m = M.shape
    # nb_dummies=1
    M_star = M - 2 * Lambda  # modified cost matrix

    # trick to fasten the computation: select only the subset of columns/lines
    # that can have marginals greater than 0 (that is to say M < 0)
    idx_x = np.where(np.min(M_star, axis=1) < eps)[0]
    idx_y = np.where(np.min(M_star, axis=0) < eps)[0]

    mu_extended = np.append(
        mu[idx_x],
        [(np.sum(mu) - np.sum(mu[idx_x]) + np.sum(nu)) / nb_dummies] * nb_dummies,
    )
    nu_extended = np.append(
        nu[idx_y],
        [(np.sum(nu) - np.sum(nu[idx_y]) + np.sum(mu)) / nb_dummies] * nb_dummies,
    )

    # print(mu_extended.sum())
    # print(nu_extended.sum())

    M_extended = np.zeros(
        (mu_extended.shape[0], nu_extended.shape[0]), dtype=np.float64
    )
    M_extended[: idx_x.shape[0], : idx_y.shape[0]] = M_star[np.ix_(idx_x, idx_y)]
    gamma_extended, log_dict = emd_lp(
        mu_extended, nu_extended, M_extended, log=log, **kwargs
    )
    Gamma = np.zeros((n, m))
    Gamma[np.ix_(idx_x, idx_y)] = gamma_extended[:-nb_dummies, :-nb_dummies]
    if log:
        log_dict["Lambda"] = Lambda
        log_dict["mass in opt"] = Gamma.sum()

    return Gamma, log_dict


def partial_wasserstein_lagrange(
    a, b, M, reg_m=None, nb_dummies=1, log=False, **kwargs
):
    r"""
    Solves the partial optimal transport problem for the quadratic cost
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, (\mathbf{M} - \lambda) \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &
             \leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}


    or equivalently (see Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X.
    (2018). An interpolating distance between optimal transport and Fisher–Rao
    metrics. Foundations of Computational Mathematics, 18(1), 1-44.)

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F  +
        \sqrt{\frac{\lambda}{2} (\|\gamma \mathbf{1} - \mathbf{a}\|_1 + \|\gamma^T \mathbf{1} - \mathbf{b}\|_1)}

        s.t. \ \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\lambda` is the lagrangian cost. Tuning its value allows attaining
      a given mass to be transported `m`

    The formulation of the problem has been proposed in
    :ref:`[28] <references-partial-wasserstein-lagrange>`


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension `dim_b`
    M : np.ndarray (dim_a, dim_b)
        cost matrix for the quadratic cost
    reg_m : float, optional
        Lagrangian cost
    nb_dummies : int, optional, default:1
        number of reservoir points to be added (to avoid numerical
        instabilities, increase its value if an error is raised)
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the emd solver


    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).

    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------

    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(partial_wasserstein_lagrange(a,b,M), 2)
    array([[0.1, 0. ],
           [0. , 0.1]])
    >>> np.round(partial_wasserstein_lagrange(a,b,M,reg_m=2), 2)
    array([[0.1, 0. ],
           [0. , 0. ]])


    .. _references-partial-wasserstein-lagrange:
    References
    ----------
    .. [28] Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
       optimal transport and Monge-Ampere obstacle problems. Annals of
       mathematics, 673-730.

    See Also
    --------
    ot.partial.partial_wasserstein : Partial Wasserstein with fixed mass
    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(a, b, M)

    if nx.sum(a) > 1 + 1e-15 or nx.sum(b) > 1 + 1e-15:  # 1e-15 for numerical errors
        raise ValueError("Problem infeasible. Check that a and b are in the " "simplex")

    if reg_m is None:
        reg_m = float(nx.max(M)) + 1
    if reg_m < -nx.max(M):
        return nx.zeros((len(a), len(b)), type_as=M)

    a0, b0, M0 = a, b, M
    # convert to humpy
    a, b, M = nx.to_numpy(a, b, M)

    eps = 1e-20
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    M_star = M - reg_m  # modified cost matrix

    # trick to fasten the computation: select only the subset of columns/lines
    # that can have marginals greater than 0 (that is to say M < 0)
    idx_x = np.where(np.min(M_star, axis=1) < eps)[0]
    idx_y = np.where(np.min(M_star, axis=0) < eps)[0]

    # extend a, b, M with "reservoir" or "dummy" points
    M_extended = np.zeros((len(idx_x) + nb_dummies, len(idx_y) + nb_dummies))
    M_extended[: len(idx_x), : len(idx_y)] = M_star[np.ix_(idx_x, idx_y)]

    a_extended = np.append(
        a[idx_x], [(np.sum(a) - np.sum(a[idx_x]) + np.sum(b)) / nb_dummies] * nb_dummies
    )
    b_extended = np.append(
        b[idx_y], [(np.sum(b) - np.sum(b[idx_y]) + np.sum(a)) / nb_dummies] * nb_dummies
    )

    gamma_extended, log_emd = emd(
        a_extended, b_extended, M_extended, log=True, **kwargs
    )
    gamma = np.zeros((len(a), len(b)))
    gamma[np.ix_(idx_x, idx_y)] = gamma_extended[:-nb_dummies, :-nb_dummies]

    # convert back to backend
    gamma = nx.from_numpy(gamma, type_as=M0)

    if log_emd["warning"] is not None:
        raise ValueError(
            "Error in the EMD resolution: try to increase the" " number of dummy points"
        )
        log_emd["cost"] = nx.sum(gamma * M0)
        log_emd["u"] = nx.from_numpy(log_emd["u"], type_as=a0)
        log_emd["v"] = nx.from_numpy(log_emd["v"], type_as=b0)

    return gamma, None


#


def opt_pr(mu, nu, M, mass, log=False, **kwargs):
    if mass > min(mu.sum(), nu.sum()):
        warnings.warn("opt_pr, mass constraint error")
    Lambda, A = 1.0, 1.0
    n, m = M.shape
    mu1, nu1 = np.zeros(n + 1), np.zeros(m + 1)
    mu1[0:n], nu1[0:m] = mu, nu
    mu1[-1], nu1[-1] = np.sum(nu) - mass, np.sum(mu) - mass
    M1 = np.zeros((n + 1, m + 1), dtype=np.float64)
    M1[0:n, 0:m] = M
    M1[:, m], M1[n, :] = Lambda, Lambda
    M1[n, m] = 2 * Lambda + A

    gamma1, log_dict = emd(mu1, nu1, M1, log=log, **kwargs)
    gamma = gamma1[0:n, 0:m]
    # print('log is', log)
    #     if log:
    #         log_dict['mu in opt'] = mu
    #         log_dict['nu in opt'] = nu
    #         log_dict['mass in opt']=gamma.sum()
    #         log_dict['mass in input']=mass
    # #         log['cost'] =cost
    # #         log['penualty'] = penualty
    # # #        log['warning'] = result_code_string
    #         log_dict['M'] = M
    # #        return nx.from_numpy(G, type_as=type_as), log

    # #    cost=np.sum(M*gamma)

    return gamma


@nb.njit(["float64[:,:](int64,int64,int64)"], fastmath=True)
def random_projections(d, n_projections, Type=0):
    """
    input:
    d: int
    n_projections: int

    output:
    projections: d*n torch tensor

    """
    np.random.seed(0)
    if Type == 0:
        Gaussian_vector = np.random.normal(
            0, 1, size=(d, n_projections)
        )  # .astype(np.float64)
        projections = Gaussian_vector / np.sqrt(np.sum(np.square(Gaussian_vector), 0))
        projections = projections.T

    elif Type == 1:
        r = np.int64(n_projections / d) + 1
        projections = np.zeros((d * r, d))  # ,dtype=np.float64)
        for i in range(r):
            H = np.random.randn(d, d)  # .astype(np.float64)
            Q, R = np.linalg.qr(H)
            projections[i * d : (i + 1) * d] = Q
        projections = projections[0:n_projections]
    return projections


@nb.njit()
def cost_function(x, y):
    """
    case 1:
        input:
            x: float number
            y: float number
        output:
            (x-y)**2: float number
    case 2:
        input:
            x: n*1 float np array
            y: n*1 float np array
        output:
            (x-y)**2 n*1 float np array, whose i-th entry is (x_i-y_i)**2
    """
    #    V=np.square(x-y) #**p
    V = np.power(x - y, 2)
    return V


# @nb.njit(['float32[:,:](float32[:])','float64[:,:](float64[:])'],fastmath=True)
# def transpose(X):
#     Dtype=X.dtype
#     n=X.shape[0]
#     XT=np.zeros((n,1),Dtype)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT


@nb.njit(
    ["float32[:,:](float32[:],float32[:])", "float64[:,:](float64[:],float64[:])"],
    fastmath=True,
)
def cost_matrix(X, Y):
    """
    input:
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.

    """
    X1 = np.expand_dims(X, 1)
    Y1 = np.expand_dims(Y, 0)
    M = cost_function(X1, Y1)
    return M


# @nb.njit(fastmath=True)
@nb.njit(fastmath=True)
def cost_matrix_d(X, Y, loss="square"):
    """
    input:
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.

    """
    #    n,d=X.shape
    #    m=Y.shape[0]
    #    M=np.zeros((n,m))
    # for i in range(d):
    #     C=cost_function(X[:,i:i+1],Y[:,i])
    #     M+=C
    X1 = np.expand_dims(X, 1)
    Y1 = np.expand_dims(Y, 0)
    if loss == "square":
        M = np.sum((X1 - Y1) ** 2, 2)
    if loss == "dot":
        M = np.sum(-2 * (X1 * Y1), 2)
    return M


# @nb.jit()
def lot_embedding(X0, X1, p0, p1, numItermax=100000, numThreads=10):
    # C = np.asarray(C, dtype=np.float64, order='C')
    X0 = np.ascontiguousarray(X0)
    X1 = np.ascontiguousarray(X1)
    p0 = np.ascontiguousarray(p0)
    p1 = np.ascontiguousarray(p1)
    C = cost_matrix_d(X0, X1)
    gamma = ot.lp.emd(
        p0, p1, C, numItermax=numItermax, numThreads=10
    )  # exact linear program
    # gamma, cost, u, v, result_code = emd_c(p0, p1, C, numItermax, numThreads)
    # result_code_string = check_result(result_code)
    N0, d = X0.shape
    X1_hat = gamma.dot(X1) / np.expand_dims(p0, 1)
    U1 = X1_hat - X0
    return U1


# #@nb.jit()
# def opt_lp(mu,nu,cost_M,Lambda,numItermax=100000,numThreads=10,log=False):
#     #mu=np.ascontiguousarray(mu)
#     #nu=np.ascontiguousarray(nu)
#     n,m=cost_M.shape
#     mass_mu=np.sum(mu)
#     mass_nu=np.sum(nu)
#     mu1=np.zeros(n+1)
#     nu1=np.zeros(m+1)
#     mu1[0:n]=mu
#     nu1[0:m]=nu
#     mu1[-1]=mass_nu
#     nu1[-1]=mass_mu
#     cost_M1=np.zeros((n+1,m+1))
#     cost_M1[0:n,0:m]=cost_M-2*Lambda
#     cost_M1 = np.asarray(cost_M1, dtype=np.float64, order='C')
#     gamma1=ot.lp.emd(mu1,nu1,cost_M1,numItermax=numItermax,numThreads=10,log=False)


# #    result_code_string = check_result(result_code)
#     gamma=gamma1[0:n,0:m]
#     cost=np.sum(cost_M*gamma)
#     penualty=Lambda*(np.sum(mu)+np.sum(nu)-2*np.sum(gamma))
#     if log:
#         log = {}
#         log['mu'] = mu
#         log['nu'] = nu
#         log['Lambda']=Lambda
#         log['cost'] =cost
#         log['penualty'] = penualty
# #        log['warning'] = result_code_string
#         log['cost_M'] = cost_M
# #        return nx.from_numpy(G, type_as=type_as), log
#     else:
#         log={"none"}

#     return gamma,log


# def opt_pr(X,Y,mu,nu,mass,numItermax=100000):
#     n,d=X.shape
#     m=Y.shape[0]
#     cost_M=cost_matrix_d(X,Y)
#     gamma=ot.partial.partial_wasserstein(mu,nu,cost_M,m=mass,nb_dummies=n+m,numItermax=numItermax)
#     cost=np.sum(cost_M*gamma)
#     return cost,gamma


def lopt_embedding(X0, X1, p0, p1, Lambda, numItermax=100000, numThreads=10):
    X0 = np.ascontiguousarray(X0)
    X1 = np.ascontiguousarray(X1)
    cost_M = cost_matrix_d(X0, X1)
    cost, gamma, penualty = opt_lp(
        p0, p1, cost_M, Lambda, numItermax=numItermax, numThreads=10
    )
    N0 = X0.shape[0]
    domain = np.sum(gamma, 1) > 1e-10
    p1_hat = np.sum(gamma, 1)  # martial of plan
    # compute barycentric projetion
    X1_hat = X0.copy()
    X1_hat[domain] = gamma.dot(X1)[domain] / np.expand_dims(p1_hat, 1)[domain]

    # separate barycentric projection into U_1 and p1_hat,M1
    U1 = X1_hat - X0
    M1 = np.sum(p1) - np.sum(p1_hat)
    p1_perp = p1 - np.sum(gamma, 0)
    return U1, p1_hat, M1, p1_perp


def lopt_embedding_pr(Xi, X0, p1, p0, Lambda):
    n, d = X0.shape
    cost, gamma = opt_pr(X0, Xi, p0, p1, Lambda)
    n = X0.shape[0]
    domain = np.sum(gamma, 1) > 1e-10
    p1_hat = np.sum(gamma, 1)
    Xi_hat = np.full((n, d), 0)
    Xi_hat[domain] = gamma.dot(Xi)[domain] / np.expand_dims(p1_hat, 1)[domain]
    U1 = Xi_hat - X0
    return U1, p1_hat


# def vector_norm(U1,p1_hat):
#     norm2=np.sum((U1.T)**2*p1_hat[domain])
#     return norm2

# def vector_penalty(p0_hat,p1_hat,M1,Lambda):
#     penalty=Lambda*(np.abs(p0_hat-p1_hat)+M1)
#     return penalty

# def vector_minus(U1,U2,p1_hat,P2_hat):
#     p1j_hat=np.minimum(p1_hat,P2_hat)
#     diff=np.full(n,0)
#     diff=U1-U2
#     return diff,P_ij


def lopt(U1, U2, p1_hat, p2_hat, Lambda, M1=0.0, M2=0.0):
    p12_hat = np.minimum(p1_hat, p2_hat)
    norm2 = np.sum(np.minimum(np.sum((U1 - U2) ** 2, 1), 2 * Lambda) * p12_hat)
    penualty = Lambda * (np.sum(np.abs(p1_hat - p2_hat)) + M1 + M2)
    return norm2, penualty


def lot_barycenter(
    Xi_list,
    pi_list,
    X0_init,
    p0,
    weights,
    numItermax=1000,
    numItermax_lp=100000,
    stopThr=1e-7,
    numThreads=10,
):
    K = weights.shape[0]
    N0, d = X0_init.shape
    Ui_list = np.zeros((K, N0, d))
    weights = np.ascontiguousarray(weights)
    weights = weights.reshape(K, 1, 1)
    X0 = X0_init
    for iter in range(numItermax):
        for i in range(K):
            Xi = Xi_list[i]
            pi = pi_list[i]
            Ui = lot_embedding(
                X0, Xi, p0, pi, numItermax=numItermax_lp, numThreads=numThreads
            )
            Ui_list[i] = Ui
        U_bar = np.sum(Ui_list * weights, 0)
        X_bar = X0 + U_bar
        error = np.sum((X_bar - X0) ** 2)
        X0 = X_bar
        if error <= stopThr:
            break

    return X0


def lopt_barycenter(
    Xi_list,
    pi_list,
    X0_init,
    p0,
    weights,
    Lambda,
    numItermax=100000,
    stopThr=1e-4,
    numThreads=10,
):
    K = weights.shape[0]
    N0, d = X0_init.shape
    Ui_list = np.zeros((K, N0, d))
    weights = np.ascontiguousarray(weights)
    weights = weights.reshape((K, 1, 1))
    X0 = X0_init
    for iter in range(numItermax):
        for i in range(K):
            Xi = Xi_list[i]
            pi = pi_list[i]
            Ui, pi_hat, Mi, nu_perp = lopt_embedding(
                X0, Xi, p0, pi, Lambda, numItermax=numItermax, numThreads=numThreads
            )
            Ui_list[i] = Ui
        U_bar = np.sum(Ui_list * weights, 0)
        X_bar = X0 + U_bar
        error = np.sum((X_bar - X0) ** 2) / np.linalg.norm(X0)
        X0 = X_bar
        if error <= stopThr:
            break
    return X0


# def vector_norm(Vi,p0_Ti,total_mass,Lambda):
#     domain=p0_Ti>0
#     Vi_take=Vi[domain]
#     if len(Vi.shape)==1:
#         norm=np.sum((Vi_take)**2*p0_Ti[domain])
#     else:
#         norm=np.sum(np.sum((Vi_take)**2,1)*p0_Ti[domain])
#     penualty=Lambda*(total_mass-np.sum(p0_Ti[domain]))
#     return norm, penualty
