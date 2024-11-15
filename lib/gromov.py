import os
import torch
import ot


from .opt import *

import numpy as np
import numba as nb
import warnings
import time
from ot.backend import get_backend, NumpyBackend
from ot.lp import emd
#print('load package')


from ot.partial import gwgrad_partial, gwloss_partial

import warnings

# This function is adapted from PythonOT package
@nb.njit(cache=True)
def tensor_dot_param(C1, C2, Lambda=0, loss="square_loss"):
    if loss == "square_loss":

        def f1(r1):
            return r1**2 - 2 * Lambda

        def f2(r2):
            return r2**2

        def h1(r1):
            return r1

        def h2(r2):
            return 2 * r2

    # else:
    #     warnings.warn("loss function error")

    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)

    return fC1, fC2, hC1, hC2


# @nb.njit(cache=True)
def tensor_dot_func(fC1, fC2, hC1, hC2, Gamma):
    # Gamma=np.ascontiguousarray(Gamma)
    n, m = Gamma.shape
    # Gamma_1=
    # Gamma_2=)
    C1 = fC1.dot(Gamma.sum(1).reshape((-1, 1)))  # .dot(np.ones((1,m)))
    C2 = Gamma.sum(0).dot(fC2.T)
    tensor_dot = (C1 + C2) - hC1.dot(Gamma).dot(hC2.T)
    return tensor_dot

# This function is adapted from PythonOT package
# @nb.njit(cache=True)
def gwgrad_partial1(C1, C2, T, loss="square"):
    """Compute the GW gradient. Note: we can not use the trick in :ref:`[12] <references-gwgrad-partial>`
    as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient


    .. _references-gwgrad-partial:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    # T=np.ascontiguousarray(T)
    if loss == "square":
        cC1 = np.dot(C1**2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
        cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2**2)
        constC = cC1 + cC2
        A = -2 * np.dot(C1, T).dot(C2.T)
        tens = constC + A
    elif loss == "dot":
        constC = 0
        A = -2 * np.dot(C1, T).dot(C2.T)
        tens = constC + A
    return tens


# This function is adapted from PythonOT package
@nb.njit(cache=True)
def gwloss_partial1(C1, C2, T):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    GW loss
    """
    g = gwgrad_partial1(C1, C2, T) * 0.5
    return np.sum(g * T)

# This function is adapted from PythonOT package
def partial_gromov_wasserstein(
    C1,
    C2,
    p,
    q,
    m=None,
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=100000,
    numItermax_gw=1000,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    if G0 is None:
        G0 = np.outer(p, q) * m / (np.sum(p) * np.sum(q))
    
    n1,n2=p.shape[0],q.shape[0]
    #dim_G_extended = (n1 + nb_dummies, n2 + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)
    M_emd = np.zeros((n1+nb_dummies,n2+nb_dummies))

    cpt = 0
    err = 1

    if log:
        log = {"err": []}
    iter_num = 0
    while err > tol and cpt < numItermax_gw:
        iter_num += 1
        Gprev = np.copy(G0)

        M = gwgrad_partial(C1, C2, G0)
       
        M_emd[:n1, :n2] = M
        M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 2
        #M_emd = np.asarray(M_emd, dtype=np.float64)
        # M_emd[-1,-1]+=1

        Gc, logemd = emd(
            p_extended,
            q_extended,
            M_emd,
            numItermax=numItermax,
            numThreads=thres,
            log=True,
            **kwargs
        )

        # if logemd['warning'] is not None:
        #     raise ValueError("Error in the EMD resolution: try to increase the"
        #                      " number of dummy points")

        G0 = Gc[: len(p), : len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        deltaG = G0 - Gprev
        if line_search:
            a = 2 * gwloss_partial(C1, C2, deltaG)
            b = 2 * np.sum(M * deltaG)

            if a > 0:  # due to numerical precision
                if b > 0:
                    gamma = 0
                    cpt = numItermax_gw
                else:
                    gamma = min(1, np.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    gamma = 1
                else:
                    gamma = 0
                    cpt = numItermax_gw
        else:
            gamma = 1

        G0 = Gprev + gamma * deltaG
        cpt += 1

    if log:
        log["partial_gw_dist"] = gwloss_partial(C1, C2, G0)
        return G0[: len(p), : len(q)], log
    else:
        return G0[: len(p), : len(q)]  # ,iter_num


# This function is adapted from PythonOT package
def gromov_wasserstein(
    C1,
    C2,
    p,
    q,
    G0=None,
    thres=1,
    numItermax=100000,
    numItermax_gw=1000,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """
    np.testing.assert_almost_equal(
        p.sum(0), q.sum(0), err_msg="a and b vector must have the same sum", decimal=6
    )
    C1=np.array(C1,dtype=np.float64)
    C2=np.array(C2,dtype=np.float64)
    if G0 is None:
        G0 = np.outer(p, q) / np.sum(p)

    cpt = 0
    err = 1

    if log:
        log = {"err": []}
    iter_num = 0
    while err > tol and cpt < numItermax_gw:
        iter_num += 1
        Gprev = np.copy(G0)

        M = gwgrad_partial(C1, C2, G0)

        Gc, logemd = emd(
            p, q, M, numItermax=numItermax, numThreads=thres, log=True, **kwargs
        )
        G0 = Gc
        # if logemd['warning'] is not None:
        #     raise ValueError("Error in the EMD resolution: try to increase the"
        #                      " number of dummy points")

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        deltaG = G0 - Gprev
        if line_search:
            a = 2 * gwloss_partial(C1, C2, deltaG)
            b = 2 * np.sum(M * deltaG)

            if a > 0:  # due to numerical precision
                if b > 0:
                    gamma = 0
                    cpt = numItermax_gw
                else:
                    gamma = min(1, np.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    gamma = 1
                else:
                    gamma = 0
                    cpt = numItermax_gw
        else:
            gamma = 1

        G0 = Gprev + gamma * deltaG
        cpt += 1

    if log:
        log["gw_dist"] = gwloss_partial(C1, C2, G0)
        return G0[: len(p), : len(q)], log
    else:
        return G0[: len(p), : len(q)]  # ,iter_num


@nb.njit()
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


@nb.njit()
def tensor_dot_ori(M, Gamma):
    n, m = Gamma.shape
    gradient = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    gradient[i, j] += M[i, j, i1, j1] * Gamma[i1, j1]
    return gradient


@nb.njit(cache=True)
def construct_M(C1, C2):
    n, m = C1.shape[0], C2.shape[0]
    M = np.zeros((n, m, n, m))
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    M[i, j, i1, j1] = (C1[i, i1]-C2[j, j1])**2
    return M


# @nb.njit(cache=True)
def C12_func(Gamma, fC12):
    (fC1, fC2) = fC12
    n, m = Gamma.shape
    Gamma_1 = Gamma.dot(np.ones((m, 1)))
    Gamma_2 = Gamma.T.dot(np.ones((n, 1)))
    C_12 = fC1.dot(Gamma_1).dot(np.ones((1, m))) + np.ones((n, 1)).dot(Gamma_2.T).dot(
        fC2.T
    )
    return C_12


def partial_gromov_ver1(
    C1,
    C2,
    p,
    q,
    Lambda,
    G0=None,
    nb_dummies=1,
    thres=1,
    numItermax_gw=1000,
    numItermax=None,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    seed=0,
    truncate=True,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    # if m is None:
    #     m = np.min((np.sum(p), np.sum(q)))
    # elif m < 0:
    #     raise ValueError("Problem infeasible. Parameter m should be greater"
    #                      " than 0.")
    # elif m > np.min((np.sum(p), np.sum(q))):
    #     raise ValueError("Problem infeasible. Parameter m should lower or"
    #                      " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q)*np.min((np.sum(p),np.sum(q)))/(np.sum(p)*np.sum(q))
    #print('G0 sum is',np.sum(G0))

    cpt = 0
    err = 1

    if log:
        log_dict = {"err": [], "G0_mass": [], "Gprev_mass": []}

    fC1, fC2, hC1, hC2 = tensor_dot_param(C1, C2, Lambda, loss="square_loss")
    fC1, fC2, hC1, hC2 = (
        np.ascontiguousarray(fC1),
        np.ascontiguousarray(fC2),
        np.ascontiguousarray(hC1),
        np.ascontiguousarray(hC2),
    )
    C1, C2 = np.ascontiguousarray(C1), np.ascontiguousarray(C2)
    iter_num = 0
    n, m = C1.shape[0], C2.shape[0]
    if numItermax is None:
        numItermax = n * 100
    p_sum, q_sum = p.sum(), q.sum()
    G0_orig = np.zeros((n, m))

    mu_extended, nu_extended, M_extended = (
        np.zeros(n + 1),
        np.zeros(m + 1),
        np.full((n + 1, m + 1),-1e-15),
    )
    mu_extended[0:n], mu_extended[-1] = p, q_sum
    nu_extended[0:m], nu_extended[-1] = q, p_sum

    while err > tol and cpt < numItermax_gw:
        # iter_num+=1
        Gprev = G0.copy()

        Mt_circ_G = tensor_dot_func(fC1, fC2, hC1, hC2, Gprev)
        #        reg=2*Lambda*np.sum(Gprev)
        # gwgrad_partial1(C1, C2, Gprev)-reg
        # M_tilde_circ_gamma=M_circ_gamma-reg

        # opt solver:
        # Flamary's trick to fasten the computation: select only the subset of columns/lines

        #        G0,innerlog_=opt_lp(p,q,Grad,Lambda=0,log=log,numItermax=numItermax,**kwargs)

        #        eps=reg
        #        M_extended[:,-1],M_extended[-1,:]=reg,reg

        M_extended[0:n, 0:m] = Mt_circ_G  # -reg

        # M_extended[:idx_x.shape[0], :idx_y.shape[0]]= M_star[np.ix_(idx_x, idx_y)]
        gamma_extended, log_dict = emd_lp(
            mu_extended,
            nu_extended,
            M_extended,
            numItermax=numItermax,
            log=log,
            **kwargs
        )

        G0 = G0_orig.copy()
        G0[0:n, 0:m] = gamma_extended[:-1, :-1]
        # G0[np.ix_(idx_x, idx_y)] = gamma_extended[:-nb_dummies, :-nb_dummies]
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        # line search
        deltaG = G0 - Gprev

        # line search
        if line_search:

            # M_circ_deltaG=gwgrad_partial1(C1, C2, deltaG)
            # deltaG_sum=np.sum(deltaG)
            # a=np.sum(M_circ_deltaG*deltaG)-2*Lambda*deltaG_sum**2
            # b= 2 * (np.sum(M_circ_gamma * deltaG)-reg*deltaG_sum)
            Mt_circ_deltaG = tensor_dot_func(fC1, fC2, hC1, hC2, deltaG)
            a = np.sum(Mt_circ_deltaG * deltaG)
            b = 2 * (np.sum(Mt_circ_G * deltaG))
            if a > 0:  # due to numerical precision
                if b >= 0:
                    alpha = 0
                    cpt = numItermax_gw
                else:
                    alpha = min(1, np.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    alpha = 1
                else:
                    alpha = 0
                    cpt = numItermax_gw
        else:
            alpha = 1

        G0 = Gprev + alpha * deltaG
        cpt += 1
    if log:
        log_dict.update(innerlog_)
        return G0, log_dict  # ,iter_num
    else:
        return G0  # ,iter_num


def partial_gromov_ver2(
    C1,
    C2,
    p,
    q,
    Lambda=0.0,
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=None,
    numItermax_gw=1000,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    # if m is None:
    #     m = np.min((np.sum(p), np.sum(q)))
    # elif m < 0:
    #     raise ValueError("Problem infeasible. Parameter m should be greater"
    #                      " than 0.")
    # elif m > np.min((np.sum(p), np.sum(q))):
    #     raise ValueError("Problem infeasible. Parameter m should lower or"
    #                      " equal than min(|a|_1, |b|_1).")

    # if G0 is None:
    #     G0 = np.outer(p, q)

    n = p.shape[0]
    m = q.shape[0]
    if numItermax is None:
        numItermax = n * 100
    dim_G_extended = (n + nb_dummies, m + nb_dummies)
    q_extended = np.append(q, [(np.sum(p)) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q)) / nb_dummies] * nb_dummies)
    cpt = 0
    err = 1

    G0 = np.outer(p_extended, q_extended)
    if log:
        log_dict = {"err": []}

    #    fC1,fC2,hC1,hC2=tensor_dot_param(C1,C2,loss='square_loss')
    #    fC1,fC2,hC1,hC2=np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2)

    iter_num = 0
    Grad_extended = np.zeros(dim_G_extended)
    C1, C2 = np.ascontiguousarray(C1), np.ascontiguousarray(C2)
    while err > tol and cpt < numItermax_gw:
        iter_num += 1
        Gprev = np.copy(G0)

        #       Compute the tensor product
        M_circ_gamma = (
            gwgrad_partial1(C1, C2, Gprev[0:n, 0:m])
            - 2 * Lambda * Gprev[0:n, 0:m].sum()
        )
        # tensor_dot_func(fC1,fC2,hC1,hC2,Gprev[0:n,0:m])-2*Lambda*Gprev[0:n,0:m].sum()
        # gwgrad_partial(C1, C2, Gprev[0:n,0:m])-2*Lambda*Gprev[0:n,0:m].sum()

        # Compute the gradient
        # Grad=2*M_circ_gamma
        Grad_extended[0:n, 0:m] = M_circ_gamma  # Grad
        # Grad_extended = np.ascontiguousarray(Grad_extended)
        # np.ascontiguousarray(T)
        G0, innerlog_ = emd_lp(
            p_extended,
            q_extended,
            Grad_extended,
            log=log,
            numItermax=numItermax,
            numThreads=thres,
            **kwargs
        )  # Apply ot solver

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log_dict["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        deltaG = G0 - Gprev
        deltaG0 = deltaG[0:n, 0:m]
        # line search
        if line_search:
            M_circ_deltaG = gwgrad_partial1(C1, C2, deltaG0)
            deltaG_sum = np.sum(deltaG0)
            a = np.sum(M_circ_deltaG * deltaG0) - 2 * Lambda * deltaG_sum**2
            b = 2 * (np.sum(M_circ_gamma * deltaG0))

            if a > 0:  # due to the numerical unstable issue
                if b >= 0:
                    alpha = 0
                    cpt = numItermax_gw
                else:
                    alpha = min(1, np.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    alpha = 1
                else:
                    alpha = 0
                    cpt = numItermax_gw
        else:
            alpha = 1
        #        print('alpha is',alpha)
        G0 = Gprev + alpha * deltaG
        cpt += 1

    G0 = G0[0:n, 0:m]

    if log:
        log_dict.update(innerlog_)
        #        log_dict['partial_gw_dist'] = gwloss_partial(C1, C2, G0)
        return G0, log_dict  # ,iter_num
    else:
        return G0  # ,iter_num


def partial_gromov_old_v12(
    C1,
    C2,
    p,
    q,
    Lambda,
    G0=None,
    nb_dummies=1,
    thres=1,
    numItermax_gw=1000,
    numItermax=None,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    seed=0,
    truncate=False,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    # if m is None:
    #     m = np.min((np.sum(p), np.sum(q)))
    # elif m < 0:
    #     raise ValueError("Problem infeasible. Parameter m should be greater"
    #                      " than 0.")
    # elif m > np.min((np.sum(p), np.sum(q))):
    #     raise ValueError("Problem infeasible. Parameter m should lower or"
    #                      " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q)

    cpt = 0
    err = 1

    if log:
        log_dict = {"err": [], "G0_mass": [], "Gprev_mass": []}

    # fC1,fC2,hC1,hC2=tensor_dot_param(C1,C2,loss='square_loss')
    # fC1,fC2,hC1,hC2=np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2)
    C1, C2 = np.ascontiguousarray(C1), np.ascontiguousarray(C2)
    iter_num = 0
    n, m = C1.shape[0], C2.shape[0]
    if numItermax is None:
        numItermax = n * 100
    p_sum, q_sum = p.sum(), q.sum()
    G0_orig = np.zeros((n, m))

    mu_extended, nu_extended, M_extended = (
        np.zeros(n + 1),
        np.zeros(m + 1),
        np.zeros((n + 1, m + 1)),
    )
    mu_extended[0:n], mu_extended[-1] = p, q_sum
    nu_extended[0:m], nu_extended[-1] = q, p_sum
    eps = 1e-20
    while err > tol and cpt < numItermax_gw:
        # iter_num+=1
        Gprev = G0.copy()
        reg = 2 * Lambda * np.sum(Gprev)
        M_circ_gamma = gwgrad_partial1(C1, C2, Gprev) - reg

        # M_tilde_circ_gamma=M_circ_gamma-reg
        # tensor_dot_func(fC1,fC2,hC1,hC2,Gprev)-2*Lambda*Gprev.sum()

        # opt solver:
        # Flamary's trick to fasten the computation: select only the subset of columns/lines

        #        G0,innerlog_=opt_lp(p,q,Grad,Lambda=0,log=log,numItermax=numItermax,**kwargs)

        #        eps=reg

        M_extended[:, -1], M_extended[-1, :] = reg, reg

        M_extended[0:n, 0:m] = M_circ_gamma  # -reg
        G0 = G0_orig.copy()
        if truncate == False:
            # M_extended[:idx_x.shape[0], :idx_y.shape[0]]= M_star[np.ix_(idx_x, idx_y)]
            gamma_extended, log_dict = emd_lp(
                mu_extended,
                nu_extended,
                M_extended,
                numItermax=numItermax,
                log=log,
                **kwargs
            )

            G0[0:n, 0:m] = gamma_extended[:-1, :-1]
        else:
            idx_x, idx_y = (
                np.where(np.min(M_circ_gamma, axis=1) < eps)[0],
                np.where(np.min(M_circ_gamma, axis=0) < eps)[0],
            )
            mu_s, nu_s, M_s = p[idx_x], q[idx_y], M_extended[id_x, id_y]
            gamma_extended, log_dict = emd_lp(
                mu_s, nu_s, M_s, numItermax=numItermax, log=log, **kwargs
            )

            G0[np.ix_(idx_x[:-1], idx_y[:-1])] = gamma_extended[:-1, :-1]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        # line search
        deltaG = G0 - Gprev

        # line search
        if line_search:

            M_circ_deltaG = gwgrad_partial1(C1, C2, deltaG)
            deltaG_sum = np.sum(deltaG)
            a = np.sum(M_circ_deltaG * deltaG) - 2 * Lambda * deltaG_sum**2
            # b= 2 * (np.sum(M_circ_gamma * deltaG)-reg*deltaG_sum)
            b = 2 * (np.sum(M_circ_gamma * deltaG))
            if a > 0:  # due to numerical precision
                if b >= 0:
                    alpha = 0
                    cpt = numItermax_gw
                else:
                    alpha = min(1, np.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    alpha = 1
                else:
                    alpha = 0
                    cpt = numItermax_gw
        else:
            alpha = 1

        G0 = Gprev + alpha * deltaG
        cpt += 1
    if log:
        log_dict.update(innerlog_)
        return G0, log_dict  # ,iter_num
    else:
        return G0, None  # ,iter_num


def partial_gromov_old_v1(
    C1,
    C2,
    p,
    q,
    Lambda,
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax_gw=1000,
    numItermax=None,
    tol=1e-7,
    log=False,
    verbose=False,
    line_search=True,
    seed=0,
    truncate=True,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    # if m is None:
    #     m = np.min((np.sum(p), np.sum(q)))
    # elif m < 0:
    #     raise ValueError("Problem infeasible. Parameter m should be greater"
    #                      " than 0.")
    # elif m > np.min((np.sum(p), np.sum(q))):
    #     raise ValueError("Problem infeasible. Parameter m should lower or"
    #                      " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q) * min(p.sum(), q.sum()) / (p.sum() * q.sum())
    # dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    # q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    # p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1
    np.random.seed(seed)

    if log:
        log_dict = {"err": [], "G0_mass": [], "Gprev_mass": []}

    # fC1,fC2,hC1,hC2=tensor_dot_param(C1,C2,loss='square_loss')
    # fC1,fC2,hC1,hC2=np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2)
    C1, C2 = np.ascontiguousarray(C1), np.ascontiguousarray(C2)
    iter_num = 0
    n, m = C1.shape[0], C2.shape[0]
    if numItermax is None:
        numItermax = n * 100
    p_sum, q_sum = p.sum(), q.sum()
    while err > tol and cpt < numItermax_gw:
        iter_num += 1
        Gprev = G0.copy()

        M_circ_gamma = gwgrad_partial1(C1, C2, Gprev)
        reg = 2 * Lambda * np.sum(Gprev)

        # opt solver:
        # Flamary's trick to fasten the computation: select only the subset of columns/lines

        #        G0,innerlog_=opt_lp(p,q,Grad,Lambda=0,log=log,numItermax=numItermax,**kwargs)

        eps = reg

        M_star = M_circ_gamma
        # - 2*Lambda  # modified cost matrix

        # trick to fasten the computation: select only the subset of columns/lines
        # that can have marginals greater than 0 (that is to say M < 0)
        idx_x = np.where(np.min(M_star, axis=1) < eps)[0]
        idx_y = np.where(np.min(M_star, axis=0) < eps)[0]

        mu_extended = np.append(
            p[idx_x], [(p_sum - np.sum(p[idx_x]) + q_sum) / nb_dummies] * nb_dummies
        )
        nu_extended = np.append(
            q[idx_y], [(q_sum - np.sum(q[idx_y]) + p_sum) / nb_dummies] * nb_dummies
        )

        M_extended = np.full(
            (mu_extended.shape[0], nu_extended.shape[0]), reg, dtype=np.float64
        )
        M_extended[: idx_x.shape[0], : idx_y.shape[0]] = M_star[np.ix_(idx_x, idx_y)]
        gamma_extended, log_dict = emd_lp(
            mu_extended,
            nu_extended,
            M_extended,
            numItermax=numItermax,
            log=log,
            **kwargs
        )
        G0 = np.zeros((n, m))
        G0[np.ix_(idx_x, idx_y)] = gamma_extended[:-nb_dummies, :-nb_dummies]
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0)))

        # line search
        deltaG = G0 - Gprev

        # line search
        if line_search:

            #            time1=time.time()
            #            M_circ_deltaG=gwgrad_partial1(C1, C2, deltaG)-2*Lambda*np.sum(deltaG) #.sum()
            # tensor_dot_func(fC1,fC2,hC1,hC2,deltaG)-2*Lambda*deltaG.sum()
            # gwgrad_partial(C1, C2, deltaG)-2*Lambda*deltaG.sum()
            #            a = np.sum(M_circ_deltaG*deltaG)
            #            b = 2 * np.sum((M_circ_gamma-reg) * deltaG)
            #            time2=time.time()

            M_circ_deltaG = gwgrad_partial1(C1, C2, deltaG)
            deltaG_sum = np.sum(deltaG)
            a = np.sum(M_circ_deltaG * deltaG) - 2 * Lambda * deltaG_sum**2
            b = 2 * (np.sum(M_circ_gamma * deltaG) - reg * deltaG_sum)
            if a > 0:  # due to numerical precision
                if b >= 0:
                    alpha = 0
                    cpt = numItermax_gw
                else:
                    alpha = min(1, np.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    alpha = 1
                else:
                    alpha = 0
                    cpt = numItermax_gw
        else:
            alpha = 1

        G0 = Gprev + alpha * deltaG
        cpt += 1
    if log:
        log_dict.update(innerlog_)
        return G0, log_dict  # ,iter_num
    else:
        return G0  # ,iter_num


def GW_dist(C1, C2, gamma):
    M_gamma = gwgrad_partial1(C1, C2, gamma)
    dist = np.sum(M_gamma * gamma)
    return dist


def MPGW_dist(C1, C2, gamma):
    M_gamma = gwgrad_partial1(C1, C2, gamma)
    dist = np.sum(M_gamma * gamma)
    return dist


def PGW_dist_with_penalty(C1, C2, gamma, p1, p2, Lambda):
    M_gamma = gwgrad_partial1(C1, C2, gamma)
    dist = np.sum(M_gamma * gamma)
    penalty = Lambda * (p1.sum() ** 2 + p2.sum() ** 2 - 2 * gamma.sum() ** 2)
    return dist, penalty
