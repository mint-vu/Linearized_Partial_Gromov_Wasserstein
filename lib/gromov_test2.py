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

import warnings

@nb.njit()
def gwgrad_partial(C1, C2, T):
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
    cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
    constC = cC1 + cC2
    A = -np.dot(C1, T).dot(C2.T)
    tens = constC + A
    return tens * 2



@nb.njit()
def gwloss_partial(C1, C2, T):
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
    g = gwgrad_partial(C1, C2, T) * 0.5
    return np.sum(g * T)




def partial_gromov_wasserstein(C1, C2, p, q, m=None, nb_dummies=1, G0=None,
                               thres=1,numItermax_gw=1000, numItermax=1000, tol=1e-7,
                               log=False, verbose=False, **kwargs):
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
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q)

    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < numItermax_gw):

        Gprev = np.copy(G0)

        M = gwgrad_partial(C1, C2, G0)
        M_emd = np.zeros(dim_G_extended)
        M_emd[:len(p), :len(q)] = M
        M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e2
        M_emd = np.asarray(M_emd, dtype=np.float64)

        Gc, logemd = emd(p_extended, q_extended, M_emd,numItermax=numItermax,numThreads=thres, log=True, **kwargs)

        # if logemd['warning'] is not None:
        #     raise ValueError("Error in the EMD resolution: try to increase the"
        #                      " number of dummy points")

        G0 = Gc[:len(p), :len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}|{:12s}'.format(
                        'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                 gwloss_partial(C1, C2, G0)))

        deltaG = G0 - Gprev
        a = 2*gwloss_partial(C1, C2, deltaG)
        b = 2 * np.sum(M * deltaG)
        if b > 0:  # due to numerical precision
            gamma = 0
            cpt = numItermax_gw
        elif a > 0:
            gamma = min(1, np.divide(-b, 2.0 * a))
        else:
            if (a + b) < 0:
                gamma = 1
            else:
                gamma = 0
                cpt = numItermax_gw

        G0 = Gprev + gamma * deltaG
        cpt += 1

    if log:
        log['partial_gw_dist'] = gwloss_partial(C1, C2, G0)
        return G0[:len(p), :len(q)], log
    else:
        return G0[:len(p), :len(q)]

    


@nb.njit(['float32[:,:](float32[:,:],float32[:,:])','float64[:,:](float64[:,:],float64[:,:])'],fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n=X.shape[0]
    m=Y.shape[0]
    Dtype=X.dtype
    M=np.zeros((n,m),Dtype)
    for i in range(n):
        for j in range(m):
            M[i,j]=np.sum(cost_function(X[i,:],Y[j,:]))
    return M



@nb.njit(cache=True)
def tensor_dot_ori(M,Gamma):
    n,m=Gamma.shape
    gradient=np.zeros((n,m),dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    gradient[i,j]+=M[i,j,i1,j1]*Gamma[i1,j1]
    return gradient

@nb.njit()
def construct_M(C1,C2):
    n,m=C1.shape[0],C2.shape[0]
    M=np.zeros((n,m,n,m))
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    M[i,j,i1,j1]=L(C1[i,i1],C2[j,j1])
    return M

@nb.njit()
def C12_func(Gamma,fC12):
    (fC1,fC2)=fC12
    n,m=Gamma.shape
    Gamma_1=Gamma.dot(np.ones((m,1)))
    Gamma_2=Gamma.T.dot(np.ones((n,1)))
    C_12=fC1.dot(Gamma_1).dot(np.ones((1,m)))+np.ones((n,1)).dot(Gamma_2.T).dot(fC2.T)
    return C_12
    
#@nb.njit(cache=True)
def tensor_dot_param(C1,C2,loss='square_loss'):
    if loss=='square_loss':
        def f1(r1):
            return r1**2
        def f2(r2):
            return r2**2
        def h1(r1):
            return r1
        def h2(r2):
            return 2*r2
    # else:
    #     warnings.warn("loss function error")

    fC1=f1(C1)
    fC2=f2(C2)
    hC1=h1(C1)
    hC2=h2(C2)
    
    return fC1,fC2,hC1,hC2

#@nb.njit()
def tensor_dot_func(fC1,fC2,hC1,hC2,Gamma):
    fC1,fC2,hC1,hC2,Gamma=np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2),np.ascontiguousarray(Gamma)
    n,m=Gamma.shape
#    Gamma_1=Gamma.dot(np.ones((m,1)))
#    Gamma_2=Gamma.T.dot((np.ones((n,1))))
    Gamma_1=Gamma.sum(1).reshape((n,1)) #dot(np.ones((m,1)))
    Gamma_2=Gamma.sum(0).reshape((m,1)) #T.dot(np.ones((n,1)))
    C1=fC1.dot(Gamma_1).dot(np.ones((1,m)))
    C2=np.ones((n,1)).dot(Gamma_2.T).dot(fC2.T)
    tensor_dot=C1+C2-hC1.dot(Gamma).dot(hC2.T) #-2*Lambda*np.ones((n,m))*np.sum(Gamma)
    return tensor_dot



def partial_gromov_v2(C1, C2, p, q, Lambda=0.0, nb_dummies=1, G0=None,
                               thres=1, numItermax=1000,numItermax_gw=1000, tol=1e-7,
                               log=False, verbose=False, **kwargs):
   
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
    n=p.shape[0]
    m=q.shape[0]
    dim_G_extended = (n + nb_dummies, m + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) ) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) ) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1
    
    G0=np.outer(p_extended,q_extended)
    
    if log:
        log_dict = {'err': []}

        
    fC1,fC2,hC1,hC2=tensor_dot_param(C1,C2,loss='square_loss')
    
    

    while (err > tol and cpt < numItermax_gw):

        Gprev = np.copy(G0)
        
#       Compute the tensor product
        M_circ_gamma=tensor_dot_func(fC1,fC2,hC1,hC2,Gprev[0:n,0:m])-2*Lambda*Gprev[0:n,0:m].sum()
       # M2=gwgrad_partial(C1, C2, G0)

        # Compute the gradient 
        Grad=2*M_circ_gamma
        
        Grad_extended=np.zeros(dim_G_extended)
        Grad_extended[0:n,0:m]=Grad
        G0,innerlog_=emd_lp(p_extended,q_extended,Grad_extended,log=log,numItermax=numItermax,numThreads=thres,**kwargs) #Apply ot solver 
        
        
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log_dict['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}|{:12s}'.format(
                        'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                 gwloss_partial(C1, C2, G0)))
            
        
            
        
        
        # line search 
        deltaG=G0-Gprev
        M_circ_deltaG=tensor_dot_func(fC1,fC2,hC1,hC2,deltaG[0:n,0:m])-2*Lambda*deltaG[0:n,0:m].sum()
        a = np.sum(M_circ_deltaG*deltaG[0:n,0:m])
        b = 2 * np.sum(M_circ_gamma * deltaG[0:n,0:m])
        
        if abs(a)<tol*1e-3: # due to the numerical unstable issue
            a=0.0
            
        if a > 0:  # 
            alpha=np.divide(-b, 2.0 * a)
            alpha=np.clip(alpha,0,1)
            if alpha==0:
                cpt= numItermax_gw

        else:
            if (a + b) < 0:
                alpha = 1.0
            else:

                alpha = 0.0
                cpt = numItermax_gw
#        print('alpha is',alpha)
        G0 = Gprev + alpha * deltaG
        cpt += 1
        
    
    G0=G0[0:n,0:m]

    if log:
        log_dict.update(innerlog_)
#        log_dict['partial_gw_dist'] = gwloss_partial(C1, C2, G0)
        return G0, log_dict
    else:
        return G0, None

    



def partial_gromov_v1(C1, C2, p, q, Lambda, nb_dummies=1, G0=None,
                               thres=1, numItermax_gw=1000,numItermax=10000, tol=1e-7,
                               log=False, verbose=False, **kwargs):
   
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

    # dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    # q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    # p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1
    
    if log:
        log_dict = {'err': [],'G0_mass':[],'Gprev_mass':[]}
        
    fC1,fC2,hC1,hC2=tensor_dot_param(C1,C2,loss='square_loss')
    
    mass_diff=0.0

    while (err > tol and cpt < numItermax_gw):

        Gprev = np.copy(G0)

        M_circ_gamma=tensor_dot_func(fC1,fC2,hC1,hC2,Gprev)-2*Lambda*Gprev.sum()
        #gwgrad_partial(C1, C2, Gprev)-2*Lambda*Gprev.sum()
        #
        
 #       print('M_tilde_gamma_diff is',M_tilde_gamma-M_tilde_gamma2)
        
        # Compute the gradient 
        Grad=2*M_circ_gamma
    
        G0,innerlog_=opt_lp(p,q,Grad,Lambda=0.0,log=log,numItermax=numItermax,numThreads=thres,**kwargs) #Apply opt solver 
        a,b=0,0
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log_dict['err'].append(err)
                log_dict['G0_mass'].append(G0.sum())
                log_dict['Gprev_mass'].append(Gprev.sum())  
                log_dict['a']=a
                log_dict['b']=b
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}|{:12s}'.format(
                        'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                 gwloss_partial(C1, C2, G0)))

            
        
        # line search 
        deltaG = G0 - Gprev
        
        
        M_circ_deltaG=tensor_dot_func(fC1,fC2,hC1,hC2,deltaG)-2*Lambda*deltaG.sum()
        #gwgrad_partial(C1, C2, deltaG)-2*Lambda*deltaG.sum()
        #
        a = np.sum(M_circ_deltaG*deltaG)
        b = 2 * np.sum(M_circ_gamma * deltaG)
        
        
        # line search 
        if abs(a)<tol*1e-3: # due to the numerical unstable issue
            a=0
            
        if a > 0:  # 
            alpha=np.divide(-b, 2.0 * a)
            alpha=np.clip(alpha,0,1)
        else:
            if (a + b) < 0:
                alpha = 1.0
            else:
                alpha = 0
        if alpha==0:
            cpt = numItermax_gw
            
                
#        print('alpha is',alpha)
#         mass_diff=np.sum(np.abs(G0-Gprev))
    
#         if mass_diff>0.3:
#             alpha=1.0
#         if alpha==0.0:
#             alpha=0
#             cpt = numItermax_gw
        
        G0 = Gprev + alpha * deltaG
        cpt += 1
    if log:
        log_dict.update(innerlog_)
        return G0, log_dict
    else:
        return G0, None
    