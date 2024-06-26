#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:19:52 2022

"""


import numpy as np
import torch
import os
import ot

import numba as nb
from typing import Tuple #,List
from numba.typed import List
import matplotlib.pyplot as plt



#from lib.gromove import *   

import numpy as np 
from ot.backend import get_backend, NumpyBackend
from ot.utils import list_to_array
from ot.gromov._utils import init_matrix, gwloss, gwggrad
from ot.optim import solve_1d_linesearch_quad,generic_conditional_gradient
from ot.gromov._gw import solve_gromov_linesearch
from lib.opt import opt_lp, emd_lp, opt_pr, cost_matrix_d
import numba as nb
import warnings

@nb.njit()
def GW_cost(C1,C2,Gamma):
    fC1,fC2,hC1,hC2=tensor_dot_param(C1,C2)
    tensor_dot=tensor_dot_func(fC1,fC2,hC1,hC2,Gamma,Lambda=0)
    loss=np.sum(tensor_dot*Gamma)
    return loss


@nb.njit()
def cost_function(x,y): 
    ''' 
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
    '''
#    V=np.square(x-y) #**p
    V=np.power(x-y,2)
    return V




@nb.njit(['float32[:,:](float32[:])','float64[:,:](float64[:])'],fastmath=True)
def transpose(X):
    Dtype=X.dtype
    n=X.shape[0]
    XT=np.zeros((n,1),Dtype)
    for i in range(n):
        XT[i]=X[i]
    return XT

@nb.njit(['float32[:,:](float32[:],float32[:])','float64[:,:](float64[:],float64[:])'],fastmath=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n=X.shape[0]
    XT=transpose(X)
    M=cost_function(XT,Y)
    return M


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

@nb.njit(['int64[:](int64,int64)'],fastmath=True)
def arange(start,end):
    n=end-start
    L=np.zeros(n,np.int64)
    for i in range(n):
        L[i]=i+start
    return L


#@nb.njit(['float32[:](float32[:],float32[:],float32[:])','float64[:](float64[:],float64[:],float64[:])'],fastmath=True)
def quantile_function(qs, mu_com, mu_values):
    n0=mu_com.shape[0]
    n=qs.shape[0]
    index_list=arange(0,n0)
    Dtype=mu_com.dtype
    quantile_value=np.zeros(n,dtype=Dtype)
    quantile_index=np.zeros(n,dtype=np.int64)
    for i in range(n):
        sup_index=index_list[mu_com>=qs[i]][0]
        quantile_value[i]=mu_values[sup_index]
        quantile_index[i]=sup_index
    return quantile_value, quantile_index


@nb.njit()
def L(r1,r2):
    return (r1-r2)**2

@nb.njit()
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
    
@nb.njit()
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

@nb.njit()
def tensor_dot_func(fC1,fC2,hC1,hC2,Gamma,Lambda=0):
    fC1,fC2,hC1,hC2,Gamma=np.ascontiguousarray(fC1),np.ascontiguousarray(fC2),np.ascontiguousarray(hC1),np.ascontiguousarray(hC2),np.ascontiguousarray(Gamma)
    n,m=Gamma.shape
    Gamma_1=Gamma.dot(np.ones((m,1)))
    Gamma_2=Gamma.T.dot(np.ones((n,1)))
    C_12=fC1.dot(Gamma_1).dot(np.ones((1,m)))+np.ones((n,1)).dot(Gamma_2.T).dot(fC2.T)
    tensor_dot=C_12-hC1.dot(Gamma).dot(hC2.T)-2*Lambda*np.ones((n,m))*np.sum(Gamma)
    return tensor_dot

@nb.njit()
def tensor_dot_hat_func(fC1,fC2,hC1,hC2,Gamma_hat,Lambda=0):
    tensor_dot=tensor_dot_func(fC1,fC2,hC1,hC2,Gamma_hat[0:-1,0:-1],Lambda=Lambda)
    n,m=tensor_dot.shape
    tensor_dot_hat=np.zeros((n+1,m+1))
    tensor_dot_hat[0:n,0:m]=tensor_dot
    return tensor_dot_hat 


    
def permutation_matrix(n):
    """
    Generate a random n x n permutation matrix.

    Parameters:
    n (int): The size of the matrix.

    Returns:
    numpy.ndarray: An n x n permutation matrix.
    """
    perm_matrix = np.zeros((n, n), dtype=int)
    perm_index = np.random.permutation(n)
    perm_matrix[np.arange(n), perm_index] = 1
    return perm_matrix



@nb.njit()
def pgw_grad(fC1,fC2,hC1,hC2,Gamma,Lambda):
    tensor_dot=tensor_dot_func(fC1,fC2,hC1,hC2,Gamma,Lambda)
    return 2*tensor_dot

@nb.njit()
def gw_variant_grad(fC1,fC2,hC1,hC2,Gamma_hat,Lambda):
    tensor_dot_hat=tensor_dot_hat_func(fC1,fC2,hC1,hC2,Gamma_hat,Lambda)
    return 2*tensor_dot_hat


@nb.njit()
def pgw_loss(fC1,fC2, hC1, hC2, Gamma,Lambda):
    tensor_prod=tensor_dot_func(fC1,fC2,hC1,hC2,Gamma,Lambda) 
    loss=np.sum(tensor_prod*Gamma)
    return loss

@nb.njit()
def gw_variant_loss(fC1,fC2, hC1, hC2, Gamma_hat,Lambda):
    tensor_prod=tensor_dot_func(fC1,fC2,hC1,hC2,Gamma_hat[0:-1,0:-1],Lambda)
    loss=np.sum(tensor_prod*Gamma_hat[0:-1,0:-1])
    return loss


@nb.njit()
def linesearch_quad(a, b):
    r"""
    For any convex or non-convex 1d quadratic function `f`, solve the following problem:

    .. math::

        \mathop{\arg \min}_{0 \leq x \leq 1} \quad f(x) = ax^{2} + bx + c

    Parameterse
    ----------
    a,b : float or tensors (1,)
        The coefficients of the quadratic function

    Returns
    -------
    x : float
        The optimal value which leads to the minimal cost
    """
  
    if a > 0:  # convex
        minimizer=-b/(2*a) 
        if minimizer>1:
            minimizer=1
        elif minimizer<0:
            minimizer=0

    else:  # non convex
        if a + b < 0.0:
            minimizer=1.0
        else:
            minimizer=0.
    return minimizer


@nb.njit()
def solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M, reg,
                            alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Solve the linesearch in the FW iterations for any inner loss that decomposes as in Proposition 1 in :ref:`[12] <references-solve-linesearch>`.

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns), optional
        Transformed Structure matrix in the source domain.
        For the 'square_loss' and 'kl_loss', we provide hC1 from ot.gromov.init_matrix
    C2 : array-like (nt,nt), optional
        Transformed Structure matrix in the source domain.
        For the 'square_loss' and 'kl_loss', we provide hC2 from ot.gromov.init_matrix
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        G, deltaG, C1, C2, M = list_to_array(G, deltaG, C1, C2, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = - reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG))
    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G





@nb.njit()
def pgw_linesearch(Gamma, deltaGamma, cost_G,fC1,fC2 ,hC1, hC2,Lambda):
    n,m=Gamma.shape
    tensor_dot_deltaGamma=tensor_dot_func(fC1,fC2,hC1,hC2,deltaGamma,Lambda=0.0)
    a=np.sum(tensor_dot_deltaGamma*deltaGamma)
    b=2*np.sum(tensor_dot_deltaGamma*Gamma)
    alpha = linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G



def generic_conditional_gradient(a, b, M, f, df, reg1, reg2, lp_solver, line_search, G0=None,
                                 numItermax=200, stopThr=1e-9,
                                 stopThr2=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the general regularized OT problem or its semi-relaxed version with
    conditional gradient or generalized conditional gradient depending on the
    provided linear program solver.

        The function solves the following optimization problem if set as a conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b} (optional constraint)

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`

        The function solves the following optimization problem if set a generalized conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot f(\gamma) + \mathrm{reg_2}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in :ref:`[5, 7] <references-gcg>`

    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples weights in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    f : function
        Regularization function taking a transportation matrix as argument
    df: function
        Gradient of the regularization function taking a transportation matrix as argument
    reg1 : float
        Regularization term >0
    reg2 : float,
        Entropic Regularization term >0. Ignored if set to None.
    lp_solver: function,
        linear program solver for direction finding of the (generalized) conditional gradient.
        If set to emd will solve the general regularized OT problem using cg.
        If set to lp_semi_relaxed_OT will solve the general regularized semi-relaxed OT problem using cg.
        If set to sinkhorn will solve the general regularized OT problem using generalized cg.
    line_search: function,
        Function to find the optimal step. Currently used instances are:
        line_search_armijo (generic solver). solve_gromov_linesearch for (F)GW problem.
        solve_semirelaxed_gromov_linesearch for sr(F)GW problem. gcg_linesearch for the Generalized cg.
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    .. _references_gcg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M)

    loop = 1

    if log:
        log_dict = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        # to not change G0 in place.
        G = nx.copy(G0)

    if reg2 is None:
        def cost(G):
            return nx.sum(M * G) + reg1 * f(G)
    else:
        def cost(G):
            return nx.sum(M * G) + reg1 * f(G) + reg2 * nx.sum(G * nx.log(G))
    cost_G = cost(G)
    if log:
        log_dict['loss'].append(cost_G)
        log_dict['init_df']=df(G0)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, 0, 0))

    while loop:

        it += 1
        old_cost_G = cost_G
        Mi = M + reg1 * df(G)
        
        print('it is', it)
        Gc, innerlog_ = lp_solver(a, b, Mi,log=log, **kwargs)
        
        if innerlog_ is not None:
            if 'mass in opt' in innerlog_:
                print('mass opt in logdict is', innerlog_['mass in opt'])
        print('Gc mass', Gc.sum())
        
        # line search
        deltaG = Gc - G
        mass_diff=np.abs(np.sum(deltaG))
        if mass_diff<0.01:
            alpha, fc, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)
        else:
            alpha=1
            cost_G = cost(G)
        #print('G before line search', G.sum())
        
        G = G + alpha * deltaG
        mass=np.sum(G)
        
        
        print('G mass after line search', G.sum())

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = abs_delta_cost_G / abs(cost_G) if cost_G != 0. else np.nan
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:

                loop = 0

        if log:
            log_dict['loss'].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, relative_delta_cost_G, abs_delta_cost_G))

    if log:
        log_dict.update(innerlog_)
    
        return G, log_dict
    else:
        return G, False
    
@nb.njit()
def pq_hat_construct(p,q):
    n,m=p.shape[0],q.shape[0]
    p_hat=np.zeros(n+1)
    p_hat[0:n]=p
    p_hat[-1]=np.sum(q)
    
    q_hat=np.zeros(m+1)
    q_hat[0:m]=q
    q_hat[-1]=np.sum(p)
    return p_hat,q_hat
    
    
def gromov_partial_wasserstein_v1(C1, C2, p, q,Lambda=0.0, loss_fun='square_loss', G0=None,
                       max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, log=False,**kwargs):
    
    p_hat,q_hat=pq_hat_construct(p,q)
    if G0 is None:
        G0 = np.outer(p_hat,q_hat)[0:-1,0:-1]-1e-10
    else: 
        G0 = nx.to_numpy(G0_)
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    fC1,fC2, hC1, hC2 = tensor_dot_param(C1, C2)
    # print('fC1',fC1)
    # print('fC2',fC2)
    # print('hC1',hC1)
    # print('hC2',hC2)
    tensor_dot=tensor_dot_func(fC1,fC2,hC1,hC2,G0,Lambda)
    #print('initial tensor_dot',tensor_dot)

    
    def f(G):
        return pgw_loss(fC1,fC2, hC1, hC2, G,Lambda)


    def df(G):
        return pgw_grad(fC1,fC2, hC1, hC2, G,Lambda)
    
    def lp_solver(p,q,M,log):
        return opt_lp(p,q,M,Lambda=0.0,log=log) # -2lambda has been added into df(G)


    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        return pgw_linesearch(G, deltaG, cost_G,fC1,fC2, hC1, hC2, Lambda)
    Gamma,log_dict=generic_conditional_gradient(a=p, b=q, M=0, f=f, df=df, reg1=1.0, reg2=None, lp_solver=lp_solver, line_search=line_search, G0=G0,numItermax=200, stopThr=1e-9,stopThr2=1e-9, verbose=False, log=log, **kwargs)
    return Gamma,log_dict


def gromov_partial_wasserstein_v2(C1, C2, p, q,Lambda=0.0, loss_fun='square_loss', G0=None,
                       max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, log=False,**kwargs):
    
    
    # construct p_hat, q_hat 
    p_hat,q_hat=pq_hat_construct(p,q)
    
    
    if G0 is None:
        G0 = np.outer(p_hat,q_hat)
#        G0 = np.outer(p,q) # p[:, None] * q[None, :]
    else: 
        G0 = nx.to_numpy(G0_)
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()
    fC1,fC2, hC1, hC2 = tensor_dot_param(C1, C2)
    


    
    def f(G_hat):
        return gw_variant_loss(fC1,fC2, hC1, hC2, G_hat,Lambda)


    def df(G_hat):
        return gw_variant_grad(fC1,fC2, hC1, hC2, G_hat,Lambda)
    
    def lp_solver(p,q,M,log):
        return emd(p,q,M,log=log)


    def line_search(cost, G_hat, deltaG_hat, Mi, cost_G):
        return pgw_linesearch(G_hat[0:-1,0:-1], deltaG_hat[0:-1,0:-1], cost_G,fC1,fC2, hC1, hC2, Lambda)
    Gamma_hat,log_dict=generic_conditional_gradient(a=p_hat, b=q_hat, M=0, f=f, df=df, reg1=1.0, reg2=None, lp_solver=lp_solver, line_search=line_search, G0=G0,numItermax=200, stopThr=1e-9,stopThr2=1e-9, verbose=False, log=log, **kwargs)
    Gamma=Gamma_hat[0:-1,0:-1]
    return Gamma,log_dict





def gromov_partial_wasserstein_prim(C1, C2, p, q,mass=0.0, loss_fun='square_loss', G0=None,
                       max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, log=False,**kwargs):
    


    if G0 is None:
        p_hat,q_hat=pq_hat_construct(p,q)
        G0 = np.outer(p_hat,q_hat)[0:-1,0:-1]-1e-10

#       G0 = np.outer(p,q) # p[:, None] * q[None, :]
    else: 
        G0 = nx.to_numpy(G0_)
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    fC1,fC2, hC1, hC2 = tensor_dot_param(C1, C2)

    def f(G):
        return pgw_loss(fC1,fC2, hC1, hC2, G,Lambda=0)


    def df(G):
        return pgw_grad(fC1,fC2, hC1, hC2, G,Lambda=0)
    
    def lp_solver(p,q,M,log):
        return opt_pr(p,q,M,mass=mass,log=log)


    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        return pgw_linesearch(G, deltaG, cost_G,fC1,fC2, hC1, hC2, Lambda=0.0)
    Gamma,log_dict=generic_conditional_gradient(a=p, b=q, M=0, f=f, df=df, reg1=1.0, reg2=None, lp_solver=lp_solver, line_search=line_search, G0=G0,numItermax=200, stopThr=1e-9,stopThr2=1e-9, verbose=False, log=log, **kwargs)
    return Gamma,log_dict
