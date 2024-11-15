

# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr 19 11:32:17 2022

# """

import numpy as np
import torch
import os
import ot

import numba as nb
#from typing import Tuple #,List
from numba.typed import List
import matplotlib.pyplot as plt

from ot.lp.emd_wrap import emd_c, check_result, emd_1d_sorted
epsilon=1e-10


@nb.njit(['float64[:,:](int64,int64,int64)'],fastmath=True)
def random_projections(d,n_projections,Type=0):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
    np.random.seed(0)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)) #.astype(np.float64)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((d*r,d)) #,dtype=np.float64)
        for i in range(r):
            H=np.random.randn(d,d) #.astype(np.float64)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections

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

# @nb.njit(['float32[:,:](float32[:])','float64[:,:](float64[:])'],fastmath=True)
# def transpose(X):
#     Dtype=X.dtype
#     n=X.shape[0]
#     XT=np.zeros((n,1),Dtype)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT

@nb.njit(['float32[:,:](float32[:],float32[:])','float64[:,:](float64[:],float64[:])'],fastmath=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    X1=np.expand_dims(X,1)
    Y1=np.expand_dims(Y,0)
    M=cost_function(X1,Y1)
    return M


#@nb.njit(fastmath=True)
@nb.njit(['float32[:,:](float32[:,:],float32[:,:])','float64[:,:](float64[:,:],float64[:,:])'],fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
#    n,d=X.shape
#    m=Y.shape[0]
#    M=np.zeros((n,m)) 
    # for i in range(d):
    #     C=cost_function(X[:,i:i+1],Y[:,i])
    #     M+=C
    X1=np.expand_dims(X,1)
    Y1=np.expand_dims(Y,0)
    M=np.sum(cost_function(X1,Y1),2)
    return M

#@nb.jit()
def lot_embedding(X0,X1,p0,p1,numItermax=100000,numThreads=10):
    #C = np.asarray(C, dtype=np.float64, order='C')
    X0=np.ascontiguousarray(X0)
    X1=np.ascontiguousarray(X1)
    p0=np.ascontiguousarray(p0)
    p1=np.ascontiguousarray(p1)
    C=cost_matrix_d(X0,X1)
    gamma=ot.lp.emd(p0,p1,C,numItermax=numItermax,numThreads=10) # exact linear program
    #gamma, cost, u, v, result_code = emd_c(p0, p1, C, numItermax, numThreads)
    #result_code_string = check_result(result_code)
    N0,d=X0.shape
    X1_hat=gamma.dot(X1)/np.expand_dims(p0,1)
    U1=X1_hat-X0
    return U1


#@nb.jit()
def opt_lp(mu,nu,cost_M,Lambda,numItermax=100000,numThreads=10):
    mu=np.ascontiguousarray(mu)
    nu=np.ascontiguousarray(nu)
    n,m=cost_M.shape
    mass_mu=np.sum(mu)
    mass_nu=np.sum(nu)
    mu1=np.zeros(n+1)
    nu1=np.zeros(m+1)
    mu1[0:n]=mu
    nu1[0:m]=nu
    mu1[-1]=mass_nu
    nu1[-1]=mass_mu       
    cost_M1=np.zeros((n+1,m+1))
    cost_M1[0:n,0:m]=cost_M-2*Lambda
    #cost_M1 = np.asarray(cost_M1, dtype=np.float64, order='C')
    #gamma1=ot.lp.emd(mu1,nu1,cost_M1,numItermax=numItermax,numThreads=10)
    gamma1, cost1, u, v, result_code = emd_c(mu1, nu1, cost_M1, numItermax, numThreads)
    result_code_string = check_result(result_code)
    gamma=gamma1[0:n,0:m]
    cost=np.sum(cost_M*gamma)
    #destroyed_mass=np.sum(mu)+np.sum(nu)-2*np.sum(gamma)
    penualty=Lambda*(np.sum(mu)+np.sum(nu)-2*np.sum(gamma))
    return cost,gamma,penualty


def opt_pr(X,Y,mu,nu,mass,numItermax=100000):
    n,d=X.shape
    m=Y.shape[0]
    cost_M=cost_matrix_d(X,Y)
    gamma=ot.partial.partial_wasserstein(mu,nu,cost_M,m=mass,nb_dummies=n+m,numItermax=numItermax)
    cost=np.sum(cost_M*gamma)
    return cost,gamma


def lopt_embedding(X0,X1,p0,p1,Lambda,numItermax=100000,numThreads=10):
    X0=np.ascontiguousarray(X0)
    X1=np.ascontiguousarray(X1)
    cost_M=cost_matrix_d(X0,X1)
    cost,gamma,penualty=opt_lp(p0,p1,cost_M,Lambda,numItermax=numItermax,numThreads=10)
    N0=X0.shape[0]
    domain=np.sum(gamma,1)>1e-10
    p1_hat=np.sum(gamma,1) # martial of plan 
    # compute barycentric projetion 
    X1_hat=X0.copy() 
    X1_hat[domain]=gamma.dot(X1)[domain]/np.expand_dims(p1_hat,1)[domain]
    
    # separate barycentric projection into U_1 and p1_hat,M1
    U1=X1_hat-X0
    M1=np.sum(p1)-np.sum(p1_hat)
    p1_perp=p1-np.sum(gamma,0)
    return U1,p1_hat,M1,p1_perp

def lopt_embedding_pr(Xi,X0,p1,p0,Lambda):
    n,d=X0.shape
    cost,gamma=opt_pr(X0,Xi,p0,p1,Lambda)
    n=X0.shape[0]
    domain=np.sum(gamma,1)>1e-10
    p1_hat=np.sum(gamma,1)
    Xi_hat=np.full((n,d),0)
    Xi_hat[domain]=gamma.dot(Xi)[domain]/np.expand_dims(p1_hat,1)[domain]
    U1=Xi_hat-X0
    return U1,p1_hat


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


def lopt(U1,U2,p1_hat,p2_hat,Lambda,M1=.0,M2=.0):
    p12_hat=np.minimum(p1_hat,p2_hat)
    norm2=np.sum(np.minimum(np.sum((U1-U2)**2,1),2*Lambda)*p12_hat)
    penualty=Lambda*(np.sum(np.abs(p1_hat-p2_hat))+M1+M2)
    return norm2, penualty

    
   

def lot_barycenter(Xi_list,pi_list,X0_init,p0, weights,numItermax=1000,numItermax_lp=100000,stopThr=1e-7,numThreads=10):
    K=weights.shape[0]
    N0,d=X0_init.shape
    Ui_list=np.zeros((K,N0,d))
    weights=np.ascontiguousarray(weights)
    weights=weights.reshape(K,1,1)
    X0=X0_init
    for iter in range(numItermax):
        for i in range(K):
            Xi=Xi_list[i]
            pi=pi_list[i]
            Ui=lot_embedding(X0,Xi,p0,pi,numItermax=numItermax_lp,numThreads=numThreads)
            Ui_list[i]=Ui
        U_bar=np.sum(Ui_list*weights,0)
        X_bar=X0+U_bar    
        error = np.sum((X_bar - X0)**2)
        X0=X_bar
        if error<=stopThr:
            break
        
    return X0

def lopt_barycenter(Xi_list,pi_list,X0_init,p0, weights,Lambda, numItermax=100000,stopThr=1e-4, numThreads=10):
    K=weights.shape[0]
    N0,d=X0_init.shape
    Ui_list=np.zeros((K,N0,d))
    weights=np.ascontiguousarray(weights)
    weights=weights.reshape((K,1,1))
    X0=X0_init
    for iter in range(numItermax):
        for i in range(K):
            Xi=Xi_list[i]
            pi=pi_list[i]
            Ui,pi_hat,Mi,nu_perp=lopt_embedding(X0,Xi,p0,pi,Lambda,numItermax=numItermax,numThreads=numThreads)
            Ui_list[i]=Ui
        U_bar=np.sum(Ui_list*weights,0)
        X_bar=X0+U_bar    
        error = np.sum((X_bar - X0)**2)/np.linalg.norm(X0)
        X0=X_bar
        if error<=stopThr:
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







    


