import os
import torch
import ot


from .opt import *
from .gromov import *

import numpy as np 
import numba as nb
import warnings
import time
from ot.backend import get_backend, NumpyBackend
from ot.lp import emd

import warnings

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





def GW_dist(C1,C2,gamma):        
    M_gamma=gwgrad_partial1(C1, C2, gamma)
    dist=np.sum(M_gamma*gamma)
    return dist

def MPGW_dist(C1,C2,gamma):        
    M_gamma=gwgrad_partial1(C1, C2, gamma)
    dist=np.sum(M_gamma*gamma)
    return dist

def PGW_dist_with_penalty(C1,C2,gamma,p1,p2,Lambda):
    M_gamma=gwgrad_partial1(C1, C2, gamma)
    dist=np.sum(M_gamma*gamma)
    penalty=Lambda*(p1.sum()**2+p2.sum()**2-2*gamma.sum()**2)    
    return dist,penalty

def LGW_embedding(C0,C1,X1,p0,p1,gamma,loss='square',**kwargs):
    X1_tilde=gamma.dot(X1)/np.expand_dims(p0,1)
    C1_tilde=cost_matrix_d(X1_tilde,X1_tilde,loss=loss)
    K1_tilde=C1_tilde-C0    
    return K1_tilde,X1_tilde 

def LGW_dist(K1_tilde,K2_tilde,p0):
    M=(K1_tilde-K2_tilde)**2 
    dist=p0.dot(M).dot(p0.reshape(-1,1))[0]
    return dist



def LMPGW_embedding(C0,C1,X1,p0,p1,gamma,loss='square',mass=None,**kwargs):
    if mass is None:
        mass=p0.sum()
        
    np.testing.assert_almost_equal(p0.sum(0),mass, 
                                   err_msg='p0.sum and mass must have the same value',
                                   decimal=6)

    if mass>p1.sum():
        raise ValueError("Problem infeasible. mass should less than p1.sum")
                         
    X1_tilde=gamma.dot(X1)/np.expand_dims(p0,1)
    C1_tilde=cost_matrix_d(X1_tilde,X1_tilde,loss=loss)
    K1_tilde=C1_tilde-C0
    return K1_tilde,X1_tilde

def LMPGW_dist(K1_tilde,K2_tilde,p0):
    M=(K1_tilde-K2_tilde)**2 #np.minimum((,2*Lambda)
    dist=p0.dot(M).dot(p0.reshape(-1,1))[0]
    return dist


def LPGW_embedding(C0,C1,X1,p0,p1,Lambda,gamma,loss='square',**kwargs):
    #C1,C2=cost_matrix_d(X,X,loss=loss),cost_matrix_d(Y,Y,loss=loss)
    n0,d1=C0.shape[0],X1.shape[1] 
    p1_tilde=np.sum(gamma,1)
    domain=p1_tilde>0
    X1_tilde=np.zeros((n0,d1)) 
    X1_tilde[domain]=gamma.dot(X1)[domain]/np.expand_dims(p1_tilde,1)[domain]
    C1_tilde=cost_matrix_d(X1_tilde,X1_tilde,loss=loss)
    K1_tilde=C1_tilde-C0
    mass_p1c=p1.sum()**2-p1_tilde.sum()**2
    return (K1_tilde,p1_tilde,mass_p1c),X1_tilde


def LPGW_dist(K1_tilde,K2_tilde,p1_tilde,p2_tilde,mass_p1c,mass_p2c,Lambda):
    p12=np.minimum(p1_tilde,p2_tilde)
    M=(K1_tilde-K2_tilde)**2 
    dist=p12.dot(M).dot(p12.reshape(-1,1))[0]
    penalty_1=Lambda*(p1_tilde.sum()**2+p2_tilde.sum()**2-2*p12.sum()**2)
    penalty_2=Lambda*(mass_p1c+mass_p2c)
    return dist, penalty_1+penalty_2


