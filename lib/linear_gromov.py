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

def LGW_embedding(C0,X1,p0,p1,gamma,loss='square',**kwargs):
    X1_tilde=gamma.dot(X1)/np.expand_dims(p0,1)
    C1_tilde= X_to_C(X1_tilde,loss=loss)
    K1_tilde=C1_tilde-C0    
    return K1_tilde,X1_tilde 

def LGW_dist(K1_tilde,K2_tilde,p0):
    M=(K1_tilde-K2_tilde)**2 
    dist=p0.dot(M).dot(p0.reshape(-1,1))[0]
    return dist


def X_to_C(X,loss='square'):
    if loss=='square':
        C=cost_matrix_d(X,X,loss=loss)
    elif loss=='sqrt':
        C=np.sqrt(cost_matrix_d(X,X,loss='square'))
    return C

def LMPGW_embedding(C0,X1,p0,p1,gamma,loss='square',mass=None,**kwargs):
    if mass is None:
        mass=p0.sum()
        
    np.testing.assert_almost_equal(p0.sum(0),mass, 
                                   err_msg='p0.sum and mass must have the same value',
                                   decimal=6)

    if mass>p1.sum():
        raise ValueError("Problem infeasible. mass should less than p1.sum")
                         
    X1_tilde=gamma.dot(X1)/np.expand_dims(p0,1)
    C1_tilde= X_to_C(X1_tilde,loss=loss)
    K1_tilde=C1_tilde-C0
    return K1_tilde,X1_tilde

def LMPGW_dist(K1_tilde,K2_tilde,p0):
    M=(K1_tilde-K2_tilde)**2 #np.minimum((,2*Lambda)
    dist=p0.dot(M).dot(p0.reshape(-1,1))[0]
    return dist


def LPGW_embedding(C0,X1,p0,p1,Lambda,gamma,loss='square',**kwargs):
    n0,d1=C0.shape[0],X1.shape[1] 
    p1_tilde=np.sum(gamma,1)
    domain=p1_tilde>0
    X1_tilde=np.zeros((n0,d1)) 
    X1_tilde[domain]=gamma.dot(X1)[domain]/np.expand_dims(p1_tilde,1)[domain]
    X1_tilde=gamma.dot(X1)/np.expand_dims(p0,1)
    C1_tilde= X_to_C(X1_tilde,loss=loss)
    K1_tilde=C1_tilde-C0
    mass_p1c=p1.sum()**2-p1_tilde.sum()**2
    return (K1_tilde,p1_tilde,mass_p1c),X1_tilde


#Compute LPGW distances for all reference measures
def LPGW_dist(embedding1,embedding2,Lambda=0):
    
    (K1_tilde,p1_tilde,mass_p1c),(K2_tilde,p2_tilde,mass_p2c)=embedding1,embedding2
    
    p12=np.minimum(p1_tilde,p2_tilde)
    M=(K1_tilde-K2_tilde)**2 
    dist=p12.dot(M).dot(p12.reshape(-1,1))[0]
    penalty_1=Lambda*(p1_tilde.sum()**2+p2_tilde.sum()**2-2*p12.sum()**2)
    penalty_2=Lambda*(mass_p1c+mass_p2c)
    return dist, penalty_1+penalty_2





def LGW(T1,T2,sigma, metric = "sqsq_loss",normalized=False):
    if metric == "sqsq_loss":
        M1 = ot.dist(T1,T1)
        M2 = ot.dist(T2,T2)
        if normalized:
            M1 = M1/np.max(M1)
            M2 = M2/np.max(M2)
    elif metric == "sq_loss":
        M1 = ot.dist(T1,T1,metric="euclidean")
        M2 = ot.dist(T2,T2,metric="euclidean")
        if normalized:
            M1 = M1/np.max(M1)
            M2 = M2/np.max(M2)
    else:
        raise Exception("metric not known")
    return np.sqrt(np.sum(np.multiply((M1-M2)**2,np.outer(sigma,sigma))))
    #return np.sum(np.multiply(np.linalg.norm(M1-M2)**2,np.outer(sigma,sigma)))

#functions
def lgw_procedure(M_ref,height_ref,posns,Ms,heights,max_iter = 1000,loss='square',emb=None):
    st = time.time()
    assert loss in ["square", "graph",'sqrt']
    N = len(Ms)
    n=height_ref.shape[0]
    if emb is None:
        embeddings=[] # embeddings 
        bps = [] #barycentric projections
       
        height_ref=height_ref.copy()
        for i in range(0,N):
            height=heights[i].copy()
                
            #P = ot.gromov.gromov_wasserstein(M_ref,Ms[i],height_ref/height_ref.sum(),heights[i]/heights[i].sum(),"square_loss",log=True,numItermax = max_iter, stopThr = 1e-20, stopThr2 = 1e-20,armijo=False)[0]
            G=gromov_wasserstein(M_ref, Ms[i], height_ref, height, G0=None, thres=1, numItermax=500*n, tol=1e-5,log=False, verbose=False,line_search=True)
    
            embedding,bp=LGW_embedding(M_ref,posns[i],height_ref,heights[i],gamma=G,loss=loss)
            embeddings.append(embedding)
            bps.append(bp)
    else:
        embeddings,bps=emb

    #LGW computation
    lgw = np.zeros((N,N))
    for i in range(N):
        embedding1=embeddings[i]
        for j in range(i + 1, N):
            embedding2=embeddings[j]
            trans=LGW_dist(embedding1,embedding2,height_ref)
            lgw[i, j] = np.sqrt(trans)
    lgw += lgw.T
    et =time.time()
    return lgw, et-st,(embeddings,bps)


#functions
def lpgw_procedure(M_ref,height_ref,posns,Ms,heights,Lambda,loss = 'square',emb=None):
    st = time.time()
    assert loss in ["square",'sqrt', "graph"]
    N = len(Ms)
  

    st = time.time()
    n=M_ref.shape[0]
    if emb is None:
        embeddings=[] # embeddings 
        bps = [] #barycentric projections
        for i in range(0,N):
            #GW computation
    
            G = partial_gromov_ver1(M_ref, Ms[i], height_ref, heights[i], G0=None, Lambda=Lambda, thres=1, numItermax=500*n, tol=1e-5,log=False, verbose=False,line_search=True)
            #euclidean barycentric projection
            embedding,bp=LPGW_embedding(M_ref,posns[i],height_ref,heights[i],Lambda=Lambda,gamma=G,loss=loss)
            embeddings.append(embedding)
            bps.append(bp)

    else:
        embeddings,bps=emb
    #LGW computation
    lpgw_trans = np.zeros((N,N))
    lpgw_penalty=np.zeros((N,N))
    for i in range(N):
        embedding1=embeddings[i]
        for j in range(i + 1, N):
            embedding2=embeddings[j]
            trans,penalty=LPGW_dist(embedding1,embedding2,Lambda)
            lpgw_trans[i, j] = trans
            lpgw_penalty[i,j]  = penalty
    
    lpgw_trans += lpgw_trans.T
    lpgw_penalty += lpgw_penalty.T
    et =time.time()
    return lpgw_trans,lpgw_penalty, et-st,(embeddings,bps)


#functions
def lmpgw_procedure(M_ref,height_ref,posns,Ms,heights,mass,loss = 'square',emb=None):
    st = time.time()
    assert loss in ["square",'sqrt', "graph"]
    N = len(Ms)

    assert abs(height_ref.sum()-mass)<1e-9
  

    st = time.time()
    n=M_ref.shape[0]
    if emb is None:
        embeddings=[] # embeddings 
        bps = [] #barycentric projections
        for i in range(0,N):
            #GW computation
    
            G = partial_gromov_wasserstein(M_ref, Ms[i], height_ref, heights[i], G0=None, m=mass, thres=1, numItermax=500*n, tol=1e-5,log=False, verbose=False,line_search=True)
            #euclidean barycentric projection
            embedding,bp=LMPGW_embedding(M_ref,posns[i],height_ref,heights[i],m=mass,gamma=G,loss=loss)
            embeddings.append(embedding)
            bps.append(bp)

    else:
        embeddings,bps=emb
    #LGW computation
    lpgw_trans = np.zeros((N,N))
    for i in range(N):
        embedding1=embeddings[i]
        for j in range(i + 1, N):
            embedding2=embeddings[j]
            trans=LMPGW_dist(embedding1,embedding2,height_ref)
            lpgw_trans[i, j] = trans
    
    lpgw_trans += lpgw_trans.T
    et =time.time()
    return lpgw_trans, et-st,(embeddings,bps)
