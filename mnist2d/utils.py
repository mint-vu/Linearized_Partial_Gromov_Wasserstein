import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import torch
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
import ot
from tqdm import tqdm
import sys
sys.path.append("../")
from lib.linear_gromov import *
from lib.gromov_barycenter import *
from lib.opt import opt_lp,emd_lp
from lib.gromov import *
from sklearn import manifold
from sklearn.decomposition import PCA

from sklearn.datasets import load_digits
from sklearn.manifold import MDS
import ot

from sklearn import manifold

import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import plotly
import plotly.express as px
import sklearn
import pandas as pd
import random

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra



def sample_images_by_label(X, Y, target_label, sample_size=10,rd_seed=0):
    """
    Displays a specified number of random sample images from the MNIST dataset with the given label.

    loss_parameters:
    X (numpy.ndarray): The MNIST dataset of images.
    Y (numpy.ndarray): The labels corresponding to the images.
    target_label (int): The label of the images to sample.
    sample_size (int): The number of images to display. Default is 10.

    Returns:
    None
    """
    # Find indices of images with the specified label
    indices = np.where(Y == target_label)[0]
    np.random.rand(rd_seed)
    if len(indices) == 0:
        print(f"No images found with label {target_label}.")
        return

    # Adjust sample size if there are fewer images available
    if sample_size > len(indices):
        print(f"Only {len(indices)} images available with label {target_label}. Displaying all.")
        sample_size = len(indices)

    # Randomly select indices from the found indices
    sample_indices = np.random.choice(indices, size=sample_size, replace=False)

    return X[sample_indices], Y[sample_indices]

def split_pmf(X):
    b=X[:,2]
    X1=X[b!=-1,0:2]
    pmf=b[b!=-1]
    pmf=pmf/pmf.sum()
    return X1,pmf

def center(X):
    return X- X.mean(0)

def rotate(X,seed=0):
    X= X- X.mean(0)
    np.random.seed(seed)
    theta=np.random.rand(1)[0]*2*np.pi
    c,s=np.cos(theta),np.sin(theta)
    rot_d=np.array([[c,-s],[s,c]])
    flip = np.random.randint(0,2)
    X_rot=X.dot(rot_d)
    if flip ==1:
        flip_d=np.array([[0,1],[1,0]])
        X_rot=X_rot.dot(flip)
    return X_rot

def add_noise(X,pmf,eta=0,mass=1,seed=0,low_bound=-20,upper_bound=25):
    if eta==0:
        return X,pmf
    
    np.random.seed(seed)
    n_pts = X.shape[0]
    n_noise = int(n_pts * eta)
    mass=pmf.sum()
    mass_noise=eta*mass    
    noise_pts = np.random.randint(low_bound, upper_bound, size=(n_noise, 2))
    pmf_noise=np.ones(n_noise)/n_noise*mass_noise 

    X1=np.concatenate((X,noise_pts))
    pmf1=np.concatenate((pmf,pmf_noise))
    return X1,pmf1




# sample the training data 
def sample_dataset(train_X,train_y,labels,sample_size=500,seed=0):
    np.random.rand(seed)
    train_X_sample=[]
    train_Y_sample=[]
    train_X_sample_pos=[]
    train_X_sample_pmf=[]
    for label in labels:
        X_sample,Y_sample=sample_images_by_label(train_X,train_y,label,sample_size=sample_size)
        train_X_sample.append(X_sample)
        train_Y_sample.append(Y_sample)
    train_X_sample,train_Y_sample=np.concatenate(train_X_sample),np.concatenate(train_Y_sample)

    for i,X in tqdm(enumerate(train_X_sample)):
        pos,pmf=split_pmf(X)
        pos= pos- pos.mean(0)
        train_X_sample_pos.append(pos)
        train_X_sample_pmf.append(pmf)
    return train_X_sample_pos,train_X_sample_pmf,train_Y_sample

def process_dataset(test_X_sample_pos,test_X_sample_pmf,eta=0,seed=0):
    test_X_sample_pos1,test_X_sample_pmf1=[],[]
    for i,(pos,pmf) in tqdm(enumerate(zip(test_X_sample_pos,test_X_sample_pmf))):
        pos2=rotate(pos,seed=i+seed)
        pos3,pmf3=add_noise(pos2,pmf,eta=eta,mass=1,seed=0,low_bound=-25,upper_bound=25)
        test_X_sample_pos1.append(pos3)
        test_X_sample_pmf1.append(pmf3)
    return test_X_sample_pos1,test_X_sample_pmf1



def pos_to_mm_space(pos,loss='sqrt',loss_param=None):
    if loss == 'sqrt':
        M=np.sqrt(cost_matrix_d(pos,pos))
    if loss== 'square':
        M=cost_matrix_d(pos,pos)
    elif loss=='graph':
        n=loss_param['n_neighbors']
        knnG1 = kneighbors_graph(pos, n_neighbors=n,mode='distance')
        M = dijkstra(knnG1,directed=False)
    n=M.shape[0]
    pmf=np.ones(n)/n
    return M,pmf



def upper_triangle_flatten_matrices(k_matrices):
    K, n, _ = k_matrices.shape
    result = []
    # Get the indices of the upper triangular part, excluding the diagonal
   
    upper_triangle_indices = np.triu_indices(n, k=0)
    # Loop through each of the K matrices and extract the upper triangle
    for i in range(K):
        
        flattened_upper_half = k_matrices[i][upper_triangle_indices]
        result.append(flattened_upper_half)
    
    # Convert the result to a single flattened array
    return np.array(result)
    
def merge_embeddings(embeddings_list):
    a=embeddings_list[0]
    n=a.shape[0]
    a1=upper_triangle_flatten_matrices(a)
    merged_embedding=a1 #a.reshape((n,-1))
    for b in embeddings_list[1:]:
        b1=b.reshape((n,-1))
        merged_embedding=np.hstack((merged_embedding,b1))

    return merged_embedding
    
def GW_interporlation(X1,X2,t_list=[0,0.5,1],p1=None,p2=None,loss='sqrt',method='GW',M_ref=None,pmf_ref=None,Lambda=None,loss_param=None):
    assert method in ['GW','LGW','PGW','LPGW']
    assert loss in ['sqrt','graph']
    
    M1=X_to_C(X1,loss=loss,loss_param=loss_param)
    M2=X_to_C(X2,loss=loss,loss_param=loss_param)
    n1,n2,d=M1.shape[0],M2.shape[0],X1.shape[1]
    Ms=[M1,M2]
    ps=[p1,p2]

    clf=PCA(n_components=2)
    
    Mt_list,Xt_list=[],[]
    if M_ref is not None:
        N=M_ref.shape[0]
    else:
        N=int(np.mean([n1,n2]))+1
    if method in ['GW','LGW']:
        ps=[p/p.sum() for p in ps]
    if method =='LGW':
        if M_ref is None:
            N=int(np.mean([n1,n2]))+1
            pmf_ref=np.ones(N)/N
            M_ref=ot.gromov.gromov_barycenters(N, Cs = Ms,
                                            ps = ps, p = pmf_ref, lambdas = ot.unif(2), 
                                            max_iter=100, tol=1e-5,random_state=0,log=False)
        embedding_list=[]
        for (X,pmf,M) in zip([X1,X2],ps,Ms):
            gamma=gromov_wasserstein(M_ref, M, pmf_ref, pmf, G0=None, thres=1, numItermax=500*N, tol=1e-5,log=False, verbose=False,line_search=True)
            embedding,_=LGW_embedding(M_ref,X,pmf_ref,pmf,gamma,loss=loss,loss_param=loss_param)
            embedding_list.append(embedding)

    elif method =='LPGW':
        if Lambda is None:
            Lambda=max(M1.max(),M1.max())/2
        
        if M_ref is None:
            
            pmf_ref=np.ones(N)/N
            M_ref=pgw_barycenters(N, Cs = [M1,M2],
                                            ps = [p1,p2], p = pmf_ref, lambdas = ot.unif(2), Lambda_list=[Lambda,Lambda],
                                            max_iter=100, tol=1e-5,log=False)
        embedding_list=[]
        for (X,pmf,M) in zip([X1,X2],ps,Ms):
            gamma=partial_gromov_ver1(M_ref, M, pmf_ref, pmf, G0=None,Lambda=Lambda, thres=1, numItermax=500*N, tol=1e-5,log=False, verbose=False,line_search=True)
            embedding,_=LPGW_embedding(M_ref,X, pmf_ref, pmf, gamma,Lambda,loss=loss,loss_param=loss_param) 
            embedding_list.append(embedding[0])
            pmf_list.append(embedding[0])
        

    elif method=='GW':
        for t in t_list:
            lambdas=[1-t,t]
            #normalize qs
            C=ot.gromov.gromov_barycenters(
                N=N, Cs=Ms, ps=ps, p=pmf_ref, lambdas=lambdas,init_C=M_ref, 
                symmetric=True, armijo=False, max_iter=1000, tol=1e-5)
            Mt_list.append(C)
    elif method =='PGW':
        for t in t_list:
            lambdas=[1-t,t]
            C=pgw_barycenters(
                N=N, Cs=Ms, ps=ps, p=pmf_ref, lambdas=lambdas,Lambda_list=[Lambda,Lambda],loss_fun='square_loss',
                init_C=M_ref, max_iter=1000, tol=1e-5)
            
            Mt_list.append(C)
        #return Mt_list,
        
        
        
        
    if method in ['LGW','LPGW']:
        E1,E2=embedding_list[0],embedding_list[1]
        for t in t_list:
            Mt=M_ref+(1-t)*E1+t*E2
            Mt_list.append(Mt)
    
    if method =='LPGW':
        pmf=np.minimum(pmf_list[0],pmf_list[1])
    else:
        pmf=pmf_ref
    # convert Mt to Xt
    mds = MDS(n_components=2, metric=True,dissimilarity='precomputed',random_state=0,normalized_stress='auto')

    
    for Mt in Mt_list:
        Xt=mds.fit_transform(Mt)
        #Xt=clf.fit_transform(Xt.copy())
        Xt_list.append(Xt)
    return Xt_list,Mt_list,pmf

def add_noise(X,pmf,eta=0,mass=1,seed=0,low_bound=-20,upper_bound=25):
    if eta==0:
        return X,pmf
    
    np.random.seed(seed)
    n_pts = X.shape[0]
    n_noise = int(n_pts * eta)
    mass=pmf.sum()
    mass_noise=eta*mass    
    noise_pts = np.random.randint(low_bound, upper_bound, size=(n_noise, 2))
    pmf_noise=np.ones(n_noise)/n_noise*mass_noise 

    X1=np.concatenate((X,noise_pts))
    pmf1=np.concatenate((pmf,pmf_noise))
    return X1,pmf1

def rotation_2d(angle):
    """
    Computes a 2D rotation matrix.

    Parameters:
    angle (float): The rotation angle in radians.

    Returns:
    numpy.ndarray: A 2x2 rotation matrix.
    """
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

def flipping():
    return np.array([[0,1],[1,0]])
