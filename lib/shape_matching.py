
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import ot
from sklearn.datasets import make_moons
import os
os.chdir(".")

from lib.unbalanced_gromov_wasserstein.unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn,log_ugw_sinkhorn
from lib.unbalanced_gromov_wasserstein.unbalancedgw._vanilla_utils import ugw_cost
from lib.unbalanced_gromov_wasserstein.unbalancedgw.utils import generate_measure
from lib.unbalanced_gromov_wasserstein.unbalancedgw.batch_stable_ugw_solver import log_batch_ugw_sinkhorn
from lib.gromov import gromov_wasserstein, cost_matrix_d, partial_gromov_ver1, GW_dist, MPGW_dist, PGW_dist_with_penalty,partial_gromov_wasserstein


# @nb.njit(fastmath=True)
def generate_d_square(d, n):
    x_list = []
    for i in range(d):
        x = np.linspace(-1, 1, n)
        x_list.append(x)
    xx = np.meshgrid(*x_list, indexing="ij")
    data = np.zeros((n**d, d))
    for i in range(d):
        data[:, i] = xx[i].reshape(-1)
    return data


# @nb.njit(fastmath=True)
def generate_d_sphere(d, n, seed=20):
    np.random.seed(seed)
    N = n**d
    data = np.random.normal(size=(N, d))
    norm = np.sqrt(np.sum(np.square(data), 1)).reshape(-1, 1)
    data = data / norm
    return data



def rotation_2d(theta=0):
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
    ])


def make_yinyang_2d(N,r=1,var=0.0001,eta=0.2,theta=0.9*np.pi,beta=[1.5,-1],low=-2,high=-1):
    X = []
    c = []

    for _ in range(N):
        while True:
            x = np.random.uniform(-r, r)
            y = np.random.uniform(-r, r)

            if x**2 + y**2 <= r**2:
                if y >= 0: c_ = 0
                else: c_ = 1

                if y >= 0 and (x + r/2)**2 + y**2 <= (r/2)**2:
                    c_ = 1 - c_
                elif y < 0 and (x - r/2)**2 + y**2 <= (r/2)**2:
                    c_ = 1 - c_

                x += np.random.normal(0, var)
                y += np.random.normal(0, var)

                X.append([x, y])
                c.append(c_)
                break

    X,c = np.array(X), np.array(c)
    
    X1=X[c==0]
    X2=X[c==1]
    X2=X2.dot(rotation_2d(theta))+beta
    outliers = np.random.uniform(low=low, high=high, size=(int(N*eta), 2))
    return X1, X2, outliers

flip_M=np.array([[0,1],[1,0]])

def make_moon_2d(N=200, eta=0.10, ns=0.2,theta=0,beta=0,low=-3,high=-2,random_state=0):
    X, y = make_moons(n_samples=2*N, noise=ns,random_state=random_state)
    
    X1=X[y==0]
    X2=X[y==1]
    X2=X2.dot(rotation_2d(theta)).dot(flip_M)
    X2=X2+beta
    N=X1.shape[0]
    #outlier_idx = np.random.choice(np.where(y == 0)[0], int(N*eta))
    outliers = np.random.uniform(low=low, high=high, size=(int(N*eta), 2))
    return X1, X2,outliers

def make_3d_sphere(N=200, eta=0.2, d_min=0.5,low=-3,high=-2,random_state=0):
    circle = (np.linspace(0,2*np.pi,N+1)[0:N]).reshape((N,1)) #np.random.uniform(low=0, high=2*np.pi, size=(N, 1))
    circle = np.hstack((np.cos(circle), np.sin(circle)))
    np.random.seed(random_state)
    k=int(np.sqrt(N))
    n_remains=N-k**2
    
    angles =np.random.uniform([0,0],[np.pi,2*np.pi],(N,2))
#    phi   = np.linspace(0,np.pi,k+1)[0:k]
#    theta = np.linspace(0,2*np.pi,k+1)[0:k]
#    phi,theta=np.meshgrid(phi,theta)
#    phi,theta=phi.reshape(-1),theta.reshape(-1)
    
#    phi_1 = np.random.uniform(0, np.pi, n_remains)
#    theta_1 = np.random.uniform(0, 2*np.pi, n_remains)
    phi = angles[:,0]  #np.concatenate((phi,phi_1))
    
    theta = angles[:,1] # np.concatenate((theta,theta_1))
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere = np.stack((x, y, z), axis=1) 
    
    sphere += np.array([0, 0, 4])
    
    n_outliers = int(N*eta)
    #outliers = sphere[np.random.choice(sphere.shape[0], n_outliers, replace=True)]
    outliers = np.random.uniform(low=low, high=high, size=(n_outliers,2))
    
    
    return sphere, circle, outliers  

# def plot_2d(X1, X2):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', s=20)
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', s=20)
#     plt.scatter(X[y == 2, 0], X[y == 2, 1], c='c', s=20)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()

#@nb.njit(fastmath=True)
def generate_d_square(d,n):
    x_list=[]
    for i in range(d):
        x = np.linspace(-1, 1, n)
        x_list.append(x)
    xx= np.meshgrid(*x_list,indexing='ij')
    data=np.zeros((n**d,d))
    for i in range(d):
        data[:,i]=xx[i].reshape(-1)
    return data



def make_3d_square(k=2, eta=0.2, d_min=0.5,low=-3,high=-2):
    N=k**(2*3)
    square=generate_d_square(2,k**3)
    cubic=generate_d_square(3,k**2)
    
    cubic += np.array([0, 0, 4])
    
    n_outliers = int(N*eta)

    outliers = np.random.uniform(low=low, high=high, size=(n_outliers,2))
    
    
    return cubic,square, outliers  


def plot_2d(X1, X2):
    plt.figure(figsize=(8, 6))
    plt.scatter(X1[:,0], X1[:,1], c='r', s=20)
    plt.scatter(X2[:,0], X2[:, 1], c='b', s=20)
    #plt.scatter(X[y == 2, 0], X[y == 2, 1], c='c', s=20)
    #plt.xticks([])
    #plt.yticks([])
    plt.show()

    
def get_cost_matrices(X, Y):
    C1 = cost_matrix_d(X, X)
    C2 = cost_matrix_d(Y, Y)
    return C1, C2


def plot_plan_2d(X, Y, gamma,threshold=1e-8):
    plt.figure(figsize=(8, 6))
    N=X.shape[0]
    plt.scatter(X[:,0], X[:, 1], c='r')
    
    plt.scatter(Y[0:N,0], Y[0:N,1], c='c')
    plt.scatter(Y[N+1:,0],  Y[N+1:,1], c='b')
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            if gamma[i, j] > threshold:  plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'grey', lw=gamma[i, j]*150, alpha=0.4)

    plt.show()

def embed_3d(X):
    N,d=X.shape
    if d<3:
        X1=np.zeros((N,3))
        X1[:,0:d]=X
    else:
        X1=X.copy()
    return X1

def plot_3d(X, Y, gamma=None,threshold=1e-5):   
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    X,Y=embed_3d(X),embed_3d(Y)
    N=X.shape[0]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r')
    ax.scatter(Y[:N, 0], Y[:N, 1], Y[:N, 2], c='c')
    ax.scatter(Y[N+1:, 0], Y[N+1:, 1], Y[N+1:, 2], c='b')
    ax.axis('off')
    ax.set_facecolor('white') 
    ax.view_init(10, 90,0)
    if gamma is not None:
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if gamma[i, j] > threshold:  ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], [X[i, 2], Y[j, 2]], 'grey', lw=gamma[i, j]*150,alpha=0.4)

    plt.show()
    