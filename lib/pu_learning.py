# load data
import torch
import pickle

import numpy as np
import torch
import ot
import os

import matplotlib.pyplot as plt

os.chdir("..")
from .gromov_test import (
    partial_gromov_ver1,
    cost_matrix_d,
    tensor_dot_param,
    tensor_dot_func,
    gwgrad_partial,
    partial_gromov_wasserstein,
)
from .opt import *

import numpy as np
import numba as nb
import warnings
import time
from ot.backend import get_backend, NumpyBackend
from ot.lp import emd

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, recall_score, precision_score

from .unbalanced_gromov_wasserstein.unbalancedgw.vanilla_ugw_solver import (
    exp_ugw_sinkhorn,
    log_ugw_sinkhorn,
)
from .unbalanced_gromov_wasserstein.unbalancedgw.batch_stable_ugw_solver import (
    log_batch_ugw_sinkhorn,
)
from .unbalanced_gromov_wasserstein.unbalancedgw._vanilla_utils import (
    ugw_cost,
    init_plan,
)
from .unbalanced_gromov_wasserstein.unbalancedgw.utils import generate_measure
from .unbalanced_gromov_wasserstein.unbalancedgw._batch_utils import (
    compute_batch_flb_plan,
    compute_distance_histograms,
)
from .primal_pgw.partial_gw import compute_cost_matrices, pu_w_emd, pu_gw_emd


def data_process(name="amazon_surf"):
    # open the data file
    if name in ["MNIST", "EMNIST"]:
        data_file = torch.load("pu_learning/data/" + name + ".pt")
        (X, l) = data_file
        classes = None
    elif "surf" in name or "decaf" in name:
        with open("pu_learning/data/" + name + "_fts.pkl", "rb") as f:
            data_file = pickle.load(f)

        if "surf" in name:
            X0 = data_file["features"]
            l = data_file["labels"]
            classes = data_file["classes"]
            pca = PCA(n_components=10, random_state=0)
            pca.fit(X0.T)
            X = pca.components_.T
        elif "decaf" in name:
            X0 = data_file["fc8"]
            l = data_file["labels"]
            classes = data_file["classes"]
            pca = PCA(n_components=40, random_state=0)
            pca.fit(X0.T)
            X = pca.components_.T
    return (X, l), classes


def MNIST_figure(figure_list, label_list):
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(figure_list[i][0], cmap="gray")
        plt.title(f"Label: {label_list[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def normalize_X(X):
    div = np.max(X, axis=0) - np.min(X, axis=0)
    div[div == 0] = 1  # Avoid division by zero
    X = (X - np.min(X, axis=0)) / div
    return X


# def convert_data(dataset,name='MNIST',visual=False):
#     if name in ['MNIST','EMNIST']:
#         X_list,label_list=dataset
#             label_list_all.append(label_list)
#         embedding_list_all=np.vstack(embedding_list_all)
#         label_list_all=np.vstack(label_list_all).reshape(-1).astype(np.int64)
#     return embedding_list_all,label_list_all


# This function is adapted from "Partial Gromov-Wasserstein with applications on Positive-Unlabeled Learning" repo 
# it is modified version
def draw_pu_dataset_scar(
    dataset_p,
    dataset_u=None,
    size_p=10,
    size_u=20,
    prior=0.5,
    p_label=0,
    seed_nb=None,
    same_dataset=True,
):
    """Draw a Positive and Unlabeled dataset "at random""

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    size_p: number of points in the positive dataset

    size_u: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    seed_nb: seed

    Returns
    -------
    pandas.DataFrame of shape (n_p, d_p)
        Positive dataset

    pandas.DataFrame of shape (n_u, d_u)
        Unlabeled dataset

    pandas.Series of len (n_u)
        labels of the unlabeled dataset
    """
    x, l = dataset_p[0].copy(), dataset_p[1].copy()
    A = l == p_label
    B = l != p_label
    l[A], l[B] = 1, 0
    x = normalize_X(x)

    size_u_p = int(prior * size_u)
    size_u_n = size_u - size_u_p

    xp_t = x[l == 1]
    tp_t = l[l == 1]

    xp, xp_other, _, tp_o = train_test_split(
        xp_t, tp_t, train_size=size_p, random_state=seed_nb
    )
    # print('xp_other shape',xp_other.shape)
    if same_dataset or dataset_u is None:
        xup, _, lup, _ = train_test_split(
            xp_other, tp_o, train_size=size_u_p, random_state=seed_nb
        )
    else:
        x, l = dataset_u[0].copy(), dataset_u[1].copy()
        x = normalize_X(x)
        A = l == p_label
        B = l != p_label
        l[A], l[B] = 1, 0
        # x, t = make_data(dataset=dataset_u)

        # div = np.max(x, axis=0) - np.min(x, axis=0)
        # div[div == 0] = 1
        # x = (x - np.min(x, axis=0)) / div
        xp_other = x[l == 1]
        tp_o = l[l == 1]
        xup, _, lup, _ = train_test_split(
            xp_other, tp_o, train_size=size_u_p, random_state=seed_nb
        )

    xn_t = x[l == 0]
    tn_t = l[l == 0]
    xun, _, lun, _ = train_test_split(
        xn_t, tn_t, train_size=size_u_n, random_state=seed_nb
    )

    xu = np.concatenate([xup, xun], axis=0)
    yu = np.concatenate((np.ones(len(xup)), np.zeros(len(xun)))).astype(np.int64)
    yu_2 = np.concatenate((lup, lun))
    # print(np.linalg.norm(yu-yu_2))
    return xp, xu, yu_2


def init_pgw_param(C1, C2, r):
    n, m = C1.shape[0], C2.shape[0]
    q = np.ones(m) / m
    p = np.ones(n) / n * r  # make the mass of p to be r
    mass = np.min((p.sum(), r))
    return p, q, mass


def gamma_to_l(G, r):
    n, m = G.shape
    G_2 = G.sum(0)
    quantile = np.quantile(G_2, 1 - r)
    l_G = np.zeros(m)
    l_G[G_2 >= quantile] = 1
    return l_G

#This function is adapted from repo "unbalanced_gromov_wasserstein"
def init_param_ugw(C1, C2):
    n, m = C1.shape[0], C2.shape[0]
    n_pos, n_unl = n, m
    nb_try = 1
    mu = (torch.ones([n_pos]) / n_pos).expand(nb_try, -1)
    nu = (torch.ones([n_unl]) / n_unl).expand(nb_try, -1)

    grid_eps = [2.0**k for k in range(-9, -8, 1)]
    grid_rho = [2.0**k for k in range(-10, -4, 1)]
    eps = grid_eps[0]
    rho = grid_rho[0]
    rho2 = grid_rho[0]
    Cx = torch.from_numpy(C1).to(torch.float32).reshape((nb_try, n, n))
    Cy = torch.from_numpy(C2).to(torch.float32).reshape((nb_try, m, m))
    return mu, nu, eps, rho, rho2, Cx, Cy


#This function is adapted from repo "unbalanced_gromov_wasserstein"
def init_flb_uot(C1, C2):
    mu, nu, eps, rho, rho2, Cx, Cy = init_param_ugw(C1, C2)
    _, _, init_plan = compute_batch_flb_plan(
        mu,
        Cx,
        nu,
        Cy,
        eps=eps,
        rho=rho,
        rho2=rho2,
        nits_sinkhorn=50000,
        tol_sinkhorn=1e-5,
    )

    return init_plan[0].numpy().astype(np.float64)

#This function is adapted from repo "Partial Gromov-Wasserstein with applications on Positive-Unlabeled Learning" 
def init_flb_pot(C1, C2, p, q, r, Lambda=30.0, n=100):
    p, q, mass = init_pgw_param(C1, C2, r)
    S1, S2 = C1.mean(0), C2.mean(0)
    C = cost_matrix(S1, S2)
    gamma, _ = opt_lp(p, q, C, Lambda=Lambda, numItermax=n * 500)

    return gamma


def pu_prediction_gw(C1, C2, r=0.2, G0=None, method="pgw", param={"Lambda": 30.0}):
    C1, C2 = C1.astype(np.float64), C2.astype(np.float64)
    # C1,C2=cost_matrix_d(X_p,X_p),cost_matrix_d(X_u,X_u)
    n, m = C1.shape[0], C2.shape[0]
    size_p = int(m * r)
    if size_p != n:
        print("# of positives in X_p and X_u are different, we suggest to modify them")
    if method == "gw":
        p = np.ones(n) / n
    if method == "primal_pgw":
        p, q, mass = init_pgw_param(C1, C2, r)
        #       mass=min(r*np.sum(q),np.sum(p)) # this used to avoid numerical issue
        C1, C2 = C1.astype(np.float64), C2.astype(np.float64)
        gamma = partial_gromov_wasserstein(
            C1,
            C2,
            p,
            q,
            m=mass,
            G0=G0,
            numItermax=n * 500,
            nb_dummies=1,
            line_search=False,
        )

    if method == "pgw":
        Lambda = param["Lambda"]
        p, q, mass = init_pgw_param(C1, C2, r)
        C1, C2 = C1.astype(np.float64), C2.astype(np.float64)
        gamma, _ = partial_gromov_v11(
            C1,
            C2,
            p,
            q,
            Lambda=Lambda,
            G0=G0,
            numItermax=n * 500,
            nb_dummies=1,
            line_search=False,
        )
    if method == "ugw":
        mu, nu, eps, rho, rho2, Cx, Cy = init_param_ugw(C1, C2)
        if "rho" in param:
            rho = param["rho"]
            rho2 = rho
        if "eps" in param:
            eps = param["eps"]
        # need to try different rho for better performance
        #        rho=0.0023 surf A
        if type(G0) == np.ndarray:
            init_plan = torch.from_numpy(G0).to(torch.float32).reshape((1, n, m))
        elif type(G0) == torch.Tensor:
            init_plan = G0
        gamma = log_batch_ugw_sinkhorn(
            mu,
            Cx,
            nu,
            Cy,
            init=init_plan,
            eps=eps,
            rho=rho,
            rho2=rho2,
            nits_plan=3000,
            tol_plan=1e-5,
            nits_sinkhorn=3000,
            tol_sinkhorn=1e-6,
        )
        print("gamma_mass_diff", gamma.sum() - r)
        gamma = gamma[0]
    return gamma
