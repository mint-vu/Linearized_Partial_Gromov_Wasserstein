#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:20:32 2023

"""
import numpy as np
import os
import sys
import numba as nb
import matplotlib.pyplot as plt


def PCA_numpy(X, k):
    N, N0, d = X.shape
    N = X.shape[0]
    X = X.reshape(N, -1)
    X_c = X - X.mean(0)
    Cov = np.cov(X_c.T) * (N - 1) / N
    eig, V = np.linalg.eig(Cov)
    a = X_c.dot(V)[:, 0:k]
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    # print(eig)
    V = V[:, idx][:, 0:k]
    a = np.dot(X, V)
    return a, eig, V


def sampling(X_list, N, N0, k):
    sample_list = np.zeros(k * N, dtype=object)
    i = 0
    for i in range(k):
        rand_ind = np.random.randint(low=N0 * i, high=N0 * (i + 1), size=(N,))
        sample_list[i * N : (i + 1) * N] = X_list[rand_ind]
        i += 1
    return sample_list


@nb.njit(nb.int64[:](nb.int64[:, :]), fastmath=True)
def gamma_to_T(gamma):
    n, m = gamma.shape
    T = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ind = np.where(gamma[i, :] == 1)[0]
        if ind.shape[0] == 1:
            T[i] = ind[0]
    return T


def scatter_2d(
    Xt,
    pt,
    X_d=np.array([]),
    p_d=0,
    X_c=np.array([]),
    p_c=0,
    xlim=None,
    ylim=None,
    color="blue",
    marker="o",
    name=None,
    Type=1,
):
    s = 2000
    plt.figure(1, figsize=(5.5, 5.5))

    plt.scatter(Xt[:, 0], Xt[:, 1], c=color, marker="o", s=s * pt, alpha=1)
    if X_d.shape[0] > 0:
        plt.scatter(X_d[:, 0], X_d[:, 1], c=color, marker="^", s=s * p_d, alpha=1)
    if X_c.shape[0] > 0:
        plt.scatter(X_c[:, 0], X_c[:, 1], c=color, marker="v", s=s * p_c, alpha=1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if name != None:
        plt.savefig(name, dpi=200, format="pdf", bbox_inches="tight")
    plt.show()


def scatter_2d_sub(
    ax,
    Xt,
    pt,
    X_d=np.array([]),
    p_d=0,
    X_c=np.array([]),
    p_c=0,
    color="blue",
    marker="o",
    xlim=None,
    ylim=None,
    label=None,
):
    s = 2000
    # plt.figure(1, figsize=(5.5,5.5))
    ax.scatter(
        Xt[:, 0], Xt[:, 1], c=color, marker=marker, s=s * pt, alpha=1, label=label
    )
    if X_d.shape[0] > 0:
        ax.scatter(X_d[:, 0], X_d[:, 1], c=color, marker=marker, s=s * p_d, alpha=1)
    if X_c.shape[0] > 0:
        ax.scatter(X_c[:, 0], X_c[:, 1], c=color, marker=marker, s=s * p_c, alpha=1)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if type == 0:
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
