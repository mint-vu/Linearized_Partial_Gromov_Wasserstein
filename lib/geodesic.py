import torch
import numpy as np
import os
import sys
import numba as nb
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_swiss_roll, make_moons, make_circles
import ot

# from lib.library import *
from .lib_lopt import *


# geodesic
def ot_geodesic(X0, U1, t_list):
    N, d = X0.shape
    tN = t_list.shape[0]
    Xt_list = np.zeros((tN, N, d))
    for i in range(tN):
        t = t_list[i]
        Xt_list[i] = X0 + t * U1
    return Xt_list


def lot_geodesic(X0, U1, U2, t_list):
    N, d = X0.shape
    tN = t_list.shape[0]
    Xt_list = np.zeros((tN, N, d))
    for i in range(tN):
        t = t_list[i]
        Ut = (1 - t) * U1 + t * U2
        Xt_list[i] = X0 + Ut
    return Xt_list


def opt_interpolation(X0, U1, p0, p1_hat, t_list):
    N, d = X0.shape
    tN = t_list.shape[0]
    Xt_list = np.zeros((tN, N, d))

    D = p1_hat > 0

    for i in range(tN):
        t = t_list[i]
        Ut = t * U1
        Xt_list[i] = X0 + Ut
    Xt_list = Xt_list[:, D, :]
    X_d = X0[np.invert(D)]
    pt = p1_hat[D]
    p_d = (p0 - p1_hat)[np.invert(D)]

    return Xt_list, pt, X_d, p_d


def lopt_interpolation(X0, U1, U2, p1_hat, p2_hat, t_list):
    N, d = X0.shape
    tN = t_list.shape[0]
    Xt_list = np.zeros((tN, N, d))

    p12_hat = np.minimum(p1_hat, p2_hat)
    D1 = p1_hat > 0
    D2 = p2_hat > 0
    D12 = np.logical_and(D1, D2)

    D_d = np.logical_and(D1, np.invert(D12))
    D_c = np.logical_and(D2, np.invert(D12))  # domain of desrtoyed point

    X_d = (X0 + U1)[D_d]
    X_c = (X0 + U2)[D_c]
    for i in range(tN):
        t = t_list[i]
        Ut = (1 - t) * U1 + t * U2
        Xt_list[i] = X0 + Ut
    Xt_list = Xt_list[:, D12, :]
    pt = p12_hat[D12]
    p_d = (p1_hat - p12_hat)[D_d]
    p_c = (p2_hat - p12_hat)[D_c]
    return Xt_list, pt, X_d, p_d, X_c, p_c


# opt interporlation


def opt_interpolation_T(X0, X1, p0, p1, T, t_list):
    N0, d = X0.shape
    N1 = X1.shape[0]
    Tn = t_list.shape[0]
    D0 = np.arange(N0)
    D0_trans = D0[T >= 0]
    D0_perp = D0[T < 0]
    D1 = np.arange(N1)
    D1_trans = T[D0_trans]
    mask = np.in1d(D1, D1_trans)
    D1_perp = D1[np.invert(mask)]
    X0_trans = X0[D0_trans]
    X1_trans = X1[D1_trans]

    p0_trans = np.zeros(N0)
    p1_trans = np.zeros(N1)
    p01_trans_take = np.minimum(p0[D0_trans], p1[D1_trans])
    p0_trans[D0_trans] = p01_trans_take
    p1_trans[D1_trans] = p01_trans_take

    Xt_list = np.zeros((Tn, D0_trans.shape[0], d))
    i_t = 0
    for t in t_list:
        Xt_list[i_t] = (1 - t) * X0_trans + t * X1_trans
        i_t += 1

    p0_remain = p0 - p0_trans
    p1_remain = p1 - p1_trans
    D0_remain = p0_remain > 0
    D1_remain = p1_remain > 0
    X_d = X0[D0_remain]
    X_c = X1[D1_remain]
    p0_remain_take = p0_remain[D0_remain]
    p1_remain_take = p1_remain[D1_remain]

    return Xt_list, p01_trans_take, X_d, X_c, p0_remain_take, p1_remain_take


# HK geodesic
def cost_hk(x, y):
    s = 1 / np.sqrt(2)
    cost = s * np.linalg.norm(x - y)
    return cost


def phi_hk(cost, m0, m1, t):
    if t == 0 or m1 == 0:
        return 0
    elif t == 1 or m0 == 0:
        return cost
    elif t < 1 and t > 0:
        return np.arccos(
            ((1 - t) * np.sqrt(m0) + t * np.sqrt(m1) * np.cos(cost))
            / np.sqrt(M(cost, m0, m1, t))
        )


def M_hk(cost, m0, m1, t):
    return (
        (1 - t) ** 2 * m0
        + t**2 * m1
        + 2 * t * (1 - t) * np.sqrt(m0 * m1) * np.cos(cost)
    )


def X_hk(x0, x1, m0, m1, t):
    cost_hk = cost_hk(x0, x1)
    return x0 + (x1 - x0) * phi_hk(cost_hk, m0, m1, t) / cost_hk


def HK_geodesic_T(X0, X1, p0, p1, T, t_list):
    N0, d = X0.shape
    N1 = X1.shape[0]
    D0 = np.arange(N0)
    D0_trans = D0[T >= 0]
    D0_perp = D0[T < 0]

    D1 = np.arange(N1)

    D1_trans = T[D0_trans]
    mask = np.in1d(D1, D1_trans)
    D1_perp = D1[np.invert(mask)]

    X0_trans = X0[D0_trans]
    X1_trans = X1[D1_trans]
    p0_trans = p0[D0_trans]
    p1_trans = p1[D1_trans]
    Tn = t_list.shape[0]
    Xt_list = np.zeros((Tn, D0_trans.shape[0], d))
    Mt_list = np.zeros((Tn, D0_trans.shape[0]))
    i_t = 0
    for t in t_list:
        i_loc = 0
        for x0, x1, m0, m1 in zip(X0_trans, X1_trans, p0_trans, p1_trans):
            cost = cost_hk(x0, x1)
            Xt_list[i_t, i_loc] = X(x0, x1, m0, m1, t)
            Mt_list[i_t, i_loc] = M(cost, m0, m1, t)
            i_loc += 1
        i_t += 1
    X_d = X0[D0_perp]
    p0_perp = p0[D0_perp]
    X_c = X1[D1_perp]
    p1_perp = p1[D1_perp]
    return Xt_list, Mt_list, X_d, X_c, p0_perp, p1_perp
