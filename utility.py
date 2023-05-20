# My Utility : auxiliars functions

import pandas as pd
import numpy as np

# load parameters to train the SNN


def load_cnf():
    cnf = np.genfromtxt("cnf.csv")
    return cnf


# Initialize weights for SNN-SGDM
def iniWs(L, nodes):
    W = []
    V = []
    for i in range(L):
        w = iniW(nodes[i+1], nodes[i])
        W.append(w)
        V.append(np.zeros_like(w))
    return (W, V)


# Initialize weights for one-layer
def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return w


# Feed-forward of SNN
def forward(x, W, n_function):

    L = len(W)
    A = [0] * (L+1)
    Z = [0] * L
    A[0] = x.T

    for i in range(0, L):
        if i == L-1:
            f_act = 5
        else:
            f_act = n_function
        Z[i] = np.dot(W[i], A[i])
        A[i+1] = act_function(Z[i], f_act)

    return A[-1], Z


# Feed-Backward of SNN
def gradW(act,x, ye, z, W, param):
    L = len(W)
    gW = []
    N = ye.shape[0]
    hidden_f = int(param[6])

    for i in reversed(range(1, L+1)):
        if i == L:  # primera iteracion osea capa de salida
            dA = (act - ye.T)
            dZ = dA * derivate_act(z[i-1], 5)
            grad = np.dot(dZ, act_function(z[i-2],5).T)
        else:
            dZ = dA * derivate_act(z[i-1], hidden_f)
            if i == 1:
                grad = np.dot(dZ,x.T)
            else:
                grad = np.dot(dZ, act_function(z[i-2],hidden_f).T)
        dA = np.dot(W[i-1].T, dZ)
        gW.append(grad)
    cost = (1/(2*N)) * ((act - ye.T)**2)

    # se retorna dado vuelta ya que parte el 3 y baja hasta el 1
    return gW[::-1], cost


# Update W and V
def updWV_sgdm(W, V, gW, param):
    mu = param[9]
    beta = param[10]
    for i in range(len(W)):
        V[i] = beta * V[i] + mu * gW[i]
        W[i] = W[i] - V[i]

    return W, V


# Measure
def metricas(x, y):
    cm = np.zeros((y.shape[1], x.shape[1]))
    for real, predicted in zip(y, x):
        cm[np.argmax(real)][np.argmax(predicted)] += 1
    f_score = []
    for index, feature in enumerate(cm):
        TP = feature[index]
        FP = cm.sum(axis=0)[index] - TP
        FN = cm.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append((2 * (precision * recall) / (precision + recall)))
    f_score.append(np.array(f_score).mean())



def act_function(Z: np.ndarray, act_func: int):
    if act_func == 1:
        return np.maximum(0, Z)
    if act_func == 2:
        return np.maximum(0.01 * Z, Z)
    if act_func == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(Z > 0, Z, alpha*(np.exp(Z)-1)) * lam
    if act_func == 5:
        return 1 / (1 + np.exp(-Z))

# Derivatives of the activation funciton
def derivate_act(A: np.ndarray, act_func: int):
    if act_func == 1:
        return np.where(A >= 0, 1, 0)
    if act_func == 2:
        return np.where(A >= 0, 1, 0.01)
    if act_func == 3:
        return np.where(A >= 0, 1, 0.01 * np.exp(A))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(A > 0, 1, alpha*np.exp(A)) * lam
    if act_func == 5:
        s = act_function(A, act_func)
        return s * (1 - s)
# -----------------------------------------------------------------------
