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
        w = iniW(nodes[i], nodes[i+1])
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
    a = x
    A = [a]
    for i in range(len(W)):
        w = W[i]
        if i == len(W)-1:
            a_i = (act_function(w.T, a.T, 5))
        else:
            a_i = (act_function(w.T, a.T, n_function))
        A.append(a_i)
        a = a_i
    return A

# Feed-Backward of SNN


def gradW(act, ye, W, param):
    mu = param[9]
    M = int(param[8])
    e = (ye - act[len(act)-1])/ye.shape[0]
    hidden_act_function = int(param[6])
    gW = []
    W_i = []

    for i in reversed(range(len(W))):  # reversed para tomar primero la salida
        w = W[i]
        z = np.dot(w.T, act[i].T)
        if i == len(W)-1:  # primera iteracion osea capa de salida
            derivate_z = derivate_act(z, 5)
            gamma = np.multiply(e, derivate_z)
        else:
            derivate_z = derivate_act(z, hidden_act_function)
            if (hidden_act_function == 5):
                derivate_z = derivate_z.T
            gamma = np.multiply(np.dot(gamma, w_prev.T), derivate_z.T)

        grad = np.dot(gamma.T, act[i])
        gW.append(grad)
        w_i = w - mu*grad.T
        W_i.append(w_i)
        w_prev = w
    cost = (1/2*M)*(act[len(act)-1]-ye)**2
    return gW[::-1], cost

# Update W and V


def updWV_sgdm(W, V, gW, param):
    mu = param[9]
    beta = param[10]
    W_i = []
    V_i = []
    for i in range(len(W)):
        v_i = beta * V[i] - mu*gW[i].T
        w_i = W[i] - v_i
        W_i.append(w_i)
        V_i.append(v_i)
    return (W_i, V_i)

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

    return (cm, np.array(f_score))


# Activation function
def act_function(w, X, function_number):
    z = np.dot(w.T, X)
    if (function_number == 1):
        h_z = ReLu_function(z).T
    if (function_number == 2):
        h_z = L_ReLu_function(z).T
    if (function_number == 3):
        h_z = ELU_function(z).T
    if (function_number == 4):
        h_z = SELU_function(z).T
    if (function_number == 5):
        h_z = sigmoidal_function(z).T
    return (h_z)


def derivate_act(z, function_number):
    if (function_number == 1):
        h_z = d_ReLu_function(z)
    elif (function_number == 2):
        h_z = d_L_ReLu_function(z)
    elif (function_number == 3):
        h_z = d_ELU_function(z)
    elif (function_number == 4):
        h_z = d_SELU_function(z)
    elif (function_number == 5):
        h_z = d_sigmoidal_function(z)
    return (h_z)


def output_activation(v, h):
    z = np.dot(v, h.T)
    y = 1/(1+np.exp(-z))
    return y.T


def ReLu_function(x):
    return np.where(x > 0, x, 0)


def L_ReLu_function(x):
    return np.where(x < 0, 0.01*x, x)


def ELU_function(x):
    a = 1.6732
    return np.where(x > 0, x, a*(np.exp(x)-1))


def SELU_function(x):
    a = 1.6732
    lam = 1.0507
    return np.where(x > 0, x*lam, a*(np.exp(x)-1))


def sigmoidal_function(z):
    return 1.0/(1.0+np.exp(-z))


def d_ReLu_function(x):
    return np.maximum(0, x)


def d_L_ReLu_function(x):
    return np.where(x < 0, 0.01*x, x)


def d_ELU_function(x):
    a = 1.6732
    return np.where(x > 0, 1, a*np.exp(x))


def d_SELU_function(x):
    lam = 1.0507
    a = 1.6732
    return np.where(x > 0, 1, a*np.exp(x))*lam


def d_sigmoidal_function(z):
    return (np.multiply(1/(1+np.exp(-z)), 1-(1/(1+np.exp(-z))))).T
# -----------------------------------------------------------------------
