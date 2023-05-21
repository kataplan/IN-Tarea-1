# My Utility : auxiliars functions

import pandas as pd
import numpy as np

# load parameters to train the SNN


def load_cnf():
    cnf = np.genfromtxt("cnf.csv")
    return cnf


# Initialize weights for SNN-SGDM
def iniWs(x, y, param):
    n_input_nodes = x.shape[0]
    n_output_nodes = int(param[0])
    n_hidden_1 = int(param[4])
    n_hidden_2 = int(param[5])
    # SetUp pesos
    L = 3

    nodes = [n_input_nodes]

    if n_hidden_1 > 0:
        nodes.append(n_hidden_1)
    else:
        L -= -1
    if n_hidden_2 > 0:
        nodes.append(n_hidden_2)
    else:
        L -= 1
    nodes.append(n_output_nodes)

    W = []
    V = []
    for i in range(L):
        w = iniW(nodes[i+1], nodes[i])
        W.append(w)
        V.append(np.zeros_like(w))
    return (W, V)


# Initialize weights for one-layer
def iniW(next, prev):
    r = np.sqrt(6 / (next + prev))
    w = np.random.rand(next, prev)
    w = w * 2 * r - r
    return w


def sort_data_random(X, Y):
    sort_indices = np.argsort(X[0, :])
    return X[:, sort_indices], Y[:, sort_indices]

# Feed-forward of SNN
def forward(x, W, param):
    n_function = int(param[6])
    L = len(W)
    A = [0] * (L+1)
    Z = [0] * L
    A[0] = x

    for i in range(0, L):
        if i == L-1:
            f_act = 5
        else:
            f_act = n_function
        Z[i] = np.dot(W[i], A[i])
        A[i+1] = act_function(Z[i], f_act)

    return A[-1], Z

# Activation function
def act_function(Z, act_func):
    if act_func == 1:
        return np.maximum(0, Z)
    if act_func == 2:
        return np.maximum(0.01 * Z, Z)
    if act_func == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1)) * lam
    if act_func == 5:
        return 1 / (1 + np.exp(-Z))


# Feed-Backward of SNN
def gradW(X, AL, Z, Ye, W, params):
    N = Ye.shape[1]
    h_act_function = params[6]
    cost = np.sum((1 / (2 * N)) * ((AL - Ye) ** 2))
    L = len(Z)
    gW = [0]*L
    for i in reversed(range(1, L + 1)):
        if i == L:
            dA = (AL - Ye)
            dZ = dA * deriva_act(Z[L - 1], 5)
            gW[L-1] = np.dot(dZ, act_function(Z[i - 2], 5).T)
        else:
            dZ = dA * deriva_act(Z[i - 1], h_act_function)
            if i == 1:
                gW_i = np.dot(dZ, X.T)
            else:
                gW_i = np.dot(dZ, act_function(Z[i - 2], h_act_function).T)
            gW[i-1] = gW_i
        dA = np.dot(W[i - 1].T, dZ)

    return gW, cost

# Update W and V


def updWV_sgdm(W, V, gW, params):
    L = len(W)
    beta_coef = params[10]
    learning_rate = params[9]
    for i in range(L):
        V[i] = beta_coef * V[i] + learning_rate * gW[i]
        W[i] = W[i] - V[i]
    return V, W

# Measure


def metricas(Y, Y_predict):
    cm = confusion_matrix(Y, Y_predict)
    precision = np.nan_to_num(
        cm.diagonal() / cm.sum(axis=0), nan=0.0, posinf=1.0)
    recall = np.nan_to_num(cm.diagonal() / cm.sum(axis=1), nan=0.0, posinf=1.0)
    f_score = np.nan_to_num(
        2 * ((precision * recall) / (precision + recall)), nan=0.0, posinf=1.0)
    return cm, np.append(f_score, np.mean(f_score))

# Confusion matrix


def confusion_matrix(Y, Y_predict):
    num_classes = Y.shape[0]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    max_indices = np.argmax(Y_predict, axis=0)
    Y_pred = np.zeros_like(Y_predict)
    Y_pred[max_indices, np.arange(Y_predict.shape[1])] = 1
    for true_label in range(num_classes):
        for predicted_label in range(num_classes):
            confusion_matrix[true_label, predicted_label] = np.sum(
                (Y[true_label, :] == 1) & (Y_pred[predicted_label, :] == 1))
    return confusion_matrix
# -----------------------------------------------------------------------
# Activation function


def act_function(Z, act_func):
    if act_func == 1:
        return np.maximum(0, Z)
    if act_func == 2:
        return np.maximum(0.01 * Z, Z)
    if act_func == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1)) * lam
    if act_func == 5:
        return 1 / (1 + np.exp(-Z))

# Derivatives of the activation function


def deriva_act(A, act_func):
    if act_func == 1:
        return np.where(A >= 0, 1, 0)
    if act_func == 2:
        return np.where(A >= 0, 1, 0.01)
    if act_func == 3:
        return np.where(A >= 0, 1, 0.01 * np.exp(A))
    if act_func == 4:
        lam = 1.0507
        alpha = 1.6732
        return np.where(A > 0, 1, alpha * np.exp(A)) * lam
    if act_func == 5:
        s = act_function(A, act_func)
        return s * (1 - s)
