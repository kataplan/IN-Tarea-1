# My Utility : auxiliars functions

import pandas as pd
import numpy as np

# load parameters to train the SNN


def load_cnf():
    cnf = np.genfromtxt("cnf.csv")
    return cnf


# Initialize weights for SNN-SGDM
def iniWs(input_dim, params):
    h_nodes_1 = int(params[4])
    h_nodes_2 = int(params[5])
    L_nodes = int(params[0])
    W = [iniW(h_nodes_1, input_dim)]
    V = [np.zeros((h_nodes_1, input_dim))]
    if h_nodes_2 > 0:
        W.append(iniW(h_nodes_2, h_nodes_1))
        V.append(np.zeros((h_nodes_2, h_nodes_1)))
        W.append(iniW(L_nodes, h_nodes_2))
        V.append(np.zeros((L_nodes, h_nodes_2)))
    else:
        W.append(iniW(L_nodes, h_nodes_1))
        V.append(np.zeros((L_nodes, h_nodes_1)))

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
def forward(Xe, W, params):
    h_act_function = params[6]
    h_nodes_2 = params[5]
    Z1 = np.dot(W[0], Xe)
    Z = [Z1]
    A1 = act_function(Z1, h_act_function)
    Z2 = np.dot(W[1], A1)
    Z.append(Z2)
    if h_nodes_2 == 0:
        return act_function(Z2, 5), Z
    A2 = act_function(Z2, int(h_act_function))
    Z3 = np.dot(W[2], A2)
    Z.append(Z3)
    return act_function(Z3, 5), Z

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

# Feed-Backward of SNN
def gradW(X, AL, Z, Ye, W, params):
    N = Ye.shape[1]
    h_act_function = params[6]
    cost = (1 / (2 * N)) * ((AL - Ye) ** 2)
    L = len(Z)
    gW = {}
    for i in reversed(range(1, L + 1)):
        if i == L:
            dA = (AL - Ye)
            dZ = dA * deriva_act(Z[L - 1], 5)
            gW[f"gW{L}"] = np.dot(dZ, act_function(Z[i - 2], 5).T)
        else:
            dZ = dA * deriva_act(Z[i - 1], h_act_function)
            if i == 1:
                gW_i = np.dot(dZ, X.T)
            else:
                gW_i = np.dot(dZ, act_function(Z[i - 2], h_act_function).T)
            gW[f"gW{i}"] = gW_i
        dA = np.dot(W[i - 1].T, dZ)
    return gW, cost

# Update W and V
def updWV_sgdm(W, V, gW, params):
    L = len(W)
    beta_coef = params[10]
    learning_rate = params[9]
    for i in range(1, L + 1):
        V[i - 1] = beta_coef * V[i - 1] + learning_rate * gW[f"gW{i}"]
        W[i - 1] = W[i - 1] - V[i - 1]
    return V, W

# Measure
def metricas(Y, Y_predict):
    cm = confusion_matrix(Y, Y_predict)
    print(cm.shape)
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