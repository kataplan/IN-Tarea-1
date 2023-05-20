# SNN's Training :

import pandas as pd
import numpy as np
import utility as ut


# Save weights and MSE  of the SNN
def save_w_mse():
    ...
    return


# miniBatch-SGDM's Training
def trn_minibatch(x, y, W, V, param):
    M = int(param[8])
    N = x.shape[0]
    nBatch = (N // M)
    act_f = int(param[6])

    # Training loop
    for n in range(1, nBatch+1):
        Idx = get_Idx_n_Batch(n, M)
        xe = x[Idx, :]
        ye = y[Idx, :]
        act, z = ut.forward(xe, W, act_f)
        gW, cost = ut.gradW(act,xe, ye, z, W, param)
        W, V = ut.updWV_sgdm(W, V, gW, param)

    return cost, W, V


# SNN's Training
def train(x, y, param):

    n_input_nodes = x.shape[1]
    n_output_nodes = y[0].shape[0]
    n_hidden_1 = int(param[4])
    n_hidden_2 = int(param[5])
    iter = int(param[11])
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
        L -=1
    nodes.append(n_output_nodes)
    W, V = ut.iniWs(L, nodes)
    MSE = []

    for i in range(iter):
        x, y = sort_data_ramdom(x, y)
        cost, W, V = trn_minibatch(x, y, W, V, param)
        MSE.append(np.mean(cost))
        if (i % 10) == 0:
            print("Iterar-SGD:", i, MSE[i])
    return W, MSE


# Function to get the indices of the n-th batch
def get_Idx_n_Batch(n, M):
    start = (n-1)*M
    end = n*M
    return np.arange(start, end)


def sort_data_ramdom(X, Y):
    XY = np.concatenate((X, Y), axis=1)
    np.random.shuffle(XY)
    X_new, Y_new = np.split(XY, [X.shape[1]], axis=1)
    return X_new, Y_new


# Load data to train the SNN
def load_data_trn(param):
    n = int(param[0])
    data = np.genfromtxt('dtrn.csv', delimiter=',')
    x = data[:, :-n]
    y = data[:, -n:]
    return (x, y)


def save_w_cost(W, Cost):
    np.savez('w_snn.npz', *W)
    np.savetxt("costo.csv", Cost, delimiter=",", fmt="%.10f")
    return


# Beginning ...
def main():
    param = ut.load_cnf()
    xe, ye = load_data_trn(param)
    W, Cost = train(xe, ye, param)
    save_w_cost(W, Cost)


if __name__ == '__main__':
    main()
