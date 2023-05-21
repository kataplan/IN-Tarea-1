# SNN's Training :

import pandas as pd
import numpy as np
import utility as ut


# Save weights and MSE  of the SNN
def save_w_cost(W, cost):
    np.savez("w_snn.npz", *W)
    np.savetxt("costo.csv", cost, fmt="%.10f")


def get_Idx_n_Batch(batch_size, i):
    return np.arange(i * batch_size, i * batch_size + batch_size)

def trn_minibatch(X, Y, W, V, params):
    batch_size = int(params[8])
    nbatch = int(X.shape[1] / batch_size)
    for i in range(nbatch):
        idx = get_Idx_n_Batch(batch_size, i)
        Xe = X[:, idx]
        Ye = Y[:, idx]
        AL, Z = ut.forward(Xe, W, params)
        gW, cost = ut.gradW(Xe, AL, Z, Ye, W, params)
        V, W = ut.updWV_sgdm(W, V, gW, params)
    return cost, W, V

def train(X, Y, params):
    W, V = ut.iniWs(X.shape[0], params)
    max_iter = int(params[11])
    mse = []
    for i in range(max_iter):
        X, Y = ut.sort_data_random(X, Y)
        cost, W, V = trn_minibatch(X, Y, W, V, params)
        mse.append(np.mean(cost))
        if i % 10 == 0:
            print(f"Iterar-SGD: {i}, {mse[i]}")
    return W, np.array(mse)

# Load data to train the SNN
def load_data_trn():
    X = np.loadtxt("X_train.csv", delimiter=",")
    Y = np.loadtxt("Y_train.csv", delimiter=",")
    return X, Y


def main():
    param = ut.load_cnf()
    X, Y = load_data_trn()
    W, Cost = train(X.T, Y.T, param)
    save_w_cost(W, Cost)

if __name__ == '__main__':
    main()