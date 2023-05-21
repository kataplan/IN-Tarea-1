import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm, Fsc):
    np.savetxt('cmatriz.csv', cm, fmt="%d")
    np.savetxt('fscores.csv', Fsc, fmt="%1.25f")


def load_w():
    pesos = np.load('w_snn.npz', allow_pickle=True)
    W = []
    for i in range(len(pesos)):
        W.append(pesos[f'arr_{i}'])
    return W


def load_data_test():
    x = np.loadtxt("xv.csv", delimiter=",")
    y = np.loadtxt("yv.csv", delimiter=",")
    return x, y


# Beginning ...
def main():
    param = ut.load_cnf()
    xe, ye = load_data_test()
    W = load_w()
    Y_pred, _ = ut.forward(xe.T, W, param)
    cm, Fsc = ut.metricas(ye.T, Y_pred)
    save_measure(cm, Fsc)


if __name__ == '__main__':
    main()
