import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm, Fsc):
    np.savetxt('cmatriz.csv', cm, fmt="%d")
    np.savetxt('fscores.csv', Fsc, fmt="%1.25f")


def load_w():
    W = np.load('w_snn.npz', allow_pickle=True)
    w = W['W']
    return w


def load_data_test(param):
    n = int(param[0])
    data = np.genfromtxt('dtrn.csv', delimiter=',')
    x = np.array(data[:,:-n])
    y = np.array(data[:,-n:])
    return x,y
    

# Beginning ...
def main():
    param = ut.load_cnf()
    xv, yv = load_data_test(param)
    W = load_w()
    zv = ut.forward(xv, W, int(param[6]))
    cm, Fsc = ut.metricas(yv, zv[len(zv)-1])
    save_measure(cm, Fsc)


if __name__ == '__main__':
    main()
