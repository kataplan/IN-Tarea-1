import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm, Fsc):
    np.savetxt('cmatriz.csv', cm, fmt="%d")
    np.savetxt('fscores.csv', Fsc, fmt="%1.25f")


def load_w():
    pesos = np.load('w_snn.npz', allow_pickle=True)
    W =[]
    for i in range(len(pesos)):
        W.append(pesos[f'arr_{i}'])
    return W



def load_data_test(param):
    n = int(param[0])
    data = np.genfromtxt('dtst.csv', delimiter=',')
    x = np.array(data[:,:-n])
    y = np.array(data[:,-n:])
    return x,y
    

# Beginning ...
def main():
    param = ut.load_cnf()
    xv, yv = load_data_test(param)
    W = load_w()
    zv,_ = ut.forward(xv, W, int(param[6]))
    
    print("yvshape", yv.T.shape)
    
    print("shape.zv", zv.shape)
    cm, Fsc = ut.metricas(yv.T, zv)
    save_measure(cm, Fsc)


if __name__ == '__main__':
    main()
