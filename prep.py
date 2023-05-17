import pandas as pd
import numpy as np
import utility as ut
import os


# Save Data from  Hankel's features
def save_data(X, Y, param):
    p = param[7]
    dtrn, dtst = create_dtrn_dtst(X, Y, p)  # p: denota porcentaje de training.
    np.savetxt("dtrn.csv", dtrn, delimiter=",")
    np.savetxt("dtst.csv", dtst, delimiter=",")

    return


# normalize data
def data_norm(X):
    a = 0.01
    b = 0.99
    for i in range(X.shape[1]):
        xmin = np.min(X[:, i])
        xmax = np.max(X[:, i])
        column_norm = (X[:, i] - xmin) / (xmax - xmin) * (b - a) + a
        if i == 0:
            x_norm = column_norm
        else:
            x_norm = np.column_stack((x_norm, column_norm))

    return x_norm


def normalize(x, a=0.01, b=0.99):
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        x = ((x - x_min) / (x_max - x_min)) * (b - a) + a
    else:
        x = a
    return x


def entropy_spectral(X):
    N = X.shape[0]
    Ix = int(np.sqrt(N))
    amplitudes = np.abs(np.fft.fft(X))
    amplitudes_norm = normalize(amplitudes)
    x_max = 1
    x_min = 0.01
    step_range = (x_max - x_min) / Ix
    entropy = 0
    for i in range(Ix):
        lower_bound = x_min + step_range * i
        upper_bound = lower_bound + step_range

        quantity = np.where(
            np.logical_and(amplitudes_norm >= lower_bound, amplitudes_norm < upper_bound)
        )[0].shape[0]

        if quantity != 0:
            prob = quantity / N
            entropy =+  prob * np.log2(prob)
            

    return -entropy



# Binary Label
def binary_label(i, n):
    label = [0] * n
    label[i] = 1
    return np.array(label)



def calculate_dyadic_component(H):
    a = np.concatenate((H[0], H[1, -1:]))
    b = np.concatenate((H[0, :1], H[1]))
    c = (a + b) / 2
    return c



def create_hankel_matrix(frame: np.array, L=2):
    n = len(frame)
    K = n - L + 1
    H_frame = np.zeros((L, K))
    for i in range(L):
        H_frame[i] = frame[i:i + K]
    return H_frame


# Hankel-SVD
def hankel_svd(frame, j):
    C = []
    Sc = []
    H_matrix = create_hankel_matrix(frame, 2)
    recursive_level(H_matrix, j, 1, C, Sc)
    return C, Sc


def recursive_level(H, max_level, level, C, S_components):
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    h_0 = s[0] * U[:, 0].reshape(-1, 1) * Vt[0, :].reshape(1, -1)
    h_1 = s[1] * U[:, 1].reshape(-1, 1) * Vt[1, :].reshape(1, -1)

    if max_level == level:
        c_1 = calculate_dyadic_component(h_0)
        c_2 = calculate_dyadic_component(h_1)
        C.append(c_1)
        C.append(c_2)
        S_components.append(s[0])
        S_components.append(s[1])
    else:
        recursive_level(h_0, max_level, level + 1, C, S_components)
        recursive_level(h_1, max_level, level + 1, C, S_components)


def hankel_features(X, Param):
    n_frame = int(Param[1])
    l_frame = int(Param[2])
    level = int(Param[3])  # nivel de descomposición
    frames = np.zeros((l_frame, n_frame))

    F = np.empty((n_frame, 2 ** (level + 1)))
    for j in range(n_frame):
        start_idx = j * l_frame
        end_idx = start_idx + l_frame
        frames = X[start_idx:end_idx]
        c, Sc = hankel_svd(frames, level)

        # Compute entropy of spectral amplitudes
        entropy_c = np.array([])
        for c_i in c:
            entropy_c_i = entropy_spectral(c_i)
            entropy_c = np.append(entropy_c, entropy_c_i)
        # Compute SVD
        F[j] = np.concatenate((entropy_c, Sc))
    return F


# Obtain j-th variables of the i-th class
def data_class(df_list, j, i):
    df = df_list[i]
    return df[:, j]


# Create Features from Data
# lista de matrices(clases)
def create_features(Dat_list, param):
    nbr_class = len(Dat_list)
    for i in range(nbr_class):
        nbr_variable = Dat_list[i].shape[1]
        datF = np.array([])
        for j in range(nbr_variable):
            # Retorna j-th variable de i-th class
            Xj = data_class(Dat_list, j, i)
            Fj = hankel_features(Xj, param)
            if j == 0:
                datF = Fj
            else:
                datF = np.concatenate((datF, Fj))

        label = binary_label(i, nbr_class)
        if i == 0:
            X = datF
            Y = label.reshape(1, -1)
        else:
            Y = np.vstack((Y, label.reshape(1, -1)))
            X = np.concatenate((X, datF), axis=0)
    return X, Y


def create_dtrn_dtst(X, binary_matrix, p):
    for i in range(binary_matrix.shape[1]):
        if i == 0:
            Y = np.tile(binary_matrix[:, i], 480)
        else:
            Y = np.vstack((Y, np.tile(binary_matrix[:, i], 480)))

    XY = np.concatenate((X.T, Y)).T
    # Reordenar aleatoriamente las posiciones de la data
    np.random.shuffle(XY)

    # Dividir la data X y data Y
    train_size = int(XY.shape[0]*p)

    dtrn = XY[:train_size, :]
    dtst = XY[train_size:, :]

    return dtrn, dtst

# Load data from ClassXX.csv


def load_data():
    # Directorio donde se encuentran los archivos
    dir_path = "Data/"

    # Obtener la lista de archivos en el directorio
    files = os.listdir(dir_path)

    # Filtrar los archivos que siguen el patrón "classX.csv"
    files = [f for f in files if f.startswith("class") and f.endswith(".csv")]

    # Leer cada archivo y almacenarlo en un dataframe
    dataframes = []
    for file in files:
        filepath = os.path.join(dir_path, file)
        df = np.loadtxt(filepath, delimiter=',')
        dataframes.append(df)

    return dataframes

# Beginning


def main():
    Param = ut.load_cnf()
    Data = load_data()
    InputDat, OutDat = create_features(Data, Param)
    InputDat = data_norm(InputDat)
    save_data(InputDat, OutDat, Param)


if __name__ == '__main__':
    main()
