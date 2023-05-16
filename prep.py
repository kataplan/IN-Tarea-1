import pandas as pd
import numpy as np
import utility as ut
import os


# Save Data from  Hankel's features
def save_data(X, Y, param):
    p = param[7]
    dtrn, dtst = create_dtrn_dtst(X, Y, p)  # p: denota porcentaje de training.

    np.savetxt("dtrn.csv", dtrn, delimiter=",", fmt="%f")
    np.savetxt("dtst.csv", dtst, delimiter=",", fmt="%f")

    return

# normalize data


def data_norm(X):
    xmin = np.min(X)
    xmax = np.max(X)
    a = 0.01
    b = 0.99
    datos_norm = (X - xmin) / (xmax - xmin) * (b - a) + a
    return datos_norm

# Binary Label


def binary_label(i, n):
    label = [0] * n
    label[i] = 1
    return label


# Hankel-SVD
def hankel_svd(frame, level):
    L = 2  # matriz diadica

    frames = [frame]
    for i in range(level):
        hankels = []
        # se consigue las matriz de hankel para el primer frame y los c que llegan
        for f in frames:
            h = create_hankel_matrix(f, L)
            hankels.append(h)

        c = []
        for h in hankels:

            c_0, c_1 = decomposition_svd(h)
            c.append(c_0.reshape(-1))
            c.append(c_1.reshape(-1))

        frames = c.copy()
    return c


def calculate_dyadic_component(H: np.ndarray):
    c = []
    j = 0
    k = 1
    r = H.shape[1]
    for i in range(1, r):
        component = (H[j, i] + H[k, i-1]) / 2
        c.append(component)
    c.insert(0, H[0, 0])
    c.append(H[1, r-1])
    return np.array(c).reshape(1, -1)


def decomposition_svd(h_matrix):
    U, s, vh = np.linalg.svd(h_matrix, full_matrices=False)

    M_0 = s[0] * U[:, 0].reshape(-1, 1) * vh[0, :].reshape(1, -1)
    M_1 = s[1] * U[:, 1].reshape(-1, 1) * vh[1, :].reshape(1, -1)

    c_0 = calculate_dyadic_component(M_0)
    c_1 = calculate_dyadic_component(M_1)

    return c_0, c_1


def create_hankel_matrix(frame: np.array, L=2):
    n = len(frame)

    K = n - L + 1
    H_frame = np.zeros((L, K))
    for i in range(L):
        for j in range(K):
            if i + j < n:
                H_frame[i][j] = frame[i + j]
    return H_frame


def elements_in_range(x, lower_bound, upper_bound):
    cont = 0
    for num in x:
        if lower_bound <= num <= upper_bound:
            cont += 1

    return cont


def entropy_spectral(X):
    amplitudes = np.abs(np.fft.fft(X))

    I_x = int(np.sqrt(len(amplitudes)))
    x_max = np.max(amplitudes)
    x_min = np.min(amplitudes)
    x_range = x_max - x_min

    entropy = 0
    prob = 0

    step_range = x_range / I_x
    lower_bound = x_min

    for _ in range(I_x):
        upper_bound = lower_bound + step_range
        prob = 0 if elements_in_range(amplitudes, lower_bound, upper_bound) == 0 else elements_in_range(
            amplitudes, lower_bound, upper_bound) / len(amplitudes)
        entropy += prob*np.log2(prob)

    return -entropy


def hankel_features(X, Param):
    n_frame = int(Param[1])
    l_frame = int(Param[2])
    level = int(Param[3])  # nivel de descomposición
    frames = np.zeros((l_frame, n_frame))

    F = []
    for j in range(n_frame):
        start_idx = j * l_frame
        end_idx = start_idx + l_frame
        frames[:, j] = X[start_idx:end_idx]

        c = hankel_svd(frames[:, j], level)

        # Verificar si esto está bien
        # Compute entropy of spectral amplitudes
        entropy_c = np.array([])
        for c_i in c:
            entropy_c_i = entropy_spectral(c_i)
            entropy_c = np.append(entropy_c, entropy_c_i)
        # Compute SVD
        U, S, V = np.linalg.svd(c, full_matrices=False)
        # Concatenate features
        F.append([entropy_c, S])

    return np.array(F).reshape(-1, 2*2**level)


# Obtain j-th variables of the i-th class
def data_class(df_list, j, i):
    df = df_list[i]
    return df.iloc[:, j]

# Create Features from Data
# lista de matrices(clases)


def create_features(Dat_list, param):
    nbr_class = len(Dat_list)
    Y = []
    X = []
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
        Y.append(label)
        X.append(datF)

    return X, Y


def add_array_rows(X, Y):
    Y_rep = np.tile(Y, (X.shape[0], 1))  # replicar Y n veces
    return np.hstack((X, Y_rep))


def create_dtrn_dtst(X_list: np.array, Y_list, p):

    XY = np.array([])
    len(Y_list[1])
    for i in range(len(Y_list[1])):
        if (i == 0):
            XY = add_array_rows(X_list[i], Y_list[i])
            continue
        XY = np.concatenate((XY, add_array_rows(X_list[i], Y_list[i])))

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
        df = pd.read_csv(filepath, header=None)
        dataframes.append(df)

    # Combinar todos los dataframes en uno solo
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
