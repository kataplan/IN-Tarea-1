import numpy as np
import utility as ut
import os


def save_data(x, y, p):
    X_train, Y_train, X_test, Y_test = create_dtrn_dtst(x, y, p)
    np.savetxt("xe.csv", X_train, delimiter=",", fmt="%.10f")
    np.savetxt("ye.csv", Y_train, delimiter=",", fmt="%.10f")
    np.savetxt("xv.csv", X_test, delimiter=",", fmt="%.10f")
    np.savetxt("yv.csv", Y_test, delimiter=",", fmt="%.10f")


def create_dtrn_dtst(X, Y, p):
    M = np.concatenate((X, Y), axis=1)
    np.random.shuffle(M)
    split_index = int(M.shape[0] * p)
    trn_set, test_set = M[:split_index, :], M[split_index:, :]
    X_train, Y_train = trn_set[:, :X.shape[1]], trn_set[:, -Y.shape[1]:]
    X_test, Y_test = test_set[:, :X.shape[1]], test_set[:, -Y.shape[1]:]
    return X_train, Y_train, X_test, Y_test


def data_norm(X):
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    a = 0.01
    b = 0.99
    return ((X - x_min) * (1 / (x_max - x_min))) * (b - a) + a


def binary_label(i, m, n):
    binary_array = np.zeros((m, n))
    binary_array[:, i] = 1
    return binary_array


def entropy_spectral(components):
    amplitudes = []
    entropies = []

    for c in components:
        dft = np.fft.fft(c)
        amplitude = np.abs(dft[:, :int(dft.shape[1] / 2)])
        amplitudes.append(amplitude)

    for amplitude in amplitudes:
        a_min = np.min(amplitude)
        a_max = np.max(amplitude)
        a_range = a_max - a_min
        partitions = int(np.trunc(np.sqrt(amplitude.shape[1])))
        step_range = a_range / partitions
        probabilities = []
        for i in range(partitions):
            i_min = a_min + step_range * i
            i_max = a_min + step_range * (i + 1)
            probablity = len(amplitude[(amplitude >= i_min) & (
                amplitude <= i_max)]) / amplitude.shape[1]
            probabilities.append(probablity)
        p = np.array(probabilities)
        nonzero_prob = p[p != 0]
        entropy = -np.sum(nonzero_prob * np.log2(nonzero_prob))
        entropies.append(entropy)

    return np.array(entropies)


def create_hankel_matrix(frame, l):
    n = frame.shape[1]
    k = n - l + 1
    h_0 = frame[0, :-1].reshape(1, k)
    h_1 = frame[0, 1:].reshape(1, k)
    return np.concatenate((h_0, h_1), axis=0).reshape(l, k)


def calculate_dyadic_component(hankel):
    c = []
    j = 0
    k = 1
    r = hankel.shape[1]
    for i in range(1, r):
        component = (hankel[j, i] + hankel[k, i - 1]) / 2
        c.append(component)
    c.insert(0, hankel[0, 0])
    c.append(hankel[1, r - 1])
    return np.array(c).reshape(1, -1)


def hankel_svd(frame, levels):
    l = 2
    components = []
    hankels = []
    frames = [frame]

    for i in range(levels):
        for f in frames:
            hankel = create_hankel_matrix(f, l)
            hankels.append(hankel)
        c = []
        frames = []
        for h in hankels:
            u, s, vh = np.linalg.svd(h, full_matrices=False)
            vT = vh.T
            h_0 = s[0] * u[:, 0].reshape(-1, 1) * vT[:, 0].reshape(-1, 1).T
            c_0 = calculate_dyadic_component(h_0)
            h_1 = s[1] * u[:, 1].reshape(-1, 1) * vT[:, 1].reshape(-1, 1).T
            c_1 = calculate_dyadic_component(h_1)
            frames.extend((c_0, c_1))
            c.extend((c_0, c_1))
        components.append(c)
        hankels = []
    return components


def create_features(X, Param):
    features = []
    labels = []
    total_classes = len(X)
    for i, value in enumerate(X):
        data_f = []
        for j in range(value.shape[1]):
            X = data_class(value, j)
            F = hankel_features(X, Param)
            data_f.append(F)
        y_shape = np.concatenate(data_f).shape[0]
        features.append(np.concatenate(data_f))
        label = binary_label(i, y_shape, total_classes)
        labels.append(label)
    X = np.concatenate(features)
    Y = np.concatenate(labels)
    return X, Y


def hankel_features(X, param):
    n_frame = int(param[1])
    frame_size = int(param[2])
    decomp_level = int(param[3])
    features = []

    for i in range(n_frame):
        frame = X[:, i * frame_size:(i * frame_size) + frame_size]
        c = hankel_svd(frame, decomp_level)
        entropies = entropy_spectral(c[decomp_level-1])
        components_matrix = np.concatenate(c[decomp_level-1])
        S = np.linalg.svd(components_matrix, compute_uv=False)
        f = np.concatenate((entropies, S)).reshape(1, -1)
        features.append(f)

    return np.concatenate(features)


def data_class(x, j):
    return x[:, j].reshape(1, -1)


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


def load_cnf():
    return ut.load_cnf()


def main():
    Param = load_cnf()
    Data = load_data()
    InputDat, OutDat = create_features(Data, Param)
    InputDat = data_norm(InputDat)
    save_data(InputDat, OutDat, Param[7])


if __name__ == '__main__':
    main()
