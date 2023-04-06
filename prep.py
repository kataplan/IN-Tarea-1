import pandas     as pd
import numpy      as np
import utility    as ut
import os

# Save Data from  Hankel's features
def save_data(X,Y):
  np.savetxt("dtrain.csv", X, delimiter=",", fmt="%f")
  np.savetxt("dtest.csv", Y, delimiter=",", fmt="%f")

  return

# normalize data 
def data_norm(X):
  # Calculamos la media de cada columna
  media = np.mean(X, axis=0)

  # Calculamos la desviación estándar de cada columna
  std = np.std(X, axis=0)

  # Normalizamos la matriz
  X_norm = (X - media) / std

  return X_norm


# Binary Label
def binary_label(i):
  Label = np.zeros((1, 2))
  Label[0, i - 1] = 1
  return Label

# Fourier spectral entropy
def entropy_spectral(S):

  return -np.sum(S*np.log(S), axis=1)

# Hankel-SVD
def hankel_svd(Hankel_matrix):
  return np.linalg.svd(Hankel_matrix)

# Hankel's features 
def hankel_features(X,Param):
  n_frame = int(Param[1])
  l_frame = int(Param[2])
  N = len(X)
  nOverlap = l_frame // 2
  nShift = l_frame - nOverlap
  nFFT = l_frame

  # Create Hankel matrix
  H = np.zeros((l_frame, n_frame))
  for j in range(n_frame):
      start_idx = j * nShift
      end_idx = start_idx + l_frame
      H[:, j] = X[start_idx:end_idx]

  # Compute SVD and truncate to 2J singular values
  U, S, V = np.linalg.svd(H, full_matrices=False)
  S = S[:2 * n_frame]

  # Compute entropy of spectral amplitudes
  p = np.abs(np.fft.fft(H, axis=0)[:nFFT // 2 + 1, :]) ** 2
  p = p / np.sum(p, axis=0)
  Entropy_C = -np.sum(p * np.log2(p + 1e-10), axis=0)

  # Concatenate features
  F = np.concatenate((Entropy_C, S))
  F = F.reshape(1, -1)

  return F


# Obtain j-th variables of the i-th class
def data_class(df_list,j,i):
  df = df_list[i]
  return df.iloc[:,j]

# Create Features from Data
#lista de matrices(clases)
def create_features(Dat_list,param):
  p = param[9]
  nbr_class = len(Dat_list)
  Y=[]
  X=[]
  for i in range( nbr_class):
    nbr_variable = Dat_list[i].shape[1]
    datF = []
    for j in range( nbr_variable):
        Xj = data_class(Dat_list, j, i) # Retorna j-th variable de i-th class
        Fj = hankel_features(Xj, param)
        datF.append(Fj)
    label = binary_label(i)
    Y = stack_label(label)
    X = stack_features(datF)
  print(X)
  X = data_norm(X)
  print(X)
  return create_dtrn_dtst(X, Y, p) # p: denota porcentaje de training.


def stack_label(Label):
  
  return np.array(Label).reshape(-1,1)

def stack_features(F):
    return np.array(F).reshape(1, -1)



def create_dtrn_dtst(X, Y, p):
  # Obtener la cantidad total de muestras
  N = X.shape[0]

  # Calcular el número de muestras de entrenamiento a partir del porcentaje dado
  n_train = int(np.round(p * N))

  # Generar índices aleatorios sin repetición para los conjuntos de entrenamiento y prueba
  idx_train = np.random.choice(np.arange(N), size=n_train, replace=False)
  idx_test = np.setdiff1d(np.arange(N), idx_train)

  # Dividir los datos de entrada en entrenamiento y prueba
  x_train, y_train = X[idx_train], Y[idx_train]
  x_test, y_test = X[idx_test], Y[idx_test]

  # Unir x_train y y_train
  dtrn = np.concatenate((x_train, y_train.reshape(-1,1)), axis=1)
  
  # Unir x_test y y_test
  dtst = np.concatenate((x_test, y_test.reshape(-1,1)), axis=1)

  return dtrn,dtst

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
      df = pd.read_csv(filepath)
      dataframes.append(df)

  # Combinar todos los dataframes en uno solo
  return dataframes

# Beginning 
def main():        
    Param           = ut.load_cnf()	
    Data            = load_data()	
    InputDat,OutDat = create_features(Data, Param)
    #InputDat        = data_norm(InputDat)
    save_data(InputDat,OutDat)


if __name__ == '__main__':   
	 main()

