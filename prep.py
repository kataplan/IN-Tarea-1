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
  xmin = np.min(X)
  xmax = np.max(X)
  a = 0.01
  b = 0.99
  datos_norm = (X - xmin) / (xmax - xmin) * (b - a) + a
  return datos_norm

# Binary Label
def binary_label(i):
  Label = np.zeros((1, 2))
  Label[0, i - 1] = 1
  return Label

# Fourier spectral entropy
def entropy_spectral(S):

  return -np.sum(S*np.log(S), axis=1)

# Hankel-SVD
def hankel_svd(frame, level):
 L = 2 # matriz diadica
 hankels = []
 frames = [frame]
 for i in range(level):
    for f in frames:
      hankels.append(create_h_matrix(f,L))
    frames = []
    c=[]
    for h in hankels:
       c_0,c_1 = descomposition_svd(h)
       h_0 = create_h_matrix(c_0)
       h_1 = create_h_matrix(c_1)
       frames.append(h_0,h_1)
       c.append(c_0)
       c.append(c_1)
 return

def calculate_dyadic_component(H: np.ndarray):
  c = []
  j = 0
  k = 1
  r = H.shape[1]
  for i in range(1,r):
      component = (H[j,i] + H[k,i-1]) / 2
      c.append(component)
  c.insert(0, H[0,0])
  c.append(H[1,r-1])
  return np.array(c).reshape(1,-1)

def descomposition_svd(h_matrix):
  U, S, V = np.linalg(h_matrix)
  M_0 = S[:,0]*U[:,0]*V[:,0]
  M_1 = S[:,1]*U[:,1]*V[:,1]
  c_0 = calculate_dyadic_component(M_0)
  c_1 = calculate_dyadic_component(M_1)

  return c_0,c_1

def create_h_matrix(matriz,l):
  n = matriz.shape[1]
  h_0 = matriz[0,:-1].reshape(1,n-l+1)
  h_1 = matriz[0,1:].reshape(1,n-l+1)
  h_matrix = np.concatenate((h_0,h_1),axis=0).reshape(1,n-l+1)
  return h_matrix

# Hankel's features
#X = columna de matriz de clase
def hankel_features(X,Param):
  n_frame = int(Param[1])
  l_frame = int(Param[2])
  level = int(Param[3]) #nivel de descomposición 

  # Create Hankel matrix
  H = np.zeros((l_frame, n_frame))
  for j in range(n_frame):
      start_idx = j * l_frame
      end_idx = start_idx + l_frame
      H[:, j] = X[start_idx:end_idx]
      c = hankel_svd(H,level)

  print(H.shape)
  # Compute SVD and truncate to 2J singular values
  U, S, V = np.linalg.svd(H, full_matrices=False)
  S = S[:2 * n_frame]

  # Compute entropy of spectral amplitudes
  p = np.abs(np.fft.fft(H, axis=0)[:l_frame // 2 + 1, :]) ** 2
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
  for i in range(nbr_class):
    nbr_variable = Dat_list[i].shape[1]
    datF = []
    for j in range( nbr_variable):
        Xj = data_class(Dat_list, j, i) # Retorna j-th variable de i-th class
        Fj = hankel_features(Xj, param)
        datF.append(Fj)
    label = binary_label(i)
    Y.append(label)
    X.append(datF)
 
  #X = data_norm(X)
  dtrn, dtst = create_dtrn_dtst(X, Y, p) # p: denota porcentaje de training.
  return dtrn, dtst

def matrix_to_dataframe(X):
  print(X.shape)
  num_rows, num_cols = X.shape
  df = pd.DataFrame(columns=range(num_cols))
  for i in range(num_cols):
      df[i] = X[:, i]
  return df

def create_dtrn_dtst(X, Y, p):
  df = matrix_to_dataframe(X)
  print(len(Y))
  df['Y'] = Y
  n_train = int(len(df) * p)
  df_train = df[:n_train]
  df_test = df[n_train:]
  dtrn = df_train.to_numpy()
  dtst = df_test.to_numpy()
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
    print(InputDat)
    save_data(InputDat,OutDat)


if __name__ == '__main__':   
	 main()

