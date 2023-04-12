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

 frames = [frame]
 for i in range(level):
  hankels = []
  #se consigue las matriz de hankel para el primer frame y los c que llegan
  for f in frames:
    hankels.append(create_hankel_matrix(f,L))
  c = []

  for h in hankels:
    c_0,c_1 = decomposition_svd(h)
    c.append(c_0.reshape(-1))
    c.append(c_1.reshape(-1))
  frames = c.copy()  

 return c

def calculate_dyadic_component(H: np.ndarray):
  r = H.shape[0]     
  c = [H[0, 0], (H[1, 0] + H[0, 1]) / 2]     
  for i in range(1, r - 1):         
    c.append((H[i, i-1] + H[i-1, i]) / 2)     
  c.append((H[r-2, r-1] + H[r-1, r-2]) / 2)     
  c.append(H[1, r-1])     
  return np.array(c)[np.newaxis, :]



def decomposition_svd(h_matrix):
  U, s, vh = np.linalg.svd(h_matrix, full_matrices=False)

  M_0 =  s[0] * U[:, 0].reshape(-1, 1) * vh[0, :].reshape(1, -1)
  M_1 =  s[1] * U[:, 1].reshape(-1, 1) * vh[1, :].reshape(1, -1)
  
  c_0 = calculate_dyadic_component(M_0)
  c_1 = calculate_dyadic_component(M_1)
  
  return c_0,c_1



def create_hankel_matrix(frame:np.array, L=2):
  n= len(frame)
  K = n - L + 1
  H_frame = np.zeros((L, K))
  for i in range(L):
    for j in range(K):
      if i + j < n:
        H_frame[i][j] = frame[i + j]
  return H_frame

# Hankel's features
#X = columna de matriz de clase
def hankel_features(X,Param):
  n_frame = int(Param[1])
  l_frame = int(Param[2])
  level = int(Param[3]) #nivel de descomposición 

  frames = np.zeros((l_frame, n_frame))
 
  F = []
  for j in range(n_frame):
    start_idx = j * l_frame
    end_idx = start_idx + l_frame
    frames[:, j] = X[start_idx:end_idx] 
    c = hankel_svd(frames[:,j],level)

    # Compute entropy of spectral amplitudes
    # Verificar si esto está bien
    p = np.abs(np.fft.fft(c, axis=0)[:l_frame // 2 + 1, :]) ** 2
    p = p / np.sum(p, axis=0)
    Entropy_C = -np.sum(p * np.log2(p + 1e-10), axis=0)

    # Compute SVD 
    U, S, V = np.linalg.svd(c, full_matrices=False)
  
    # Concatenate features
    F.append([Entropy_C, S])
  
  
  return np.array(F).reshape(-1,2)


# Obtain j-th variables of the i-th class
def data_class(df_list,j,i):
  df = df_list[i]
  return df.iloc[:,j]

# Create Features from Data
#lista de matrices(clases)
def create_features(Dat_list,param):
  p = param[7]
  nbr_class = len(Dat_list)
  Y=[]
  X= np.array([])
  for i in range(nbr_class):
    nbr_variable = Dat_list[i].shape[1]
    datF = np.array([])
    for j in range( nbr_variable):
      Xj = data_class(Dat_list, j, i) # Retorna j-th variable de i-th class
      Fj = hankel_features(Xj, param)
      if j == 0:
        datF = Fj
      else:
        datF = np.concatenate((datF,Fj))
    
    label = binary_label(i)
    Y.append(label)
    if i == 0:
      X = datF
    else:
      X = np.concatenate((X,datF),axis=1)
  
 
  X = data_norm(X)
  
  dtrn, dtst = create_dtrn_dtst(X, Y, p) # p: denota porcentaje de training.
  return dtrn, dtst



def create_dtrn_dtst(X:np.array, Y, p):

  Y = np.array(Y).reshape(-1)
  print(X.shape)
  print(Y.shape)

  # Reordenar aleatoriamente las posiciones de la data
  np.random.shuffle(X)
  
  # Dividir la data X y data Y
  train_size = int(X.shape[0]*p)
  
  xe = X[:, :train_size]
  ye = Y[:train_size]

  xv = X[:,train_size:]
  yv = Y[train_size:]
  
  # Crear archivo de training csv
  dtrn = pd.DataFrame(np.hstack((xe, ye)))
  dtrn.to_csv('dtrain.csv', index=False,header=None)
  print(dtrn)
  # Crear archivo de testing csv
  dtst = pd.DataFrame(np.hstack((xv, yv)),)
  dtst.to_csv('dtest.csv', index=False,header=None)

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
      df = pd.read_csv(filepath,header= None)
      dataframes.append(df)

  # Combinar todos los dataframes en uno solo
  return dataframes

# Beginning 
def main():        
    Param           = ut.load_cnf() 
    Data            = load_data()	
    InputDat,OutDat = create_features(Data, Param)
    InputDat        = data_norm(InputDat)
    save_data(InputDat,OutDat)


if __name__ == '__main__':   
	 main()

