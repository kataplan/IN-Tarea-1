import pandas     as pd
import numpy      as np
import utility    as ut
import os

# Save Data from  Hankel's features
def save_data(X,Y):
      
  return

# normalize data 
def data_norm():
  
  return 

# Binary Label
def binary_label():
  
  return


# Fourier spectral entropy
def entropy_spectral():
  
  return 

# Hankel-SVD
def hankel_svd():
      
  return() 

# Hankel's features 
def hankel_features(X,Param):
      
  return() 


# Obtain j-th variables of the i-th class
def data_class(x,j,i):
    
  return() 


# Create Features from Data
#lista de matrices(clases)
def create_features(X,Param):
  n_frame = Param[1]
  l_frame = Param[2]

  print(n_frame)
  
  return() 


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
    InputDat        = data_norm(InputDat)
    save_data(InputDat,OutDat)


if __name__ == '__main__':   
	 main()

