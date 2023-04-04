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
def create_features(X,Param):
    
  return() 


# Load data from ClassXX.csv
def load_data():
  main_dir = os.listdir()
  # Filtrar las caroetas que siguen el patrón "DataX"
  dirs = [d for d in main_dir if d.startswith("Data")] 
  
  for dir in dirs:
    # Obtener la lista de archivos en el directorio
    files = os.listdir(dir)

    # Filtrar los archivos que siguen el patrón "classX.csv"
    files = [f for f in files if f.startswith("class") and f.endswith(".csv")]

    # Leer cada archivo y almacenarlo en un dataframe
    df_list = []
    for file in files:
        filepath = os.path.join(dir, file)
        df = pd.read_csv(filepath,header=None)
        df_list.append(df)
    print(df_list)
  
  return df_list 

# Parameters for pre-proc.
def load_cnf():
    cnf = pd.read_csv('cnf.csv')
    print(cnf)
    return cnf

# Beginning 
def main():        
    Param           = ut.load_cnf()	
    Data            = load_data()	
    InputDat,OutDat = create_features(Data, Param)
    InputDat        = data_norm(InputDat)
    save_data(InputDat,OutDat)


if __name__ == '__main__':   
	 main()

