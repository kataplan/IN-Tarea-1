# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  
#load parameters to train the SNN
def load_cnf():
    cnf = np.genfromtxt("cnf.csv")
    return cnf


# Initialize weights for SNN-SGDM
def iniWs(L,nodes):    
    W = []
    for i in range(L):
        W.append(iniW(nodes[i],nodes[i+1]))
    return(W)

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward():
        
    return() 

#Activation function
def forward(x,W,n_function):        
    a = x
    A = [a]
    for w in W:
        a_i =( w * act_function(w,a,n_function) ) 
        A.append(a_i)
        a = a_i
    return A
# Derivatives of the activation funciton
def deriva_act():
    
    return()

#Feed-Backward of SNN
def gradW():    
        
    return()    

# Update W and V
def updWV_sgdm():
        
    return()

# Measure
def metricas(x,y):
        
    return()
    
#Confusion matrix
def confusion_matrix(z,y):
        
    return(cm)

#Activation function
def act_function(w,X,function_number):
    z = np.dot(w, X.T)
    if(function_number==1):
        h_z = ReLu_function(z).T
    if(function_number==2):
        h_z = L_ReLu_function(z).T
    if(function_number==3):
        h_z = ELU_function(z).T
    if(function_number==4):
        h_z = SELU_function(z).T
    if(function_number==5):
        h_z = sigmoidal_function(z).T
    return(h_z)

def output_activation(v,h):
    z = np.dot(v, h.T)
    y = 1/(1+np.exp(-z))
    return y.T

def ReLu_function(x):
    return np.where(x>0,x,0)

def L_ReLu_function(x):
    return np.where(x<0,0.01*x,x)

def ELU_function(x):
    a = 1.6732
    return np.where(x>0,x,a*(np.exp(x)-1))

def SELU_function(x):
    a = 1.6732
    lam =1.0507
    return np.where(x>0,x*lam,a*(np.exp(x)-1))
      
def sigmoidal_function(z):
    return 1.0/(1.0+np.exp(-z))

def d_ReLu_function(x):
    return np.maximum(0,x)

def d_L_ReLu_function(x):
    return np.where(x<0,0.01*x,x)

def d_ELU_function(x):
    a = 1.6732
    return np.where(x>0,1, a*np.exp(x))

def d_SELU_function(x):
    lam = 1.0507; 
    a = 1.6732
    return np.where(x>0, 1, a*np.exp(x))*lam
      
def d_sigmoidal_function(z):
    return (np.multiply(1/(1+np.exp(-z)),1-(1/(1+np.exp(-z))))).T
#-----------------------------------------------------------------------
