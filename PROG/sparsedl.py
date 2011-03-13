#!/usr/bin/python
import scipy 
from scipy import linalg, optimize

def NN_matrix(N, random_function = scipy.random.lognormal):
    hop_values = random_function(size=(N-1))
    diag_values = scipy.zeros(N)
    diag_values[1:] -= hop_values
    diag_values[:-1] -= hop_values
    return scipy.diagflat(diag_values) +  scipy.diagflat(hop_values,-1) +  scipy.diagflat(hop_values,1)
    

def rho(t,rho0,W, index=None):
    if index==None:
        return scipy.dot(linalg.expm2(W*t), rho0)	
    else:
        return scipy.dot(linalg.expm2(W*t), rho0)[index]

#based on scipy 0.9's minpack
def _general_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata

def cvfit( f, xdata, ydata, p0):
    """  Fit a curve. p0 is the initial guess (put ones if uncertain),
     f should accept len(p0)+1 arguments. 
    """
    res = optimize.leastsq(_general_function,p0,args=(xdata,ydata,f))
    return res[0]
    
def strexp(x,a,b):
    return scipy.exp(-(x/a)**b)

def mean(x,y):
    """  Returns the mean of a ditribution with x as location and y as value """
    return scipy.sum(x*y)/scipy.sum(y)

def var(x,y):
    """  Returns the Variance of a ditribution with x as location and y as value """
    return mean(x**2,y) - mean(x,y)**2

def initial(nodes):
    """  Some convient initial values, usefull for an interactive session. 
        e.g   >>>  rho0, A, xcoord = initial(200)  """
    rho0 = scipy.zeros(nodes)
    rho0[nodes//2] =1
    A = NN_matrix(nodes)
    xcoord = scipy.linspace(-nodes/2,nodes/2-1,nodes)
    return rho0,A,xcoord

    
    

