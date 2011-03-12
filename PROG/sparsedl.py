#!/usr/bin/python
import scipy 
from scipy import sparse
from scipy import linalg, optimize

def mk_banded_sparse(N,b):
    random_values = scipy.random.random((2*b-1,N))
    offsets = scipy.array(range(1-b,b))
    return sparse.dia_matrix((random_values,offsets), shape=(N,N))

def NN_sparse(N):
    A = scipy.zeros((3,N))
    A[0,:-1] = scipy.random.lognormal(size=(N-1))
    A[2,:] = A[0,:]
    A[1,:] = -A[0,:] 
    A[1,1:] -= A[0,:-1]
    offsets = scipy.array((-1,0,1))
    return sparse.dia_matrix((A,offsets), shape=(N,N))

def NN_dense(N):
    hop_values = scipy.random.lognormal(size=(N-1))
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
    return sum(x*y)/sum(y)

def var(x,y):
    mu = mean(x,y)
    return sum((x-mu)**2*y)/sum(y)
    
    

