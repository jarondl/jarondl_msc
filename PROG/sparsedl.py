#!/usr/bin/python
"""  This module includes all of the functions that I use. It can be imported in two ways:
        The recomended way:
            import sparsedl
        or the other way:
            from sparsedl import *
"""
import scipy 
import numpy
from scipy import linalg, optimize, special

def NN_matrix(N,hop_values=None,**kwargs):
    """ Build a NxN matrix.
        Input: - N :  the size of the matrix
               - hop_values: the omegas. If none are supplied, creates them with lognormal distribution
               - **kwargs : the rest of the keyworded arguments are passed to the lognormal function,
                                so you can define sigma and mu.
          log normal construction (e.g. mu=0, sigma=7)."""
    if hop_values==None: 
        hop_values = lognormal_construction(N-1,**kwargs)
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

def lognormal_construction(N, mu=0, sigma=1):
    """ Create log-normal distribution, with N elements, mu average and sigma width.
        The construction begins with a lineary spaced vector from 0 to 1. Then
          the inverse CDF of a normal distribution is applied to the vector. 
        The result is permutated, and its piecewise exponent is returned."""
    uniform = scipy.linspace(0.0001,0.9999,N)
    y = -scipy.sqrt(2)*special.erfcinv(2*uniform)
    rescaled_y = mu+sigma*y
    perm_y = scipy.random.permutation(rescaled_y)
    return scipy.exp(perm_y)

def surv(eigen_values, times):
    """ Calcultae survival probability by the sum of exponents equation"""
    op = numpy.outer(eigen_values,times)
    exps = numpy.exp(op)
    return exps.sum(axis=0) / len(eigen_values)

    
    

