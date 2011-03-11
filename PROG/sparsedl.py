#!/usr/bin/python
import scipy 
from scipy import sparse
from scipy import linalg

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

def rho(W, rho0, t):
    return scipy.dot(linalg.expm2(W*t), rho0)	

