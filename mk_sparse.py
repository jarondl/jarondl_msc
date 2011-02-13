#!/usr/bin/python
import scipy 
from scipy import sparse
def mk_banded_sparse(N,b):
    random_values = scipy.random.random((2*b-1,N))
    offsets = scipy.array(range(1-b,b))
    return sparse.dia_matrix((random_values,offsets), shape=(N,N))

