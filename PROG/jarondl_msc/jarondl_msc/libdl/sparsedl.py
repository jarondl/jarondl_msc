#!/usr/bin/python
"""  This module includes all of the functions that I use. To import use:
            import sparsedl

"""
from __future__ import division  # makes true division instead of integer division
import scipy
import numpy
import numpy as np
from numpy import sqrt, cos, pi, diagflat, log
from scipy import optimize, special, linalg
from scipy.sparse import spdiags
###from scipy.maxentropy import logsumexp 
from scipy.misc import logsumexp
import logging

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)
FLOAT_EPS = np.finfo(np.float).eps

#set up logging:
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
     
    
##############  Create and manipulate matrices



def create_sparse_matrix(N, rates, b=1):
    """  Creates a sparse matrix out of the rates.

    :param N: The size of the matrix will be NxN
    :param rates: Should be flat with N*b elements
    :param b: The bandwidth
    """

    sp = spdiags(rates.reshape([b, N]), range(1, b + 1), N, N)  # create the above the diagonal lines
    sp = sp + sp.transpose()
    sp = sp - spdiags(sp.sum(axis=0), 0, N, N)
    return sp


def create_sparse_matrix_periodic(N, rates, b = 1):
    """  Creates a sparse matrix, but with periodic boundary conditions
    """

    rerates = rates.reshape([b, N])
    sp = spdiags(rerates, range(1, b + 1), N, N)  # create the above the diagonal lines
    sp = sp + spdiags(rerates[::-1, ::-1], range(N-b, N), N, N)
    sp = sp + sp.transpose()
    sp = sp - spdiags(sp.sum(axis=0), 0, N, N)
    return sp


def permute_tri(mat):
    """  Radnomly permutes the upper triangle of mat, and then transposes it to the lower triangle
    """
    upper_triangle_indices = numpy.triu(numpy.ones(mat.shape), k=1).nonzero()
    new_upper = numpy.random.permutation(mat[upper_triangle_indices])
    retval = numpy.zeros(mat.shape)
    retval[upper_triangle_indices] = new_upper
    retval += retval.T
    return retval

def permute_diagonals(mat):
    """ Randomly permutes every diagonal of the upper triangle
    """
    retval = numpy.zeros(mat.shape)
    for k in range(1, mat.shape[0]):
        retval += diagflat(numpy.random.permutation( mat.diagonal(k)), k)
    retval += retval.T
    zero_sum(retval)
    return retval


def permute_rows(mat):
    """ Randomly permute every row of the upper triangle of mat, and then transposes it to the lower triangle.
    """
    mat_size = mat.shape[0]
    retval = numpy.zeros(mat.shape)
    for row in range(mat_size):
        retval[row,(row+1):] = numpy.random.permutation(mat[row,(row+1):])
    ### notice, only permutes columns!
    retval += retval.T
    return retval

def zero_sum(mat, tol=1E-12):
    """ Set the diagonals of matrix, so the sum of each row eqauls zero, with
            tolerance :param:`tol` .

        :param mat: A symmetric 2d array.
        :type mat: numpy.ndarray
        :param tol: the tolerance for non zero values.
        :returns: True if matrix was already zero sum, False otherwise.
    """
    row_sum = mat.sum(axis=1)
    if (numpy.max(numpy.abs(row_sum)) > tol):
        mat -= numpy.diagflat(row_sum)
        maxdev  = numpy.max((mat.sum(axis=0), mat.sum(axis=1)))
        if maxdev > tol:
            raise Exception("Failed to make sums zero, is the matrix symmetric?")
    return mat

def create_shift_matrix(N):
    """ Creates a N//2 shift matrix. The same as D^{N//2}
    """
    return np.eye(N,k=N//2) + np.eye(N,k=N//2-N)
#########  Create random arrays

def lognormal_construction(N, mu=0, sigma=1, **kwargs):
    """ Create log-normal distribution, with N elements, mu average and sigma width.
        The construction begins with a lineary spaced vector from 0 to 1. Then
        the inverse CDF of a normal distribution is applied to the vector.
        The result is permutated, and its piecewise exponent is returned.
    """
    uniform = scipy.linspace(0.0001, 0.9999, N)
    y = -scipy.sqrt(2) * special.erfcinv(2 * uniform)
    rescaled_y = mu + (sigma * y)
    perm_y = scipy.random.permutation(rescaled_y)
    return scipy.exp(perm_y)  # Element-wise exponentiation.


def resnet(W, b, periodic=False):
    """

    """
    N = W.shape[0]
    N_high = N-(b+1)
    N_low = b
    if periodic:
        N_high, N_low = N//2, 0
    I = numpy.zeros(N)
    I[[N_low, N_high]] =  [1, -1]
    #v = spsolve(W, I)
    #v= linalg.solve(W.todense(), I)
    invW = linalg.pinv(W)
    v = numpy.dot(invW, I)
    return (N_high- N_low)/(v[N_high] - v[N_low])

def avg_2d_resnet(W, dis, r_0, sampling=20):
    """
    """
    indices = numpy.arange(sampling) # the points are not sorted, so it is random.
    invW = linalg.pinv(W)
    N = W.shape[0]
    G = numpy.zeros(sampling)
    for n in indices:
        I = numpy.zeros(N)
        I[[0, n+1]] =  [-1, 1]
        v = numpy.dot(invW, I)
        G[n] = (v[0]-v[n+1])**(-1)
    sigma = G*log(dis[0,indices+1]/r_0)/pi
    return (sigma.sum()/sampling)/r_0**2


def rho(t, rho0, W, index=None):
    """ Calculate the probability vector at time t.

    :param t: The time
    :param rho0: Initial condition
    :param W: Transition matrix
    :type rho0: numpy 1d array
    :type W: numpy 2d array
    """
    if index == None:
        return scipy.dot(linalg.expm2(W * t), rho0)
    else:
        return scipy.dot(linalg.expm2(W * t), rho0)[index]


def analytic_alter(a, b, m):
    """ Returns the m's eigenvalue of the alternating a, b model. (m=>m/N)
    """
    return (a+b) - sqrt(a**2+b**2+2*a*b*cos(2*pi*m))



def strexp(x, a, b):
    """ The streched exponential function, :math:`e^{-(\\frac{x}{a})^b}`

    >>> strexp(1, 2, 2)
    0.77880078307140488
    """
    return numpy.exp(-(x / a) ** b)


def mean(x, y):
    """  Returns the mean of a ditribution with x as location and y as value
    """
    return numpy.sum(x * y, axis=0) / numpy.sum(y, axis=0)



def var(x, y):
    """  Returns the Variance of a ditribution with x as location and y as value

    >>> sparsedl.var([1, 2, 3, 4, 5, 6], [1, 1, 2, 1, 1, 2])
    2.9375
    """
    xarr = numpy.array(x)  # Make sure we have numpy arrays and not lists
    yarr = numpy.array(y)
    return mean(xarr ** 2, yarr) - mean(xarr, yarr) ** 2


def logavg(X, **kwargs):
    return np.exp(np.average(np.log(X), **kwargs))

def sparsity(mat):
    """  Calculate the sparsity parameters of a matrix
    """
    matsum =  mat.sum()
    avg = matsum / mat.size
    sqr_avg = (mat**2).sum() / mat.size
    s = avg**2 / sqr_avg
    p = ((mat > avg).sum())/mat.size
    #q = mat.flatten()[mat.size//2] / avg
    return s, p#, q

def rnn(mat):
    """  Find the average NN distance. The algorithm is to add infinty to the
         diagonal, and average over the minimum of each row.
    """
    N = mat.shape[0]
    infdiag = numpy.diagflat(numpy.ones(N)*numpy.inf)
    return numpy.mean((mat+infdiag).min(axis=0))


def surv(eigen_values, times):
    """ Calculate survival probability by the sum of exponents equation"""
    op = numpy.outer(eigen_values, times)
    exps = numpy.exp(op)
    return exps.sum(axis=0) / len(eigen_values)

def safe_surv(eigen_values, times):
    """ Calculate survival probability by the sum of exponents equation"""
    op = numpy.outer(eigen_values, times)
    return logsumexp(op) / len(eigen_values)



def sorted_eigh(matrix):
    """ """
    eigvals, eigvecs = linalg.eigh(matrix)
    sort_indices = numpy.argsort(eigvals)
    return eigvals[sort_indices[::-1]], eigvecs[:,sort_indices[::-1]]


def sorted_eigvalsh(matrix):
    """ """
    eigvals =  - linalg.eigvalsh(matrix)
    eigvals.sort()
    return -eigvals

def descrete_spatial_fourier2(k , points, values):
    """ """
    ikr = 1j*numpy.dot(k, points.T)
    f_e_ikr = values*numpy.exp(ikr)
    return f_e_ikr.sum()

def banded_ones(N, bandwidth):
    """ returns a NxN banded matrix of ones
    """
    return numpy.tri(N, k=bandwidth)*numpy.tri(N,k=bandwidth).T

def periodic_banded_ones(N, bandwidth, periodic=True):
    """ returns a NxN banded matrix of ones
    """
    if not periodic:
        return banded_ones(N, bandwidth)
    if (N <= bandwidth*2):
        warning("Bandwidth more than N!")
        return numpy.ones([N,N])
    return (numpy.tri(N, k=bandwidth)*numpy.tri(N,k=bandwidth).T +
           numpy.tri(N, k=(bandwidth-N)) + numpy.tri(N,k=(bandwidth-N)).T)
           
def pi_phasor(N):
    """ returns a matrix with 1 at N/2 diagonals,
        and -1 at the rest. e.g.:
         1, 1,-1,-1
         1, 1, 1,-1
        -1, 1, 1, 1
        -1,-1, 1, 1
        For a one dimensional network, it gives a pi phase boundary condition.
        """
    M = np.ones((N,N))
    return M - 2 * ( np.tri(N,k=-N//2) + np.tri(N,k=-N//2).T)
    
           
def boundary_phasor(N,phi):
    """ returns a matrix with 1 at N/2 diagonals,
         and exp(i*phi) at the rest. e.g.:
         1, 1,-1,-1
         1, 1, 1,-1
        -1, 1, 1, 1
        -1,-1, 1, 1
        For a one dimensional network, it gives a pi phase boundary condition.
        """
    if phi == 0 :
        """ Keep things real if possible"""
        return np.ones((N,N))
    M = np.ones((N,N))*(1+0j)
    #return M - 2 * ( np.tri(N,k=-N//2) + np.tri(N,k=-N//2).T)
    M += (np.exp(-phi*1j)-1)*np.tri(N,k=-N//2)
    M += (np.exp(phi*1j)-1)*np.tri(N,k=-N//2).T
    return M
    
def symmetric_sign_randomizer(N):
    """ returns a symmetric matrix with random 1 and -1 values """
    M = np.tril(np.random.random_integers(0,1,size=(N,N))*2-np.ones((N,N)),k=-1)
    return M + M.T
    
    
def window_avg_mtrx(N, win_size=4):
    """ Create the matrix neccesary for window averaging
    """
    M = np.tri(N, k=win_size//2)*(np.tri(N, k=win_size//2).T)
    return ( M / M.sum(axis=0)).T
    
def mode_center(modes):
    """ return the estimated center of the modes """
    #return np.average(np.arange(modes.shape[0]), axis=0, weights=abs(modes))
    x = np.tile(np.arange(modes.shape[0]), [modes.shape[1],1]).T
    return mean(x, abs(modes)**2)

