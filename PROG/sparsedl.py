#!/usr/bin/python
"""  This module includes all of the functions that I use. To import use:
            import sparsedl

"""
from __future__ import division  # makes true division instead of integer division
import scipy
import numpy
from numpy import sqrt, cos, pi, diagflat, log
from scipy import optimize, special, linalg
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.maxentropy import logsumexp

### Raise all float errors 
numpy.seterr(all='warn')

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
    upper = numpy.triu(mat, k=1)
    mat_size = mat.shape[0]
    retval = numpy.zeros(mat.shape)
    for row in range(mat_size):
        retval[row,(row+1):] = numpy.random.permutation(mat[row,(row+1):])
    ### notice, only permutes columns!
    retval += retval.T
    return retval

def zero_sum(mat, tol=1E-12):
    """  Set the diagonals of matrix, so the sum of each row eqauls zero, with
            tolerance :param:`tol`.

        :param mat: A symmetric 2d array.
        :type mat: numpy.ndarray
        :param tol: the tolerance for non zero values.
        :returns: True if matrix was already zero sum, False otherwise.
    """
    row_sum = mat.sum(axis=1)
    if (numpy.max(numpy.abs(row_sum)) < tol):
        return True
    else:
        mat -= numpy.diagflat(row_sum)
        maxdev  = numpy.max((mat.sum(axis=0), mat.sum(axis=1)))
        if numpy.max((mat.sum(axis=0), mat.sum(axis=1))) > tol:
            print(mat)
            print(mat-mat.T)
            print(numpy.max((mat.sum(axis=0), mat.sum(axis=1))))
        #    raise Exception("Failed to make sums zero, is the matrix symmetric?")
        return False

def new_zero_sum(mat):
    """ Same as zero_sum, up to a diagonal constant
    """
    N = mat.shape[0] # the matrix is NxN
    row_sum = mat.sum(axis=1)
    row_avg = row_sum.sum() / N
    mat -= (numpy.diagflat(row_sum) - numpy.eye(N)*row_avg)
    return row_avg
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


def resnet(W, b):
    """

    """
    N = W.shape[0]
    N_high = N-(b+1)
    N_low = b
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


def _general_function(params, xdata, ydata, function):
    #based on scipy 0.9's minpack
    return function(xdata, *params) - ydata


def cvfit(f, xdata, ydata, p0):
    """  Fit data to a function.

    :param f: A function that accepts 1+len(p0) arguments, first one will be x
    :param xdata: X values
    :param ydata: Y values
    :param p0: The initial guess, put [1, 1, 1, ..] if uncertain.

    """
    res = optimize.leastsq(_general_function, p0, args=(xdata, ydata, f))
    return res[0]


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
    return numpy.sum(x * y) / numpy.sum(y)



def var(x, y):
    """  Returns the Variance of a ditribution with x as location and y as value

    >>> sparsedl.var([1, 2, 3, 4, 5, 6], [1, 1, 2, 1, 1, 2])
    2.9375
    """
    xarr = numpy.array(x)  # Make sure we have numpy arrays and not lists
    yarr = numpy.array(y)
    return mean(xarr ** 2, yarr) - mean(xarr, yarr) ** 2


def initial(nodes):
    """  Some convient initial values, usefull for an interactive session.
        e.g   >>>  rho0, A, xcoord = initial(200)  """
    rho0 = scipy.zeros(nodes)
    rho0[nodes // 2] = 1
    A = NN_matrix(nodes)
    xcoord = scipy.linspace(-nodes / 2, (nodes / 2) - 1, nodes)
    return rho0, A, xcoord



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
    return eigvals    

def descrete_spatial_fourier2(k , points, values):
    """ """
    ikr = 1j*numpy.dot(k, points.T)
    f_e_ikr = values*numpy.exp(ikr)
    return f_e_ikr.sum()

def banded_ones(N, bandwidth):
    """ returns a NxN banded matrix of ones
    """
    return numpy.tri(N, k=bandwidth)*numpy.tri(N,k=bandwidth).T
