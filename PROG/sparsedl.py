#!/usr/bin/python
"""  This module includes all of the functions that I use. To import use:
            import sparsedl

"""
from __future__ import division  # makes true division instead of integer division
import scipy
import numpy
from scipy import optimize, special, linalg
from scipy.sparse import spdiags


def NN_matrix(N, b=1, **kwargs):
    """ Build a NxN matrix.

    :param N: The size of the matrix
    :param b: The Bandwidth
    :param **kwargs: The rest of the keyworded arguments are passed
                        to the lognormal function, so you can define
                        sigma and mu.
    :returns: a NxN numpy array

    Uses :py:func:`lognormal_construction`.

    """
    matrix = numpy.zeros([N, N])
    for diag in range(1, b + 1):
        off_diag_values = lognormal_construction(N - diag, **kwargs)
        matrix += (numpy.diagflat(off_diag_values, diag) +
                   numpy.diagflat(off_diag_values, - diag))
    diag_values = matrix.sum(axis=1)  # the diagonal contains minus the sum of rows
    matrix -= numpy.diagflat(diag_values)
    return matrix


def lognormal_sparse_matrix(N, b=1, **kwargs):
    """ Build a NxN matrix, using sparse matrices.

    :param N: The size of the matrix
    :param b: The Bandwidth
    :param **kwargs: The rest of the keyworded arguments are passed
                        to the lognormal function, so you can define
                        sigma and mu.
    :returns: a NxN numpy array

    Uses :py:func:`lognormal_construction`.

    """
    diag_data = lognormal_construction(N * b, **kwargs).reshape([b, N])
    sp = spdiags(diag_data, range(1, b + 1), N, N)  # create the above the diagonal lines
    sp = sp + sp.transpose()
    sp = sp - spdiags(sp.sum(axis=0), 0, N, N)
    return sp


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
    :param p0: The initial guess, put [1,1,1,..] if uncertain.

    """
    res = optimize.leastsq(_general_function, p0, args=(xdata, ydata, f))
    return res[0]


def strexp(x, a, b):
    """ The streched exponential function, :math:`e^{-(\\frac{x}{a})^b}`

    >>> strexp(1,2,2)
    0.77880078307140488
    """
    return numpy.exp(-(x / a) ** b)


def mean(x, y):
    """  Returns the mean of a ditribution with x as location and y as value
    """
    return numpy.sum(x * y) / numpy.sum(y)


def var(x, y):
    """  Returns the Variance of a ditribution with x as location and y as value

    >>> sparsedl.var([1,2,3,4,5,6],[1,1,2,1,1,2])
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


def zero_sum(mat, tol=1E-12):
    """  Change the diagonal elements of mat so that the sum of each row becomes 0
    """
    row_sum = mat.sum(axis=1)
    if (numpy.max(row_sum) < tol):
        return True
    else:
        mat -= numpy.diagflat(row_sum)
        if numpy.max((mat.sum(axis=0), mat.sum(axis=1))) > tol:
            raise Exception("Failed to make sums zero, is the matrix symmetric?")
        return False


def lognormal_construction(N, mu=0, sigma=1, **kwargs):
    """ Create log-normal distribution, with N elements, mu average and sigma width.
        The construction begins with a lineary spaced vector from 0 to 1. Then
          the inverse CDF of a normal distribution is applied to the vector.
        The result is permutated, and its piecewise exponent is returned."""
    uniform = scipy.linspace(0.0001, 0.9999, N)
    y = -scipy.sqrt(2) * special.erfcinv(2 * uniform)
    rescaled_y = mu + (sigma * y)
    perm_y = scipy.random.permutation(rescaled_y) 
    return scipy.exp(perm_y)  # Element-wise exponentiation. 


def exponent_minus1(N,nxi = 0.5):
    """ """
    uniform = numpy.linspace(0.0001,0.9999, N)
    y = uniform**(1/nxi)
    perm_y = numpy.random.permutation(y)
    sp = spdiags(perm_y, 1, N, N)  # create the above the diagonal line
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
    zero_sum(retval)
    return retval

def sparsity(mat):
    """  Calculate the sparsity parameters of a matrix
    """
    matsum =  mat.sum()
    avg = matsum / mat.size
    sqr_avg = (mat**2).sum() / mat.size
    s = avg**2 / sqr_avg
    p = ((mat > avg).sum())/mat.size
    #q = mat.flatten()[mat.size//2] / avg
    return s,p#,q


def surv(eigen_values, times):
    """ Calculate survival probability by the sum of exponents equation"""
    op = numpy.outer(eigen_values, times)
    exps = numpy.exp(op)
    return exps.sum(axis=0) / len(eigen_values)
