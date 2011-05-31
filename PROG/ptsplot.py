#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.  
"""
from scipy.sparse import linalg as splinalg
from numpy import linalg, random
#from argparse import ArgumentParser
from matplotlib.colors import LogNorm

import numpy

import sparsedl
import plotdl
import geometry

    
def p_lognormal_band(ax, N=100, b=1, **kwargs):
    """ Plot p (survival) as a function of time, with lognormal banded transition matrices.
        The sparsitiy is constant (set via the lognormal sigma), 
        while b assumes values between 1 and 10.
        ARGUMENT LIST:
            **kwargs, arguments passed on to create the banded lognormal matrix. (N,b, sigma,mu)
    """
    
    t= sparsedl.numpy.logspace(-1,1,100,endpoint=False)
    for b in range(1,10):
        s     = sparsedl.lognormal_sparse_matrix(N,b,**kwargs).todense()
        vals  = linalg.eigvalsh(s)
        survs = sparsedl.surv(vals,t)
        ax.loglog(t,survs,label= r"$b = {0}$".format(b))
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival", legend=True)


def spreading_plots(ax, N=100):
    """
    """
    t= sparsedl.numpy.linspace(0,4,100)
    rho0 = numpy.zeros(N)
    rho0[N//2] =1
    xcoord = numpy.linspace(-N,N,N)
    
    for b in range(1,10):
        W = sparsedl.lognormal_sparse_matrix(N,b).todense()
        S = []
        for time in t:
            S += [sparsedl.var(xcoord, sparsedl.rho(time, rho0, W))]
        ax.semilogy(t,S,label= r"$b = {0}$".format(b))
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$S(t)$", title = r"Spreading", legend=True)


def eigenvalues_lognormal(ax, N=100, b=1):
    """  Plot the eigenvalues for a lognormal sparse banded matrix
    """
    W = sparsedl.lognormal_sparse_matrix(N,b).todense()
    eigvals = - linalg.eigvalsh(W)[1:]  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    diffusion = 0.1 * numpy.logspace(0,2,N-1)
    ax.loglog(eigvals, numpy.logspace(0,2,N-1))
    ax.loglog(diffusion, numpy.logspace(0,2,N-1))
    plotdl.set_all(ax, title="lognormal, $b=1$")


def eigenvalues_uniform(ax, N=100, b=4):
    """  Plot the eigenvalues for a uniform random matrix
    """
    W = numpy.random.uniform(-1,1,N**2).reshape([N,N])
    eigvals = linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    ax.loglog(eigvals, numpy.logspace(0,2,N))
    plotdl.set_all(ax, title=r"uniform, $[-1,1]$")

    

def torus_plots(ax_eig,ax_surv, N_points=100,dimensions=(10,10),end_log_time=1):
    """  Create A_ij for points on a torus via e^(-r_ij). 
        :param N_points: Number of points, defaults to 100
        :type N_points: int
        :param dimensions: The 2d dimensions, as a 2-tuple. defaults to (10,10)

    """

    
    torus = geometry.Torus(dimensions)
    points = torus.generate_points(N_points)
    dis =  geometry.distance_matrix(points, torus.distance)
    
    ex1 = numpy.exp(-dis)
    sparsedl.zero_sum(ex1)
    ex2 = sparsedl.permute_tri(ex1)
    eigvals1 = linalg.eigvalsh(ex1)
    eigvals2 = linalg.eigvalsh(ex2)
    eigvals1.sort()
    eigvals2.sort()
    
    ax_eig.plot(eigvals1, numpy.linspace(0,N_points,N_points), label="original")
    ax_eig.plot(eigvals2, numpy.linspace(0,N_points,N_points), label="permuted")
    #plotdl.set_all(ax_eig, xlabel="Eigenvalue", ylabel="Cummulative distribution")
    ax_eig.legend(loc='lower right')
    ax_eig.set_xlim(right=0)

    
    t= numpy.logspace(-2,end_log_time,100,endpoint=False)
    survs1 = sparsedl.surv(eigvals1,t)
    survs2 = sparsedl.surv(eigvals2,t)
   
    ax_surv.loglog(t, survs1)
    ax_surv.loglog(t, survs2)
    ax_surv.set_ylim(bottom=0.01)
    
    #plotdl.set_all(ax_torus_surv, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival")
    #ax_torus_eigvals.plot(eigvals,numpy.linspace(0,N_points,N_points))
    #plotdl.set_all(ax_torus_eigvals, xlabel="Eigenvalue", title="Cummulative eigenvalue distribution")


def clip(nparray,lower_bound):
    """  """
    return numpy.max((nparray > lower_bound).nonzero())
    
    
def loop_torus():   
    """
    """
    fig = plotdl.Figure()
    fig.subplots_adjust(top=0.99, bottom=0.05)
    for i in range(1,10,2):
        random.seed(i)
        torus_plots(fig.add_subplot(5,2,i), fig.add_subplot(5,2,i+1))
    plotdl.savefig(fig, "8tori", size_factor=(1,2))
    

def torus_permutation_noax(N_points=100,dimensions=(10,10),filename="torus_perm"):
    """
    """
    torus = geometry.Torus(dimensions)
    points = torus.generate_points(N_points)
    dis =  geometry.distance_matrix(points, torus.distance)

    fig = plotdl.Figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    ex1 = numpy.exp(-dis)
    sparsedl.zero_sum(ex1)
    ex2 = sparsedl.permute_tri(ex1)
    mat1 = ax1.matshow(ex1) #, norm=LogNorm(vmin=numpy.min(ex1), vmax=numpy.max(ex1))
    mat2 = ax2.matshow(ex2)
    #fig.colorbar(mat1)
    #fig.colorbar(mat2)
    plotdl.savefig(fig, filename)


def all_plots(seed= 1, **kwargs):
    """
    """
    random.seed(seed)
    plotdl.plot_to_file( p_lognormal_band, "P_lognormal_band")
    random.seed(seed)
    plotdl.plot_to_file( spreading_plots, "spreading")
    random.seed(seed)
    plotdl.plot_2subplots_to_file( eigenvalues_lognormal, eigenvalues_uniform, "eigvals", suptitle="Cummulative eigenvalue distribution")
    loop_torus( )
    random.seed(1)
    #plotdl.plot_twin_subplots_to_file( torus_permutation)


    
    


if __name__ ==  "__main__":
    all_plots()
