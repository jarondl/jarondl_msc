#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.  
"""
from scipy.sparse import linalg as splinalg
from numpy import linalg, random
#from argparse import ArgumentParser

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


def eigenvalues_lognormal(ax, N=100, b=4):
    """  Plot the eigenvalues for a lognormal sparse banded matrix
    """
    W = sparsedl.lognormal_sparse_matrix(N,b).todense()
    eigvals = linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    ax.plot(eigvals, numpy.linspace(0,N,N))
    plotdl.set_all(ax, title="lognormal")


def eigenvalues_uniform(ax, N=100, b=4):
    """  Plot the eigenvalues for a uniform random matrix
    """
    W = numpy.random.uniform(-1,1,N**2).reshape([N,N])
    eigvals = linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    ax.plot(eigvals, numpy.linspace(0,N,N))
    plotdl.set_all(ax, title=r"uniform, $[-1,1]$")

    

def torus_plots(N_points=100,dimensions=(10,10),endtime=1,ergodic_end=True,filename="distance"):
    """  Create A_ij for points on a torus via e^(-r_ij). 
        :param N_points: Number of points, defaults to 100
        :type N_points: int
        :param dimensions: The 2d dimensions, as a 2-tuple. defaults to (10,10)

    """
    fig = plotdl.Figure()
    ax_torus_surv = fig.add_subplot(1,2,1)
    ax_torus_eigvals = fig.add_subplot(1,2,2)
    
    torus = geometry.Torus(dimensions)
    points = torus.generate_points(N_points)
    #dis =  geometry.distance_matrix(points, torus.distance)
    dis = geometry.distance_matrix(points, geometry.euclid)
    ex = numpy.exp(-dis)
    sparsedl.zero_sum(ex)
    eigvals = linalg.eigvalsh(ex)
    eigvals.sort()
    
    t= numpy.logspace(-2,endtime,100,endpoint=False)
    survs = sparsedl.surv(eigvals,t)
   
    if ergodic_end:
        clip_idx = clip(survs, 1.01/(N_points))
        while clip_idx < (0.8*N_points):
                t= numpy.logspace(-2,numpy.log(t[clip_idx]),100,endpoint=False)
                survs = sparsedl.surv(eigvals,t)
                clip_idx = clip(survs, 1.01/(N_points))
    diffusion = numpy.sqrt(t)**(-1)
    ax_torus_surv.loglog(t, survs)
    ax_torus_surv.loglog(t,diffusion)
    plotdl.set_all(ax_torus_surv, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival")
    ax_torus_eigvals.plot(eigvals,numpy.linspace(0,N_points,N_points))
    plotdl.set_all(ax_torus_eigvals, xlabel="Eigenvalue", title="Cummulative eigenvalue distribution")
    
    plotdl.savefig(fig,filename)
    return fig,survs

def clip(nparray,lower_bound):
    """  """
    return numpy.max((nparray > lower_bound).nonzero())
    
    
    
    

def all_plots(seed= 1, **kwargs):
    random.seed(seed)
    plotdl.plot_to_file( p_lognormal_band, "P_lognormal_band")
    random.seed(seed)
    plotdl.plot_to_file( spreading_plots, "spreading")
    random.seed(seed)
    plotdl.plot_2subplots_to_file( eigenvalues_lognormal, eigenvalues_uniform, "eigvals", suptitle="Cummulative eigenvalue distribution")
    #eigenvalue_plots()
    #random.seed(seed)
    #spreading_plots()
    #random.seed(seed)
    #p_lognormal_band()
    #random.seed(seed)
    #torus_plots()
    
    


if __name__ ==  "__main__":
    all_plots()
