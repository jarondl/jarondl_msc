#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.  
"""
from scipy.sparse import linalg as splinalg
from numpy import linalg, random
from argparse import ArgumentParser

import numpy

import sparsedl
import plotdl
import geometry

    
def p_lognormal_band(N=100,b=1, filename="P_lognormal_band",**kwargs):
    """ Plot p (survival) as a function of time, with lognormal banded transition matrices.
        The sparsitiy is constant (set via the lognormal sigma), 
        while b assumes values between 1 and 10.
        ARGUMENT LIST:
            **kwargs, arguments passed on to create the banded lognormal matrix. (N,b, sigma,mu)
    """
    fig = plotdl.Figure()
    ax = fig.add_subplot(1,1,1)
    
    t= sparsedl.numpy.linspace(0.1,10,100,endpoint=False)
    for b in range(1,10):
        s     = sparsedl.lognormal_sparse_matrix(N,b,**kwargs).todense()
        vals  = linalg.eigvalsh(s)
        survs = sparsedl.surv(vals,t)
        ax.loglog(t,survs,label= r"$b = {0}$".format(b))
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival", legend=True)
    plotdl.savefig(fig, filename)
    return fig

def spreading_plots(N=100,filename="spreading"):
    fig = plotdl.Figure()
    ax = fig.add_subplot(111)
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
    plotdl.savefig(fig,filename)
    return fig

def eigenvalue_plots(N=100,b=4,filename="eigvals", **kwargs):
    fig = plotdl.Figure()
    fig.suptitle("Cummulative eigenvalue distribution")
    
    lognormal_plot = {"title" : "lognormal"}
    lognormal_plot["W"] = sparsedl.lognormal_sparse_matrix(N,b).todense()
    lognormal_plot["ax"] = fig.add_subplot(1,2,1)
    
    uniform_plot = {"title":r"uniform, $[-1,1]$"}
    uniform_plot["W"] = numpy.random.uniform(-1,1,N**2).reshape([N,N])
    uniform_plot["ax"] = fig.add_subplot(1,2,2)

    for plotset in [uniform_plot, lognormal_plot] :
        eigvals = linalg.eigvalsh(plotset["W"])
        eigvals.sort()
        ax =plotset["ax"]
        ax.plot(eigvals,numpy.linspace(0,N,N))
        ax.set_title(plotset["title"])
    plotdl.savefig(fig, filename)
    return fig

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
    eigenvalue_plots()
    random.seed(seed)
    spreading_plots()
    random.seed(seed)
    p_lognormal_band()
    random.seed(seed)
    torus_plots()
    
    


if __name__ ==  "__main__":
    parser = ArgumentParser(description="Make plots", epilog="try a subcommand with -h to see its help")
    parser.add_argument('-s','--seed', help="random seed to make pseudo random numbers",type=int, default=1, dest='seed')

    subparsers = parser.add_subparsers()
    
    parser_all  = subparsers.add_parser('ALL', help="Create all plots with default arguments")
    parser_all.set_defaults(function = all_plots)

    parser_p_lognormal_band = subparsers.add_parser('lognormal', help=p_lognormal_band.__doc__)
    parser_p_lognormal_band.add_argument('-N', type=int,default=100, dest='N')
    parser_p_lognormal_band.set_defaults(function = p_lognormal_band)

    parser_eigenvalue = subparsers.add_parser('eigenvalues', help=eigenvalue_plots.__doc__)
    parser_eigenvalue.add_argument('-N', type=int,default=100, dest='N')
    parser_eigenvalue.set_defaults(function = eigenvalue_plots)

    parser_torus = subparsers.add_parser('torus', help=torus_plots.__doc__)
    parser_torus.add_argument('-N', type=int,default=100, dest='N')
    parser_torus.set_defaults(function = torus_plots)


    args = parser.parse_args()
    random.seed(seed=args.seed)
    args.function(**vars(args))
    
    #numpy.random.seed(1)
    #spread_surv_subplots()
    
