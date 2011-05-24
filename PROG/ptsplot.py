#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.  
"""
from scipy.sparse import linalg as splinalg
from numpy import linalg
from argparse import ArgumentParser
import numpy

import sparsedl
import plotdl


    
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

def spreading_without_scaling(N=100,filename="s_without_scaling"):
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


def eigenvalue_plots(N=100,b=1,filename="eigvals"):
    fig = plotdl.Figure()
    
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
        ax.plot(eigvals,numpy.linspace(0,100,100))
        ax.set_title(plotset["title"])
    plotdl.savefig(fig, filename)

    


if __name__ ==  "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_p_lognormal_band = subparsers.add_parser('lognormal', help=p_lognormal_band.__doc__)
    parser_p_lognormal_band.add_argument('-N', type=int,default=100, dest='N')
    parser_p_lognormal_band.set_defaults(function = p_lognormal_band)

    
    args = parser.parse_args()
    args.function(**vars(args))
    
    #numpy.random.seed(1)
    #spread_surv_subplots()
    
