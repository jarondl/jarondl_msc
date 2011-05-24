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


    
def p_s_lognormal_band(ax_p,ax_s,N=100,b=1,seed=None,**kwargs):
    """ Plot p (survival) and s (spreading) as a function of time, with lognormal banded transition matrices.
        The sparsitiy is constant (set via the lognormal sigma), 
        while b assumes values between 1 and 10.
        ARGUMENT LIST:
            ax_p - an axis to plot p on
            ax_s - an axis to plot s on
            **kwargs, arguments passed on to create the banded lognormal matrix. (N,b, sigma,mu)
    """
    t= sparsedl.numpy.linspace(0,4,100)
    for b in range(1,10):
        s     = sparsedl.lognormal_sparse_matrix(N,b, seed,**kwargs).todense()
        vals  = linalg.eigvalsh(s)
        survs = sparsedl.surv(vals,t)
        spreads = survs**(-2)
        ax_p.loglog(t,survs,label= r"$b = {0}$".format(b))
        ax_s.loglog(t,spreads,label= r"$b = {0}$".format(b))
    plotdl.set_all(ax_p, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival", legend=True)
    plotdl.set_all(ax_s, xlabel = "$t$", ylabel = r"$S(t)$", title = r"Spreading",legend=True)

def spreading_without_scaling(N=100):
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
    plotdl.set_all(xlabel = "$t$", ylabel = r"$S(t)$", title = r"Spreading", legend=True)
    plotdl.savefig(fig,"s_without_scaling")


def spread_surv_subplots():
    """ Plot two side by side plots, of the survival and spreading as a function of time, 
            for various values of b.
        The transition matirces are banded, with lognormal distribution.
    """
    fig = plotdl.Figure()
    fig.subplots_adjust(wspace=0.5)
    ax_survs = fig.add_subplot(1,2,1)
    ax_spread = fig.add_subplot(1,2,2)
    p_s_lognormal_band(ax_survs, ax_spread)
    plotdl.savefig(fig,"p_s_lognormal_band")

def eigenvalue_plots(N=100,b=1):
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
    plotdl.savefig(fig, "eigvals")

    


if __name__ ==  "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_p_s_lognormal_band = subparsers.add_parser('lognormal', help=spread_surv_subplots.__doc__)
    parser_p_s_lognormal_band.add_argument('-N', type=int, dest='N')

    
    parser.parse_args()
    #numpy.random.seed(1)
    #spread_surv_subplots()
    
