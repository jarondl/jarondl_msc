#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.  
"""
import sparsedl
import plotdl
from scipy.sparse import linalg as splinalg
from numpy import linalg

def surv_lognormal_band(t,N=100,b=1,seed=None,**kwargs):
    """ Creates a lognormal banded matrix and calculates the survival.
        ARGUMENT LIST:
            t - time
            N - size of matrix
            b - bandwidth
            seed - random seed to create pseudo random results
            **kwargs - arguments passed to "lognormal_sparse_matrix" , including sigma and mu
        """
    s = sparsedl.lognormal_sparse_matrix(N,b, seed,**kwargs)
    try : 
        vals,vecs  = splinalg.eigen_symmetric(s, k=50,which='LA')  # finds 50 largest eigavlues (largest algebraic, so least negative)
    except AttributeError:  # The older version of scipy doesn't solve sparse eigenvalue problems...
        vals  = linalg.eigvalsh(s.todense())   # 
    
    return sparsedl.surv(vals,t)
    
def p_s_lognormal_band(ax_p,ax_s,**kwargs):
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
        survs = surv_lognormal_band(t,b=b)
        spreads = survs**(-2)
        ax_p.semilogy(t,survs,label= r"$b = {0}$".format(b))
        ax_s.semilogy(t,spreads,label= r"$b = {0}$".format(b))
    ax_p.set_xlabel("$t$")
    ax_p.set_ylabel(r"$\mathcal{P}(t)$")
    ax_p.set_title("Survival")
    ax_p.legend()
    ax_s.set_xlabel("$t$")
    ax_s.set_ylabel(r"$S(t)$")
    ax_s.set_title(r"Spreading")
    ax_s.legend()



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
    plotdl.savefig(fig,"p_s_lognormal_band",size=[8,4])


if __name__ ==  "__main__":
    spread_surv_subplots()
    

