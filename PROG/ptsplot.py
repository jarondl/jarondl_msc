#!/usr/bin/python
""" Survival and spreading for log normal distribution.  
"""
import sparsedl
import plotdl
from scipy.sparse import linalg

def surv_lognormal_band(t,N=100,b=1,seed=None):
    """plot s(t) and p(t) for lognormal banded matrix"""
    s = sparsedl.lognormal_sparse_matrix(N,b, seed)
    vals,vecs  = linalg.eigen_symmetric(s, k=50,which='LA')  # finds 50 largest eigavlues (largest algebraic, so least negative)
    
    return sparsedl.surv(vals,t)
    
def plot_surv_lognormal_band():
    fig = plotdl.Figure()
    ax_survs = fig.add_subplot(1,1,1)
    t= sparsedl.numpy.linspace(0,4,100)
    for b in range(1,10):
        survs = surv_lognormal_band(t,b=b)
        ax_survs.semilogy(t,survs,label= r"$b = {0}$".format(b))
    ax_survs.set_xlabel("$t$")
    ax_survs.set_ylabel(r"$\mathcal{P}(t)$")
    ax_survs.legend()
    plotdl.savefig(fig,"p_lognormal_band.png")













if __name__ ==  "__main__":
    plot_surv_lognormal_band()
    

