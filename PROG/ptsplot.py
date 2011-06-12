#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.  
"""
#from scipy.sparse import linalg as splinalg
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
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival", legend_loc="best")


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
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$S(t)$", title = r"Spreading", legend_loc="best")


def eigenvalues_lognormal(ax, N=100, b=1):
    """  Plot the eigenvalues for a lognormal sparse banded matrix
    """
    W = sparsedl.lognormal_sparse_matrix(N,b).todense()
    eigvals = - linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    eigvals = eigvals[1:]  ## The first eigenvalue is zero, which does problems with loglog plots
    diffusion_space = numpy.logspace(numpy.log10(numpy.min(eigvals)),numpy.log10(numpy.max(eigvals)),N-1)
    diffusion = numpy.sqrt( diffusion_space)
    ax.loglog(eigvals, numpy.linspace(0,1,N-1),marker='.',linestyle='', label="Normalized cummulative eigenvalues (divided by N)")
    ax.loglog(diffusion_space,diffusion, linestyle='--',label = "Square root")
    plotdl.set_all(ax, title="lognormal, $b={0}$".format(b), legend_loc="upper left")
    
def eigenvalues_NN(ax, N=100, NN_data=None):
    """  Plot the eigenvalues for a Nearest neighbor matrix
    """
    if NN_data is None:
        NN_data = numpy.ones(N)
    W = sparsedl.create_sparse_matrix(N,NN_data,b=1).todense()
    eigvals = - linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    eigvals = eigvals[1:]  ## The first eigenvalue is zero, which does problems with loglog plots
    diffusion_space = numpy.logspace(numpy.log10(numpy.min(eigvals)),numpy.log10(numpy.max(eigvals)),N-1)
    diffusion = numpy.sqrt( diffusion_space)
    ax.loglog(eigvals, numpy.linspace(0,1,N-1),marker='.',linestyle='', label="Normalized cummulative eigenvalues (divided by N)")
    ax.loglog(diffusion_space,diffusion, linestyle='--',label = "Square root")
    plotdl.set_all(ax, title="lognormal, $b={0}$".format(b), legend_loc="upper left")


def eigenvalues_exponent_minus1(ax, N=100, nxi=0.3):
    """  Plot the eigenvalues for a :math:p(w) = w^{n\\xi-1}n\\xi:
    """
    W = sparsedl.exponent_minus1(N,nxi=nxi).todense()
    eigvals = - linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    eigvals = eigvals[2:]/N  ## The first eigenvalue is zero, which does problems with loglog plots
    power_law_space = numpy.logspace(numpy.log10(numpy.min(eigvals[1:])),numpy.log10(numpy.max(eigvals)),N-1)
    power_law = (power_law_space**(nxi))
    ax.loglog(eigvals, numpy.linspace(0,1,N-2),marker='.',linestyle='', label="Cummulative eigenvalues (divided by N)")
    ax.loglog(power_law_space, power_law, linestyle='--',label=r"\lambda^{n\xi}")
    plotdl.set_all(ax, title=r"$p(w) = w^{n\xi-1}n\xi $ Where $n\xi=$"+str(nxi), legend_loc="lower right")
    
    


def eigenvalues_uniform(ax, N=100):
    """  Plot the eigenvalues for a uniform random matrix
    """
    W = numpy.random.uniform(-1,1,N**2).reshape([N,N])
    eigvals = linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    ax.plot(eigvals, numpy.linspace(0,N,N), label="Cummulative eigenvalue distribution",marker='.',linestyle='')
    
    R=numpy.max(eigvals)
    #R=2.0
    semicircle = numpy.sqrt(numpy.ones(N)*R**2 - numpy.linspace(-R,R,N)**2)#/(2*numpy.pi)
    cum_semicircle = numpy.cumsum(semicircle) 
    print(numpy.max(cum_semicircle))
    cum_semicircle = cum_semicircle / numpy.max(cum_semicircle)*N
    ax.plot(numpy.linspace(-R,R,N), semicircle, linestyle="--", label=r"Semi circle, with $R \approx {0:.2}$".format(R))
    ax.plot(numpy.linspace(-R,R,N), cum_semicircle,linestyle="--", label = r"Cummulative semicircle, with $R \approx {0:.2}$".format(R))

    plotdl.set_all(ax, title=r"uniform, $[-1,1]$", legend_loc="upper left")

    

def torus_plots_eig_surv(ax_eig,ax_surv, N_points=100,dimensions=(10,10),end_log_time=1):
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
    eigvals1 = -linalg.eigvalsh(ex1)
    eigvals2 = -linalg.eigvalsh(ex2)
    eigvals1.sort()
    eigvals2.sort()
    #eigvals1 = eigvals1[1:] # The zero is problematic for the plots
    #eigvals2 = eigvals2[1:] # The zero is problematic for the plots
    
    ax_eig.plot(eigvals1, numpy.linspace(0,N_points,N_points), label="original")
    ax_eig.plot(eigvals2, numpy.linspace(0,N_points,N_points), label="permuted")
    ax_eig.legend(loc='lower right')
    ax_eig.set_xlim(right=0)

    
    t= numpy.logspace(-2,end_log_time,100,endpoint=False)
    survs1 = sparsedl.surv(-eigvals1,t)
    survs2 = sparsedl.surv(-eigvals2,t)
   
    ax_surv.loglog(t, survs1)
    ax_surv.loglog(t, survs2)
    ax_surv.set_ylim(bottom=0.01)
    
    #plotdl.set_all(ax_torus_surv, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival")
    #ax_torus_eigvals.plot(eigvals,numpy.linspace(0,N_points,N_points))
    #plotdl.set_all(ax_torus_eigvals, xlabel="Eigenvalue", title="Cummulative eigenvalue distribution")


def torus_plots_eig(ax_eig, N_points=100,dimensions=(10,10),end_log_time=1):
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
    eigvals1 = -linalg.eigvalsh(ex1)   # NOTE : minus sign
    eigvals2 = -linalg.eigvalsh(ex2)
    eigvals1.sort()
    eigvals2.sort()
    eigvals1 = eigvals1[1:] # The zero is problematic for the plots
    eigvals2 = eigvals2[1:] # The zero is problematic for the plots
    #minvallog = numpy.log10(min(numpy.min(eigvals1),numpy.min( eigvals2)))
    #maxvallog = numpy.log10(max(numpy.max(eigvals1),numpy.max( eigvals2)))
    #theory_space = numpy.logspace(minvallog,maxvallog,N_points)
    #theory = 1-numpy.exp(-0.1*numpy.log(theory_space)**2)
    #ax_eig.loglog(theory_space,theory,label="theory", linestyle="--")
    
    ax_eig.loglog(eigvals1, numpy.linspace(0,1,N_points-1), label="original", marker='.', linestyle='')
    ax_eig.loglog(eigvals2, numpy.linspace(0,1,N_points-1), label="permuted", marker='.', linestyle='')
    ax_eig.legend(loc='lower right')
    ax_eig.set_xlim(right=0)
    


def clip(nparray,lower_bound):
    """  """
    return numpy.max((nparray > lower_bound).nonzero())
    
    
def loop_torus_eig_surv():   
    """ Create 5 pairs subplots of `torus_plots`
    """
    fig = plotdl.Figure()
    fig.subplots_adjust(top=0.99, bottom=0.05)
    for i in range(1,10,2):
        random.seed(i)
        torus_plots(fig.add_subplot(5,2,i), fig.add_subplot(5,2,i+1))
    plotdl.savefig(fig, "8tori", size_factor=(1,2))


def loop_torus_eig():   
    """ Create 5 pairs subplots of `torus_plots`
    """
    fig = plotdl.Figure()
    fig.subplots_adjust(top=0.99, bottom=0.05)
    for i in range(1,6):
        random.seed(i)
        torus_plots_eig(fig.add_subplot(5,1,i))
    plotdl.savefig(fig, "8tori_eig", size_factor=(1,2))

    

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
    """  Create all of the figures. Please note that it might take some time.
    """
    random.seed(seed)
    plotdl.plot_to_file( p_lognormal_band, "P_lognormal_band")
    random.seed(seed)
    plotdl.plot_to_file( spreading_plots, "spreading")
    random.seed(seed)
    #plotdl.plot_2subplots_to_file( eigenvalues_lognormal, eigenvalues_uniform, "eigvals", suptitle="Cummulative eigenvalue distribution")
    plotdl.plot_to_file(eigenvalues_lognormal, "eigvals_lognormal")
    plotdl.plot_to_file(eigenvalues_lognormal, "eigvals_lognormal_b_5", b=5)
    plotdl.plot_to_file(eigenvalues_lognormal, "eigvals_lognormal_b_10", b=10)
    plotdl.plot_to_file(eigenvalues_lognormal, "eigvals_lognormal_b_100", b=100)
    plotdl.plot_to_file(eigenvalues_uniform,  "eigvals_uniform")
    random.seed(seed)
    loop_torus_eig( )

    #plotdl.plot_twin_subplots_to_file( torus_permutation)


    
    


if __name__ ==  "__main__":
    all_plots()
