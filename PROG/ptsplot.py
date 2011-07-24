#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
#from scipy.sparse import linalg as splinalg
from numpy import linalg, random, pi, log10, sqrt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
from copy import deepcopy

import numpy

import sparsedl
import plotdl
import geometry


def p_lognormal_band(ax, N=100, b=1, **kwargs):
    """ Plot p (survival) as a function of time, with lognormal banded transition matrices.
        The sparsitiy is constant (set via the lognormal sigma), 
        while b assumes values between 1 and 10.

        :param \*\*kwargs: arguments passed on to create the banded lognormal matrix. (N, b, sigma, mu)
    """
    t = sparsedl.numpy.logspace(-1, 1, 100, endpoint=False)
    for b in range(1, 10):
        rates = sparsedl.lognormal_construction(N * b, **kwargs)
        s     = sparsedl.create_sparse_matrix(N, rates, b).todense()
        vals  = linalg.eigvalsh(s)
        survs = sparsedl.surv(vals, t)
        ax.loglog(t, survs, label= r"$b = {0}$".format(b))
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$\mathcal{P}(t)$", title = "Survival", legend_loc="best")


def spreading_plots(ax, N=100, **kwargs):
    """
    """
    t= sparsedl.numpy.linspace(0, 4, 100)
    rho0 = numpy.zeros(N)
    rho0[N//2] =1
    xcoord = numpy.linspace(-N, N, N)

    for b in range(1, 10):
        rates = sparsedl.lognormal_construction(N * b, **kwargs)
        W     = sparsedl.create_sparse_matrix(N, rates, b).todense()
        S = []
        for time in t:
            S += [sparsedl.var(xcoord, sparsedl.rho(time, rho0, W))]
        ax.semilogy(t, S, label= r"$b = {0}$".format(b))
    plotdl.set_all(ax, xlabel = "$t$", ylabel = r"$S(t)$", title = r"Spreading", legend_loc="best")


############################     Eigen value plot functions   ###############
## Here follows a set of of plot functions to plot eigenvalues. The first argument is always a matplotlib axes instance, to plot on.

def eigenvalues_lognormal(ax, N=200, b_list=(1, ), **kwargs):
    """  Plot the eigenvalues for a lognormal sparse banded matrix
    """
    for b in b_list:
        rates = sparsedl.lognormal_construction(N * b, **kwargs)
        W     = sparsedl.create_sparse_matrix(N, rates, b).todense()
        D = sparsedl.resnet(W, b)
        eigvals = eigenvalues_cummulative(ax, W, label = "b = {0}, D = {1}".format(b, D))  ## Plots the eigenvalues.

        diffusion_plot(ax, D, eigvals)
    plotdl.set_all(ax, title="lognormal, N={N}".format(N=N), legend_loc="upper left")


def eigenvalues_ones(ax, N=200, b_list=(1, )):
    """  Plot the eigenvalues for a banded matrix with ones only, for each b.
    """
    for b in b_list:
        rates = numpy.ones(N*b)
        W = sparsedl.create_sparse_matrix(N, rates, b).todense()
        if not sparsedl.zero_sum(W):
            raise Exception("The W matrix is not zero summed. How did you make it?")
        D = sparsedl.resnet(W, b)
        label = "b = {0}, D = {1}".format(b, D)
        eigvals = eigenvalues_cummulative(ax, W, label)
        diffusion_plot(ax, D, eigvals)
        #ones_analytic_plot(ax, N)
    #ax.set_xscale('linear')
    #ax.set_yscale('linear')
    plotdl.set_all(ax, title="All ones, N = {N}".format(N=N), legend_loc="best")


def eigenvalues_alter(ax, N=200, w1 = 3, w2 = 8, b_list=(1, )):
    """  Plot the eigenvalues for a banded matrix with alternating 3 and 8, for each b.
    """
    for b in b_list:
        rates = numpy.zeros(N*b)
        rates[::2] = 3
        rates[1::2] = 8
        W = sparsedl.create_sparse_matrix(N, rates, b).todense()
        if not sparsedl.zero_sum(W):
            raise Exception("The W matrix is not zero summed. How did you make it?")
        D = sparsedl.resnet(W, b)
        label = "b = {0}, D = {1}".format(b, D)
        eigvals = eigenvalues_cummulative(ax, W, label)
        diffusion_plot(ax, D, eigvals)
    alter_analytic_plot(ax, 3, 8, N)
    plotdl.set_all(ax, title="Alternating 3-8, N = {N}".format(N=N), legend_loc="best")


def eigenvalues_box(ax, N=200, w1 = 3, w2 = 8, b_list=(1, )):
    """  Plot the eigenvalues for a banded matrix with alternating 3 and 8, for each b.
    """
    for b in b_list:
        rates = numpy.random.uniform(3, 8, N*b)
        W = sparsedl.create_sparse_matrix(N, rates, b).todense()
        if not sparsedl.zero_sum(W):
            raise Exception("The W matrix is not zero summed. How did you make it?")
        D = sparsedl.resnet(W, b)
        label = "b = {0}, D = {1}".format(b, D)
        eigvals = eigenvalues_cummulative(ax, W, label)
        diffusion_plot(ax, D, eigvals)
    plotdl.set_all(ax, title="Box distibution 3-8, N = {N}".format(N=N), legend_loc="best")


def eigenvalues_uniform(ax, N=100):
    """  Plot the eigenvalues for a uniform random matrix
    """
    W = numpy.random.uniform(-1, 1, N**2).reshape([N, N])
    eigvals = linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    ax.plot(eigvals, numpy.linspace(0, N, N), label="Cummulative eigenvalue distribution", marker='.', linestyle='')

    R=numpy.max(eigvals)
    #R=2.0
    semicircle = numpy.sqrt(numpy.ones(N)*R**2 - numpy.linspace(-R, R, N)**2)#/(2*pi)
    cum_semicircle = numpy.cumsum(semicircle)
    cum_semicircle = cum_semicircle / numpy.max(cum_semicircle)*N
    ax.plot(numpy.linspace(-R, R, N), semicircle, linestyle="--", label=r"Semi circle, with $R \approx {0:.2}$".format(R))
    ax.plot(numpy.linspace(-R, R, N), cum_semicircle, linestyle="--", label = r"Cummulative semicircle, with $R \approx {0:.2}$".format(R))

    plotdl.set_all(ax, title=r"uniform, $[-1, 1]$", legend_loc="upper left")


###############  Meta-eigenvalue #########
def eigenvalues_cummulative(ax, matrix, label):
    """  Plot the cummulative density of the eigenvalues
    """
    N = matrix.shape[0]
    eigvals = -linalg.eigvalsh(matrix)
    eigvals.sort()
    eigvals = eigvals[1:]  ## The zero (or nearly zero) is a problematic eigenvalue.
    assert  eigvals[0] >0, ("All eigenvalues [except the first] should be positive" + str(eigvals))
    ax.loglog(eigvals, numpy.linspace(1/(N-1), 1, N-1), marker=".", linestyle='', label=label)
    return eigvals
    
def cummulative_plot(ax, values, label=None):
    """  Plot cummulative values.
    """
    N = len(values)
    ax.plot(numpy.sort(values), numpy.linspace(1/N, 1, N), marker=".", linestyle='', label=label)

def plot_matrix_and_colorbar(ax, matrix):
    """
    """
    #vals, vecs = sparsedl.sorted_eigh(matrix)
    ms = ax.matshow(matrix, norm=LogNorm(vmin=10**(-10) ))
    ax.figure.colorbar(ms)

################ Plots related to the eigenvalue plots ############3

def diffusion_plot(ax, D, eigvals):
    """ """
    max_log_value = numpy.log10(numpy.min((numpy.max(eigvals), D*pi**2)))
    diffusion_space = numpy.logspace(numpy.log10(numpy.min(eigvals)), max_log_value, 100)
    diffusion = numpy.sqrt(diffusion_space/(D))/pi
    ax.loglog(diffusion_space, diffusion, linestyle='--', label="")#label = r"Square root, $\sqrt{\frac{\lambda}{(D)}}/\pi$")


def alter_analytic_plot(ax, a, b, N):
    """
    """
    space = numpy.linspace(1/N, 0.5, N // 2 )  # removed -1
    alter = sparsedl.analytic_alter(a, b, space)
    alter.sort()
    ax.loglog(alter, space, linestyle='', marker='+', label = r"Analytic alternating model")


def ones_analytic_plot(ax, N):
    """
    """
    space = numpy.linspace(1/N, 1, N)
    analytic1 = 2*(1-numpy.cos(pi*space))/ N
    approx_space = numpy.linspace(0, 1/N)
    analytic2 = numpy.arccos(1-N*approx_space/2)/pi
    approx = sqrt(N*approx_space)/pi
    #analytic.sort()
    ax.loglog(analytic1, space, linestyle='', marker='+', label = r"Analytic : $2(1-\cos(\pi n))$")
    ax.plot(approx_space, analytic2, linestyle='--', label=r"Analytic, $\cos^{-1}$")
    ax.plot(approx_space, approx, linestyle='', marker = '+', label=r"Approximation, $\sqrt{N*n}/\pi$")


####################   Sample Plots ################

def sample_plots_eig(ax_eig, sample, distance_matrix = None, xi = 1, end_log_time=1 ,show_theory=False):
    """  Create A_ij for points on a torus via e^(-r_ij).

        :param N_points: Number of points, defaults to 100
        :type N_points: int
        :param dimensions: The 2d dimensions, as a 2-tuple. defaults to (10, 10)

    """


    points = sample.points
    n = sample.number_of_points / sample.volume
    print("n = {0}, xi = {1}, n*xi = {2}, n*xi^2={3}".format(n, xi, n*xi, n*xi**2))
    if distance_matrix is None:
        dis =  geometry.fast_periodic_distance_matrix(points, sample.dimensions)
    else:
        dis = distance_matrix

    ex1 = numpy.exp(-dis/xi)
    sparsedl.zero_sum(ex1)
    assert sparsedl.zero_sum(ex1)
    ex2 = sparsedl.permute_tri(ex1)
    sparsedl.zero_sum(ex2)
    assert sparsedl.zero_sum(ex2)
    eigvals = []
    eigvals += [eigenvalues_cummulative(ax_eig, ex1, "Original values")]
    eigvals += [eigenvalues_cummulative(ax_eig, ex2, "Permuted values")]
    diagvals = - ex1.diagonal()
    diagvals.sort()
    diagvals = diagvals[1:]
    if show_theory:
        #minvallog = numpy.log10(numpy.min(eigvals[0]))
        #maxvallog = min(( numpy.log10(numpy.max(eigvals[0])) , log10(2)))
        left_end = numpy.log10(2) - sqrt(log10(sample.number_of_points)/(pi*n*xi**2))
        right_end = numpy.log10(2)
        #print "min : {0} max : {1}  minlog : {2}  maxlog : {3}".format(numpy.min(eigvals[0]),numpy.max(eigvals[0]), minvallog, maxvallog)
        theory_space= numpy.logspace(left_end, right_end,100)
        theory = numpy.exp(-(pi)*n*(xi*numpy.log(theory_space/2))**2)

        xlim, ylim = ax_eig.get_xlim(), ax_eig.get_ylim()
        ax_eig.loglog(theory_space, theory, label="theory", linestyle="--")
        ax_eig.set_xlim(xlim)
        ax_eig.set_ylim(ylim)
    plotdl.set_all(ax_eig, title = r"A {0}, {2}, $w = e^{{-r/\xi}}$, N ={1}, $\xi = {3}$, $n\xi^2 = {4}$".format(sample.description, sample.number_of_points, sample.dimensions, xi, n*xi**2), legend_loc='lower right')


def sample_2d_theory(ax, sample, epsilon):
    """ """
    n = sample.number_of_points/sample.volume
    xi = epsilon * n**(-1/sample.d)
    left_end = numpy.log10(2) - sqrt(log10(sample.number_of_points)/(pi*n*xi**2))
    right_end = numpy.log10(2)
    #print "min : {0} max : {1}  minlog : {2}  maxlog : {3}".format(numpy.min(eigvals[0]),numpy.max(eigvals[0]), minvallog, maxvallog)
    theory_space= numpy.logspace(left_end, right_end,100)
    theory = numpy.exp(-(pi)*n*(xi*numpy.log(theory_space/2))**2)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(theory_space, theory, label=r"theory $\epsilon = {0}$".format(epsilon), linestyle="--")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def sample_collect_eigenvalues(sample, N=1000, epsilon=0.1, number_of_runs=10):
    """
    """
    collected_eigenvalues = []
    n = N/sample.volume
    xi = epsilon * n**(-1/sample.d)
    print n, xi, epsilon
    for i in range(number_of_runs):
        print( "{0:03} / {1}".format(i+1, number_of_runs))
        sample.generate_points(N)
        ex1 = sample_exp_matrix(sample, epsilon)
        collected_eigenvalues += [ -linalg.eigvals(ex1)]
    all_eigenvalues = numpy.concatenate(collected_eigenvalues)
    return all_eigenvalues

def sample_exp_matrix(sample, epsilon=0.1):
    """
    """
    n = sample.number_of_points/ sample.volume
    xi = epsilon * n**(-1/sample.d)
    dis = geometry.fast_periodic_distance_matrix(sample.points, sample.dimensions)
    ex1 = numpy.exp(-dis/xi)
    sparsedl.zero_sum(ex1)
    return ex1


def loop_torus_eig():
    """ Create 5 pairs subplots of `torus_plots`
    """
    fig = plotdl.Figure()
    fig.subplots_adjust(top=0.99, bottom=0.05)
    for i in range(1, 6):
        random.seed(i)
        torus_plots_eig(fig.add_subplot(5, 1, i), N_points=250)
    plotdl.savefig(fig, "8tori_eig", size_factor=(1, 2))


def torus_3_plots(N=200):
    """
    """
    ax1 = plotdl.new_ax_for_file()
    torus = geometry.Torus((100,100), N)
    sample_plots_eig(ax1, torus, xi = 1)
    plotdl.save_ax(ax1, "torus")
    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    plotdl.save_ax(ax1, "torus_linear")

    ax2 = plotdl.new_ax_for_file()
    ax2.scatter(torus.xpoints, torus.ypoints)
    plotdl.set_all(ax2, title="Scatter plot of the points")
    plotdl.save_ax(ax2, "torus_scatter")
    
    ax3 = plotdl.new_ax_for_file()
    



def sheet_3_plots(N=200):
    """ non periodic 2d surface
    """
    ax1 = plotdl.new_ax_for_file()
    torus = geometry.Torus((100,100), N)
    dis = geometry.fast_distance_matrix(torus.points)
    sample_plots_eig(ax1, torus)
    plotdl.save_ax(ax1, "sheet")
    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    plotdl.save_ax(ax1, "sheet_linear")

def line_3_plots(N=200):
    """
    """
    ax1 = plotdl.new_ax_for_file()
    line = geometry.PeriodicLine(10, N)
    sample_plots_eig(ax1, line)
    plotdl.save_ax(ax1, "line")
    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    plotdl.save_ax(ax1, "line_linear")



#########  Torus animation #####
def torus_show_state(ax, time, torus ,xi=1):
    """
    """
    N = torus.number_of_points
    # Create initial condition rho
    rho0 = numpy.zeros(N)
    rho0[0] = 1
        
    # Create rate matrix W
    dis =  geometry.fast_periodic_distance_matrix(torus.points, torus.dimensions)
    W = numpy.exp(-dis/xi)
    sparsedl.zero_sum(W)

    # 
    rho = sparsedl.rho(time, rho0, W) 
    ax.scatter( torus.xpoints, torus.ypoints, c=rho)
    

def torus_plot_rho(ax, rho, torus, num):
    """
    """
    sct = ax.scatter(torus.xpoints, torus.ypoints, edgecolors='none',
            c=rho, norm=LogNorm( vmin=(1/torus.number_of_points)/1000, vmax =1))
    
    if num==0:
        ax.get_figure().colorbar(sct)

def replot_rho_factory(ax, rhos, torus):
    """
    """
    def replot_rho(slider_position):
        torus_plot_rho(ax, rhos[:,int(slider_position)], torus, 1)
    return replot_rho

def torus_plot_several_rhos(fig,rhos, torus):
    """
    """
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    torus_plot_rho(ax, rhos[:,0], torus, 0)
    replot_rho = replot_rho_factory(ax, rhos, torus)
    axsl = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    sl = Slider(axsl, "eigenmode", 0,rhos.shape[1],0)
    sl.on_changed(replot_rho)


def torus_list_of_rhos(torus, times, xi=1):
    """
    """
    N = torus.number_of_points
    rho0 = numpy.zeros(N)
    rho0[0] = 1
        
    # Create rate matrix W
    dis =  geometry.fast_periodic_distance_matrix(torus.points, torus.dimensions)
    W = numpy.exp(-dis/xi)
    sparsedl.zero_sum(W)

    # 
    rhos = []
    for t in times:
        rhos += [sparsedl.rho(t, rho0, W)]
    return rhos

def torus_time():
    """
    """
    times = numpy.linspace(0,1,100)
    torus = geometry.Torus((10,10),100)
    rhos = torus_list_of_rhos(torus, times)
    plotdl.animate(torus_plot_rho, "test", rhos, torus=torus)


##########
def exp_models_sample(sample=geometry.Torus((1,1)), number_of_points=300, number_of_realizations = 10):
    """
    """
    ax_exp = plotdl.new_ax_for_file()
    epsilon_list = (0.05, 0.1,0.5,1,1.5,2,5,10)
    ## 2d - torus
    if number_of_realizations >1 :
        plot_title = "{0} with {1} points, eigenvalues of ${2}$ realizations, $n=1$".format(sample.description,number_of_points, number_of_realizations )
    else:
        plot_title = "{0} with {1} points, eigenvalues for a single realization, $n=1$".format(sample.description,number_of_points)
    hist_bins = sqrt(number_of_points*number_of_realizations)
    
    logvals = {} # empty dict
    for epsilon in epsilon_list:
        eigvals = sample_collect_eigenvalues(sample, number_of_points, epsilon, number_of_realizations)
        logvals[epsilon]= numpy.log(numpy.sort(eigvals)[number_of_realizations:])
        # I'm using str to avoid trouble with 0.05000000003
    
    #all
    for epsilon in epsilon_list:
        cummulative_plot(ax_exp, logvals[epsilon], label=r"$\epsilon = {0}$".format(epsilon))
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}_{1}_0semilogx".format(sample.short_name, number_of_realizations))
    ax_exp.set_yscale('log')
    plotdl.save_ax(ax_exp, "exp_{0}_{1}_0loglog".format(sample.short_name, number_of_realizations))
    ax_exp.clear()
       
    ### low density
    for epsilon in (0.05, 0.1):
        cummulative_plot(ax_exp, logvals[epsilon], label=r"$\epsilon = {0}$".format(epsilon))
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}_{1}_low_semilogx".format(sample.short_name, number_of_realizations))
    ax_exp.set_yscale('log')
    plotdl.save_ax(ax_exp, "exp_{0}_{1}_low_loglog".format(sample.short_name, number_of_realizations))
    ax_exp.clear()
    #histogram
    for epsilon in (0.05, 0.1):
        ax_exp.hist(logvals[epsilon], bins = hist_bins, label=r"$\epsilon = {0}$".format(epsilon), histtype='step', normed=True)
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}_{1}_low_zhist".format(sample.short_name, number_of_realizations))
    ax_exp.clear()

    #high density
    for epsilon in (5, 10):
        cummulative_plot(ax_exp, logvals[epsilon], label=r"$\epsilon = {0}$".format(epsilon))
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp, "exp_{0}_{1}_high_semilogx".format(sample.short_name, number_of_realizations))
    ax_exp.set_yscale('log')
    plotdl.save_ax(ax_exp, "exp_{0}_{1}_high_loglog".format(sample.short_name, number_of_realizations))
    ax_exp.clear()
    #histogram
    for epsilon in (0.05, 0.1):
        ax_exp.hist(logvals[epsilon], bins = hist_bins,
            label=r"$\epsilon = {0}$".format(epsilon), histtype='step', normed=True)
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}_{1}_high_zhist".format(sample.short_name, number_of_realizations))

def participation_number(ax, matrix):
    """
    """
    pn = ((matrix**2).sum(axis=0))**(-1)
    ax.plot(pn)

def plot_several_vectors(fig, matrix, vec_indices):
    """
    """
    num_of_vectors = len(vec_indices)
    axes = {} # empty_dict
    
    for n,m in enumerate(vec_indices):
        if n==0:
            axes[n] = fig.add_subplot(num_of_vectors,1,n+1)
        else:
            axes[n] = fig.add_subplot(num_of_vectors,1,n+1, sharex=axes[0], sharey=axes[0])
        axes[n].plot(matrix[:,m], label = "eigenmode {0}".format(m))
        axes[n].legend()

    
######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    random.seed(seed)
    ax = plotdl.new_ax_for_file()
    p_lognormal_band(ax)
    plotdl.save_ax(ax, "P_lognormal_band")
    ax.clear()

    random.seed(seed)
    eigenvalues_lognormal(ax, b_list=(1, 5, 10, 20, 40))
    plotdl.save_ax(ax, "eigvals_lognormal_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_lognormal_normal")
    ax.clear()

    eigenvalues_ones(ax, N=200, b_list=(1, 5, 10, 20, 30))
    plotdl.save_ax(ax, "eigvals_ones_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_ones_normal")
    ax.clear()

    eigenvalues_alter(ax, N=200, b_list=(1, 5, 10, 20, 30))
    plotdl.save_ax(ax, "eigvals_alter_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_alter_normal")
    ax.clear()

    random.seed(seed)
    eigenvalues_box(ax, N=200, b_list=(1, 5, 10, 20, 30))
    plotdl.save_ax(ax, "eigvals_box_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_box_normal")
    ax.clear()

    random.seed(seed)
    spreading_plots(ax)
    plotdl.save_ax(ax, "spreading")
    ax.clear()

    eigenvalues_uniform(ax)
    plotdl.save_ax(ax, "eigvals_uniform")
    random.seed(seed)
    torus_3_plots()
    ax.clear()
    
    random.seed(seed)
    exp_models_sample(sample=geometry.Torus((1,1)), number_of_points=300, number_of_realizations = 10)
    exp_models_sample(sample=geometry.PeriodicLine((1)), number_of_points=300, number_of_realizations = 10)
    



if __name__ ==  "__main__":
    all_plots()
