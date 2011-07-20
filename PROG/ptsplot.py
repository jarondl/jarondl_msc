#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
#from scipy.sparse import linalg as splinalg
from numpy import linalg, random, pi, log10, sqrt
from matplotlib.colors import LogNorm
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


def eigenvalues_exponent_minus1(ax, N=100, nxi=0.3):
    """  Plot the eigenvalues for a :math:p(w) = w^{n\\xi-1}n\\xi:
    """
    W = sparsedl.exponent_minus1(N, nxi=nxi).todense()
    eigvals = - linalg.eigvalsh(W)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    eigvals = eigvals[2:]/N  ## The first eigenvalue is zero, which does problems with loglog plots
    power_law_space = numpy.logspace(numpy.log10(numpy.min(eigvals[1:])), numpy.log10(numpy.max(eigvals)), N-1)
    power_law = (power_law_space**(nxi))
    ax.loglog(eigvals, numpy.linspace(0, 1, N-2), marker='.', linestyle='', label="Cummulative eigenvalues (divided by N)")
    ax.loglog(power_law_space, power_law, linestyle='--', label=r"\lambda^{n\xi}")
    plotdl.set_all(ax, title=r"$p(w) = w^{n\xi-1}n\xi $ Where $n\xi=$"+str(nxi), legend_loc="lower right")





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


def eigenvalues_lognormal_normal_axis(ax, N=200, b_list=(1, )):
    """  Plot the eigenvalues for a lognormal sparse banded matrix
    """
    eigenvalues_lognormal(ax, N=N, b_list=b_list)
    ax.set_xscale('linear')
    ax.set_yscale('linear')


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
    ax.plot(values, numpy.linspace(1/N, 1, N), marker=".", linestyle='', label=label)

def eigenvalues_density(ax, matrix, label):
    """  Plot the density of the eigenvalues
    """
    N = matrix.shape[0]
    eigvals = -linalg.eigvalsh(matrix)
    eigvals.sort()
    eigvals = eigvals[1:]  ## The zero (or nearly zero) is a problematic eigenvalue.
    assert  eigvals[0] >0, ("All eigenvalues [except the first] should be positive" + str(eigvals))
    left_lim = log10(eigvals[0])
    right_lim = log10(eigvals[-1])
    #density_space = eigvals[:-1]
    log_lambda = numpy.log(eigvals)
    density = ((log_lambda[1:] - log_lambda[:-1])**(-1))
    ax.plot(log_lambda[:-1], density, marker=".", linestyle='', label=label)
    return eigvals

def plot_density(ax, values):
    """
    """
    values.sort()
    density = (values[1:]-values[:-1])/len(values)
    ax.plot(values[:-1], density, marker=".", linestyle='')
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
        points = sample.points
        dis = geometry.fast_periodic_distance_matrix(points, sample.dimensions)
        ex1 = numpy.exp(-dis/xi)
        sparsedl.zero_sum(ex1)
        collected_eigenvalues += [ -linalg.eigvals(ex1)]
    all_eigenvalues = numpy.concatenate(collected_eigenvalues)
    return all_eigenvalues

def sample_cummulative_avg(ax, epsilon_list=(0.1,), number_of_points=500, number_of_realizations=20, sample=geometry.Torus((1,1)), ax_hist=None):
    """
    """
    for epsilon in epsilon_list:
        eigvals = sample_collect_eigenvalues(sample, number_of_points, epsilon, number_of_realizations)
        logvals = numpy.log(numpy.sort(eigvals)[number_of_realizations:])
        cummulative_plot(ax, logvals, label=r"$\epsilon = {0}$".format(epsilon))
        if ax_hist is not None:
            ax_hist.hist(logvals, bins = sqrt(number_of_points*number_of_realizations), label=r"$\epsilon = {0}$".format(epsilon), histtype='step', normed=True)
    if ax_hist is not None:
        ax_hist.legend()

def torus_avg(ax_eig, N_points=1000, dimensions=(1, 1), xi = 0.0032, end_log_time=1, avg_N=10):
    """
    """
    sum_of_eigvals = []
    for i in range(avg_N):
        torus = geometry.Torus(dimensions)
        points = torus.generate_points(N_points)
        n = N_points / (dimensions[0]*dimensions[1])
        print("n = {0}, xi = {1}, n*xi = {2}, n*xi^2={3}".format(n, xi, n*xi, n*xi**2))
        dis =  geometry.fast_periodic_distance_matrix(points, torus.dimensions)
        rnn = sparsedl.rnn(dis)
        print("Rnn = "+str(rnn)+ " xi/rnn = "+str(xi/rnn))

        ex1 = numpy.exp(-dis/xi)
        sparsedl.zero_sum(ex1)
        ex2 = sparsedl.permute_tri(ex1)
        ex3 = sparsedl.keep_only_nn(ex1)  # new addition, keep only nn
        #print(ex3)
        ex4 = numpy.copy(ex3)
        print(sparsedl.zero_sum(ex4))
        #print(ex3.diagonal() - ex4.diagonal())
        eigvals = []
        for ex in (ex1, ex2, ex3, ex4):
            eig = -linalg.eigvals(ex)
            eig.sort()
            eig = eig[1:] # The zero is problematic for the plots
            eigvals += [eig]
        diagvals = - ex1.diagonal()
        diagvals.sort()
        diagvals = diagvals[1:]
        eigvals += [diagvals]  #### Only the diagonal values. Should resemble the others.
        minvallog = numpy.log10(min(numpy.min(eigvals[0]), numpy.min( eigvals[1])))
        maxvallog = numpy.log10(max(numpy.max(eigvals[0]), numpy.max( eigvals[1])))
        theory_space = numpy.logspace(0, 2, 100)
        theory = 1 - numpy.exp(-(pi/2)*((xi/rnn)*numpy.log(theory_space/2))**2)
        if sum_of_eigvals == [] :
            sum_of_eigvals = deepcopy(eigvals)
        else:
            for i in range(5):
                sum_of_eigvals[i] += eigvals[i]
    avg_eigvals = [ values/ avg_N for values in sum_of_eigvals]

    ax_eig.loglog(avg_eigvals[0], numpy.linspace(0, 1, N_points-1), label="original", marker='.', linestyle='')
    ax_eig.loglog(avg_eigvals[1], numpy.linspace(0, 1, N_points-1), label="permuted", marker='.', linestyle='')
    ax_eig.loglog(avg_eigvals[2], numpy.linspace(0, 1, N_points-1), label="only nn, same diagonal", marker='.', linestyle='')
    ax_eig.loglog(avg_eigvals[3], numpy.linspace(0, 1, N_points-1), label="only nn, zero-summed", marker='.', linestyle='')
    ax_eig.loglog(avg_eigvals[4], numpy.linspace(0, 1, N_points-1), label="Only the diagonals", marker='.', linestyle='')

    xlim, ylim = ax_eig.get_xlim(), ax_eig.get_ylim()
    #ax_eig.loglog(theory_space, theory, label="theory", linestyle="--")
    ax_eig.legend(loc='upper left')
    ax_eig.set_xlim(xlim)
    ax_eig.set_ylim(ylim)



def loop_torus_eig():
    """ Create 5 pairs subplots of `torus_plots`
    """
    fig = plotdl.Figure()
    fig.subplots_adjust(top=0.99, bottom=0.05)
    for i in range(1, 6):
        random.seed(i)
        torus_plots_eig(fig.add_subplot(5, 1, i), N_points=250)
    plotdl.savefig(fig, "8tori_eig", size_factor=(1, 2))


def torus_permutation_noax(N_points=100, dimensions=(10, 10), filename="torus_perm"):
    """
    """
    torus = geometry.Torus(dimensions)
    points = torus.generate_points(N_points)
    dis =  geometry.fast_periodic_distance_matrix(points, torus.dimensions)

    fig = plotdl.Figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ex1 = numpy.exp(-dis)
    sparsedl.zero_sum(ex1)
    ex2 = sparsedl.permute_tri(ex1)
    mat1 = ax1.matshow(ex1) #, norm=LogNorm(vmin=numpy.min(ex1), vmax=numpy.max(ex1))
    mat2 = ax2.matshow(ex2)
    #fig.colorbar(mat1)
    #fig.colorbar(mat2)
    plotdl.savefig(fig, filename)

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
            c=rho, norm=LogNorm( vmin=1/torus.number_of_points, vmax =1))
    
    if num==0:
        ax.get_figure().colorbar(sct)

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
    rho = []
    for t in times:
        rho += [sparsedl.rho(t, rho0, W)]
    return rho

def torus_time():
    """
    """
    times = numpy.linspace(0,1,100)
    torus = geometry.Torus((10,10),100)
    rhos = torus_list_of_rhos(torus, times)
    plotdl.animate(torus_plot_rho, "test", rhos, torus=torus)

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
    
    ####################
    random.seed(seed)
    torus = geometry.Torus((1,1))
    sample_cummulative_avg(ax,(0.1,0.5,1,1.5,2,5), 300, 10, torus)
    plotdl.set_all(ax, title="2d surface with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax,"avg_torus_0semilogx")
    ax.set_yscale('log')
    plotdl.save_ax(ax, "avg_torus_0loglog")
    ax.clear()
    
    # high density region (high xi)(with hist)
    ax_hist = plotdl.new_ax_for_file()
    random.seed(seed)
    torus = geometry.Torus((1,1))
    sample_cummulative_avg(ax,(5,10), 300, 10, torus, ax_hist=ax_hist)
    plotdl.set_all(ax, title="2d surface with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax,"avg_torus_high_semilogx")
    ax.set_yscale('log')
    plotdl.save_ax(ax, "avg_torus_high_loglog")
    ax.clear()
    #hist
    plotdl.set_all(ax_hist, title="2d surface with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_hist,"avg_torus_high_zhist")
    ax_hist.clear()
    
    # low density region (low xi)(with hist)
    random.seed(seed)
    torus = geometry.Torus((1,1))
    sample_cummulative_avg(ax,(0.05,0.1), 300, 10, torus, ax_hist=ax_hist)
    plotdl.set_all(ax, title="2d surface with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax,"avg_torus_low_semilogx")
    ax.set_yscale('log')
    plotdl.save_ax(ax, "avg_torus_low_loglog")
    ax.clear()
    #hist
    plotdl.set_all(ax_hist, title="2d surface with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_hist,"avg_torus_low_zhist")
    ax_hist.clear()

    random.seed(seed)
    line = geometry.PeriodicLine(1)
    sample_cummulative_avg(ax,(0.1,0.5,1,1.5,2,5), 300, 10, line)
    plotdl.set_all(ax, title="1d line with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax,"avg_line_0semilogx")
    ax.set_yscale('log')
    plotdl.save_ax(ax, "avg_line_0loglog")
    ax.clear()
    
    #high density region: +hist
    random.seed(seed)
    line = geometry.PeriodicLine(1)
    sample_cummulative_avg(ax,(5,10), 300, 10, line, ax_hist=ax_hist)
    plotdl.set_all(ax, title="1d line with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax,"avg_line_high_semilogx")
    ax.set_yscale('log')
    plotdl.save_ax(ax, "avg_line_high_loglog")
    ax.clear()
    #hist
    plotdl.set_all(ax_hist, title="1d line with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_hist,"avg_line_high_zhist")
    ax_hist.clear()
    
    #low density region: +hist
    random.seed(seed)
    line = geometry.PeriodicLine(1)
    sample_cummulative_avg(ax,(0.05,0.1), 300, 10, line, ax_hist=ax_hist)
    plotdl.set_all(ax, title="1d line with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax,"avg_line_low_semilogx")
    ax.set_yscale('log')
    plotdl.save_ax(ax, "avg_line_low_loglog")
    ax.clear()
    #hist
    plotdl.set_all(ax_hist, title="1d line with 300 points, eigenvalues of $10$ realizations, $n=1$", xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_hist,"avg_line_low_zhist")
    ax_hist.clear()
    



if __name__ ==  "__main__":
    all_plots()
