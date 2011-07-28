#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
#from scipy.sparse import linalg as splinalg
from numpy import linalg, random, pi, log10, sqrt
from matplotlib.colors import LogNorm
from matplotlib.cm import summer
from matplotlib.widgets import Slider
from copy import deepcopy

import numpy

import sparsedl
import plotdl
import geometry
from geometry import Sample
from eigenvalue_plots import eigenvalues_cummulative
from plotdl import cummulative_plot

### Raise all float errors 
numpy.seterr(all='warn')


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
        print(numpy.sort(vals))
        #survs = sparsedl.surv(vals, t) ## has problem with e^{-800}
        survs = sparsedl.safe_surv(vals, t)
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



####################   Sample Plots ################

def sample_plots_eig(ax_eig, sample, distance_matrix = None, epsilon = 0.1, end_log_time=1 ,show_theory=False):
    """  Create A_ij for points on a torus: e^(-r_ij).

        :param N_points: Number of points, defaults to 100
        :type N_points: int
        :param dimensions: The 2d dimensions, as a 2-tuple. defaults to (10, 10)

    """


    points = sample.points
    n = sample.number_of_points / sample.volume
    xi = sample.epsilon_to_xi(epsilon)
    print("n = {0}, xi = {1}, n*xi = {2}, n*xi^2={3}".format(n, xi, n*xi, n*xi**2))
    if distance_matrix is None:
        dis =  sample.periodic_distance_matrix()
    else:
        dis = distance_matrix

    ex1 = numpy.exp(-dis/xi)
    sparsedl.zero_sum(ex1)
    assert sparsedl.zero_sum(ex1)
    ex2 = sparsedl.permute_tri(ex1)
    sparsedl.zero_sum(ex2)
    assert sparsedl.zero_sum(ex2)
    ex3 = sparsedl.permute_rows(ex1)
    sparsedl.zero_sum(ex3)
    assert sparsedl.zero_sum(ex3)
    eigvals = []
    eigvals += [eigenvalues_cummulative(ax_eig, ex1, "Original values")]
    eigvals += [eigenvalues_cummulative(ax_eig, ex2, "Permuted values")]
    eigvals += [eigenvalues_cummulative(ax_eig, ex3, "Permuted rows only")]
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
    plotdl.set_all(ax_eig, title = r"{0}d, $w = e^{{-r/\xi}}$, $N ={1}$, $\epsilon={2}$".format(sample.d, sample.number_of_points, epsilon), legend_loc='lower right')


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
    xi = sample.epsilon_to_xi(epsilon)
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
    torus = Sample((1,1), N)
    sample_plots_eig(ax1, torus)
    plotdl.save_ax(ax1, "torus")
    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    plotdl.save_ax(ax1, "torus_linear")

    ax2 = plotdl.new_ax_for_file()
    ax2.scatter(torus.points[:,0], torus.points[:,1])
    plotdl.set_all(ax2, title="Scatter plot of the points")
    plotdl.save_ax(ax2, "torus_scatter")
    
    ax3 = plotdl.new_ax_for_file()
    



def sheet_3_plots(N=200):
    """ non periodic 2d surface
    """
    ax1 = plotdl.new_ax_for_file()
    torus = Sample((100,100), N)
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
    line = Sample(10, N)
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
    

def torus_plot_rho(ax, rho, torus, colorbar=False ):
    """
    """
    sct = ax.scatter(torus.xpoints, torus.ypoints, edgecolors='none',
            c=rho, norm=LogNorm( vmin=(1/torus.number_of_points)/1000, vmax =1))
    
    if colorbar:
        ax.get_figure().colorbar(sct)

def replot_rho_factory(ax, rhos, torus, eigvals):
    """
    """
    def replot_rho(slider_position):
        ax.clear()
        pos = int(slider_position)
        eigval = eigvals[pos]
        participation_number = ((rhos[:,pos]**2).sum(axis = 0))**(-1)
        torus_plot_rho(ax, rhos[:,pos], torus, colorbar=False)
        ax.set_title(r"\#${0}, \lambda={1}, PN = {2}$".format(pos, eigval, participation_number))
        plotdl.draw()
    return replot_rho

def torus_rhos_slider(fig,rhos, torus, eigvals):
    """
    """
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    torus_plot_rho(ax, rhos[:,0], torus, colorbar=True)
    replot_rho = replot_rho_factory(ax, rhos, torus, eigvals)
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
    torus = Sample((10,10),100)
    rhos = torus_list_of_rhos(torus, times)
    plotdl.animate(torus_plot_rho, "test", rhos, torus=torus)


##########
def exp_models_sample(sample=Sample((1,1)), number_of_points=300, number_of_realizations = 10):
    """
    """
    ax_exp = plotdl.new_ax_for_file()
    epsilon_list = (0.05, 0.1,0.5,1,1.5,2,5,10)
    ## 2d - torus
    if number_of_realizations >1 :
        plot_title = "{0}d with {1} points, eigenvalues of ${2}$ realizations, $n=1$".format(sample.d,number_of_points, number_of_realizations )
    else:
        plot_title = "{0}d with {1} points, eigenvalues for a single realization, $n=1$".format(sample.d,number_of_points)
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
    plotdl.save_ax(ax_exp,"exp_{0}d_{1:02}_0semilogx".format(sample.d, number_of_realizations))
    ax_exp.set_yscale('log')
    plotdl.save_ax(ax_exp, "exp_{0}d_{1:02}_0loglog".format(sample.d, number_of_realizations))
    ax_exp.clear()
       
    ### low density
    for epsilon in (0.05, 0.1):
        cummulative_plot(ax_exp, logvals[epsilon], label=r"$\epsilon = {0}$".format(epsilon))
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}d_{1:02}_low_semilogx".format(sample.d, number_of_realizations))
    ax_exp.set_yscale('log')
    plotdl.save_ax(ax_exp, "exp_{0}d_{1:02}_low_loglog".format(sample.d, number_of_realizations))
    ax_exp.clear()
    #histogram
    for epsilon in (0.05, 0.1):
        ax_exp.hist(logvals[epsilon], bins = hist_bins, label=r"$\epsilon = {0}$".format(epsilon), histtype='step', normed=True)
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}d_{1:02}_low_zhist".format(sample.d, number_of_realizations))
    ax_exp.clear()

    #high density
    for epsilon in (5, 10):
        cummulative_plot(ax_exp, logvals[epsilon], label=r"$\epsilon = {0}$".format(epsilon))
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp, "exp_{0}d_{1:02}_high_semilogx".format(sample.d, number_of_realizations))
    ax_exp.set_yscale('log')
    plotdl.save_ax(ax_exp, "exp_{0}d_{1:02}_high_loglog".format(sample.d, number_of_realizations))
    ax_exp.clear()
    #histogram
    for epsilon in (5, 10):
        ax_exp.hist(logvals[epsilon], bins = hist_bins,
            label=r"$\epsilon = {0}$".format(epsilon), histtype='step', normed=True)
    plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax_exp,"exp_{0}d_{1:02}_high_zhist".format(sample.d, number_of_realizations))

def participation_number(ax, matrix):
    """
    """
    pn = ((matrix**2).sum(axis=0))**(-1)
    ax.plot(pn)
    

def plot_several_vectors(fig, matrix, vec_indices, x_values = None):
    """
    """
    num_of_vectors = len(vec_indices)
    axes = {} # empty_dict
    
    for n,m in enumerate(vec_indices):
        if n==0:
            axes[n] = fig.add_subplot(num_of_vectors,1,n+1)
        else:
            axes[n] = fig.add_subplot(num_of_vectors,1,n+1, sharex=axes[0], sharey=axes[0])
        if x_values is None:
            axes[n].plot(matrix[:,m], label = "eigenmode {0}".format(m))
        else:
            axes[n].plot(x_values, matrix[:,m], label = "eigenmode {0}".format(m))
        axes[n].legend()

def sample_participation_number(ax, sample, epsilon=0.1):
    """
    """
    exp_mat = sample_exp_matrix(sample, epsilon)
    eigvals, eigvecs = sparsedl.sorted_eigh(exp_mat)
    pn = ((eigvecs**4).sum(axis=0))**(-1)
    ax.plot(pn, label="PN - participation number")
    ax.axhline(y=1, label="1 - the minimal PN possible", linestyle="--", color="red")
    ax.axhline(y=2, label="2 - dimer", linestyle="--", color="green")

    dots_to_annotate = (pn > (sample.number_of_points/3)).nonzero()[0]
    _arrowprops=dict(arrowstyle="->")
    for dot in dots_to_annotate:
        print dot, pn[dot]
        ax.annotate(r"${x},{y}, \lambda={l}$".format(x=dot, y=pn[dot], l= eigvals[dot]), xy=(dot,pn[dot]), xycoords='data', xytext=(20,20), 
            textcoords='offset points', arrowprops=_arrowprops)
    
    
######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    ax = plotdl.new_ax_for_file()

    #random.seed(seed)
    #p_lognormal_band(ax)
    #plotdl.save_ax(ax, "P_lognormal_band")
    #ax.clear()

    random.seed(seed)
    spreading_plots(ax)
    plotdl.save_ax(ax, "spreading")
    ax.clear()


    random.seed(seed)
    torus_3_plots()
    ax.clear()
    
    random.seed(seed)
    exp_models_sample(sample=Sample((1,1)), number_of_points=300, number_of_realizations = 10)
    exp_models_sample(sample=Sample((1,1)), number_of_points=300, number_of_realizations = 1)
    exp_models_sample(sample=Sample((1)), number_of_points=300, number_of_realizations = 10)
    



if __name__ ==  "__main__":
    all_plots()
