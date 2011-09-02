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
from sparsedl import sorted_eigvalsh
from plotdl import cummulative_plot

### Raise all float errors 
numpy.seterr(all='warn')
EXP_MAX_NEG = numpy.log(numpy.finfo( numpy.float).tiny)


####################   Sample Plots ################

def plot_exp_model_permutation(ax, sample, epsilon = 0.1, end_log_time=1 ,show_theory=False):
    """  Create A_ij for points on a torus: e^(-r_ij).

        :param N_points: Number of points, defaults to 100
        :type N_points: int
        :param dimensions: The 2d dimensions, as a 2-tuple. defaults to (10, 10)

    """

    ex1 = exp_model_matrix(sample, epsilon)
    sparsedl.zero_sum(ex1)
    ex2 = sparsedl.permute_tri(ex1)
    sparsedl.zero_sum(ex2)
    ex3 = sparsedl.permute_diagonals(ex1)
    sparsedl.zero_sum(ex3)
    
    cummulative_plot(ax, sorted_eigvalsh(ex1)[1:], "Original values")
    cummulative_plot(ax, sorted_eigvalsh(ex2)[1:], "Permuted values")
    cummulative_plot(ax, sorted_eigvalsh(ex3)[1:], "Diagonaly permuted values")

    plotdl.set_all(ax, title = r"{0}d, $w = e^{{-r/\xi}}$, $N ={1}$, $\epsilon={2}$".format(sample.d, sample.number_of_points, epsilon),
                   legend_loc='lower right', xlabel=r"$\lambda$", ylabel=r"$C(\lambda)$")


def sample_2d_theory(ax, sample, epsilon):
    """ """
    n = sample.number_of_points/sample.volume
    xi = sample.epsilon_to_xi(epsilon)
    left_end = numpy.log10(2) - sqrt(log10(sample.number_of_points)/(pi*n*xi**2))
    right_end = numpy.log10(2)
    #print "min : {0} max : {1}  minlog : {2}  maxlog : {3}".format(numpy.min(eigvals[0]),numpy.max(eigvals[0]), minvallog, maxvallog)
    theory_space= numpy.logspace(left_end, right_end,100)
    theory = numpy.exp(-(pi)*n*(xi*numpy.log(theory_space/2))**2)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(theory_space, theory, label=r"theory $\epsilon = {0}$".format(epsilon), linestyle="--")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def sample_collect_eigenvalues(sample, epsilon=0.1, number_of_runs=10):
    """
    """
    collected_eigenvalues = []
    for i in range(number_of_runs):
        print( "{0:03} / {1}".format(i+1, number_of_runs))
        sample.generate_points()
        ex1 = exp_model_matrix(sample, epsilon)
        collected_eigenvalues += [ -linalg.eigvals(ex1)]
    all_eigenvalues = numpy.concatenate(collected_eigenvalues)
    return all_eigenvalues

def exp_model_matrix(sample, epsilon=0.1): ## rename from sample exp
    """ Creats W_{nm} = exp((r_0-r_{nm})/xi). The matrix is zero summed and should be symmetric.

        :param sample: The sample, with generated points.
        :type sample: geometry.Sample
        :param epsilon: the epsilon, defaults to 0.1
    """
    xi = sample.epsilon_to_xi(epsilon)
    dis = sample.periodic_distance_matrix()
    ## check for minimal exp probelm (arround exp(-800))
    #underflows = (-dis/xi < EXP_MAX_NEG).sum()
    #if underflows != 0:
    #    print "### {0} underflows out of {1} ".format(underflows, dis.size)

    # new -renormalization
    r_0 = dis.sum()/(dis.size)
    print(r_0)
    r_0_mat = numpy.ones(dis.shape)*r_0
    ex1 = numpy.exp((r_0_mat-dis)/xi)
#    ex1 = ex1*numpy.exp(r_0/xi)
    #sparsedl.zero_sum(ex1)
    lamb_0 = sparsedl.new_zero_sum(ex1)
    #  Check for symmetry
    print (ex1-ex1.T).nonzero()
    print(lamb_0)
    print (ex1.max(), numpy.abs(ex1).min())
    #assert (ex1 == ex1.T).all()
    return ex1 - numpy.eye(ex1.shape[0])*lamb_0


def torus_3_plots(N=200):
    """
    """
    ax1 = plotdl.new_ax_for_file()
    torus = Sample((1,1), N)
    plot_exp_model_permutation(ax1, torus)
    plotdl.save_ax(ax1, "torus")
    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    plotdl.save_ax(ax1, "torus_linear")

    ax2 = plotdl.new_ax_for_file()
    ax2.scatter(torus.points[:,0], torus.points[:,1])
    plotdl.set_all(ax2, title="Scatter plot of the points")
    plotdl.save_ax(ax2, "torus_scatter")



##########


#### Sample one realization with diffusion, and same points for all realizations##
def exp_models_D(sample, epsilon_ranges=((0.05, 0.1,0.5,1,1.5,2,5,10),(0.05, 0.1),(5,10)), epsilon_range_names=("all","low","high")):
    """ creates files
    """
    #temp:
    number_of_realizations = 1
    ax_exp = plotdl.new_ax_for_file()
    epsilon_list = set()
    epsilon_list.update(*epsilon_ranges)
    print(epsilon_list)
    #epsilon_list = (0.05, 0.1,0.5,1,1.5,2,5,10)
    plot_title = "{0}d with {1} points, eigenvalues for a single realization, $n=1$".format(sample.d,sample.number_of_points)
    hist_bins = sqrt(sample.number_of_points*number_of_realizations)
    
    ## First we create all the eigenvalues for all epsilons and realizations
    logvals = {} # empty dict
    rate_matrix = {}
    for epsilon in epsilon_list:
        ex1 = exp_model_matrix(sample, epsilon)
        eigvals= -linalg.eigvals(ex1)
        logvals[epsilon]= numpy.log(numpy.sort(eigvals)[1:])
        rate_matrix[epsilon] = ex1.copy()
        
    
    for range_name, epsilon_list in zip(epsilon_range_names, epsilon_ranges):
        for epsilon in sorted(epsilon_list):
            cummulative_plot(ax_exp, logvals[epsilon], label=r"$\epsilon = {0}$".format(epsilon))
            ### diffusion
            diff_coef = sparsedl.resnet(rate_matrix[epsilon], 1)
            maxvallog = numpy.min((numpy.log(diff_coef*pi**2), logvals[epsilon][-1]))
            print("maxvallog  " + str(numpy.log(diff_coef*pi**2))+ "  "+str(logvals[epsilon][-1]))
            diffusion_space = numpy.exp(numpy.linspace(logvals[epsilon][0],maxvallog, 100))
            diffusion = numpy.sqrt(diffusion_space/(diff_coef))/pi
            ax_exp.plot(numpy.log(diffusion_space), diffusion, linestyle='--', label="")
                
        plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$C(\lambda)$", legend_loc="best")
        plotdl.save_ax(ax_exp, "exp_{0}d_{1:02}_{2}_semilogx".format(sample.d, number_of_realizations, range_name))
        ax_exp.set_yscale('log')
        plotdl.save_ax(ax_exp, "exp_{0}d_{1:02}_{2}_loglog".format(sample.d, number_of_realizations, range_name))
        ax_exp.clear()
        ### Histogram
        for epsilon in sorted(epsilon_list):
            ax_exp.hist(logvals[epsilon], bins = hist_bins, label=r"$\epsilon = {0}$".format(epsilon), histtype='step', normed=True)
        plotdl.set_all(ax_exp, title=plot_title, xlabel="$\log\lambda$", ylabel="$P(\lambda)$", legend_loc="best")
        plotdl.save_ax(ax_exp,"exp_{0}d_{1:02}_{2}_zhist".format(sample.d, number_of_realizations, range_name))
        ax_exp.clear()



def plot_participation_number(ax, matrix):
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
    exp_mat = exp_model_matrix(sample, epsilon)
    eigvals, eigvecs = sparsedl.sorted_eigh(exp_mat)
    pn = ((eigvecs**4).sum(axis=0))**(-1)
    ax.plot(pn, label="PN - participation number")
    ax.axhline(y=1, label="1 - the minimal PN possible", linestyle="--", color="red")
    ax.axhline(y=2, label="2 - dimer", linestyle="--", color="green")


def plotf_eig_matshow_pn(sample = Sample(1,200), epsilon=20):#used to be high_epsilons
    """
    """
    number_of_points = sample.number_of_points
    fig = plotdl.Figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharex=ax1)
#    ax3 = fig.add_subplot(2,1,3)#,sharex=ax1)
    ex = exp_model_matrix(sample, epsilon=epsilon)
    v,w = sparsedl.sorted_eigh(ex)
    cummulative_plot(ax1, (-v)[1:])
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    plotdl.set_all(ax1, title="Cummulative eigenvalues $N={0}, \epsilon = {1}$".format(number_of_points, epsilon), xlabel=r"$\lambda$")
    #plotdl.save_ax(ax, "exp_1d_{0}_{1}_eigvals".format(number_of_points, epsilon))
    #ax2.set_xscale('log')
    ax2.set_ylim(bottom=0)

    pn = ((w**4).sum(axis=0))**(-1)
    ax2.plot((-v[1:]),pn[1:], label="PN - participation number", marker=".", linestyle='')
    ax2.axhline(y=1, label="1 - the minimal PN possible", linestyle="--", color="red")
    ax2.axhline(y=2, label="2 - dimer", linestyle="--", color="green")
    ax2.set_yscale('symlog', linthreshx=1)
    plotdl.set_all(ax2, title=r"$N={0}, \epsilon = {1}$".format(number_of_points, epsilon))
    #plotdl.save_ax(ax2, "exp_1d_{0}_{1}_participation".format(number_of_points, epsilon))
    plotdl.save_fig(fig, "exp_1d_{0}_{1}_test".format(number_of_points, epsilon))

    #ax.clear()
    ax3 = plotdl.new_ax_for_file()
    plotdl.matshow_cb(ax3, w**2, vmin=10**(-10), colorbar=True)
    plotdl.set_all(ax3, title=r"$N={0}, \epsilon = {1}$".format(number_of_points, epsilon))
    plotdl.save_ax(ax3, "exp_1d_{0}_{1}_matshow".format(number_of_points, epsilon))

    #plotdl.save_fig(fig, "exp_1d_{0}_{1}_test".format(number_of_points, epsilon))

######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    ax = plotdl.new_ax_for_file()


    line = Sample(1,200)
    random.seed(1)
    plotf_eig_matshow_pn(line, epsilon=0.5)
    random.seed(1)
    plotf_eig_matshow_pn(line, epsilon=20)
    random.seed(1)
    plotf_eig_matshow_pn(line, epsilon=5)

    random.seed(seed)
    torus_3_plots()
    ax.clear()
    
    random.seed(seed)
    exp_models_D(Sample((1,),300), epsilon_ranges=((0.5,0.8,1,1.5,2,5,10),(0.5,0.8),(5,10)))
    exp_models_D(Sample((1,1),300))
    exp_models_D(Sample((1,1,1),300))
    
    

if __name__ ==  "__main__":
    all_plots()
