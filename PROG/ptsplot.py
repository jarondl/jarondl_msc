#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
#from scipy.sparse import linalg as splinalg
from numpy import linalg, random, pi, log10, sqrt, log
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
from sparsedl import sorted_eigvalsh, banded_ones
from plotdl import cummulative_plot

### Raise all float errors 
numpy.seterr(all='warn')
EXP_MAX_NEG = numpy.log(numpy.finfo( numpy.float).tiny)

def resnet_1d_plot(ax, diff_coef, x1, x2, y1, y2):
    """ Plots 1d diffusion, in the current xlims
    """
    #right bound:
    xr = min((x2,  diff_coef*pi**2))
    xl = x1
    diffusion_space = numpy.linspace(xl, xr, 100)
    diffusion = sqrt((diffusion_space)/(diff_coef))/pi
    
    ax.plot(diffusion_space, diffusion, linestyle='--', label=r"$D \approx {0:.3G}$".format(diff_coef))


def resnet_1d_logplot(ax, diff_coef, x1, x2, y1, y2,label):
    """ Plots 1d diffusion, treating the x value as log10.
    """
    #right bound:
    xr = min((x2,  diff_coef*pi**2))
    xl = x1
    diffusion_space = numpy.linspace(log10(xl), log10(xr), 100)
    diffusion = (sqrt((10**diffusion_space)/(diff_coef))/pi)
    ax.plot(diffusion_space, diffusion, linestyle='--', label=label)
    

def power_law_logplot(ax, power, coeff, bbox,label):
    """ Plots 1d diffusion, treating the x value as log10.
    """
    #right bound:
    x1,x2,x3,x4 = bbox
    xr_log = min((log10(x2), -log10(coeff)/power))
    xl = x1
    power_space = numpy.linspace(log10(xl), xr_log, 100)
    power_law = coeff*(10**power_space)**(power)
    ax.plot(power_space, power_law, linestyle='--', label=label)


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
    n = sample.n
    r_0 = sample.r_0
    xi = sample.epsilon_to_xi(epsilon)
    left_end = numpy.log10(2) - sqrt(log10(sample.number_of_points)/(pi*n*xi**2))
    right_end = numpy.log10(2)
    #print "min : {0} max : {1}  minlog : {2}  maxlog : {3}".format(numpy.min(eigvals[0]),numpy.max(eigvals[0]), minvallog, maxvallog)
    theory_space= numpy.logspace(left_end, right_end,100)
    theory = numpy.exp(-(-pi*xi/(2*r_0)*log(10**theory_space))**2)

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

def exp_model_matrix(sample, epsilon=0.1, bandwidth=None): ## rename from sample exp
    """ Creats W_{nm} = exp((r_0-r_{nm})/xi). The matrix is zero summed and should be symmetric.

        :param sample: The sample, with generated points.
        :type sample: geometry.Sample
        :param epsilon: the epsilon, defaults to 0.1
    """
    xi = sample.epsilon_to_xi(epsilon)
    dis = sample.periodic_distance_matrix()
    if sample.d ==  1:
        if bandwidth is None:
            dis = dis*banded_ones(dis.shape[0], 1)
        else:
            dis = dis*banded_ones(dis.shape[0], bandwidth)
    # new -renormalization
    r_0_N = sample.r_0
    r_0 = dis.sum()/(dis.shape[0]**2)
    print("r_0_N", r_0_N)
    print("old_r_0", r_0)
    r_0_mat = numpy.ones(dis.shape)*r_0
    ex1 = numpy.exp((r_0_mat-dis)/xi)
    # handle bandwidth for 1d. the default is 1.
    if sample.d ==  1:
        if bandwidth is None:
            ex1 = ex1*banded_ones(dis.shape[0], 1)
        elif bandwidth != 0: #### Zero means ignore bandwidth
            ex1 = ex1*banded_ones(dis.shape[0], bandwidth)
    sparsedl.zero_sum(ex1)
    #assert (ex1 == ex1.T).all()
    return ex1 #- numpy.eye(ex1.shape[0])*lamb_0


def torus_3_plots(N=200,epsilon=0.1):
    """
    """
    ax1 = plotdl.new_ax_for_file()
    torus = Sample((1,1), N)
    plot_exp_model_permutation(ax1, torus, epsilon=epsilon)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
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
        
def plot_quasi_1d(ax, sample, bandwidth_list, epsilon=10):
    """
    """
    ex =  exp_model_matrix(sample, epsilon=10, bandwidth=0)
    for bandwidth in bandwidth_list:
        rate_matrix = ex*banded_ones(ex.shape[0], bandwidth)
        sparsedl.zero_sum(rate_matrix)
        diff_coef = sparsedl.resnet(rate_matrix, bandwidth)
        #eigvals= -linalg.eigvals(rate_matrix)
        #eigvals.sort()
        eigvals = sorted_eigvalsh(rate_matrix)
        logvals = log10((eigvals)[1:])
        cummulative_plot(ax, logvals, label=r"$\epsilon = {0:.3G}, b={1:.3G}$".format(epsilon, bandwidth))
        bbox = [eigvals[1], eigvals[-1], 1/len(eigvals),  1]
        print(bbox)
        power_law_logplot(ax, 0.5, 1/(sqrt(diff_coef)*pi), bbox, label=r"$\frac{{D}}{{r_0^2}} \approx {0:.3G}$".format(diff_coef))
        #diffusion_plot(ax, diff_coef, eigvals)
    plotdl.set_all(ax, title="", legend_loc="best", xlabel=r"$\log_{10}\lambda$", ylabel=r"$C(\lambda)$")
    ax.set_yscale('log')



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
    ax1.label_outer()
   
    fig.subplots_adjust(hspace=0.001)
#    ax3 = fig.add_subplot(2,1,3)#,sharex=ax1)
    ex = exp_model_matrix(sample, epsilon=epsilon)
    v,w = sparsedl.sorted_eigh(ex)
    logvals = log10((-v)[1:])
    cummulative_plot(ax1, logvals, label=r"$\epsilon = {0:.3G}$".format(epsilon))
    #D = sparsedl.resnet(ex,1)
    D1 =  ((epsilon-1)/(epsilon))
    v_not_including_first = (-v)[1:]
    bbox = [((v_not_including_first)[v_not_including_first>0])[0], (-v)[-1], 1/len(v),  1]
    if sample.d ==1 :
        if D1 > 0:
            power_law_logplot(ax1, 0.5, 1/(sqrt(D1)*pi), bbox, label=r"$\frac{{D}}{{r_0^2}} = \frac{{\epsilon-1}}{{\epsilon}} \approx {0:.3G}$".format(D1))
        else:
            D_resnet = sparsedl.resnet(ex,1)
            power_law_logplot(ax1, 0.5, 1/(sqrt(D_resnet)*pi), bbox, label=r"$ResNet D = {0:.3G}$".format(D_resnet))
            power_law_logplot(ax1, epsilon/2, 1, bbox, label=r"$\lambda^{{\epsilon/2}}$".format())
            power_law_logplot(ax1, epsilon, 1, bbox, label=r"$\lambda^{{\epsilon}}$".format())
    if sample.d == 2:
        D_resnet = sparsedl.avg_2d_resnet(ex,sample.periodic_distance_matrix(),sample.r_0)
        power_law_logplot(ax1, 0.5, 1/(sqrt(D_resnet)*pi), bbox, label=r"$ResNet D = {0:.3G}$".format(D_resnet))
        #sample_2d_theory(ax1, sample, epsilon)
    pn = ((w**4).sum(axis=0))**(-1)
    ax2.plot(logvals,pn[1:], marker=".", linestyle='')
    ax2.axhline(y=2, label="2 - dimer", linestyle="--", color="green")
    plotdl.set_all(ax1, ylabel=r"$C(\lambda)$", legend_loc="best")  
    plotdl.set_all(ax2, ylabel=r"PN", xlabel=r"$\log_{10}\lambda$", legend_loc="best")
    ax1.set_yscale('log')
    ax2.set_yscale('log')    
    # There are two overlaping ticks, so we remove both
    ax1.set_yticks(ax1.get_yticks()[1:])
    ax2.set_yticks(ax2.get_yticks()[:-1])    
    plotdl.save_fig(fig, "exp_{1}d_{0}_pn".format(epsilon, sample.d))

    #ax.clear()
    ax3 = plotdl.new_ax_for_file()
    plotdl.matshow_cb(ax3, w**2, vmin=10**(-10), colorbar=True)
    plotdl.set_all(ax3, title=r"$N={0}, \epsilon = {1}$".format(number_of_points, epsilon))
    plotdl.save_ax(ax3, "exp_{2}d_{1}_matshow".format(number_of_points, epsilon,sample.d))

    #plotdl.save_fig(fig, "exp_1d_{0}_{1}_test".format(number_of_points, epsilon))

######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    ax = plotdl.new_ax_for_file()

    #### 1d PN and matshow
    line = Sample(1,800)
    random.seed(1)
    plotf_eig_matshow_pn(line, epsilon=0.2)
    random.seed(1)
    plotf_eig_matshow_pn(line, epsilon=1.5)
    random.seed(1)
    plotf_eig_matshow_pn(line, epsilon=5)

    #### 2d PN and matshow
    tor = Sample((1,1),800)
    random.seed(1)
    plotf_eig_matshow_pn(tor, epsilon=0.2)
    random.seed(1)
    plotf_eig_matshow_pn(tor, epsilon=1.5)
    random.seed(1)
    plotf_eig_matshow_pn(tor, epsilon=5)

    #### quasi-1d
    random.seed(seed)
    plot_quasi_1d(ax, line, bandwidth_list=(1,2,4,8,16), epsilon=10)
    plotdl.save_ax(ax, "quasi_1d")
    ax.clear()

    random.seed(seed)
    torus_3_plots()
    ax.clear()
    
    random.seed(seed)
    exp_models_D(Sample((1,),300), epsilon_ranges=((0.5,0.8,1,1.5,2,5,10),(0.5,0.8),(5,10)))
    exp_models_D(Sample((1,1),300))
    exp_models_D(Sample((1,1,1),300))
    
    

if __name__ ==  "__main__":
    all_plots()
