#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
#from scipy.sparse import linalg as splinalg
from numpy import linalg, random, pi, log10, sqrt, log, exp, sort
from scipy.special import gamma
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
from sparsedl import sorted_eigvalsh, banded_ones, periodic_banded_ones
from plotdl import cummulative_plot

### Raise all float errors 
numpy.seterr(all='warn')
EXP_MAX_NEG = numpy.log(numpy.finfo( numpy.float).tiny)


def power_law_logplot(ax, power, coeff, logbbox,label, **kwargs):
    """ Plots 1d diffusion, treating the x value as log10.
    """
    x1,x2,y1,y2 = logbbox
    xl_log = min(x1, log10(y1/coeff)/power)
    xr_log = min(x2, log10(y2/coeff)/power)

    power_space = numpy.linspace(xl_log, xr_log, 100)
    
    power_law = coeff*(10**power_space)**(power)
    ax.plot(power_space, power_law, linestyle='--', label=label, **kwargs)

def plot_func_logplot(ax, func, logbboxs, label, **kwargs):
    """ Plots function func in a logscale within the bounding box given by logbbox
    """
    

def exponent_law_logplot(ax, logbbox, label, x0=0.1, **kwargs):
    """
    """
    x1,x2,y1,y2 = logbbox
    #xr_log = min((log10(x2)))#, log10(y2/coeff)/power))
    #xl_log = max((log10(x1)))#, log10(y1/coeff)/power))
    xr_log = x2
    xl_log = x0

    exp_space = numpy.linspace(xl_log, xr_log, 100)
    exp_law = 1 - exp(-2*(pi)*((exp_space-x0))**2)
    ax.plot(exp_space, exp_law, linestyle='--', label=label, **kwargs)

####################   Sample Plots ################


def exp_model_matrix(sample, epsilon=0.1, bandwidth=None): ## rename from sample exp
    """ Creats W_{nm} = exp((r_0-r_{nm})/xi). The matrix is zero summed and should be symmetric.

        :param sample: The sample, with generated points.
        :type sample: geometry.Sample
        :param epsilon: the epsilon, defaults to 0.1
    """
    xi = sample.epsilon_to_xi(epsilon)
    dis = sample.non_periodic_distance_matrix()
    # new -renormalization
    r_0 = sample.r_0()
    #r_0 = dis.sum()/(dis.shape[0]**2)
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


def plot_quasi_1d(ax, sample, bandwidth_list, epsilon=10):
    """ diffusion doesn't work yet
    """
    for bandwidth in bandwidth_list:
        model =  ExpModel_1d(sample, epsilon=10, bandwidth1d=bandwidth)
        cummulative_plot(ax, model.logvals, label=r"$\epsilon = {0:.3G}, b={1:.3G}$".format(epsilon, bandwidth))
        diff_coef = sparsedl.resnet(model.ex, bandwidth, periodic=False)
        power_law_logplot(ax, 0.5, 1/(sqrt(diff_coef)*pi), model.logbbox, label=r"$\frac{{D}}{{r_0^2}} \approx {0:.3G}$".format(diff_coef))
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

class ExpModel(object):
    def __init__(self, sample, epsilon, basename="exp_{dimensions}d_{epsilon}",bandwidth1d = None):
        """ Take sample and epsilon, and calc eigvals and eigmodes"""
        self.epsilon = epsilon
        self.xi = sample.epsilon_to_xi(epsilon)
        self.sample = sample
        self.ex = exp_model_matrix(sample, epsilon=epsilon, bandwidth=bandwidth1d)
        self.eigvals, self.logvals, self.eig_matrix = self.calc_eigmodes(self.ex)
        self.vals_dict = {"epsilon" : epsilon, "dimensions" : sample.d, "number_of_points" : sample.number_of_points()}
        self.permuted = False
        self.basename = basename.format(**self.vals_dict)
        self.logbbox = [self.logvals[1], self.logvals[-1], 1/len(self.logvals), 1]

    def permute_and_store(self):
        """ Permute the rates and set perm_logvals and perm_eig_matrix """
        self.permuted = True
        perm_ex = sparsedl.permute_tri(self.ex)
        sparsedl.zero_sum(perm_ex)
        self.perm_eigvals, self.perm_logvals, self.perm_eig_matrix = self.calc_eigmodes(perm_ex)
        #return (self.perm_logvals, self.perm_eig_matrix)
        
    def calc_eigmodes(self, ex):
        """ calculate eigmodes, and return logvals and eigmodes"""
        v,w = sparsedl.sorted_eigh(ex)
        return (v, log10((-v)[1:]), w)
    
    def plot_diff(self, ax, label = r"$\frac{{D}}{{r_0^2}} = {D:.3G} $", **kwargs):
        """ """
        D = self.diff_coef()
        d2 = self.sample.d / 2
        prefactor = 1/((d2)*gamma(d2)*((4*pi*D)**(d2)))
        power_law_logplot(ax, d2, prefactor, self.logbbox, label=label.format(D=D, **self.vals_dict), **kwargs)

class ExpModel_1d(ExpModel):
    """ Subclassing exp model for 1d """
    def diff_coef(self):
        D = ((self.epsilon-1)/(self.epsilon))
        if D < 0 :
            D = sparsedl.resnet(self.ex,1)
        return D
    def plot_rate_density(self, ax, label=r"$\lambda^\epsilon$", **kwargs):
        """ """
        power_law_logplot(ax, self.epsilon, 1, self.logbbox, label=label.format(**self.vals_dict), color="green")
        N = self.sample.number_of_points()
        brates = numpy.triu(self.ex,k=1).ravel().copy()
        brates.sort()
#        brates *= exp(-1)
        cummulative_plot(ax, log10(brates[-N:-1]), label="high rates ", color='purple')
        
class ExpModel_Bloch_1d(ExpModel_1d):
    def diff_coef(self):
        return 1
    def plot_rate_density(self, ax, label=r"$\lambda^\epsilon$", **kwargs):
        """ """
        power_law_logplot(ax, self.epsilon, 1, self.logbbox, label=label.format(**self.vals_dict), color="green")


class ExpModel_2d(ExpModel):
    """ Subclassing exp model for 2d """
    def diff_coef(self):
        return self.epsilon*4
    def old_plot_rate_density(self, ax, x0=0.1, label=r"$\lambda^\epsilon$", **kwargs):

        exponent_law_logplot(ax,self.logbbox, label, x0, **kwargs)

    def plot_rate_density(self, ax, label=r"high rates ", **kwargs):
        N = self.sample.number_of_points()
        brates = numpy.triu(self.ex, k=1).ravel().copy()
        brates.sort()
        cummulative_plot(ax, log10(brates[-N:-1]), label=label, color='purple')


class ExpModel_Bloch_2d(ExpModel_2d):
    def diff_coef(self):
        return self.epsilon*4
    def plot_rate_density(self, ax, label=r"$\lambda^\epsilon$", **kwargs):
        """ """
        pass


def plot_pn(ax, model, **kwargs):
    """ """
    pn = ((model.eig_matrix**4).sum(axis=0))**(-1)
    ax.plot(model.logvals,pn[1:], marker=".", linestyle='', **kwargs)

def plot_permuted_pn(ax, model, **kwargs):
    """ """
    if not model.permuted :
        model.permute_and_store()
    pn = ((model.perm_eig_matrix**4).sum(axis=0))**(-1)
    ax.plot(model.perm_logvals,pn[1:], marker=".", linestyle='', **kwargs)

def plot_logvals(ax, model, label = r"$\epsilon = {epsilon:.3G}$", **kwargs):
    """ """
    cummulative_plot(ax, model.logvals, label=label.format(**model.vals_dict), **kwargs)

def plot_permuted_logvals(ax, model, label = r"Permuted", **kwargs):
    """ """
    if not model.permuted :
        model.permute_and_store()
    cummulative_plot(ax, model.perm_logvals, label=label.format(**model.vals_dict), **kwargs)

def plot_diffusion(ax, model, label="diff", **kwargs):
    """ """
    D = model.diff_coef()
    bbox = [-model.eigvals[1],-model.eigvals[-1], 1/len(model.eigvals),  1]
    print("bbox = " , bbox)
    print ("D = ", D)
    power_law_logplot(ax, 1, D, bbox, label=label.format(**model.vals_dict), **kwargs)


def scatter_eigmode(ax, model, n, keepnorm=False):
    """ """
    if keepnorm:
        vdict = {'vmin' :model.eig_matrix.min(), 'vmax':model.eig_matrix.max()}
    else:
        vdict = {}
    sample = model.sample
    ax.scatter(sample.points[:,0], sample.points[:,1], c=model.eig_matrix[:,n], edgecolors='none', **vdict)
    
def plot_diag_eigen(ax, model, **kwargs):

    cummulative_plot(ax, sort(-model.ex.diagonal()), label=r"Main diagonal, $\epsilon = {epsilon}$".format(**model.vals_dict), **kwargs)
    cummulative_plot(ax, sort(-model.eigvals), label=r"Eigenvalues, $\epsilon = {epsilon}$".format(**model.vals_dict), **kwargs)


def plotf_logvals_pn(model):
    """ """
    fig = plotdl.Figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharex=ax1)
    ax1.label_outer()
    fig.subplots_adjust(hspace=0.001)
    plot_logvals(ax1, model)
    model.plot_diff(ax1, color="red")
    model.plot_rate_density(ax1, color="purple")
    if model.sample.d ==2:
        plot_permuted_logvals(ax1, model, color="green")
    plot_pn(ax2, model)
    if model.sample.d ==2:
        plot_permuted_pn(ax2, model, color="green") 
    ax2.axhline(y=2, label="2 - dimer", linestyle="--", color="green")
    plotdl.set_all(ax1, ylabel=r"$C(\lambda)$", legend_loc="best")  
    plotdl.set_all(ax2, ylabel=r"PN", xlabel=r"$\log_{10}\lambda$", legend_loc="best")
    ax1.set_yscale('log')
    ax2.set_yscale('log')    
    # There are two overlaping ticks, so we remove both
    ax1.set_yticks(ax1.get_yticks()[1:])
    ax2.set_yticks(ax2.get_yticks()[:-1])
    plotdl.save_fig(fig, model.basename + "_pn")

def plotf_matshow(model):
    """ """
    ax = plotdl.new_ax_for_file()
    w = model.eig_matrix
    plotdl.matshow_cb(ax, w**2, vmin=10**(-10), colorbar=True)
    plotdl.set_all(ax, title=r"$N={number_of_points}, \epsilon = {epsilon}$".format(**model.vals_dict))
    plotdl.save_ax(ax, model.basename + "_mat")


def plotf_all_raw_rates(sample, epsilons=(0.2,1,5)):
    ax = plotdl.new_ax_for_file()
    for eps in epsilons:
        ax.cla()
        b = ExpModel_2d(sample, eps)
        N = sample.number_of_points()
        brates = b.ex.ravel().copy()
        brates.sort()
        cummulative_plot(ax, brates[-N:-1])
        ax.set_xscale('log')
        plotdl.save_ax(ax, "raw_rates_{epsilon}".format(epsilon=eps))


def create_bloch_sample_1d(N):
    """
    """
    bloch = Sample(1,N)
    bloch.points = numpy.linspace(0,1,N, endpoint=False)
    return bloch

def create_bloch_sample_2d(N):
    """
    """
    bloch = Sample((1,1),N*N)
    pts = numpy.linspace(0,N,N*N, endpoint=False)
    pts = numpy.mod(pts,1)
    x = pts
    y = numpy.sort(pts)
    bloch.points = numpy.array((x,y)).T
    return bloch

######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    ax = plotdl.new_ax_for_file()

    #### 1d PN and matshow
    line = Sample(1,900)
    for epsilon in (0.2, 1.5,5):
        random.seed(1)
        line_model = ExpModel_1d(line, epsilon)
        plotf_logvals_pn(line_model)
        plotf_matshow(line_model)
    
    # 1d bloch:
    bloch1d = create_bloch_sample_1d(900)
    for epsilon in (0.2,0.8,1.5,5):
        plotf_logvals_pn(ExpModel_Bloch_1d(bloch1d, epsilon, basename="bloch_1d_{epsilon}"))

    #### 2d PN and matshow
    sample2d = Sample((1,1),900)
    for epsilon in (0.2, 1.5, 5):
        random.seed(1)
        model2d = ExpModel_2d(sample2d, epsilon)
        plotf_logvals_pn(model2d)
        plotf_matshow(model2d)
    # 2d bloch:
    bloch2d = create_bloch_sample_2d(30)
    for epsilon in (0.2,0.8,1.5,5):
        plotf_logvals_pn(ExpModel_Bloch_2d(bloch2d, epsilon, basename="bloch_2d_{epsilon}"))
        
    #### quasi-1d
    random.seed(seed)
    plot_quasi_1d(ax, line, bandwidth_list=(1,2,4,8,16), epsilon=10)
    plotdl.save_ax(ax, "quasi_1d")
    ax.clear()

    #random.seed(seed)
    #torus_3_plots()
    #ax.clear()
    
    #random.seed(seed)
    #exp_models_D(Sample((1,),300), epsilon_ranges=((0.5,0.8,1,1.5,2,5,10),(0.5,0.8),(5,10)))
    #exp_models_D(Sample((1,1),300))
    #exp_models_D(Sample((1,1,1),300))
    
    

if __name__ ==  "__main__":
    all_plots()
