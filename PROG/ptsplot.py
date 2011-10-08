#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

#from scipy.sparse import linalg as splinalg
from numpy import random, pi, log10, sqrt,  exp, sort, eye, nanmin, nanmax, log, cos
from scipy.special import gamma

import numpy as np

import sparsedl
import plotdl
from geometry import Sample
from sparsedl import sorted_eigvalsh, banded_ones, periodic_banded_ones
from plotdl import cummulative_plot

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)


def power_law_logplot(ax, power, coeff, logxlim,label, **kwargs):
    """ Plots 1d diffusion, treating the x value as log10.
    """
    x1,x2 = logxlim
    power_space = np.linspace(x1, x2, 100)
    power_law = coeff*(10**power_space)**(power)
    return ax.plot(power_space, power_law, linestyle='--', label=label, **kwargs)

def plot_func_logplot(ax, func, logxlim, label, **kwargs):
    """ Plots function func in a logscale within the bounding box given by logbbox
    """
    x1, x2 = logxlim
    func_space = np.linspace(x1,x2,200)
    func_y = func(10**func_space)
    #print(func_y)
    return ax.plot(func_space, func_y, linestyle='--', label=label, **kwargs)

def plot_func(ax, func, xlim, label, **kwargs):
    """ Plots function func in a logscale within the bounding box given by logbbox
    """
    x1, x2 = xlim
    func_space = np.linspace(x1,x2,200)
    func_y = func(func_space)
    #print(func_y)
    return ax.plot(func_space, func_y, linestyle='--', label=label, **kwargs)



####################   Sample Plots ################


def exp_model_matrix(sample, epsilon=0.1, bandwidth=None, periodic=False): ## rename from sample exp
    """ Creats W_{nm} = exp((r_0-r_{nm})/xi). The matrix is zero summed and should be symmetric.

        :param sample: The sample, with generated points.
        :type sample: geometry.Sample
        :param epsilon: the epsilon, defaults to 0.1
    """
    # handle bandwidth for 1d. the default is 1.
    ex1 = (sample.exponent_1_minus_r(periodic))**(1/epsilon)
    if sample.d ==  1:
        if bandwidth is None:
            ex1 = ex1*periodic_banded_ones(ex1.shape[0], 1)
        elif bandwidth != 0: #### Zero means ignore bandwidth
            ex1 = ex1*periodic_banded_ones(ex1.shape[0], bandwidth)
    sparsedl.zero_sum(ex1)
    #assert (ex1 == ex1.T).all()
    return ex1 #- np.eye(ex1.shape[0])*lamb_0


def plot_quasi_1d(ax, sample, bandwidth_list, epsilon=10):
    """ diffusion doesn't work yet
    """
    for bandwidth in bandwidth_list:
        model =  ExpModel_1d(sample, epsilon=10, bandwidth1d=bandwidth)
        cummulative_plot(ax, model.logvals, label=r"$\epsilon = {0:.3G}, b={1:.3G}$".format(epsilon, bandwidth))
        diff_coef = sparsedl.resnet(model.ex, bandwidth, periodic=False)
        power_law_logplot(ax, 0.5, 1/(sqrt(diff_coef)*pi), model.logxlim, label=r"$\frac{{D}}{{r_0^2}} \approx {0:.3G}$".format(diff_coef))
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
    def __init__(self, sample, epsilon, basename="exp_{dimensions}d_{epsilon}",bandwidth1d = None, periodic=True):
        """ Take sample and epsilon, and calc eigvals and eigmodes"""
        self.epsilon = epsilon
        self.sample = sample
        self.periodic=periodic
        self.ex = exp_model_matrix(sample, epsilon=epsilon, bandwidth=bandwidth1d, periodic=periodic)
        self.eigvals, self.logvals, self.eig_matrix = self.calc_eigmodes(self.ex)
        self.vals_dict = {"epsilon" : epsilon, "dimensions" : sample.d, "number_of_points" : sample.number_of_points()}
        self.permuted = False
        self.basename = basename.format(**self.vals_dict)
        #self.logxlim = self.logvals[[1,-1]]
        self.logxlim = [nanmin(self.logvals), nanmax(self.logvals)]

    def permute_and_store(self):
        """ Permute the rates and set perm_logvals and perm_eig_matrix """
        self.permuted = True
        perm_ex = sparsedl.permute_tri(self.ex)
        sparsedl.zero_sum(perm_ex)
        self.perm_eigvals, self.perm_logvals, self.perm_eig_matrix = self.calc_eigmodes(perm_ex)
        #return (self.perm_logvals, self.perm_eig_matrix)
    def maximal_rate_per_row(self):
        return self.ex.max(axis=0)

    def calc_eigmodes(self, ex):
        """ calculate eigmodes, and return logvals and eigmodes"""
        v,w = sparsedl.sorted_eigh(ex)
        return (v, log10((-v)[1:]), w)

    def plot_diff(self, ax, label = r"$\frac{{D}}{{r_0^2}} = {D:.3G} $", **kwargs):
        """ """
        #D = self.diff_coef()#exp(1/self.epsilon)/(4*pi)#1#self.epsilon#
        r = self.sample.normalized_distance_matrix(self.periodic)
        D = (r*self.ex**2).sum(axis=0).mean()
        
        d2 = self.sample.d / 2
        #d2  = self.sample.d
        prefactor = 1/((d2)*gamma(d2)*((4*pi*D)**(d2)))
        #power_law_logplot(ax, d2, prefactor, self.logxlim, label=label.format(D=D, **self.vals_dict), **kwargs)
        f = lambda x: prefactor*x**d2
        plot_func_logplot(ax, f, self.logxlim, "D = {D:.3G}".format(D=D))

class ExpModel_1d(ExpModel):
    """ Subclassing exp model for 1d """
    def diff_coef(self):
        D = ((self.epsilon-1)/(self.epsilon))
        if D < 0 :
            D = sparsedl.resnet(self.ex,1)
        return D
    def plot_rate_density(self, ax, label=r"Max. rate / row", **kwargs):
        """ """
        N = self.sample.number_of_points()
        brates = nanmax(self.ex, axis=0)
        logbrates = log10(brates)
        if (nanmin(logbrates) < self.logxlim[1]) and (nanmax(logbrates) > self.logxlim[0]):
            print "len(logbrates)", len(logbrates)
            cummulative_plot(ax, sort(logbrates), label=label, color='purple')
            plot_func_logplot(ax, lambda w: exp(-2*(1-self.epsilon*log(w))),self.logxlim, r"$e^{-2\cdot(1-\epsilon\ln(w))}$")
            plot_func_logplot(ax, lambda w: exp(-(1-self.epsilon*log(w*0.5))),self.logxlim, r"$e^{-\cdot(1-\epsilon\ln(\frac{w}{2}))}$")


class ExpModel_Bloch_1d(ExpModel_1d):
    def diff_coef(self):
        return 1
    def plot_rate_density(self, ax, label=r"$\lambda^\epsilon$", **kwargs):
        """ """
        #power_law_logplot(ax, self.epsilon, 1, self.logxlim, label=label.format(**self.vals_dict), color="green")
        N = self.sample.number_of_points()
        nn, xx = np.meshgrid(np.arange(N), np.arange(N))
        ev = sort((exp(1-nn/self.epsilon)*cos(2*nn*xx*pi/N)).sum(axis=1))
        cummulative_plot(ax, sort(ev),color="green")


class ExpModel_2d(ExpModel):
    """ Subclassing exp model for 2d """
    def diff_coef(self):
        return self.epsilon*4
    def old_diff_plot(self, ax, label = r"$\frac{{D}}{{r_0^2}} = {D:.3G} $", **kwargs):
        """ """
        D = self.epsilon*4
        d2 = self.sample.d / 2
        prefactor = 1/((d2)*gamma(d2)*((4*pi*D)**(d2)))
        power_law_logplot(ax, d2, prefactor, self.logxlim, label=label.format(D=D, **self.vals_dict))

    def plot_rate_density(self, ax, label=r"Max. rate / row", **kwargs):
        N = self.sample.number_of_points()
        brates = self.ex.max(axis=0)
        logbrates = log10(brates)
        if (nanmin(logbrates) < self.logxlim[1]) and (nanmax(logbrates) > self.logxlim[0]):
            cummulative_plot(ax, sort(logbrates), label=label, color='purple')
            plot_func_logplot(ax, lambda w: exp(-pi*(1-self.epsilon*log(w))**2),self.logxlim, r"$e^{-\pi\cdot(1-\epsilon\ln(w))^2}$")
            plot_func_logplot(ax, lambda w: exp(-0.5*pi*(1-self.epsilon*log(0.5*w))**2),self.logxlim, r"$e^{-\frac{\pi}{2}\cdot(1-\epsilon\ln(\frac{w}{2}))^2}$")


class ExpModel_Bloch_2d(ExpModel_2d):
    def old_diff_coef(self):
        return self.epsilon*4
    def diff_coef(self):
        r = self.sample.periodic_distance_matrix()
        D = (self.ex*r**2).sum(axis=0).mean()
        return D
    def plot_rate_density(self, ax, label=r"$\lambda^\epsilon$", **kwargs):
        """ """
        pass
    
class ExpModel_alter_1d(ExpModel_1d):
    def __init__(self, sample, epsilon, basename="exp_alter_{dimensions}d_{epsilon}",bandwidth1d = None, periodic=True):
        """ This should be similar to ExpModel.__init__, and eventually recombined there.
            This model has its alternating sites nulled"""
        self.epsilon = epsilon
        self.sample = sample
        self.periodic  = periodic
        N = self.sample.number_of_points()
        self.ex = exp_model_matrix(sample, epsilon=epsilon, bandwidth=1)
        offd = np.arange(1, N-1, 2 )
        self.ex[[offd, offd-1]] = 0
        self.ex[[offd-1, offd]] = 0
        sparsedl.zero_sum(self.ex)
        self.eigvals, self.logvals, self.eig_matrix = self.calc_eigmodes(self.ex)
        self.eigvals = self.eigvals[N//2:]
        self.logvals = self.logvals[N//2:]
        self.eig_matrix = self.eig_matrix[:,N//2:]
        self.vals_dict = {"epsilon" : epsilon, "dimensions" : sample.d, "number_of_points" : sample.number_of_points()}
        self.permuted = False
        self.basename = basename.format(**self.vals_dict)
        #self.logxlim = self.logvals[[1,-1]]
        self.logxlim = [nanmin(self.logvals), nanmax(self.logvals)]



def plot_pn(ax, model, **kwargs):
    """ """
    pn = ((model.eig_matrix**4).sum(axis=0))**(-1)
    return ax.plot(model.logvals,pn[1:], marker=".", linestyle='', **kwargs)

def plot_permuted_pn(ax, model, **kwargs):
    """ """
    if not model.permuted :
        model.permute_and_store()
    pn = ((model.perm_eig_matrix**4).sum(axis=0))**(-1)
    return ax.plot(model.perm_logvals,pn[1:], marker=".", linestyle='', **kwargs)

def plot_logvals(ax, model, label = r"$\epsilon = {epsilon:.3G}$", **kwargs):
    """ """
    return cummulative_plot(ax, model.logvals, label=label.format(**model.vals_dict), **kwargs)
    
def plot_log_decay(ax, model, label = r"$\gamma$" ,**kwargs):
    """ """
    return cummulative_plot(ax, log10(sort(-model.ex.diagonal())), label = label,**kwargs)

def plot_permuted_logvals(ax, model, label = r"Permuted", **kwargs):
    """ """
    if not model.permuted :
        model.permute_and_store()
    return cummulative_plot(ax, model.perm_logvals, label=label.format(**model.vals_dict), **kwargs)


def scatter_eigmode(ax, model, n, keepnorm=False):
    """ """
    if keepnorm:
        vdict = {'vmin' :model.eig_matrix.min(), 'vmax':model.eig_matrix.max()}
    else:
        vdict = {}
    sample = model.sample
    return ax.scatter(sample.points[:,0], sample.points[:,1], c=model.eig_matrix[:,n], edgecolors='none', **vdict)

def plot_diag_eigen(ax, model, **kwargs):

    line1, = cummulative_plot(ax, sort(-model.ex.diagonal()), label=r"Main diagonal, $\epsilon = {epsilon}$".format(**model.vals_dict), **kwargs)
    line2, = cummulative_plot(ax, sort(-model.eigvals), label=r"Eigenvalues, $\epsilon = {epsilon}$".format(**model.vals_dict), **kwargs)
    return [line1, line2]


def plotf_logvals_pn(model):
    """ """
    lines_for_scale = [] # I use this list to autoscale only with certain lines
    fig = plotdl.Figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharex=ax1)
    ax1.label_outer()
    fig.subplots_adjust(hspace=0.001)
    lines_for_scale += plot_logvals(ax1, model)
    ### new - test
    lines_for_scale += plot_log_decay(ax1, model, color="magenta")
    ###
    model.plot_diff(ax1, color="red")
    model.plot_rate_density(ax1, color="purple")
    if model.sample.d ==2:
        #lines_for_scale += plot_permuted_logvals(ax1, model, color="green")
        pass
    plot_pn(ax2, model, zorder=2.5)
    if model.sample.d ==2:
        #plot_permuted_pn(ax2, model, color="green")
        pass
    ax2.axhline(y=2, label="2 - dimer", linestyle="--", color="green")
    plotdl.set_all(ax1, ylabel=r"$C(\lambda)$", legend_loc="best")
    plotdl.set_all(ax2, ylabel=r"PN", xlabel=r"$\log_{10}\lambda$", legend_loc="best")
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    # There are two overlaping ticks, so we remove both
    ax1.set_yticks(ax1.get_yticks()[1:])
    ax2.set_yticks(ax2.get_yticks()[:-1])
    plotdl.autoscale_based_on(ax1, lines_for_scale)
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
        brates = b.ex.flatten()
        brates.sort()
        cummulative_plot(ax, brates[-N:-1])
        ax.set_xscale('log')
        plotdl.save_ax(ax, "raw_rates_{epsilon}".format(epsilon=eps))

def plotf_distance_statistics(N=1000):
    """ plot to file the nearest distance statistics and theory for 1d 2d 3d. """
    dict1d = { 'sample' : Sample(1,N),
               'fit_func' : lambda r: 1-exp(-r*2),
               'fit_func_txt' : r"$1-e^{-2x}$",
               'filename': "dist_1d_1000"}
    dict2d = { 'sample' : Sample((1,1),N),
               'fit_func' : lambda r: 1-exp(-(pi)*(r**2)),
               'fit_func_txt' : r"$1-e^{-\pi\cdot x^2}$",
               'filename': "dist_2d_1000"}
    dict3d = { 'sample' : Sample((1,1,1),N),
               'fit_func' : lambda r: 1-exp(-(4*pi/3)*(r**3)),
               'fit_func_txt' : r"$1-e^{-\frac{4\pi}{3}\cdot x^3}$",
               'filename': "dist_3d_1000"}

    ax = plotdl.new_ax_for_file()


    for di in (dict1d, dict2d, dict3d):
        dist = (di['sample'].normalized_distance_matrix() +N*eye(N) ).min(axis=0)
        dist.sort()
        cummulative_plot(ax, dist)
        plot_func(ax, di['fit_func'], dist[[0,-1]], di['fit_func_txt'])
        plotdl.set_all(ax, xlabel=r"$\frac{r}{r_0}$", ylabel=r"$C(\frac{r}{r_0})$", legend_loc="best")
        plotdl.save_ax(ax, di['filename'])
        ax.cla()
        

def create_bloch_sample_1d(N):
    """
    """
    bloch = Sample(1,N)
    bloch.points = np.linspace(0,1,N, endpoint=False)
    return bloch

def create_bloch_sample_2d(N):
    """
    """
    bloch = Sample((1,1),N*N)
    pts = np.linspace(0,N,N*N, endpoint=False)
    pts = np.mod(pts,1)
    x = pts
    y = np.sort(pts)
    bloch.points = np.array((x,y)).T
    return bloch

######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """

    plotf_distance_statistics()
    
    ax = plotdl.new_ax_for_file()

    #### 1d PN and matshow
    line = Sample(1,900)
    for epsilon in (0.05, 0.2, 1.5,5):
        line_model = ExpModel_1d(line, epsilon)
        plotf_logvals_pn(line_model)
        #plotf_matshow(line_model)
    for epsilon in (0.2,5):
        alter_model = ExpModel_alter_1d(line, epsilon,basename="exp_alter_{dimensions}d_{epsilon}")
        plotf_logvals_pn(alter_model)
    # 1d bloch:
    bloch1d = create_bloch_sample_1d(900)
    for epsilon in (0.2,0.8,1.5,5):
        plotf_logvals_pn(ExpModel_Bloch_1d(bloch1d, epsilon, basename="bloch_1d_{epsilon}"))

    #### 2d PN and matshow
    sample2d = Sample((1,1),900)
    for epsilon in (0.05, 0.2, 1.5, 5):
        random.seed(1)
        model2d = ExpModel_2d(sample2d, epsilon)
        plotf_logvals_pn(model2d)
        #plotf_matshow(model2d)
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
