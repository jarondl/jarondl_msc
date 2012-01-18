#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

import itertools
import logging

#from scipy.sparse import linalg as splinalg
from numpy import random, pi, log10, sqrt,  exp, sort, eye, nanmin, nanmax, log, cos, sinc
from scipy.special import gamma

import numpy as np

import sparsedl
import plotdl
from geometry import Sample
from sparsedl import sorted_eigvalsh, banded_ones, periodic_banded_ones, zero_sum, lazyprop
from plotdl import cummulative_plot

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logger = logging.getLogger(__name__)
info = logger.info
warning = logger.warning
debug = logger.debug


#Setting up some lambda functions:
#  D* = D - DELTA_D  # comes from "rigorous VRH"
DELTA_D = lambda eps, rstar : 2*pi*(exp(1/eps)*eps*(6*eps**3-exp(-rstar/eps)*(6*eps**3+ 6*eps**2*rstar + 3*eps*rstar**2 + rstar**3)) - rstar**4*exp((1-rstar)/eps)/4)

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

def plot_func(ax, func, xlim, **kwargs):
    """ Plots function func in a logscale within the bounding box given by logbbox
    """
    x1, x2 = xlim
    func_space = np.linspace(x1,x2,200)
    func_y = func(func_space)
    kwargs.setdefault('linestyle','--') # add linestyle if it's not in kwargs already
    #print(func_y)
    return ax.plot(func_space, func_y,  **kwargs)



####################   Sample Plots ################


def exp_model_matrix(sample, epsilon=0.1, convention=1, bandwidth=None, periodic=False): ## rename from sample exp
    """ Creats W_{nm} = exp((r_0-r_{nm})/xi). The matrix is zero summed and should be symmetric.

        :param sample: The sample, with generated points.
        :type sample: geometry.Sample
        :param epsilon: the epsilon, defaults to 0.1
    """
    # handle bandwidth for 1d. the default is 1.
    if convention == 1:
        ex1 = (sample.exponent_1_minus_r(periodic))**(1/epsilon)
    else:
        ex1 = (sample.exponent_minus_r(periodic))**(1/epsilon)
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
    pn = ((matrix**4).sum(axis=0))**(-1)
    return ax.plot(pn[1:], marker=".", linestyle='')

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

############ ExpModel matrices


class ExpModel(object):
    """ The current table of inheritance is:
        
      --+  ExpModel
        +-+- ExpModel_1d
          +--- ExpModel_Bloch_1d
        +---
    """
    def __init__(self, sample, epsilon, basename="exp_{dimensions}d_{epsilon}",bandwidth1d = None, periodic=True, convention=1):
        """ Take sample and epsilon, and calc eigvals and eigmodes"""
        self.epsilon = epsilon
        self.sample = sample
        self.convention = convention
        self.periodic=periodic
        self.bandwidth1d = bandwidth1d
        self.vals_dict = {"epsilon" : epsilon, "dimensions" : sample.d, "number_of_points" : sample.number_of_points()}
        self.permuted = False
        self.basename = basename.format(**self.vals_dict)
        #self.logxlim = self.logvals[[1,-1]]

    def rate_matrix(self, convention):
        """  Return the rate matrix of the model. It is here mainly to be subclassed.
        """
        raise Exception("Unimplemented")

    def permute_and_store(self):
        """ Permute the rates and set perm_logvals and perm_eig_matrix """
        self.permuted = True
        perm_ex = sparsedl.permute_tri(self.ex)
        sparsedl.zero_sum(perm_ex)
        self.perm_eigvals, self.perm_logvals, self.perm_eig_matrix = self.calc_eigmodes(perm_ex)
        #return (self.perm_logvals, self.perm_eig_matrix)
    def maximal_rate_per_row(self):
        return self.ex.max(axis=0)
    
    @lazyprop
    def eigvals(self):
	    return sparsedl.sorted_eigvalsh(self.ex)

    @lazyprop
    def eig_matrix(self):
        (self.eigvals, w) = sparsedl.sorted_eigh(self.ex)
        return w

    @lazyprop
    def logvals(self):
        return log10((-self.eigvals)[1:])

    @lazyprop
    def ex(self):
        return self.rate_matrix(self.convention)

    @lazyprop
    def xlim(self):
        return [-self.eigvals[1], -self.eigvals[-1]]

    @lazyprop
    def logxlim(self):
        return [nanmin(self.logvals), nanmax(self.logvals)]
    
    def plot_diff(self, ax, label = r"$\frac{{D}}{{r_0^2}} = {D:.3G} $", **kwargs):
        """ """
        #D = self.diff_coef()#exp(1/self.epsilon)/(4*pi)#1#self.epsilon#
        r = self.sample.normalized_distance_matrix(self.periodic)
        D = (r*self.ex**2).sum(axis=0).mean()
        
        d2 = self.sample.d / 2.0
        #d2  = self.sample.d
        prefactor = 1/((d2)*gamma(d2)*((4*pi*D)**(d2)))
        #power_law_logplot(ax, d2, prefactor, self.logxlim, label=label.format(D=D, **self.vals_dict), **kwargs)
        f = lambda x: prefactor*x**d2
        plot_func_logplot(ax, f, self.logxlim, "D = {D:.3G}".format(D=D))

class ExpModel_1d(ExpModel):
    """ Subclassing exp model for 1d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        if self.bandwidth1d is None:
            ex1 = ex1*periodic_banded_ones(ex1.shape[0], 1)
        elif self.bandwidth1d != 0: #### Zero means ignore bandwidth
            ex1 = ex1*periodic_banded_ones(ex1.shape[0], self.bandwidth1d)
        sparsedl.zero_sum(ex1)
        return ex1 


    def plot_alexander(self, ax, convention=1, **kwargs):
        """ plots Alexander's solution """
        epsilon = self.epsilon
        if epsilon > 1:
            f = lambda x: sqrt( x * exp(-convention/epsilon) *epsilon / (epsilon - 1)) / pi
        else:
            f = lambda x: exp(-convention)*sinc(epsilon/(epsilon+1))*x**(epsilon/(epsilon+1))
        plot_func(ax, f, self.xlim, **kwargs)
        
    def diff_coef(self):
        D = ((self.epsilon-1)/(self.epsilon))
        if D < 0 :
            D = sparsedl.resnet(self.ex,1)
        return D
    def plot_rate_density(self, ax, label=r"Max. rate / row",convention=1, **kwargs):
        """ """
        N = self.sample.number_of_points()
        brates = nanmax(self.ex, axis=0)
        logbrates = log10(brates)
        if (nanmin(logbrates) < self.logxlim[1]) and (nanmax(logbrates) > self.logxlim[0]):
            #print "len(logbrates)", len(logbrates)
            cummulative_plot(ax, sort(logbrates), label=label, color='purple')
            plot_func_logplot(ax, lambda w: exp(-2*(convention-self.epsilon*log(w))),
                self.logxlim, r"$e^{{-2\cdot({}-\epsilon\ln(w))}}$".format(convention))
            plot_func_logplot(ax, lambda w: exp(-(convention-self.epsilon*log(w*0.5))),
                self.logxlim, r"$e^{{-\cdot({}-\epsilon\ln(\frac{{w}}{{2}}))}}$".format(convention))
    def plot_theoretical_eigvals(self, ax):
        N = self.sample.number_of_points()
        qx = 2*pi/N*np.arange(N)
        z = sort(2*(cos(qx)+1 ).flatten())[1:]  # the 1: is to remove the 0 mode
        cummulative_plot(ax, z, label="$2+2\cos(q_x)$" ,color="red", marker="x")


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
    def plot_theoretical_eigvals(self, ax):
        N = self.sample.number_of_points()
        qx = 2*pi/N*np.arange(N)
        z = sort(2*(cos(qx)+1 ).flatten())[1:]  # the 1: is to remove the 0 mode
        cummulative_plot(ax, z, label="$2+2\cos(q_x)$" ,color="red", marker="x")


class ExpModel_2d(ExpModel):
    """ Subclassing exp model for 2d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        sparsedl.zero_sum(ex1)
        return ex1 

    def LRT_diff_coef(self, convention = 1):
        return 6*pi*exp(convention/self.epsilon)*self.epsilon**4

    def fit_diff_coef(self, convention = 1):
        fitN = self.sample.number_of_points() - 1
        y = np.linspace(1.0/(fitN),1,fitN)
        ar = np.arange(fitN)
        #w = (ar%4==3 )*exp(-ar/3.0)
        ### Trying to cheat less : 
        w = exp(-ar/3.0)
        x = -self.eigvals[1:]
        #prefactor  = sparsedl.cvfit((lambda x,a : x+a), log(x), log(y), [0],w)
        #D = exp(-prefactor)/(2*pi)
        ## Keep things simple:
        D = sparsedl.cvfit((lambda x, D: x/(2*pi*D)), x, y, [1], w)
        return D


    def plot_rate_density(self, ax, label=r"Max. rate / row", convention=1, **kwargs):
        N = self.sample.number_of_points()
        brates = self.ex.max(axis=0)
        logbrates = log10(brates)
        if (nanmin(logbrates) < self.logxlim[1]) and (nanmax(logbrates) > self.logxlim[0]):
            cummulative_plot(ax, sort(logbrates), label=label, color='purple')
            plot_func_logplot(ax, lambda w: exp(-pi*(convention-self.epsilon*log(w))**2),
                self.logxlim, r"$e^{{-\pi\cdot({}-\epsilon\ln(w))^2}}$".format(convention))
            plot_func_logplot(ax, lambda w: exp(-0.5*pi*(convention-self.epsilon*log(0.5*w))**2),
                self.logxlim, r"$e^{{-\frac{{\pi}}{{2}}\cdot({}-\epsilon\ln(\frac{{w}}{{2}}))^2}}$".format(convention))

    def plot_theoretical_eigvals(self, ax):
        N = sqrt(self.sample.number_of_points())
        qy, qx = np.meshgrid(2*pi/N*np.arange(N),2*pi/N*np.arange(N))
        z = sort(2*(cos(qx) + cos(qy) +2 ).flatten())[1:]  # the 1: is to remove 0
        cummulative_plot(ax, z, label="$4+2\cos(q_x)+2\cos(q_y) $" ,color="red", marker="x")


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
    
class ExpModel_Bloch_2d_only4nn(ExpModel_2d):
    def rate_matrix(self,convention):
        r = self.sample.normalized_distance_matrix(self.periodic)
        ex = exp(1-r)*(r<1.001)
        zero_sum(ex)
        return ex
        
    def plot_theoretical_eigvals(self, ax):
        N = sqrt(self.sample.number_of_points())
        qy, qx = np.meshgrid(2*pi/N*np.arange(N),2*pi/N*np.arange(N))
        z = sort(2*(cos(qx) + cos(qy) +2 ).flatten())[1:]  # the 1: is to remove 0
        cummulative_plot(ax, z, label="$4+2\cos(q_x)+2\cos(q_y) $" ,color="red", marker="x")
        
class ExpModel_Bloch_2d_only4nn_randomized(ExpModel_2d):
    def rate_matrix(self, convention):
        """ 4 nn (u,d,l,r)"""
        r = self.sample.normalized_distance_matrix(self.periodic)
        ## r is normalized, so r=1 means n.n. 
        # lower triangle nearest neighbor
        lnn = np.tri(r.shape[0])*(r>0.99)*(r<1.001)
        #W = exp(1-np.sqrt(-log(np.linspace(0,1, 2*r.shape[0]+1)[1:])/pi))**(1/self.epsilon)
        ex = np.zeros(r.shape)
        W = exp(convention - np.sqrt( -log(np.linspace(0, 1, ex[lnn==1].shape[0] + 1)[1:])/pi))**(1/self.epsilon)

        debug("ex[lnn=1].shape = %d", ex[lnn==1].shape)
        #print W.shape
        ex[lnn==1] = np.random.permutation(W)
        sym_ex = ex + ex.T
        zero_sum(sym_ex)
        return sym_ex

class ExpModel_Bloch_1d_only2nn_randomized(ExpModel_1d):
    def rate_matrix(self,convention):
        """ 2 nn (l,r)"""
        r = self.sample.normalized_distance_matrix(self.periodic)
        ## r is normalized, so r=1 means n.n. 
        # lower triangle nearest neighbor
        lnn = np.tri(r.shape[0])*(r>0.99)*(r<1.001)
        #W = exp(1-np.sqrt(-log(np.linspace(0,1, 2*r.shape[0]+1)[1:])/pi))**(1/self.epsilon)
        ex = np.zeros(r.shape)
#        W = exp(1/self.epsilon)*np.linspace(0,1, ex[lnn==1].shape[0]+1)[1:]**(1/(2*self.epsilon))
        W = exp(convention/self.epsilon)*np.linspace(0,1, ex[lnn==1].shape[0]+1)[1:]**(1/(self.epsilon))

        #print ex[lnn==1].shape
        #print W.shape
        ex[lnn==1] = np.random.permutation(W)
        sym_ex = ex + ex.T
        zero_sum(sym_ex)
        return sym_ex

    
class ExpModel_alter_1d(ExpModel_1d):
    def rate_matrix(self,convention):
        N = self.sample.number_of_points()
        ex = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        ex = ex*periodic_banded_ones(ex.shape[0], 1)
        sparsedl.zero_sum(ex)
        offd = np.arange(1, N-1, 2 )
        ex[[offd, offd-1]] = 0
        ex[[offd-1, offd]] = 0
        return ex



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

def scatter_eigmode_slider(fig, model):
    """ """
    ax = fig.add_axes([0.05,0.15,0.9,0.8])
    n_range = np.arange(1,model.sample.number_of_points())
    ax_slider = fig.add_axes([0.05,0.01,0.9,0.03])
    slider = plotdl.Slider(ax_slider, 'N', 1,model.sample.number_of_points(), valinit=1, valfmt="%d")
    def update_num(num):
        n = int(slider.val)
        scatter_eigmode(ax, model, n)
        plotdl.draw()
    slider.on_changed(update_num)

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
    
def plot_eigvals_theory(ax, model):
    """ """
    cummulative_plot(ax, -model.eigvals[1:], label="$\lambda$",color="blue")
    model.plot_theoretical_eigvals(ax)
    plotdl.set_all(ax, ylabel=r"$C(\lambda)$", legend_loc="best")

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
        plot_func(ax, di['fit_func'], dist[[0,-1]], label=di['fit_func_txt'])
        plotdl.set_all(ax, xlabel=r"$\frac{r}{r_0}$", ylabel=r"$C(\frac{r}{r_0})$", legend_loc="best")
        plotdl.save_ax(ax, di['filename'])
        ax.cla()
        

def create_bloch_sample_1d(N):
    """
    """
    bloch = Sample(1)
    bloch.points = np.linspace(0,1,N, endpoint=False)
    return bloch

def create_bloch_sample_2d(N):
    """ Create NxN 2d bloch sample.  (matrix size is N^4)
    """
    bloch = Sample((1,1))
    pts = np.linspace(0,N,N*N, endpoint=False)
    pts = np.mod(pts,1)
    x = pts
    y = np.sort(pts)
    bloch.points = np.array((x,y)).T
    return bloch

def nn_mesh(normalized_distance_matrix):
    """ Take a matrix, and pick the points with distance approx. 1.
    """
    upper_lim = normalized_distance_matrix < 1.0001
    lower_lim =  normalized_distance_matrix > 0.9999
    return upper_lim*lower_lim*normalized_distance_matrix  # The last multiplication makes the type correct (i.e. not boolean)

def plot_linear_fits(ax, models, **kwargs):
    
    Ds, epss = zip(*[(mod.fit_diff_coef(), mod.epsilon) for mod in models])
    ax.plot(epss, Ds, ".", **kwargs)

def plot_eig_scatter_and_bloch_2d(ax, epsilon_range=(5,2,1,0.5,0.2,0.1), root_number_of_sites = 30, convention=1):

    number_of_sites = root_number_of_sites**2
    ## 2d scatter same graphs as 4nn 
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sample2d = Sample((1,1),number_of_sites)
    bloch2d = create_bloch_sample_2d(root_number_of_sites)
    ex = ExpModel_2d(sample2d, epsilon=1)
    ex.plot_theoretical_eigvals(ax)
    for epsilon in epsilon_range:
        color = colors.next()
        model = ExpModel_2d(sample2d, epsilon=epsilon)
        model_bloch = ExpModel_2d(bloch2d, epsilon = epsilon)
        pl = cummulative_plot(ax,-model.eigvals[1:], r"$\epsilon={epsilon}$".format(epsilon=epsilon), color=color)
        #pl_color = pl[0].get_color()
        cummulative_plot(ax,-model_bloch.eigvals[1:], None, marker='o', mfc='none', mec=color)
        #new - try to fit a curve
        x = -model.eigvals[1:]
        #[a] = sparsedl.cvfit((lambda x,a : x+a),log(x),log(y),[0],w)
        xlim = (max((1.0/(number_of_sites-1))*12*pi*epsilon**4*exp(convention/epsilon),model.xlim[0]), min(model.xlim[1], 0.9*(12*pi*epsilon**4*exp(convention/epsilon))))
        #plot_func(ax, lambda x: x*exp(a), xlim, label="{:3}".format(a), color= color)
        #plot_func(ax, lambda x: x*exp(a), xlim,  color= color)
        info(" epsilon = %f , epsilon**4*exp(1/epsilon) = %f ",epsilon, epsilon**4*exp(1/epsilon))
        plot_func(ax, lambda x: x/(12*pi*(model.epsilon**4*exp(convention/epsilon))),xlim,  color= color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\lambda$",ylabel=r"$C(\lambda)$", legend_loc="best")

def plot_several_pn_graphs(fig, epsilon_range=(0.1,0.2,0.5,1,2,4,5)):
    """  plot pn graphs on ax1, and a pn**2 graph on ax2. """
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sample2d = Sample((1,1),900)
    ex = ExpModel_2d(sample2d, epsilon=1)
    lines_to_scale = []
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    N = len(epsilon_range)    
    for n, epsilon in enumerate(epsilon_range):
        color = colors.next()
        model = ExpModel_2d(sample2d, epsilon=epsilon)
        pn = ((model.eig_matrix**4).sum(axis=0)**(-1))

        if n==0:
            ax0 = fig.add_subplot(N,2,2*n+1)
            ax =ax0
            ax1 = fig.add_subplot(N,2,2*n+2)
            ax_r = ax1
        else:
            ax = fig.add_subplot(N,2,2*n+1, sharex = ax0, sharey=ax0)
            ax_r = fig.add_subplot(N,2,2*n+2, sharex = ax1, sharey=ax1)

        lines_to_scale += ax.plot(-model.eigvals[1:],pn[1:], label=r"$\epsilon={epsilon}$".format(epsilon=epsilon),marker=".", linestyle='', color = color)
        ax_r.plot(pn[1:], label=r"$\epsilon={epsilon}$".format(epsilon=epsilon),marker=".", linestyle='', color = color)

        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax_r.legend()
        ax_r.set_xscale('log')
        ax_r.set_yscale('log')

    plotdl.autoscale_based_on(ax0, lines_to_scale)

def plot_super_pn_graph(ax,epsilon_range=np.arange(0.1,5,0.05)):
    sample2d = Sample((1,1),900)
    super_pn = list()
    for epsilon in epsilon_range:
        model = ExpModel_2d(sample2d,epsilon=epsilon)
        pn = ((model.eig_matrix**4).sum(axis=0)**(-1))
        super_pn += [(((pn/pn.sum())**2).sum())**(-1)]
    ax.plot(epsilon_range, super_pn,'b.')
    plotdl.set_all(ax, xlabel=r"$\epsilon$", ylabel="PN(PN)")

def plot_1d_alexander_theory(ax):
    eps_0_1 = np.linspace(0,1,100)
    eps_1_5 = np.linspace(1,5,200)
    ax.plot(eps_0_1, -eps_0_1/(eps_0_1+1), color="blue", linestyle="-", label=r"$\mathcal{P}(t)\propto t^\alpha$, The plot is of $\alpha$")
    ax.plot(eps_1_5, -0.5*np.ones_like(eps_1_5), color="blue", linestyle="-")
    ax.plot(eps_1_5, exp(1/eps_1_5)*(eps_1_5-1)/eps_1_5, color="red", linestyle="-", label=r"The diffusion coefficient $D$")
    plotdl.set_all(ax, xlabel=r"$\epsilon$",legend_loc="best")

def plot_D_fit_vs_LRT(ax):
    epsilons = np.logspace(-1.5,1,40)
    sample2d = Sample((1,1),900)
    scatter_models = (ExpModel_2d(sample2d, epsilon=eps) for eps in epsilons)
    plot_linear_fits(ax, scatter_models, label="scatter")
    bloch2d = create_bloch_sample_2d(30)
    bloch_4nn_models = (ExpModel_Bloch_2d_only4nn_randomized(bloch2d, epsilon=eps) for eps in epsilons)
    plot_linear_fits(ax, bloch_4nn_models, label="Bloch 4nn")
    DLRT = lambda eps : 6*pi*exp(1/eps)*eps**4
    ax.plot(epsilons , DLRT(epsilons), linestyle="--")
    DLRT_star = lambda eps: DLRT(eps) - DELTA_D(eps, 1/sqrt(2*pi))
    ax.plot(epsilons , DLRT_star(epsilons), linestyle="--")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\epsilon$", legend_loc="best")
    #plotdl.save_ax(ax, "linear_fits")
    #ax.cla()



def plot_x_exp_x(ax,epsilon=1):
    plot_func(ax, lambda x: x*exp(-x/epsilon), [0,epsilon*5])
        

######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    ax = plotdl.new_ax_for_file()
    #plotf_distance_statistics()
    

    #plotf_eig_scatter_and_bloch_2d()
    plotd = lambda n: ax.plot([10**(n-3), 10**(n)],[10**(-3),10**(0)], ':', color='0.2')
    plot_eig_scatter_and_bloch_2d(ax)
    #[plotd(n) for n in np.arange(0,5,0.5)]
    plotdl.save_ax(ax, 'scatter_and_bloch_2d')
    ax.cla()

    plot_eig_scatter_and_bloch_2d(ax,epsilon_range=0.1*np.arange(1,5))
    #[plotd(n) for n in np.arange(0,2,0.5)]
    plotdl.save_ax(ax, 'scatter_and_bloch_2d_small')
    ax.cla()


    plot_eig_scatter_and_bloch_2d(ax,epsilon_range=(0.4,0.8,1.6,3.2,6.4))
    #[plotd(n) for n in np.arange(1,5,0.5)]
    plotdl.save_ax(ax, 'scatter_and_bloch_2d_large')
    ax.cla()

    epsilons = np.logspace(-1.5,1,40)
    sample2d = Sample((1,1),900)
    scatter_models = (ExpModel_2d(sample2d, epsilon=eps) for eps in epsilons)
    plot_linear_fits(ax, scatter_models, label="scatter")
    bloch2d = create_bloch_sample_2d(30)
    bloch_4nn_models = (ExpModel_Bloch_2d_only4nn_randomized(bloch2d, epsilon=eps) for eps in epsilons)
    plot_linear_fits(ax, bloch_4nn_models, label="Bloch 4nn")
    DLRT = lambda eps : 6*pi*exp(1/eps)*eps**4
    plot_func( ax, DLRT, [10**(-1.5), 10**1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\epsilon$", legend_loc="best")
    plotdl.save_ax(ax, "linear_fits")
    ax.cla()


    plot_super_pn_graph(ax)
    plotdl.save_ax(ax, 'super_pn')
    ax.cla()


    fig = plotdl.Figure()
    plot_several_pn_graphs(fig)
    plotdl.save_fig(fig, 'several_pn', size_factor=[1,2])


    #### 1d PN and matshow
    line = Sample(1,900)
    for epsilon in (0.05, 0.2, 1.5,5):
        line_model = ExpModel_1d(line, epsilon)
        plotf_logvals_pn(line_model)
        plot_eigvals_theory(ax, line_model)
        plotdl.save_ax(ax, "exp_1d_{epsilon}_eig".format(epsilon=epsilon))
        ax.set_xscale('log')
        ax.set_yscale('log')
        plotdl.save_ax(ax, "exp_1d_{epsilon}_log_eig".format(epsilon=epsilon))
        ax.cla()

        #plotf_matshow(line_model)
    for epsilon in (0.2,5):
        alter_model = ExpModel_alter_1d(line, epsilon,basename="exp_alter_{dimensions}d_{epsilon}")
        plotf_logvals_pn(alter_model)
    # 1d bloch:
    bloch1d = create_bloch_sample_1d(900)
    for epsilon in (0.2,0.8,1.5,5):
        plotf_logvals_pn(ExpModel_Bloch_1d(bloch1d, epsilon, basename="bloch_1d_{epsilon}"))
 
    bloch1d100 = create_bloch_sample_1d(100)
    plot_eigvals_theory(ax, ExpModel_Bloch_1d(bloch1d100, epsilon=1, basename="bloch_1d"))
    plotdl.save_ax(ax, "bloch_1d_eig")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.save_ax(ax, "bloch_1d_log_eig")
    ax.cla()
    
    #1d two epsilons and theory at once.
    line_model0_2 = ExpModel_1d(line, 0.2)
    for epsilon in (0.5,0.8,1,2,5):
        line_model = ExpModel_1d(line, epsilon)
        cummulative_plot(ax, -line_model.eigvals[1:], label="$\epsilon={epsilon}$".format(epsilon=epsilon))
    line_model0_2.plot_theoretical_eigvals(ax)

    plotdl.set_all(ax, ylabel=r"$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax, "exp_1d_5_0_2_eig")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.save_ax(ax, "exp_1d_5_0_2_log_eig")
    ax.cla()

    

    

    #### 2d PN and matshow
    sample2d = Sample((1,1),900)
    for epsilon in (0.05, 0.2, 1.5, 5):
        random.seed(1)
        model2d = ExpModel_2d(sample2d, epsilon)
        plotf_logvals_pn(model2d)
        plot_eigvals_theory(ax, model2d)
        plotdl.save_ax(ax, "exp_2d_{epsilon}_eig".format(epsilon=epsilon))
        ax.set_xscale('log')
        ax.set_yscale('log')
        plotdl.save_ax(ax, "exp_2d_{epsilon}_log_eig".format(epsilon=epsilon))
        ax.cla()        #plotf_matshow(model2d)
    # 2d bloch:
    bloch2d = create_bloch_sample_2d(30)
    plot_eigvals_theory(ax, ExpModel_Bloch_2d_only4nn(bloch2d, epsilon= 1, basename="bloch_2d")) #epsilon has no meaning here, but it's easier than reimplementing
    plotdl.save_ax(ax, "bloch_2d_eig")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.save_ax(ax, "bloch_2d_log_eig")
    ax.cla()

    for epsilon in (0.2,0.8,1.5,5):
        plotf_logvals_pn(ExpModel_Bloch_2d(bloch2d, epsilon, basename="bloch_2d_{epsilon}"))

    # four nearest neighbor randomized
    ex4nnrand = ExpModel_Bloch_2d_only4nn_randomized(bloch2d, epsilon=1, basename="bloch_2d_4nn_rand_{epsilon}")
    ex4nnrand.plot_theoretical_eigvals(ax)
    for epsilon in (5,2,1,0.5,0.2):
        model = ExpModel_Bloch_2d_only4nn_randomized(bloch2d, epsilon=epsilon, basename="bloch_2d_4nn_rand_{epsilon}")
        pl = cummulative_plot(ax,-model.eigvals[1:], r"$\epsilon={epsilon}$".format(epsilon=epsilon))
        #new - try to fit a curve
        x = -model.eigvals[1:]
        y = np.linspace(1.0/len(x),1,len(x))
        ar = np.arange(899)
        w = (ar%4==3 )*exp(-ar/10.0)
        [a] = sparsedl.cvfit((lambda x,a : x+a),log(x),log(y),[0],w)
        plot_func(ax, lambda x: x*exp(a), model.xlim, label="{:3}".format(a), color= pl[0].get_color())

    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\lambda$",ylabel=r"$C(\lambda)$", legend_loc="best")
    #[plotd(n) for n in np.arange(1,3,0.5)]
    plotdl.save_ax(ax, "bloch_2d_4nn_rand_log_eig")
    ax.cla()

    ## 2d scatter same garphs as 4nn 
    sample2d = Sample((1,1),900)
    ex = ExpModel_2d(sample2d, epsilon=1)
    ex.plot_theoretical_eigvals(ax)
    for epsilon in (5,2,1,0.5,0.2):
        model = ExpModel_2d(sample2d, epsilon=epsilon)
        model_bloch = ExpModel_2d(bloch2d, epsilon = epsilon)
        pl = cummulative_plot(ax,-model.eigvals[1:], r"$\epsilon={epsilon}$".format(epsilon=epsilon))
        pl_color = pl[0].get_color()
        cummulative_plot(ax,-model_bloch.eigvals[1:], marker='o', mfc='none', mec=pl_color)
        
        #new - try to fit a curve
        x = -model.eigvals[1:]
        y = np.linspace(1.0/len(x),1,len(x))
        ar = np.arange(899)
        w = (ar%4==3 )*exp(-ar/100.0)
        [a] = sparsedl.cvfit((lambda x,a : x+a),log(x),log(y),[0],w)
        plot_func(ax, lambda x: x*exp(a), model.xlim, label="{:3}".format(a), color= pl_color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\lambda$",ylabel=r"$C(\lambda)$", legend_loc="best")
    plotdl.save_ax(ax, "sample_scatter_bloch_log_eig")
    ax.cla()

    # 1d - two nearest neighbor randomized
    ex2nnrand = ExpModel_Bloch_1d_only2nn_randomized(bloch1d, epsilon=1)
    ex2nnrand.plot_theoretical_eigvals(ax)
    for epsilon in (5,2,1,0.6,0.5,0.4,0.2):
        model = ExpModel_Bloch_1d_only2nn_randomized(bloch1d, epsilon=epsilon)
        pl = cummulative_plot(ax,-model.eigvals[1:], r"$\epsilon={epsilon}$".format(epsilon=epsilon))
        pl_color = pl[0].get_color()
        model.plot_alexander(ax, linestyle=":", color= pl_color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\lambda$",ylabel=r"$C(\lambda)$", legend_loc="best")
    #plotd1d = lambda n: ax.plot([10**(n), 10**(n+6)],[10**(-3),10**(0)], ':', color='0.2')
    #[plotd1d(n) for n in np.arange(-6,-2,0.5)]
    plotdl.save_ax(ax, "bloch_1d_2nn_rand_log_eig")
    ax.cla()

    plot_1d_alexander_theory(ax)
    plotdl.save_ax(ax, "alexander_1d_theory")
    ax.cla()

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
