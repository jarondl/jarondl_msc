#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

import itertools
import logging
import os

#from scipy.sparse import linalg as splinalg
from numpy import random, pi, log10, sqrt,  exp, expm1, sort, eye, nanmin, nanmax, log, cos, sinc
from scipy.special import gamma
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator

import numpy as np
import scipy as sp

import sparsedl
import plotdl
from geometry import Sample
from sparsedl import sorted_eigvalsh, banded_ones, periodic_banded_ones, zero_sum, lazyprop, omega_d
from plotdl import cummulative_plot, plt

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug


#Setting up some lambda functions:
#  D* = D - DELTA_D  # comes from "rigorous VRH"
#DELTA_D = lambda eps, rstar : 2*pi*(exp(1/eps)*eps*(6*eps**3-exp(-rstar/eps)*(6*eps**3+ 6*eps**2*rstar + 3*eps*rstar**2 + rstar**3)) - rstar**4*exp((1-rstar)/eps)/4)
D_LRT = lambda eps : 6*pi*exp(1/eps)*eps**4 ###  WRONG!
DELTA_D = lambda eps, rstar : (pi/4)*exp((1-rstar)/eps)*(24*(expm1(rstar/eps))*eps**4-24*eps**3*rstar - 12*eps**2*rstar**2-4*eps*rstar**3-rstar**4)
D_ERH_0 = lambda s,rstar: exp(-rstar/s)*pi*0.5*(0.25*rstar**4 + rstar**3*s + 3*rstar**2*s**2 + 6*rstar*s**3 + 6*s**4)

D_ERH_2d_LATTICE = lambda s: (exp(-s/2)-0.5*exp(-s))/s

BANDED_D0_s_b = lambda s,b : (b*(b+1)*(2*b+1)/6.0)*expm1(-2*s)/(-2*s)  #expm1(x) = exp(x)-1 #with higher precision.
BANDED_D_ERH_s_b = lambda p: lambda s,b : (b*(b+1)*(2*b+1)/6.0)*(exp(-2*s*p/b)*(1+2*s*p/b)-exp(-2*s))/(2*s)  #expm1(x) = exp(x)-1 #with higher precision.
BANDED_D_s_b_ERH =  lambda s,b: lambda p : (b*(b+1)*(2*b+1)/6.0)*(exp(-2*s*p/b)*(1+2*s*p/b)-exp(-2*s))/(2*s)
BANDED_OOS_D0_s_b_ERH = lambda p: lambda s,b : 0.5*b*(exp(-2*s*p/b)*(1+2*s*p/b)-exp(-2*s))/(2*s)
BANDED_OOS_D0_s_b = lambda s,b : 0.5*b*expm1(-2*s)/(-2*s)

BANDED_D_s_b_ERH_NEW =  lambda s,b: lambda p : (b*(b+1)*(2*b+1)/6.0)  *  (exp(-s*p/(2*b))*(1+s*p/(2*b))-exp(-s))/(s)


def power_law_logplot(ax, power, coeff, logxlim,label, **kwargs):
    """ Plots 1d diffusion, treating the x value as log10.
    """
    x1,x2 = logxlim
    power_space = np.linspace(x1, x2, 100)
    power_law = coeff*(10**power_space)**(power)
    return ax.plot(power_space, power_law, linestyle='--', label=label, **kwargs)

def plot_func_logplot(ax, func, logxlim,  **kwargs):
    """ Plots function func in a logscale within the bounding box given by logbbox
    """
    x1, x2 = logxlim
    func_space = np.linspace(x1,x2,200)
    func_y = func(10**func_space)
    #print(func_y)
    return ax.plot(func_space, func_y, linestyle='--', **kwargs)

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
    def __init__(self, sample, epsilon, basename="exp_{dimensions}d_{epsilon}",
                 bandwidth1d = None, periodic=True, convention=1, rseed=None, phi=0):
        """ Take sample and epsilon, and calc eigvals and eigmodes"""
        self.epsilon = epsilon
        self.sample = sample
        self.convention = convention
        self.periodic=periodic
        self.bandwidth1d = bandwidth1d
        self.vals_dict = {"epsilon" : epsilon, "dimensions" : sample.d, "number_of_points" : sample.number_of_points()}
        self.permuted = False
        self.basename = basename.format(**self.vals_dict)
        self.rseed = rseed
        self.phi = phi
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
        return -self.eigvals[[1,-1]]

    @lazyprop
    def logxlim(self):
        return [nanmin(self.logvals), nanmax(self.logvals)]
        
    @lazyprop
    def PN(self):
        return ((self.eig_matrix**(4)).sum(axis=0)**(-1))[1:] # why give N-1?
        
    @lazyprop
    def PN_N(self):
        """ PN with N values.. PN is legacy?"""
        return abs((self.eig_matrix**(4)).sum(axis=0)**(-1))
    
    def plot_diff(self, ax, label = r"$\frac{{D}}{{r_0^2}} = {D:.3G} $", **kwargs):
        """ """
        #D = self.diff_coef()#exp(1/self.epsilon)/(4*pi)#1#self.epsilon#
        r = self.sample.normalized_distance_matrix(self.periodic)
        D = (r*self.ex**2).sum(axis=0).mean()#############WRONG r should be squared!!!!
        
        d2 = self.sample.d / 2.0
        #d2  = self.sample.d
        prefactor = 1/((d2)*gamma(d2)*((4*pi*D)**(d2)))
        #power_law_logplot(ax, d2, prefactor, self.logxlim, label=label.format(D=D, **self.vals_dict), **kwargs)
        f = lambda x: prefactor*x**d2
        plot_func_logplot(ax, f, self.logxlim, label ="D = {D:.3G}".format(D=D))


    def plot_PN(self, ax, convention=0, **kwargs) :
        """ plots Participation number"""
        PN = ((self.eig_matrix**(4)).sum(axis=0)**(-1))[1:]
        ev = -self.eigvals[1:]*exp(-convention/self.epsilon)
        return ax.plot(ev, PN,".", **kwargs)

    @lazyprop
    def fit_diff_coef(self):
        fitN = self.sample.number_of_points() - 1
        y = np.linspace(1.0/(fitN),1,fitN)
        ar = np.arange(fitN)
        #w = (ar%4==3 )*exp(-ar/3.0)
        ### Trying to cheat less : 
        w = ar*exp(-ar/4.0)
        x = -self.eigvals[1:]
        #prefactor  = sparsedl.cvfit((lambda x,a : x+a), log(x), log(y), [0],w)
        #D = exp(-prefactor)/(2*pi)
        ## Keep things simple:
        D = sparsedl.cvfit(self.diff_density(), x, y, [1], w)
        return D
    def diff_density(self):
        """ The expected eigenvalue cummulative distribution for a diffusive system"""
        d = self.sample.d
        return lambda x, D: (omega_d(d)/d )* (x/(4*D*pi**2))**(d/2)


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
            f = lambda x: sqrt( (x) * exp(-convention/epsilon) *epsilon / (epsilon - 1)) / pi
        else:
            f = lambda x: exp(-convention)*sinc(epsilon/(epsilon+1))*(x/2)**(epsilon/(epsilon+1))
        plot_func(ax, f, self.xlim, **kwargs)

    def plot_eigmatrix(self, ax, **kwargs):
        em = self.eig_matrix[:,1:]
        em /= em.max(axis=0)
        mshow = ax.matshow(em, vmin=-1,vmax=1)
        ax.figure.colorbar(mshow)
        

        
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
                self.logxlim, label= r"$e^{{-2\cdot({0}-\epsilon\ln(w))}}$".format(convention))
            plot_func_logplot(ax, lambda w: exp(-(convention-self.epsilon*log(w*0.5))),
                self.logxlim, label=r"$e^{{-\cdot({0}-\epsilon\ln(\frac{{w}}{{2}}))}}$".format(convention))
    def plot_theoretical_eigvals(self, ax):
        N = self.sample.number_of_points()
        qx = 2*pi/N*np.arange(N)
        z = sort(2*(cos(qx)+1 ).flatten())[1:]  # the 1: is to remove the 0 mode
        cummulative_plot(ax, z, label="$2+2\cos(q_x)$" ,color="red", marker="x")

    def inverse_resnet(self):
        if self.bandwidth1d is None:
            return sparsedl.resnet(self.ex, 1, self.periodic) /2
        else:
            return sparsedl.resnet(self.ex, self.bandwidth1d, self.periodic) /2
    @lazyprop
    def new_resnet(self):
        N = self.sample.number_of_points()
        shift = sparsedl.create_shift_matrix(N)
        invex = np.linalg.pinv(self.ex)
        retval =  (N//2)*(N//2)*(np.dot( (shift - np.eye(N)), invex)).trace()**(-1)/2
        return retval

    @lazyprop 
    def resnet3(self):
        """ Works only for periodic models at the moment! """ 
        N = self.sample.number_of_points()
        b = self.bandwidth1d
        invex = np.linalg.pinv(self.ex)
        I = np.zeros(N)
        #I[[0 + b, N//2 - b]] = [-1,1]  # We should apply the current as usual....
        I[[0, N//2]] = [-1,1]
        V = invex.dot(I)
        debug("s = {0}, b={1} ".format(self.epsilon, b))
        retval = (N//2 -2*b)*(V[0+b] - V[N//2-b])**(-1)/2.0
        return retval


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

class ExpModel_Banded_Logbox(ExpModel_1d):
    def rate_matrix(self, convention=1):
        # we map epsilon to sigma, and the distribution goes from -2\sigma to 0.
        n = self.sample.number_of_points()
        x = np.triu(periodic_banded_ones(n, self.bandwidth1d, self.periodic), 1)
        m = np.zeros_like(x)
        ##m[x==1] = np.random.permutation(np.logspace(-2*self.epsilon,0, m[x==1].size)) ###logspace was a bad idea
        m[x==1] = np.random.permutation(exp(np.linspace(-2*self.epsilon,0, m[x==1].size)))
        m += m.T
        sparsedl.zero_sum(m)
        return m

class ExpModel_Band_profile_Logbox(ExpModel_1d):
    def __init__(self, *args, **kwargs):
        no_band = lambda m: np.ones(m.sample.number_of_points())
        self.band_profile_function = kwargs.pop("band_profile_function", no_band)
        return ExpModel_1d.__init__(self, *args, **kwargs)


    def rate_matrix(self, convention=1):
        n = self.sample.number_of_points()
        x = np.tri(n, k=-1)
        m = np.zeros_like(x)
        m[x==1] = np.random.permutation(exp(np.linspace(-2*self.epsilon, 0 , m[x==1].size)))
        sm = (m + m.T)
        sm *= self.band_profile_function(self)
        sparsedl.zero_sum(sm)
        return sm

class ExpModel_1d_zerodiag(ExpModel_1d):
    """ Subclassing exp model for 1d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        if self.bandwidth1d is None:
            ex1 = ex1*periodic_banded_ones(ex1.shape[0], 1)
        elif self.bandwidth1d != 0: #### Zero means ignore bandwidth
            ex1 = ex1*periodic_banded_ones(ex1.shape[0], self.bandwidth1d)
        ##sparsedl.zero_sum(ex1)  <-- that's the difference
        return ex1 

class ExpModel_2d(ExpModel):
    """ Subclassing exp model for 2d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        sparsedl.zero_sum(ex1)
        return ex1 

    def LRT_diff_coef(self, convention = 1):
        return 6*pi*exp(convention/self.epsilon)*self.epsilon**4


    def plot_rate_density(self, ax, label=r"Max. rate / row", convention=1, **kwargs):
        N = self.sample.number_of_points()
        brates = self.ex.max(axis=0)
        logbrates = log10(brates)
        if (nanmin(logbrates) < self.logxlim[1]) and (nanmax(logbrates) > self.logxlim[0]):
            cummulative_plot(ax, sort(logbrates), label=label, color='purple')
            plot_func_logplot(ax, lambda w: exp(-pi*(convention-self.epsilon*log(w))**2),
                self.logxlim, label = r"$e^{{-\pi\cdot({0}-\epsilon\ln(w))^2}}$".format(convention))
            plot_func_logplot(ax, lambda w: exp(-0.5*pi*(convention-self.epsilon*log(0.5*w))**2),
                self.logxlim, label = r"$e^{{-\frac{{\pi}}{{2}}\cdot({0}-\epsilon\ln(\frac{{w}}{{2}}))^2}}$".format(convention))

    def plot_theoretical_eigvals(self, ax):
        N = sqrt(self.sample.number_of_points())
        qy, qx = np.meshgrid(2*pi/N*np.arange(N),2*pi/N*np.arange(N))
        z = sort(2*(cos(qx) + cos(qy) +2 ).flatten())[1:]  # the 1: is to remove 0
        cummulative_plot(ax, z, label="$4+2\cos(q_x)+2\cos(q_y) $" ,color="red", marker="x")


    @lazyprop 
    def resnet3(self):
        """ Work in progress """ 
        N = self.sample.number_of_points()
        r = self.sample.normalized_distance_matrix(self.periodic)  #r/r_0
        n1,n2 = np.unravel_index(r.argmax(), r.shape)
        r12 = r[n1,n2]
        #b = self.bandwidth1d
        invex = np.linalg.pinv(self.ex)
        I = np.zeros(N)
        #I[[0 + b, N//2 - b]] = [-1,1]  # We should apply the current as usual....
        I[[n1, n2]] = [-1,1]
        V = invex.dot(I)
        sV = sorted(V)
        
        # I'm trying to make the same twist as in the banded model.  please
        #  note that it does only work for large matrices!
        debug("s = {0} ; r = {1}; n1,n2 = {2}".format(self.epsilon, r12, (n1,n2)))
        #return (N//2 -2*b)*(V[0+b] - V[N//2-b])**(-1)/2.0
        debug(" oldstyle : {0}".format( (V[n1]-V[n2])**(-1)*np.log(r12)/pi))
        return (sV[-1]-sV[1])**(-1)*np.log(r12)/pi

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
        #W = exp(convention - np.sqrt( -log(np.linspace(0, 1, ex[lnn==1].shape[0] + 1)[1:])/pi))**(1/self.epsilon)
        W = exp( - np.linspace(0, self.epsilon, ex[lnn==1].shape[0] + 1)[1:])
        debug("ex[lnn=1].shape = {0}".format(ex[lnn==1].shape))
        #print W.shape
        ex[lnn==1] = np.random.permutation(W)
        sym_ex = ex + ex.T
        zero_sum(sym_ex)
        return sym_ex

class ExpModel_Bloch_2d_only4nn_randomized_hs(ExpModel_2d):
    # To confer with percolation
    def rate_matrix(self, convention):
        """ 4 nn (u,d,l,r)"""
        r = self.sample.normalized_distance_matrix(self.periodic)
        lnn = np.tri(r.shape[0])*(r>0.99)*(r<1.001)
        ex = np.zeros(r.shape)
        W = exp( - np.linspace(0, self.epsilon, ex[lnn==1].shape[0] + 1)[1:])
        ### Here comes the change:
        W_c = np.median(W)
        W[W<W_c] = W_c
        # That was it
        debug("ex[lnn=1].shape = {0}".format(ex[lnn==1].shape))
        ex[lnn==1] = np.random.permutation(W)
        sym_ex = ex + ex.T
        zero_sum(sym_ex)
        return sym_ex

######## WOW there is a lot to be done here...

class ExpModel_Bloch_2d_only4nn_randomized_sym(ExpModel_2d):
    # To confer with percolation
    def rate_matrix(self, convention):
        """ 4 nn (u,d,l,r)"""
        r = self.sample.normalized_distance_matrix(self.periodic)
        lnn = np.tri(r.shape[0])*(r>0.99)*(r<1.001)
        ex = np.zeros(r.shape)
        W = exp( np.linspace(-self.epsilon, self.epsilon, ex[lnn==1].shape[0] , endpoint=True))
        ### Here comes the change:
#        W_c = np.median(W)
#        W[W<W_c] = W_c
        # That was it
        debug("ex[lnn=1].shape = {0}".format(ex[lnn==1].shape))
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

class ExpModel_2d_zerodiag(ExpModel):
    """ Subclassing exp model for 2d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        ## sparsedl.zero_sum(ex1)  <-- that is commented.
        return ex1 

class ExpModel_2d_zerodiag_randint(ExpModel):
    """ Subclassing exp model for 2d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        rand_minus = (np.random.randint(2, size=ex1.size)*2 -1).reshape(ex1.shape)
        sym_rand = np.tril(rand_minus, k = -1)  +  np.tril(rand_minus, k = -1).T
        
        ## sparsedl.zero_sum(ex1)  <-- that is commented.
        return ex1 * sym_rand


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
    
    Ds, epss = zip(*[(mod.fit_diff_coef, mod.epsilon) for mod in models])
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
    #### changed to plotting the exponent of S(t) instead of P(t), as is the custom
    s_0_1 = np.linspace(0,1,40)
    s_1_5 = np.linspace(1,5,80)
    ax.plot(s_0_1, 2*s_0_1/(s_0_1+1), color="blue", linestyle="--", label=r"$\alpha$  [$S(t)\propto t^\alpha$]")
    ax.plot(s_1_5, np.ones_like(s_1_5), color="blue", linestyle="--")
    ax.plot(s_1_5, (s_1_5-1)/s_1_5, color="red", linestyle="-", label=r"$D$")
    ax.plot(s_0_1, np.zeros_like(s_0_1), color="red", linestyle="-")
    ax.axvline(1, color="black", linestyle=":")
    ax.set_ylim(-0.1,1.1)
    plotdl.set_all(ax, xlabel=r"$s$", ylabel=r"$D$, $\alpha$",legend_loc="best")

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
    DLRT_star = lambda eps: DLRT(eps) - DELTA_D(eps, 1/sqrt(pi))
    ax.plot(epsilons , DLRT_star(epsilons), linestyle="--", label=r"D^*(1/\sqrt(\pi))")
    DLRT_star1_5 = lambda eps: DLRT(eps) - DELTA_D(eps, 1)
    ax.plot(epsilons , DLRT_star1_5(epsilons), linestyle="--", label=r"D^*(1))")
    DLRT_star2 = lambda eps: DLRT(eps) - DELTA_D(eps, 4/sqrt(2*pi))
    ax.plot(epsilons , DLRT_star2(epsilons), linestyle="--", label=r"D^*(4/\sqrt(\pi))")
    DLRT_star3 = lambda eps: DLRT(eps) - DELTA_D(eps, 20/sqrt(2*pi))
    ax.plot(epsilons , DLRT_star3(epsilons), linestyle="--", label=r"D^*(20/\sqrt(\pi))")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$\epsilon$", legend_loc="best")
    #plotdl.save_ax(ax, "linear_fits")
    #ax.cla()

def get_D_fittings2d_invs(inv_s , sample_size = 2000):
    """ Get D fittings for two dimensional models.

        I recommend using inv_s = np.linspace(0.01, 20, 80)
    """
    # The models are now randomized
    models = (ExpModel_2d(Sample((1,1),sample_size), epsilon = s ) for s in inv_s**(-1))
    debug("inv_s len is {0}".format(len(inv_s)))
    two_type = [("fit",np.float64), ("resnet3",np.float64)]
    D_fits = np.fromiter(((model.fit_diff_coef, model.resnet3) 
            for model in models), dtype = two_type, count=len(inv_s))
    ## I'm changing convention to zero:
    D_fits["resnet3"] *= exp(-inv_s)
    D_fits["fit"] *= exp(-inv_s)
    
    return D_fits
    

def plot_D_fittings2d(ax, inv_s, D):
    """ plot 2d ERH numerics
    """
    D_res = D["resnet3"]
    D_fit = D["fit"]
    ax.plot(-inv_s, D_res, "r.", label=r"ResNet")
    ax.plot(-inv_s, D_fit, "y*", label=r"Spectral")
    x = np.linspace(max(inv_s), min(inv_s), 150)
    ax.plot(-x, D_ERH_0(x**(-1), 0), "b--", label=r"linear")
    ax.plot(-x, D_ERH_0(x**(-1),  sqrt(4.5/pi)),"g-", label=r"$n_c =4.5$")
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$X = -\frac{1}{s}$", legend_loc="best", ylabel=r"$D$")
    # fix the over-density of the yaxis
    ax.yaxis.set_minor_locator(plotdl.ticker.NullLocator())
    mi, ma = ax.get_ylim()
    stride = np.ceil((log10(ma)-log10(mi))/5)
    ax.yaxis.set_major_locator(plotdl.ticker.LogLocator(base=10**stride))


############# this needs urgent re-ordering!!!

def plot_D_fittings2_lattice(ax, inv_s = np.linspace(0.01, 20, 80 )):
    bloch2d = create_bloch_sample_2d(40)  ## enlarged to 2000 because the diffence is visible
    models = (ExpModel_Bloch_2d_only4nn_randomized(bloch2d, epsilon = s ) for s in inv_s**(-1))
    #models = (ExpModel_2d(Sample((1,1),2000), epsilon = s ) for s in inv_s**(-1))
    models_hs = (ExpModel_Bloch_2d_only4nn_randomized_hs(bloch2d, epsilon = s ) for s in inv_s**(-1))

#temp : replaced back to fit
    D_fits = np.fromiter((model.resnet3 for model in models), dtype = np.float64, count=len(inv_s)) # switched to resnet
    D_fits_hs = np.fromiter((model.resnet3 for model in models_hs), dtype = np.float64, count=len(inv_s)) # switched to resnet

    D_C0 = D_fits*exp(-inv_s)  ### I'm changing convention back to 0.
    D_C0_hs = D_fits_hs*exp(-inv_s)  ### I'm changing convention back to 0.
    ax.plot(-inv_s, D_C0, "r.", label=r"$D$")
    ax.plot(-inv_s, D_C0_hs, "b.", label=r"$D$")
    x = np.linspace(max(inv_s), min(inv_s), 150)
    ax.plot(-x,4*np.ones_like(x), "b--", label=r"linear")
#    ax.plot(x, D_ERH_0(x**(-1), sqrt(1/pi)), "g-", label=r"$p_c =1$")
    ax.plot(-x, D_ERH_2d_LATTICE(x**(-1)),"g-", label=r"2d erh.. $n_c =5$")
#    ax.plot(x, D_ERH_0(x**(-1), sqrt(8/pi)), "y-", label=r"$p_c =8$")
    #ax.set_xlim(max(inv_s),min(inv_s))
    #inv_formatter = lambda x, pos : "{0:.3f}".format(x**(-1))
    #ax.xaxis.set_major_formatter(FuncFormatter(inv_formatter))
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$X = -\frac{1}{s}$", legend_loc="best", ylabel=r"$D$")
    #ax.locator_params(axis='y', nbins=6)
    # fix the over-density of the yaxis
    ax.yaxis.set_minor_locator(plotdl.ticker.NullLocator())
    mi, ma = ax.get_ylim()
    stride = np.ceil((log10(ma)-log10(mi))/5)
    ax.yaxis.set_major_locator(plotdl.ticker.LogLocator(base=10**stride))


def plot_D_fittings2_1d(ax, s_space = np.linspace(0.1, 20, 80 ), b=1):
    sample1d = Sample(1,900)
    models = (ExpModel_1d(sample1d, epsilon = s , bandwidth1d=b) for s in s_space)
    three_type = [("fit",np.float64), ("resnet",np.float64),("new_resnet",np.float64)]
    D_fits = np.fromiter(((model.fit_diff_coef,model.inverse_resnet(),model.new_resnet) for model in models), dtype = three_type, count=len(s_space))
    #D_fit_0 = D_fits["fit"]*exp(-1/s_space)  ### I'm changing convention back to 0.
    #ax.plot(s_space, D_C0, "r.", label=r"$D$")
    ax.plot(s_space, D_fits["fit"]*exp(-1/s_space), ".", label=r"$D (fit), b = {0}$".format(b))
    ax.plot(s_space, D_fits["resnet"]*exp(-1/s_space), ".", label=r"$D (res-net), b = {0}$".format(b))
    ax.plot(s_space, D_fits["new_resnet"]*exp(-1/s_space), ".", label=r"$D (new res-net), b = {0}$".format(b))
    x = np.linspace(min(s_space), max(s_space), 150)
#    ax.plot(-x, D_ERH_0(x**(-1), 0), "b--", label=r"$p_c =0$")
#    ax.plot(x, D_ERH_0(x**(-1), sqrt(1/pi)), "g-", label=r"$p_c =1$")
#    ax.plot(-x, D_ERH_0(x**(-1),  sqrt(5/pi)),"g-", label=r"$p_c =5$")
#    ax.plot(x, D_ERH_0(x**(-1), sqrt(8/pi)), "y-", label=r"$p_c =8$")
    #ax.set_xlim(max(inv_s),min(inv_s))
    #inv_formatter = lambda x, pos : "{0:.3f}".format(x**(-1))
    #ax.xaxis.set_major_formatter(FuncFormatter(inv_formatter))
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"$s$", legend_loc="best", ylabel=r"$D$")


def get_D_fittings_logbox(s_space, b_space):
    """ some refractoring.. this gets all the D fittings for the banded logbox model"""
    s_grid, b_grid = np.meshgrid(np.asarray(s_space), np.asarray(b_space))
    bloch1d = create_bloch_sample_1d(1000)
    outprod = zip(s_grid.flat, b_grid.flat)
    models = (ExpModel_Banded_Logbox(bloch1d, epsilon = s , bandwidth1d=b) for (s,b) in outprod)
    two_type = [("fit",np.float64), ("new_resnet",np.float64), ("resnet3", np.float64)]
    D_fits = np.fromiter(((model.fit_diff_coef, model.new_resnet, model.resnet3) for model in models), dtype = two_type, count=s_grid.size)
    return D_fits.reshape(s_grid.shape)


def get_D_fittings_2d_lattice(s_space):
    """ this gets all the D fittings for the 2d randomized lattice model"""
    bloch2d = create_bloch_sample_2d(40)  #(40**2=1600)
    models = (ExpModel_Bloch_2d_only4nn_randomized(bloch2d, epsilon = s) for s in s_space)
    two_type = [("fit",np.float64), ("resnet3", np.float64)]
    #D_fits = np.fromiter(((model.fit_diff_coef, model.new_resnet, model.resnet3) for model in models), dtype = two_type, count=s_grid.size)
    D_fits = np.fromiter(((model.fit_diff_coef, model.resnet3) for model in models), dtype=two_type, count=s_space.size)
    return D_fits

def get_D_fittings_2d_hs(s_space):
    """ this gets all the D fittings for the 2d randomized lattice model"""
    bloch2d = create_bloch_sample_2d(40)  #(40**2=1600)
    models = (ExpModel_Bloch_2d_only4nn_randomized_hs(bloch2d, epsilon = s) for s in s_space)
    two_type = [("fit",np.float64), ("resnet3", np.float64)]
    #D_fits = np.fromiter(((model.fit_diff_coef, model.new_resnet, model.resnet3) for model in models), dtype = two_type, count=s_grid.size)
    D_fits = np.fromiter(((model.fit_diff_coef, model.resnet3) for model in models), dtype=two_type, count=s_space.size)
    return D_fits

def get_D_fittings_2d_sym(s_space):
    """ this gets all the D fittings for the 2d randomized lattice model"""
    bloch2d = create_bloch_sample_2d(40)  #(40**2=1600)
    models = (ExpModel_Bloch_2d_only4nn_randomized_sym(bloch2d, epsilon = s) for s in s_space)
    two_type = [("fit",np.float64), ("resnet3", np.float64)]
    #D_fits = np.fromiter(((model.fit_diff_coef, model.new_resnet, model.resnet3) for model in models), dtype = two_type, count=s_grid.size)
    D_fits = np.fromiter(((model.fit_diff_coef, model.resnet3) for model in models), dtype=two_type, count=s_space.size)
    return D_fits

def get_D_fittings_logbox_profile(s_space, b_space):
    """ same as above, only with one_over_square """
    def cut_one_over_square(model):
        r = model.sample.normalized_distance_matrix()
        r += np.eye(1000)
        cutoff = periodic_banded_ones(1000, model.bandwidth1d)
        return cutoff*(r**(-2))
    s_grid, b_grid = np.meshgrid(np.asarray(s_space), np.asarray(b_space))
    bloch1d = create_bloch_sample_1d(1000)
    outprod = zip(s_grid.flat, b_grid.flat)
    models = (ExpModel_Band_profile_Logbox(bloch1d, epsilon = s , bandwidth1d=b, band_profile_function=cut_one_over_square) for (s,b) in outprod)
    two_type = [("fit",np.float64), ("new_resnet",np.float64), ("resnet3", np.float64)]
    D_fits = np.fromiter(((model.fit_diff_coef, model.new_resnet, model.resnet3) for model in models), dtype = two_type, count=s_grid.size)
    return D_fits.reshape(s_grid.shape)

def plot_D_fittings2_logbox(ax, s_space = np.linspace(0.1, 20, 80 ), b=1):
    bloch1d = create_bloch_sample_1d(900)
    models = (ExpModel_Banded_Logbox(bloch1d, epsilon = s , bandwidth1d=b) for s in s_space)
    two_type = [("fit",np.float64), ("new_resnet",np.float64), ("resnet3", np.float64)]
    D_fits = np.fromiter(((model.fit_diff_coef, model.new_resnet, model.resnet3) for model in models), dtype = two_type, count=len(s_space))
    #D_fit_0 = D_fits["fit"]*exp(-1/s_space)  ### I'm changing convention back to 0. WTF!!!!!!!!!
    #ax.plot(s_space, D_C0, "r.", label=r"$D$")
    D0 = BANDED_D0_s_b(s_space,b)
    debug("last items in fit, resnet, and D0 : {0} {1} {2}".format(D_fits["fit"][-1], D_fits["new_resnet"][-1], D0[-1]))
    ax.plot(s_space, D_fits["fit"]/D0 , ".", label=r"D (fit), b = {0}".format(b))
    ax.plot(s_space, D_fits["new_resnet"]/ D0 , ".", label=r"D (resnet)")
    ax.plot(s_space, D_fits["resnet3"]/ D0 , ".", label=r"D (resnet 3 )")
    x = np.linspace(min(s_space), max(s_space), 150)
#    ax.plot(-x, D_ERH_0(x**(-1), 0), "b--", label=r"$p_c =0$")
#    ax.plot(x, D_ERH_0(x**(-1), sqrt(1/pi)), "g-", label=r"$p_c =1$")
#    ax.plot(-x, D_ERH_0(x**(-1),  sqrt(5/pi)),"g-", label=r"$p_c =5$")
#    ax.plot(x, D_ERH_0(x**(-1), sqrt(8/pi)), "y-", label=r"$p_c =8$")
    #ax.set_xlim(max(inv_s),min(inv_s))
    #inv_formatter = lambda x, pos : "{0:.3f}".format(x**(-1))
    #ax.xaxis.set_major_formatter(FuncFormatter(inv_formatter))
    ax.set_yscale('log')
    plotdl.set_all(ax, xlabel=r"s", legend_loc="best", ylabel=r"D")


def plot_banded_logbox_s_b(ax, s, b, color):
    """ plots cum_eigvals for  s  and b . also plots resnet values
    """
    bloch1d = create_bloch_sample_1d(1000)
    model = ExpModel_Banded_Logbox(bloch1d, s, bandwidth1d=b)
    cummulative_plot(ax, -model.eigvals[1:], color=color)
    x_space = np.linspace(-model.eigvals[1], -model.eigvals[100], 10)
    exp_dist = model.diff_density() # Expected diffusion density
    ax.plot( x_space, exp_dist(x_space, model.new_resnet), linestyle="-.", color=color, label="improved resnet b = {0}, s = {1}".format(b,s))
    ax.plot( x_space, exp_dist(x_space, model.fit_diff_coef), linestyle=":", color=color, label="fitting")
    ax.set_xscale('log')
    ax.set_yscale('log')


def plotf_banded_examples():
    ax = plotdl.new_ax_for_file()
    for (b,color) in zip((1,10,20,30),("red", "blue", "green", "purple")):
        plot_banded_logbox_s_b(ax, 2, b, color)
    plotdl.set_all(ax, legend_loc="best")
    plotdl.save_ax(ax, "banded_s_2")
    plotdl.save_ax(ax, "banded_s_2_big", size_factor=(3,3))
    ax.cla()

    for (s,color) in zip((1,10,20,30),("red", "blue", "green", "purple")):
        plot_banded_logbox_s_b(ax, s, 20, color)
    plotdl.set_all(ax, legend_loc="best")
    plotdl.save_ax(ax, "banded_b_20")
    plotdl.save_ax(ax, "banded_b_20_big", size_factor=(3,3))
    ax.cla()

    for (s,color) in zip((1,10,20,30),("red", "blue", "green", "purple")):
        plot_banded_logbox_s_b(ax, s, 1, color)
    plotdl.set_all(ax, legend_loc="best")
    plotdl.save_ax(ax, "banded_b_1")
    plotdl.save_ax(ax, "banded_b_1_big", size_factor=(3,3))
    ax.cla()
        

    plot_D_fittings2_logbox(ax, np.linspace(1,10,40), b=20)
    plotdl.save_ax(ax, "banded_D_of_s_b20", size_factor=(3,3))
    ax.cla()

    plot_D_fittings2_logbox(ax, np.linspace(1,10,40), b=40)
    plotdl.save_ax(ax, "banded_D_of_s_b40", size_factor=(3,3))
    ax.cla()



def plot_me_vs_amir(ax1, ax2, eps_range = (0.05, 0.1, 0.15)):
    """ Plotting the eigenvalue dist with D vs amir's result"""
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sample2d = Sample((1,1),900)
    for eps in eps_range:
        color=colors.next()
        m = ExpModel_2d(sample2d,eps)
        ev = - m.eigvals[1:]*exp(-1/eps)
        cummulative_plot(ax1, ev, label=r"$s = {0}$".format(eps), color=color)
        D = m.fit_diff_coef*exp(-1/eps)
        plot_func(ax1, lambda x: m.diff_density()(x,D), xlim=m.xlim*exp(-1/eps), color=color)
        x = np.logspace(log10(ev[1]), log10(ev[-1]))
        ax1.plot(x,exp(-0.5*pi*eps**2*log(0.5*x)**2), color=color)
        m.plot_PN(ax2, convention=1, color=color)
    ax1.set_xlim(2*exp(-sqrt(2*log(900)/(pi*min(eps_range)**2))), 2 )
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(1/900,1)
    plotdl.set_all(ax1, xlabel=r"$\lambda$", ylabel = r"$\mathcal{N}(\lambda)$", legend_loc="upper left")



def plot_randomized_2d_ev(ax1, eps_range = (0.05, 0.1, 0.15)):
    """ Plotting the eigenvalue dist, for randomized model"""
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sample2d = Sample((1,1),900)
    for eps in eps_range:
        color=colors.next()
        m = ExpModel_2d(sample2d,eps)
        m2 = ExpModel_2d(sample2d,eps)
        ### THIS IS BLACK MAGIC, dependent on value lazyness:
        rex = m2.ex
        m2.ex = sparsedl.permute_diagonals(rex)

        #
        ev = - m.eigvals[1:]*exp(-1/eps)
        ev2 = - m2.eigvals[1:]*exp(-1/eps)
        cummulative_plot(ax1, ev, label=r"$s = {0}$".format(eps), color=color)
        cummulative_plot(ax1, ev2, color=color, marker="x")
        D = m.fit_diff_coef*exp(-1/eps)
        #plot_func(ax1, lambda x: m.diff_density()(x,D), xlim=m.xlim*exp(-1/eps), color=color)
        #x = np.logspace(log10(ev[1]), log10(ev[-1]))
        #ax1.plot(x,exp(-0.5*pi*eps**2*log(0.5*x)**2), color=color)
        #m.plot_PN(ax2, convention=1, color=color)
        #m2.plot_PN(ax2, convention=1, color=color, marker="x")
    ax1.set_xlim(2*exp(-sqrt(2*log(900)/(pi*min(eps_range)**2))), 2 )
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    #ax2.set_yscale('log')
    #ax2.set_xscale('log')
    ax1.set_ylim(1/900,1)
    plotdl.set_all(ax1, xlabel=r"$\lambda$", ylabel = r"$\mathcal{N}(\lambda)$", legend_loc="upper left")


def plotf_geom():
    
    fig = plotdl.Figure()
    ax1 = fig.add_subplot(111)
    plot_randomized_2d_ev(ax1)
    plotdl.save_fig(fig, "pts_geom", size_factor=(1,1), pad=0, h_pad=0, w_pad=0)

def plot_1d_cummulative_PN(ax1, ax2, eps_range = (2, 0.4, 0.2)):
    """ Plotting the eigenvalue dist 1d"""
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sample1d = Sample((1),900)
    maxev, minev = 0,1
    for eps in eps_range:
        color=colors.next()
        m = ExpModel_1d(sample1d,eps)
        ev = - m.eigvals[1:]*exp(-1/eps)
        maxev, minev = (max(maxev,nanmax(ev)), min(minev,nanmin(ev)))
        cummulative_plot(ax1, ev, label=r"$s = {0}$".format(eps), color=color)
        D = m.fit_diff_coef*exp(-1/eps)
        #plot_func(ax, lambda x: m.diff_density()(x,D), xlim=m.xlim*exp(-1/eps), color=color)
        m.plot_alexander(ax1, convention=0, color=color)
        m.plot_PN(ax2, convention=1, color=color)
        #x = np.logspace(log10(ev[1]), log10(ev[-1]))
        #ax.plot(x,exp(-0.5*pi*eps**2*log(0.5*x)**2), color=color)
    #ax.set_xlim(2*exp(-sqrt(2*log(900)/(pi*min(eps_range)**2))), 2 )
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(1/900,1)
    ax1.set_xlim(minev,maxev)
    ax2.set_xlim(minev,maxev)

    plotdl.set_all(ax1, xlabel=r"$\lambda$", ylabel = r"$\mathcal{N}(\lambda)$", legend_loc="upper left") #mathcal had some issues..

def plot_1d_2d_panels(eps_range2d = (0.05,0.1,0.15), eps_range1d=(0.2,0.4,2,5)):
    f = plt.figure()
    ax1 = f.add_subplot(221)
    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(223, sharex = ax1)
    ax4 = f.add_subplot(224, sharex = ax2)
    ax3.axhline(2, ls="--", color="grey")
    #ax3.axhline(900, ls="--", color="grey")
    ax4.axhline(2, ls="--", color="grey")
    #ax4.axhline(900, ls="--", color="grey")
    plot_1d_cummulative_PN(ax1, ax3, eps_range1d)

    plot_me_vs_amir(ax2, ax4, eps_range2d)
    ax2.set_yticks([0])
    ax2.set_yticklabels("none", visible=False)
    ax2.set_ylabel("")
    ax1.text(1E-1, 5E-3, "$1D$")
    ax2.text(1E-2, 5E-3, "$2D$")
    ## thin-out the ticks:
    #ax1.set_xticks([0])
    #ax3.set_xticks(ax3.get_xticks()[::2])
    #ax4.set_xticks(ax4.get_xticks()[1::2])


    ax3.set_xscale('log')
    ax4.set_xscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')

    ax3.get_xaxis().set_major_locator(LogLocator(base=10000))
    ax4.get_xaxis().set_major_locator(LogLocator(base=10000))
    ax4.set_ylabel("")
    ax3.set_ylabel("$PN$")
    ax3.set_xlabel("$\lambda$")
    ax4.set_xlabel("$\lambda$")

    ax4.set_yticklabels("none",visible=False)
    #ax2.set_xticklabels("none",visible=False)
    f.subplots_adjust(hspace=0, wspace=0)
    plotdl.save_fig(f, "pts_Spectral_PN", size_factor=(2,2), pad=0, h_pad=0, w_pad=0, tight=False)


def plotf_1d_2d_Diffusion(inv_s = np.linspace(0.01, 20, 80 )):
    f = plotdl.Figure()
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    plot_1d_alexander_theory(ax1)
    D = get_D_fittings2d(inv_s = inv_s)
    plot_D_fittings2(ax2, inv_s = inv_s, D=D)

#    ax2.set_yticks([0])
#    ax2.set_yticklabels("none", visible=False)
#    ax2.set_ylabel("")
#    ax1.text(1, 5E-3, "$1D$")
#    ax2.text(1E-2, 5E-3, "$2D$")
    ## thin-out the ticks:
#    ax1.set_xticks(ax1.get_xticks()[::2])
#    ax2.set_xticks(ax2.get_xticks()[1::2])
    f.subplots_adjust(hspace=0, wspace=0)
    plotdl.save_fig(f, "two_Ds", size_factor=(2,1), pad=0, h_pad=0, w_pad=0)


def plot_banded(ax):
    bloch_sample = create_bloch_sample_1d(500)
    # To find evenly distributed s, we use fsolve
    s_range = np.linspace(0.1,1,30)
    sigma_range = sp.optimize.fsolve( lambda x: np.tanh(x)/x - s_range, x0= np.ones_like(s_range))
    bandwidth_range = np.arange(1,35)
    xx, yy = np.meshgrid(bandwidth_range, sigma_range)
    DERH = np.vectorize(lambda bandwidth, sigma : ExpModel_Banded_Logbox(bloch_sample, sigma, bandwidth1d = bandwidth).fit_diff_coef[0])
    DERH_mat  = DERH(xx,yy)
    mshow = ax.matshow(DERH_mat)
    ax.figure.colorbar(mshow)
    return DERH_mat

def plot_BANDED_D_of_S(ax,s,b,D):
    ####### PLEASE NOTE THAT OLD DATA HAS s in [0,2\sigma]
    DERH = BANDED_D_s_b_ERH_NEW(2*s,b)
    ax.plot(2*s, D["resnet3"][0]/DERH(0), "r.", label="Resistor Network")
    #ax.plot(2*s, D["fit"][0]/DERH(0), "b.", label="Spectral Analysis")
    #ax.plot(2*s, DERH(1)/DERH(0), "y-", label="$D_{ERH} \qquad [n_c=1]$")
    #ax.plot(2*s, DERH(1.1)/DERH(0), "-", label="$D_{ERH} \qquad [n_c=1.1]$")
    ax.plot(2*s, DERH(2)/DERH(0), "-", label="$D_{ERH} \qquad [n_c=2]$")
    ax.axhline(1,ls="-", color="green")
    ax.set_xlabel("$\sigma$")
    ax.set_ylabel("$g_s$")
    ax.set_yscale("log")
    ax.legend(loc="lower left")

def plot_BANDED_scatter_spectral_vs_resnet(ax, s, b, D):
    DERH = BANDED_D_s_b_ERH_NEW(2*s,b)

    xmin = min(((D["resnet3"][0]/DERH(0)).min()  ,  (D["fit"][0]/DERH(0)).min()))
    xmax = max(((D["resnet3"][0]/DERH(0)).max()  ,  (D["fit"][0]/DERH(0)).max()))
    ax.plot([xmin,xmax],[xmin,xmax], '-r', zorder=0.1)
    ax.scatter(D["resnet3"][0]/DERH(0)  ,  D["fit"][0]/DERH(0)  , color='blue', edgecolors='none', zorder=0.2)

    ax.set_xlabel("Resistor network")
    ax.set_ylabel("Spectral analysis")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(xmin,xmax)

def plot_D_matrix(figure, matrix, x, y):

    ax = figure.add_subplot(111)
    mshow = ax.matshow(matrix)

    xdict = dict(enumerate(x))
    ydict = dict(enumerate(y))
    xfmtr = FuncFormatter(lambda x,pos : "{0:0.0f}".format(xdict.get(x,0)))
    yfmtr = FuncFormatter(lambda x,pos : "{0:0.0f}".format(ydict.get(x-1,0)))
    ax.xaxis.set_major_formatter(xfmtr)
    ax.yaxis.set_major_formatter(yfmtr)
    #cbar = figure.colorbar(mshow, ax=ax,use_gridspec=True, ticks=[0,0.5,1])
    #(child_ax,kw) = plotdl.mpl.colorbar.make_axes_gridspec(ax)
    try:
        (child_ax,kw) = plotdl.mpl.colorbar.make_axes_gridspec(ax)
    except AttributeError:
        print ("Old version of matplotlib, colorbar might be mispositioned")
        (child_ax,kw) = plotdl.mpl.colorbar.make_axes(ax)
    cbar = figure.colorbar(mshow, cax=child_ax, ticks=[0,0.5,1])#,use_gridspec=True)
    #child_ax.colorbar(ticks=[0,0.5,1])
#    cbar
    return ax

def plot_three_D(D, s, b):
    s_grid, b_grid = np.meshgrid(s,b)
    for D_type in ("new_resnet", "resnet3", "fit"):
        f = plotdl.Figure()
        D0 = BANDED_D0_s_b(s_grid, b_grid)
        plot_D_matrix(f, D[D_type]/D0, s, b)
        plotdl.save_fig(f, D_type, tight=False, size_factor=(2,1.5))
        f.clf()

def plot_banded_resnet3(fig, D, s, b):
    s_grid, b_grid = np.meshgrid(s,b)
    D0 = BANDED_D0_s_b(s_grid, b_grid)
    ax = plot_D_matrix(fig, D["resnet3"]/D0, s*2, b)  #s*2 is the convention
    ax.set_xlabel("$\sigma$")
    ax.set_ylabel("$b$")



def plot_x_exp_x(ax,epsilon=1):
    plot_func(ax, lambda x: x*exp(-x/epsilon), [0,epsilon*5])
    

def plots_for_fete2013():
    """ specific plots for the negev fete 2013 """
    fig, ax = plt.subplots()
    sam = Sample((1,1),400)
    mod = ExpModel_2d(sam,0.5,periodic=True)
    gam = -mod.ex.diagonal()
    sct = ax.scatter(sam.points[:,0], sam.points[:,1], edgecolors='none', 
        c= gam, vmin=gam.min(), vmax=gam.max())
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    fig.colorbar(sct)
    fig.savefig('pts_points.png')


######## One function to plot them all
def all_plots(seed= 1, **kwargs):
    """  Create all of the figures. Please note that it might take some time.
    """
    ### create a figures dir if not already there:
    try:
        os.mkdir("figures")
    except OSError:
        pass  ## meaning it exists
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

    plot_D_fit_vs_LRT(ax)
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

#### all of the plots in the article and in the Phd research proposal
  
def article_plots(seed = 1 ):
    """ Creates all of the plots used in the first pts manuscript
        and in the research proposal. I reseed the random generator
        with seed=1 for every run to stay consistent.
    """
    ### create a figures dir if not already there:
    try:
        os.mkdir("figures")
    except OSError:
        pass  ## meaning it exists

    ax = plotdl.new_ax_for_file()
    plot_1d_alexander_theory(ax)
    plotdl.save_ax(ax, "ptsD_1D_long", size_factor=(1,2))
    ax.cla()



    random.seed(seed)
    #banded matrices for the image
    # if the data exists, use it:
    try:
        f = np.load("D_banded.npz")
        b_space = f["b_space"]
        s_space = f["s_space"]
        D = f["D"]

    except (OSError, IOError):

        # otherwise, get the data:
        b_space = np.arange(1,50)
        s_space = np.linspace(1E-4,10,98) # maybe the problem is just for one value?
        D = get_D_fittings_logbox(s_space,b_space)
        np.savez("D_banded.npz", b_space=b_space , s_space=s_space, D = D)

    ## now plot the plots

    fig = plotdl.Figure()
    plot_banded_resnet3(fig,D,s_space,b_space)
    plotdl.save_fig(fig, "ptsD_banded_Image", pad=0.4,size_factor = (1.2,1))
    fig.clf()

    ### banded matrices for the scatter plot
    try:
        f = np.load("D_banded_b10.npz")
        b_space = f["b_space"]
        s_space = f["s_space"]
        D = f["D"]

    except (OSError, IOError):

        # otherwise, get the data:
        b_space = np.array((10,))
        s_space = np.linspace(1E-4,40,400) 
        # change to 400 afterwards??
        D = get_D_fittings_logbox(s_space,b_space)
        np.savez("D_banded_b10.npz", b_space=b_space , s_space=s_space, D = D)


    plot_BANDED_D_of_S(ax,s_space,10, D)  
    plotdl.save_ax(ax, "ptsD_banded")
    ax.cla()

    plot_BANDED_scatter_spectral_vs_resnet(ax,s_space,10,D)
    plotdl.save_ax(ax, "ptsD_banded_scatter")
    ax.cla() 
    



    random.seed(seed)
    # 2d (for ERH)
    # if the data exists, use it:
    try:
        f = np.load("D_2d.npz")
        inv_s = f["inv_s"]
        D = f["D"]

    except (OSError, IOError):

        # otherwise, get the data:
        inv_s = np.linspace(0.01,20,100) # change to 100 later
        D = get_D_fittings2d_invs(inv_s, 2000) #change to 2000 later
        np.savez("D_2d.npz", inv_s = inv_s, D = D)
    

    plot_D_fittings2d(ax, inv_s, D)
    plotdl.save_ax(ax, "ptsD_2D_long", size_factor=(1,1.7))
    ax.cla()
 

    random.seed(seed)
    plotf_geom()


    random.seed(seed)
    plot_1d_2d_panels()
    
    


if __name__ ==  "__main__":
    article_plots()
