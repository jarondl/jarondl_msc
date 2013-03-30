#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for box distribution.  Anderson localization length focus
"""
from __future__ import division

import logging

import numpy as np
import scipy as sp

from numpy import exp
from scipy import linalg
import sparsedl
from sparsedl import (sorted_eigvalsh, banded_ones, periodic_banded_ones,
           zero_sum, lazyprop, omega_d, pi_phasor,boundary_phasor)


### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
     
     
     
     
     
############ NetModel matrices


class NetModel(object):
    """ NetModel
        a network model, consisting mainly of a transition matrix,
        and methods to produce eigenvalues, eigenvectors etc.
        This is different from sample, which holds geometric locations 
        of the dots.
    """
    def __init__(self, number_of_points, dis_param,rseed=None, periodic=True, phi=0):
        """ rseed is the random seed to prepare RandomState"""
        self.prng = np.random.RandomState(rseed)
        self.number_of_points = number_of_points
        self.dis_param = dis_param
        self.periodic = periodic
        self.phi = phi
        
        self.rmatrix = self.rate_matrix()

    def rate_matrix(self):
        """  Return the rate matrix of the model. It is here to be subclassed.
        """
        raise Exception("Unimplemented")
    
    @lazyprop
    def eig_vals(self):
        """ should be used only if eig_matrix is not needed! """
        return sparsedl.sorted_eigvalsh(self.rmatrix)

    @lazyprop
    def eig_matrix(self):
        (self.eig_vals, w) = sparsedl.sorted_eigh(self.rmatrix)
        return w

    @lazyprop
    def PN(self):
        return ((self.eig_matrix**(4)).sum(axis=0)**(-1))
        
        
class GeoModel(NetModel):
    """ These models have geometrical dependency on points located
        in space.
    """
    def __init__(self, sample, *args, **kwargs):
        self.sample = sample
        self._safety1 = 1  ## <<<-- very important
        super(GeoModel,self).__init__(
                sample.number_of_points(), *args, **kwargs)
    
    def rate_matrix(self):
        ex = (exp(self._safety1)*self.sample.exponent_minus_r(self.periodic))**(1/self.dis_param)
        return ex
        
    @lazyprop
    def eig_vals(self):
        """ should be used only if eig_matrix is not needed! """
        return exp(-self._safety1/self.dis_param)*sparsedl.sorted_eigvalsh(self.rmatrix)

    @lazyprop
    def eig_matrix(self):
        (vals, vecs) = sparsedl.sorted_eigh(self.rmatrix)
        self.eig_vals = exp(-self._safety1/self.dis_param)*vals
        return vecs
    






class ExpModel_1d(GeoModel):
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

class ExpModel_2d(GeoModel):
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

class ExpModel_2d_zerodiag(GeoModel):
    """ Subclassing exp model for 2d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        ## sparsedl.zero_sum(ex1)  <-- that is commented.
        return ex1 

class ExpModel_2d_zerodiag_randint(GeoModel):
    """ Subclassing exp model for 2d """
    def rate_matrix(self, convention):
        ex1 = (self.sample.exponent_minus_r(self.periodic, convention))**(1/self.epsilon)
        rand_minus = (np.random.randint(2, size=ex1.size)*2 -1).reshape(ex1.shape)
        sym_rand = np.tril(rand_minus, k = -1)  +  np.tril(rand_minus, k = -1).T
        
        ## sparsedl.zero_sum(ex1)  <-- that is commented.
        return ex1 * sym_rand

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
class Model_Anderson_banded(ExpModel_1d):
    """  Non conservative, all rates are equal to 1,
         the energies are from a box sample """
    def rate_matrix(self, convention=0):
        # we map epsilon to sigma, and the distribution goes from 0 to sigma.
        if self.rseed is not None:
            np.random.seed(self.rseed)
        n = self.sample.number_of_points()
        m = periodic_banded_ones(n, self.bandwidth1d, self.periodic)
        m += (-1)*np.eye(n) + np.diagflat(np.random.permutation(np.linspace(-self.epsilon, self.epsilon, n)))
        if (self.phi != 0):
            return m * boundary_phasor(self.sample.number_of_points(), self.phi)   
        else: return m # to keep things real
        
    def anderson_theory(self):
        return (lambda x: 6*(4-x**2)/(self.epsilon**2))

     
class Model_Anderson_rates_banded(ExpModel_1d):
    """ symmetric"""
    def base_matrix(self):
        """ this is the ordered version of the matrix, without disorder """
        n = self.sample.number_of_points()
        mat = periodic_banded_ones(n, self.bandwidth1d, self.periodic)
        np.fill_diagonal(mat, 0)
        return mat
    def disorder(self):
        """ this is box disorder added where the original matrix is 1"""
        x = np.triu(self.base_matrix(),1)
        m = np.zeros_like(x)
        m[x==1] = np.random.permutation(np.linspace(-0.5*self.epsilon,0.5*self.epsilon, m[x==1].size))
        return m + m.T

    def rate_matrix(self, convention=0):
        if self.rseed is not None:
            np.random.seed(self.rseed)
        m = self.base_matrix() + self.disorder()
        return m * boundary_phasor(self.sample.number_of_points(), self.phi)
        
class Model_Anderson_diagonal_disorder_only(Model_Anderson_rates_banded):
    def disorder(self):
        """ this is box disorder only on diagonal"""
        n = self.sample.number_of_points()
        dis = np.random.permutation(np.linspace(-0.5*self.epsilon,0.5*self.epsilon, n))
        m = np.diagflat(dis)
        return m 
        
class Model_Anderson_semidiagonal_disorder_conserv(Model_Anderson_rates_banded):
    def disorder(self):
        """ this is box disorder only on diagonal"""
        n = self.sample.number_of_points() - 1
        dis = np.random.permutation(np.linspace(-0.5*self.epsilon,0.5*self.epsilon, n))
        m = np.diagflat(dis,k=1)
        m += m.T
        zero_sum(m)
        return m 
     
class Model_Anderson_rates_conserv_banded(ExpModel_1d):
    """ symmetric"""
    def rate_matrix(self, convention=0):
        # we map epsilon to sigma, and the distribution goes from -2\sigma to 0.
        n = self.sample.number_of_points()
        x = np.triu(periodic_banded_ones(n, self.bandwidth1d, self.periodic),1)
        m = np.zeros_like(x)
        ##m[x==1] = np.random.permutation(np.logspace(-2*self.epsilon,0, m[x==1].size)) ###logspace was a bad idea
        m[x==1] = 1+np.random.permutation(np.linspace(-0.5*self.epsilon,0.5*self.epsilon, m[x==1].size))
        m += m.T
        zero_sum(m)####  <---  the change
        return m * boundary_phasor(self.sample.number_of_points(), self.phi)   

