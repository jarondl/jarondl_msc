#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
This module holds all of the models. Each model is described
as a class, with several input parameters, and several available
methods. Note that most models are inherited, meaning they add 
parameters and methods to those of their "ancestors". **ONLY** extra 
parameters are given.

This is the inheritance diagram:

.. inheritance-diagram:: jarondl_msc.models
   :parts: 1

"""
from __future__ import division

import logging

import numpy as np
import scipy as sp

from numpy import exp
from scipy import linalg
from .libdl import sparsedl
from .libdl.sparsedl import (sorted_eigvalsh, banded_ones, periodic_banded_ones,
           zero_sum, omega_d, pi_phasor,boundary_phasor)
from .libdl.tools import lazyprop


### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
     
     
     
     
     
############ NetModel matrices:parts: 1


class NetModel(object):
    """ A network model, consisting mainly of a transition matrix,
        and methods to produce eigenvalues, eigenvectors etc. **All models
        inherit from this one**.
        This is different from sample, which holds geometric locations 
        of the dots. The only necessary argument is the number of points,
        the rest can be omitted as they have default values.
        
        :param number_of_points: :math:`N` - the number of points in the model
        :param dis_param: The disorder parameter, in the ``PTA`` scope mostly
                       calles :math:`W`, but was :math:`s` in previous work.
        :param conserving: Should the sum of each row be set to 0?
        :type conserving: boolean 
        :param periodic: Should boundary conditions be periodic?
        :type periodic: boolean
        :param phi: The phase accumulated if indeed periodic
        :param model_name: Used sometimes for presentation, not recommended.
    """
    def __init__(self, number_of_points, dis_param,conserving = False, prng=None, periodic=True, phi=0, model_name=None):
        """ rseed is the random seed to prepare RandomState"""
        self.prng = (prng if prng is not None else np.random.RandomState(None))
        self.number_of_points = self.N = number_of_points
        self.dis_param = dis_param
        self.periodic = periodic
        self.phi = phi
        self.conserving = conserving
        if model_name is None:
            self.model_name = self.__class__.__name__
        self.model_name = model_name
        
    @lazyprop
    def rate_matrix(self):
        """  Return the rate matrix of the model. 
             This uses the inherited method sub_rate_matrix,
             and deals with boundary phase and conservativity.
             if not conserving it leaves diagonal as is.
        """
        ex = self.sub_rate_matrix()
        if self.conserving:
            sparsedl.zero_sum(ex)
        return ex*boundary_phasor(self.number_of_points, self.phi) 
    
    @lazyprop
    def eig_vals(self):
        """ Return the sorted eigenvalues. """
        return sparsedl.sorted_eigvalsh(self.rate_matrix)

    @lazyprop
    def eig_matrix(self):
        """ return eigen matrix, with vectors sorted by eig_vals"""
        (self.eig_vals, w) = sparsedl.sorted_eigh(self.rate_matrix)
        return w

    @lazyprop
    def PN(self):
        """ Participation number """
        return ((self.eig_matrix**(4)).sum(axis=0)**(-1))

        
        
class GeoModel(NetModel):
    """ These models have geometrical dependency on points located
        in space. The idea was to base all ``PTS`` work on this,
        but currently nothing uses it.
        
        :param sample:  a :py:class:`jarondl_msc.geometry.Sample` instance.
    """
    def __init__(self, sample, *args, **kwargs):
        self.sample = sample
        super(GeoModel,self).__init__(
                sample.number_of_points(), *args, **kwargs)
    
    def sub_rate_matrix(self):
        """ this is 'safer' because we don't reach very low values"""
        return (exp(1)*self.sample.exponent_minus_r(self.periodic))**(1/self.dis_param)

        
    @lazyprop
    def eig_vals(self):
        """ should be used only if eig_matrix is not needed! """
        return exp(-1/self.dis_param)*sparsedl.sorted_eigvalsh(self.rate_matrix)

    @lazyprop
    def eig_matrix(self):
        (vals, vecs) = sparsedl.sorted_eigh(self.rate_matrix)
        self.eig_vals = exp(-1/self.dis_param)*vals
        return vecs
    


class NetModel_1d(NetModel):
    """ A subclass for 1d models
    
    :param bandwidth: define the bandwidth. 1 is for strict 1d, 
    and the default is that all links are non zero. (infinite bandwidth?)
    """
    
    def __init__(self, *args, **kwargs):
        self.bandwidth = kwargs.pop("bandwidth", None)
        super(NetModel_1d,self).__init__(*args, **kwargs)
    @lazyprop
    def band_profile(self):
        if self.bandwidth is None:
            return 1
        else:
            return periodic_banded_ones(self.number_of_points, self.bandwidth, self.periodic)
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

class GeoModel_1d(NetModel_1d,GeoModel):

    def sub_rate_matrix(self):
        return self.band_profile*super(GeoModel_1d,self).sub_rate_matrix()
    
     
class Bloch_Banded_1d(NetModel_1d):
    """  an ordered 1d model with finite bandwidth """
         
    def sub_rate_matrix(self):
        return self.base_matrix() + self.disorder()
        
    def base_matrix(self):
        if self.bandwidth is None:
            return np.ones([self.N, self.N]) -1*np.eye(self.N)
        else:
            return  ( (-1)*np.eye(self.N) +  periodic_banded_ones(self.N, self.bandwidth, self.periodic))        
    def disorder(self):
        return 0

class Model_homogenous_banded_1d(NetModel_1d):
    """ must be subclassed by something with homogenous_disorder function """
    def sub_rate_matrix(self):
        matr = np.zeros([self.N, self.N])
        if self.bandwidth is None:
            what_to_fill = np.triu(np.ones([self.N, self.N]),k=1)
        else:
            what_to_fill = np.triu(periodic_banded_ones(self.N, self.bandwidth, self.periodic), k=1)
        l = what_to_fill.sum(axis=None)
        matr[what_to_fill==1] =  self.homogenous_disorder(l)
        ### we treat the diagonal specifically. (note the /2)
        np.fill_diagonal(matr, self.homogenous_disorder(self.N)/2)
        return (matr+matr.T)
        
class Model_Positive_Exp_banded_1d(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        return self.prng.permutation(np.exp(np.linspace(-self.dis_param, 0, N)))
        
class Model_Symmetric_Exp_banded_1d(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        disorder = self.prng.permutation(np.exp(np.linspace(-self.dis_param, 0, N))) 
        sgn = self.prng.randint(0,2,N)*2-1
        return disorder*sgn     
        
class Model_Positive_Exp_banded_1d_from_zero(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        return (self.prng.permutation(np.exp(np.linspace(-self.dis_param, 0, N))) - np.exp(-self.dis_param))/(1-np.exp(-self.dis_param))
        
def Model_Positive_Exp_banded_1d_from_zero_conservative(*args, **kwargs):
    return Model_Positive_Exp_banded_1d_from_zero(*args, conserving=True, **kwargs)
        
def Model_Positive_Exp_banded_1d_conservative(*args, **kwargs):
    return Model_Positive_Exp_banded_1d(*args, conserving=True, **kwargs)

class Model_Symmetric_Exp_banded_1d_from_zero(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        disorder = self.prng.permutation(np.exp(np.linspace(-self.dis_param, 0, N))) - np.exp(-self.dis_param)
        sgn = self.prng.randint(0,2,N)*2-1
        return disorder*sgn
        
class Model_Positive_Box_banded_1d(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        return self.prng.permutation(np.linspace(0, self.dis_param, N))
        
class Model_Positive_Box_around1_banded_1d(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        return 1+ self.prng.permutation(np.linspace(-self.dis_param, +self.dis_param, N))

class Model_Symmetric_Box_banded_1d(Model_homogenous_banded_1d):
    def homogenous_disorder(self,N):
        return self.prng.permutation(np.linspace(-self.dis_param, self.dis_param, N))

 
def Model_Positive_Box_banded_1d_conservative(*args, **kwargs):
    return Model_Positive_Box_banded_1d(*args, conserving=True, **kwargs)
 
def Model_Symmetric_Box_banded_1d_conservative(*args, **kwargs):
    return Model_Symmetric_Box_banded_1d(*args, conserving=True, **kwargs)

class Model_Anderson_DD_1d(Bloch_Banded_1d):
    r""" Diagonal Disorder 1d model, with uniform random values
    in the range :math:`\left[-\frac{1}{2}W, +\frac{1}{2}W\right]`
    """
    def disorder(self):
        return np.diagflat(self.prng.permutation(np.linspace(-0.5*self.dis_param, 0.5*self.dis_param, self.N)))
                
class Model_Anderson_S_DD_1d(Bloch_Banded_1d):
    """ sparse diagonal Disorder """
    def disorder(self):
        dis = np.exp(-np.linspace(0, self.dis_param, self.N))+(exp(-self.dis_param)-1)/self.dis_param
        return np.diagflat(self.prng.permutation(dis))

class Model_Anderson_ROD_1d(Bloch_Banded_1d):
    """ random off diagonal (k=\pm1) """
    def __init__(self, *args, **kwargs):
        self.semiconserving = kwargs.pop("semiconserving", None)
        super(Model_Anderson_ROD_1d,self).__init__(*args, **kwargs)
    def disorder(self):
        m = np.diagflat(self.prng.permutation(np.linspace(-self.dis_param, self.dis_param, self.N-1)), k=1)
        m += m.T
        if self.semiconserving:
            sparsedl.zero_sum(m)
        return m

class Model_Anderson_BD_1d(Model_Anderson_ROD_1d):
    """ Banded disorder, expcept diagonal """
    def __init__(self, *args, **kwargs):
        self.dis_band = kwargs.pop("dis_band", None)
        super(Model_Anderson_BD_1d,self).__init__(*args, **kwargs)   
    def disorder(self):
        m = self.base_matrix()
        dis = np.zeros_like(m)
        ## where is the band we want to disorder?
        if self.dis_band is None:
            where_triband = np.triu(m)==1
            
        else:
            where_triband = np.triu( (-1)*np.eye(self.N) +  periodic_banded_ones(self.N, self.dis_band, self.periodic)) ==1
        l = where_triband.sum(axis=None)
        dis[where_triband] = self.prng.permutation(np.linspace(-self.dis_param, self.dis_param, l))
        dis += dis.T
        if self.semiconserving:
            sparsedl.zero_sum(dis)
        return dis
        
class Model_Anderson_S_BD_1d(Model_Anderson_ROD_1d):
    """ sparse Banded disorder, expcept diagonal """
    
    def disorder(self):
        m = self.base_matrix()
        dis = np.zeros_like(m)
        ## where is the band we want to disorder?
        where_triband = np.triu(m)==1
        l = where_triband.sum(axis=None)
        #dis1 = np.exp(-np.linspace(0, self.dis_param, l))+(exp(-self.dis_param)-1)/self.dis_param
        dis1 = np.exp(-np.linspace(0, self.dis_param, l))
        dis[where_triband] = self.prng.permutation(dis1)
        dis += dis.T
        if self.semiconserving:
            sparsedl.zero_sum(dis)
        return dis
      

     
     
     
     
     
     
 
