#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
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
from ptsplot import ExpModel, ExpModel_1d

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug



class ExpModel_Banded_Logbox(ExpModel_1d):
    def rate_matrix(self, convention=0):
        # we map epsilon to sigma, and the distribution goes from -2\sigma to 0.
        if self.rseed is not None:
            np.random.seed(self.rseed)
        n = self.sample.number_of_points()
        x = np.triu(periodic_banded_ones(n, self.bandwidth1d, self.periodic), 1)
        m = np.zeros_like(x)
        ##m[x==1] = np.random.permutation(np.logspace(-2*self.epsilon,0, m[x==1].size)) ###logspace was a bad idea
        m[x==1] = np.random.permutation(exp(np.linspace(-2*self.epsilon,0, m[x==1].size)))
        m += m.T
        zero_sum(m)
        return m
        
    def plot_PN(self,ax,**kwargs):
        """ In this model convention=0"""
        return ExpModel_1d.plot_PN(self,ax,convention=0, **kwargs)
        

class ExpModel_Banded_Logbox_pi(ExpModel_Banded_Logbox):
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox.rate_matrix(self)
        return m*pi_phasor(self.sample.number_of_points())
    
class ExpModel_Banded_Logbox_phase(ExpModel_Banded_Logbox):
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox.rate_matrix(self)
        if (self.phi != 0):
            return m * boundary_phasor(self.sample.number_of_points(), self.phi)   
        else: return m # to keep things real


class ExpModel_Banded_Logbox_dd(ExpModel_Banded_Logbox_phase):
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox_phase.rate_matrix(self,self.phi)
        m += self.diagonal_disorder()
        return m
    
    def diagonal_disorder(self):
        """ This matrix has negative disordered values on the diagonal"""
        pinning =  - np.random.permutation(exp(np.linspace(-2*self.epsilon,0, self.sample.number_of_points())))
        return np.diagflat(pinning)
        
class ExpModel_Banded_Logbox_rd(ExpModel_Banded_Logbox_dd):#diagonal is not special...
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox.rate_matrix(self)
        np.fill_diagonal(m,0)
        m -= self.diagonal_disorder()
        return m
               
class ExpModel_Banded_Logbox_negative(ExpModel_Banded_Logbox_phase):
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox_phase.rate_matrix(self)
        # randomize the sign:
        N = self.sample.number_of_points()
        ran = sparsedl.symmetric_sign_randomizer(N)
        m2 = m*ran
        zero_sum(m2)
        return m2
        
class ExpModel_Banded_Logbox_negative_dd(ExpModel_Banded_Logbox_negative, ExpModel_Banded_Logbox_dd):
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox_negative.rate_matrix(self)
        m += self.diagonal_disorder()
        return m
        
class ExpModel_Banded_Logbox_negative_rd(ExpModel_Banded_Logbox_negative, ExpModel_Banded_Logbox_dd):#diagonal is not special...
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox_negative.rate_matrix(self)
        np.fill_diagonal(m,0)
        m -= self.diagonal_disorder()
        return m
     
        
class ExpModel_Banded_Logbox_nosym(ExpModel_1d):
    """ this model is not Hermitian!!!"""
    @lazyprop
    def eigvals(self):
        ev = -linalg.eigvals(self.ex)
        ev.sort()
        return -ev

    @lazyprop
    def eig_matrix(self):
        eigvals, eigvecs = linalg.eig(self.ex)
        sort_indices = np.argsort(eigvals)
        return eigvecs[:,sort_indices[::-1]]

    def rate_matrix(self, convention=0):
        # we map epsilon to sigma, and the distribution goes from -2\sigma to 0.
        if self.rseed is not None:
            np.random.seed(self.rseed)
        n = self.sample.number_of_points()
        x = periodic_banded_ones(n, self.bandwidth1d, self.periodic)
        m = np.zeros_like(x)
        ##m[x==1] = np.random.permutation(np.logspace(-2*self.epsilon,0, m[x==1].size)) ###logspace was a bad idea
        m[x==1] = np.random.permutation(exp(np.linspace(-2*self.epsilon,0, m[x==1].size)))
        zero_sum(m)
        if (self.phi != 0):
            return m * boundary_phasor(self.sample.number_of_points(), self.phi)   
        else: return m # to keep things real
        return m
