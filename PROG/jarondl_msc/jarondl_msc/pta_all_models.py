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

