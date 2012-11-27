#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

import logging

import numpy as np
import scipy as sp

from numpy import exp

from sparsedl import sorted_eigvalsh, banded_ones, periodic_banded_ones, zero_sum, lazyprop, omega_d
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

class ExpModel_Banded_Logbox_pinning(ExpModel_Banded_Logbox):
    def rate_matrix(self,convention=0):
        m = ExpModel_Banded_Logbox.rate_matrix(self)
        m += self.pinning_matrix()
        return m
    
    def pinning_matrix(self):
		""" This matrix has negative disordered values on the diagonal"""
		W = 0.3
		#pinning =  - np.random.permutation(np.linspace(0.5,1.5,self.sample.number_of_points()))
		# The pinning is uniform on [0,W]
		pinning =  - np.random.permutation(np.linspace(0,W,self.sample.number_of_points()))
		return np.diagflat(pinning)
