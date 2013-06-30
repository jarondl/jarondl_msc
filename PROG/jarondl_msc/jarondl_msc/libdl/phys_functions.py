#!/usr/bin/env python
"""  
This file is for functions with physical meaning.
"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging


### Warn about all float errors
np.seterr(all='warn')

#set up logging:
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
########################################################################
################  Physical functions
################  
########################################################################

def chain_g(eig_matrix, b=1):
    g = ((eig_matrix[b-1,:]**2)*(eig_matrix[-b,:]**2))/((eig_matrix[b-1,:]**2)+(eig_matrix[-b,:]**2))
    return g
    
#########  "Physical" functions

def thouless_coef(evs):
    diff = evs - evs[:,np.newaxis]
    np.fill_diagonal(diff,1)
    logsum = np.log(np.abs(diff)).sum(axis=0)#0 or 1 are equal..
    return (logsum/evs.size)**(-1)
    
def thouless_g(ev1, ev2, phi):
    """ This is divided by the spacing"""
    g = abs(ev1 - ev2) / (phi**2)
    # Approximation of  the minimal precision:
    prec = FLOAT_EPS * max(abs(ev1))* len(ev1)
    debug("precision = {0}, minimal g  = {1}".format(prec, min(g)))
    #smoothing:
    avg_spacing = window_avg_mtrx(len(ev1)-1).dot(ev1[:-1]-ev1[1:])
    # avg_spacing is now smaller than eigvals. We duplicate the last value to accomodate (quite  hackish)
    avg_spacing = np.append(avg_spacing,avg_spacing[-1])
    ga = g/avg_spacing
    return ga, prec
    
def pure_thouless_g(ev1, ev2, phi):
    """ without dividing by nothing """
    g = 2*(ev1 - ev2) / (phi)**2

    # Approximation of  the minimal precision:
    prec = FLOAT_EPS * max(abs(ev1))* len(ev1)
    debug("precision = {0}, minimal g  = {1}".format(prec, min(abs(g))))

    return g, prec
    

def A_matrix (hamiltonian,energy, velocity,gamma=1):

    qq = np.zeros_like(hamiltonian)
    qq[0,0] = qq[-1,-1] = 1
    return  energy - hamiltonian -(gamma/2)*(energy- 1j*velocity)*qq

def A_matrix_inv(hamiltonian, c, k,gamma=1):
    energy, velocity = -2*c*np.cos(k), 2*c*np.sin(k)
    A = A_matrix(hamiltonian,energy, velocity, gamma)
    return gamma*velocity*linalg.inv(A)[0,-1]
    

def heat_g(psi_1, psi_N):
    return np.nansum(2*(abs(psi_1)**2)*(abs(psi_N)**2) / ((abs(psi_1)**2) + (abs(psi_N)**2)))



        


