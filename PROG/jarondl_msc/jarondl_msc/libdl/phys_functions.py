#!/usr/bin/env python
"""  
This file is for functions with physical meaning.
"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging

from scipy import linalg, special
from scipy.optimize import curve_fit


### Warn about all float errors
np.seterr(all='warn')

#set up logging:
logger = logging.getLogger(__name__)
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)
FLOAT_EPS = np.finfo(np.float).eps


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
    
def alternative_A_matrix_inv(hamiltonian, c, k, gamma=1):
    """ Takes longer """
    energy, velocity = -2*c*np.cos(k), 2*c*np.sin(k)
    A = A_matrix(hamiltonian,energy, velocity, gamma)
    Aev, Aem = linalg.eig(A)
    return gamma*velocity*np.sum(np.conj(Aem[0,:])*(Aem[-1,:])*Aev**(-1))

def diag_approx_A_matrix_inv(hamiltonian, c, k, eig_matrix, gamma=1):
    """ Takes longer """
    energy, velocity = -2*c*np.cos(k), 2*c*np.sin(k)
    A = A_matrix(hamiltonian,energy, velocity, gamma)
    A_in_psi = (np.dot(np.dot(np.transpose(eig_matrix),A), eig_matrix)).diagonal()
    return gamma*velocity*np.sum(np.conj(eig_matrix[0,:])*(eig_matrix[-1,:])*A_in_psi**(-1))

def alter_diag_approx(eig_vals, c, k, eig_matrix):
    psi_1, psi_N = eig_matrix[0,:], eig_matrix[-1,:]
    energy, velocity = -2*c*np.cos(k), 2*c*np.sin(k)
    a_inv = energy - eig_vals - (1/2)*(energy - 1j*velocity)*(abs(psi_1)**2 + abs(psi_N)**2)
    #return np.sum(np.conj(psi_1)*psi_N*gamma*velocity/a_inv)
    return np.sum((psi_1)*psi_N*velocity/a_inv)
    
def abs_g_diag_approx(eig_vals, c, k, eig_matrix):
    psi_1, psi_N = eig_matrix[0,:], eig_matrix[-1,:]
    energy, velocity = -2*c*np.cos(k), 2*c*np.sin(k)
    a_inv = energy - eig_vals - (1/2)*(energy - 1j*velocity)*(abs(psi_1)**2 + abs(psi_N)**2)
    #return np.sum(np.conj(psi_1)*psi_N*gamma*velocity/a_inv)
    return np.sum(abs((psi_1)*psi_N*velocity/a_inv)**2)
    
def abs_g_diag_approx_of_E(eig_vals, c, energy, eig_matrix):
    psi_1, psi_N = eig_matrix[0,:], eig_matrix[-1,:]
    velocity = 4*(c**2) - energy**2
    a_inv = energy - eig_vals - (1/2)*(energy - 1j*velocity)*(abs(psi_1)**2 + abs(psi_N)**2)
    #return np.sum(np.conj(psi_1)*psi_N*gamma*velocity/a_inv)
    return np.sum(abs((psi_1)*psi_N*velocity/a_inv)**2)

def heat_g(psi_1, psi_N):
    return np.nansum(2*(abs(psi_1)**2)*(abs(psi_N)**2) / ((abs(psi_1)**2) + (abs(psi_N)**2)))

def ga(eig_matrix):
    al, ar = abs(eig_matrix[0,:])**2, abs(eig_matrix[-1,:])**2
    return 2*al*ar/(al+ar)
    
def ander_ga(N, gamma, x):
    """ Anderson expected ga"""
    return 2*N*np.exp(-2*gamma*N)/np.cosh(4*gamma*x)
    
def ga_cdf(gamma, N):
    ming = 2*gamma*N*np.exp(-2*gamma*N)/np.cosh(4*gamma*N)
    maxg = 2*gamma*N*np.exp(-2*gamma*N)
    return (ming, maxg, lambda gh: 1- (1/(2*gamma*N))*np.arccosh(2*(gamma)*N*np.exp(-2*gamma*N)*(gh**(-1))))
        
def lyap_gamma(c,s,E=0):
    """ Gamma for L-> \infty """
    c,s = np.asarray(c), np.asarray(s)
    return (s)**2/(24*(4*c**2-E**2))
    
    

def fit_diff_coef(eigen_values, d=1):
    diffusive_fit_function =  lambda x, D: (omega_d(d)/d )* (x/(4*D*np.pi**2))**(d/2)
    # We need to ignore first point
    size = len(eigen_values) - 1 
    x = -eigen_values[1:]
    y = np.linspace(1.0/(size),1,size)
    ar = np.arange(1,size+1)
    #w = (ar%4==3 )*exp(-ar/3.0)
    ### Trying to cheat less : 
    w = ar*np.exp(-ar/4.0)
    [D], _ = curve_fit(diffusive_fit_function, x, y, [1], w)
    return D

######### Mathematical aux functions
def omega_d(d):
    """ The d-dimensional solid angle """
    return 2*np.pi**(d/2)/special.gamma(d/2)

