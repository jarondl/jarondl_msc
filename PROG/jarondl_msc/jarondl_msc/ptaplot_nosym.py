#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

import itertools
import logging
import os

#from scipy.sparse import linalg as splinalg
from numpy import random, pi, log10, sqrt,  exp, expm1, sort, eye, nanmin, nanmax, log, cos, sinc, ma
from scipy.special import gamma
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator

import plotdl
import sparsedl
from plotdl import plt, tight_layout, cummulative_plot


import numpy as np
import scipy as sp
import matplotlib as mpl

import ptsplot

import pta_models


mpl.rc('figure', autolayout=True)

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)
FLOAT_EPS = np.finfo(np.float).eps

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# show warnings (remove for production)
logger.setLevel("DEBUG")

info = logger.info
warning = logger.warning
debug = logger.debug



##### Dirty hack, should be fixed by matplotlib 1.2.0
def get_LogNLocator(N = 6):
    try:
        return LogLocator(numticks=N)
    except TypeError:
        warning('using undocumented hack for log ticks')
        Log6Locator = LogLocator()
        Log6Locator.numticks = N
        return Log6Locator




###########################################
######  Numerical computations
###########################################
        


def get_ev_thoules_g_1d(number_of_sites = 1000, s=0.1,b=5, 
                        model_class=pta_models.ExpModel_Banded_Logbox_negative):
    """ This time, the idea is to use versatile models.
    """
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    win_avg_mtrx = sparsedl.window_avg_mtrx(number_of_sites - 1)
    res_type =  np.dtype([("ev",(np.float64,number_of_sites)),
                          ("PN",(np.float64,number_of_sites)),
                          ("thoules_g",(np.float64,number_of_sites)),
                          ("s",np.float64),
                          ("b",np.float64)
                          ])
    
    
    #res = np.zeros(s_grid.size, dtype = res_type) # preallocation is faster..
    res = np.zeros(1,dtype=res_type)
    n=0
    debug('eigenvals ,PN and thouless for s = {0}, b= {1}'.format(s,b))
    model = model_class(sample, epsilon=s, bandwidth1d=b, rseed=n,phi=0)
    model_phi = model_class(sample, epsilon=s, bandwidth1d=b,rseed=n,phi=pi)
    g = abs(model.eigvals - model_phi.eigvals) / (pi**2)
    # Approximation of  the minimal precision:
    prec = FLOAT_EPS * max(abs(model.eigvals))* number_of_sites  
    debug("precision = {0}, minimal g  = {1}".format(prec, min(g)))
    #g = ma.masked_less(g,prec)
    avg_spacing = win_avg_mtrx.dot(-model.eigvals[1:]+model.eigvals[:-1])
    # avg_spacing is now smaller than eigvals. We duplicate the last value to accomodate (quite  hackish)
    avg_spacing = np.append(avg_spacing,avg_spacing[-1])
    ga = ma.masked_less(g/avg_spacing,prec)
    res[n] = ( model.eigvals, model.PN_N, ma.filled(ga, fill_value=prec) , s, b)
    return res
    
def get_sym_neg(N=1000):
    g1 = get_ev_thoules_g_1d(number_of_sites=N,model_class=pta_models.ExpModel_Banded_Logbox_phase)
    g2 = get_ev_thoules_g_1d(number_of_sites=N,model_class=pta_models.ExpModel_Banded_Logbox_pinning)
    g3 = get_ev_thoules_g_1d(number_of_sites=N,model_class=pta_models.ExpModel_Banded_Logbox_negative)
    g4 = get_ev_thoules_g_1d(number_of_sites=N,model_class=pta_models.ExpModel_Banded_Logbox_negative_pinning)
    return ((g1,"RSCP"),(g2,"RSP"),(g3,"RSC"),(g4,"RS"))
 

###################################################################
############ Plotting - based on previous get functions  ##########
###################################################################

def plot_sym_neg(ax1,ax2,g_list):

    for g, label in g_list:
        N = len(g['ev'][0])
        ax1.plot(-g['ev'][0], g['thoules_g'][0],".",label=label)
        ax2.plot(-g['ev'][0], g['PN'][0]/N, ".",label=label)
    for ax in (ax1,ax2):
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(get_LogNLocator())
        ax.legend(loc='upper left')
        ax.set_xlabel(r'$\lambda$')
    ax1.set_ylabel('$g_T$')
    ax2.set_ylabel('PN')
    


#################################################################
###################  Plotting to files (using previous plot funcs) ###
#################################################################

def plotf_sym_neg():
    fig1, ax1 = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig1.subplots_adjust(left=0.1,right=0.95)
    fig2, ax2 = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig2.subplots_adjust(left=0.1,right=0.95)
    gl = get_sym_neg(N=1000)
    plot_sym_neg(ax1,ax2,gl)
    fig1.savefig("pta_sym_neg_g.pdf")
    fig2.savefig("pta_sym_neg_PN.pdf")
    
    
    

if __name__ ==  "__main__":
    plotf_sym_neg()

