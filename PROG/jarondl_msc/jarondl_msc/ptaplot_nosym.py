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
from scipy import linalg
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator

import plotdl
import sparsedl
from plotdl import plt, tight_layout, cummulative_plot,draw_if
from sparsedl import cached_get


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
                        models_and_names=((pta_models.ExpModel_Banded_Logbox_negative,"RSC"),)):
    """ This time, the idea is to use versatile models.
    """
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    win_avg_mtrx = sparsedl.window_avg_mtrx(number_of_sites - 1,win_size=20)
    res_type =  np.dtype([("ev",(np.float64,number_of_sites)),
                          ("PN",(np.float64,number_of_sites)),
                          ("thoules_g",(np.float64,number_of_sites)),
                          ("s",np.float64),
                          ("b",np.float64),
                          ("name", (np.unicode_, 16))
                          ])
    
    
    #res = np.zeros(s_grid.size, dtype = res_type) # preallocation is faster..
    res = np.zeros(len(models_and_names),dtype=res_type)
    n=0
    for n,(model_class, model_name) in enumerate( models_and_names):
        debug(' eigenvals ,PN and thouless for s = {0}, b= {1}, n={2}, name={3}'.format(s,b,n,model_name))
        model = model_class(sample, epsilon=s, bandwidth1d=b, rseed=n,phi=0)
        model_phi = model_class(sample, epsilon=s, bandwidth1d=b,rseed=n,phi=pi)
        g = abs(model.eigvals - model_phi.eigvals) / (pi**2)
        # Approximation of  the minimal precision:
        prec = FLOAT_EPS * max(abs(model.eigvals))* number_of_sites  
        #debug("precision = {0}, minimal g  = {1}".format(prec, min(g)))
        #g = ma.masked_less(g,prec)
        avg_spacing = win_avg_mtrx.dot(-model.eigvals[1:]+model.eigvals[:-1])
        # avg_spacing is now smaller than eigvals. We duplicate the last value to accomodate (quite  hackish)
        avg_spacing = np.append(avg_spacing,avg_spacing[-1])
        #ga = ma.masked_less(g/avg_spacing,prec)
        ga = g/avg_spacing
        ga[g<prec] = prec/avg_spacing
        res[n] = ( model.eigvals, model.PN_N, ga , s, b, model_name)
    return res
    

###################################################################
############ Plotting - based on previous get functions  ##########
###################################################################

def plot_sym_neg(ax1,ax2,g_models):

    for g in g_models:
        N = len(g['ev'])
        ax1.plot(-g['ev'], g['thoules_g'],".",label=g['name'])
        ax2.plot(-g['ev'], g['PN']/N, ".",label=g['name'])
        ax2.axhline(2.0/3, ls='--', color='black')
    for ax in (ax1,ax2):
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(get_LogNLocator())
        ax.legend(loc='upper left')
        ax.set_xlabel(r'$\lambda$')
    ax1.set_ylabel('$g_T$')
    ax2.set_ylabel('PN')
    
def plot_sym_neg_around_zero(ax,g_models):

    for g in g_models:
        N = len(g['ev'])
        evmin = g['ev'].max()
        ax.plot(evmin-g['ev'], g['PN']/N, ".",label=g['name'])
        ax.axhline(2.0/3, ls='--', color='black')

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(get_LogNLocator())
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(get_LogNLocator())
    ax.legend(loc='lower left')
    ax.set_xlabel(r'$\lambda-\lambda_{min}$')
    ax.set_ylabel('PN')
    
@draw_if
def plot_sym_neg_around_center(ax,g_models):

    for g in g_models:
        N = len(g['ev'])
        center = g['PN'].argmax()
        #center = N//2
        ax.plot(g['ev'][center]-g['ev'][center:], g['PN'][center:]/N, ".",label=g['name'])
        ax.axhline(2.0/3, ls='--', color='black')

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(get_LogNLocator())
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(get_LogNLocator())
    ax.legend(loc='lower left')
    ax.set_xlabel(r'$\lambda-\lambda_0$')
    ax.set_ylabel('PN')
#################################################################
###################  Plotting to files (using previous plot funcs) ###
#################################################################

def plotf_sym_neg(force_new=False):
    fig1, ax1 = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig1.subplots_adjust(left=0.1,right=0.95)
    fig2, ax2 = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig2.subplots_adjust(left=0.1,right=0.95)
    if force_new:
        try:
            os.remove("g_several_models.npz")
        except OSError:
            warning(" File does not exist, so the run is new anyway. ")
        
    gl = cached_get(get_ev_thoules_g_1d, "g_several_models.npz", number_of_sites=1000, models_and_names = [
                                                        (pta_models.ExpModel_Banded_Logbox_phase, "RSPC"),
                                                        (pta_models.ExpModel_Banded_Logbox_dd, "RSPCD"),
                                                        (pta_models.ExpModel_Banded_Logbox_rd, "RSP"),
                                                        (pta_models.ExpModel_Banded_Logbox_negative, "RSC"),
                                                        (pta_models.ExpModel_Banded_Logbox_negative_dd, "RSCD"),
                                                        (pta_models.ExpModel_Banded_Logbox_negative_rd, "RS")])#,
                                                         #(pta_models.ExpModel_Banded_Logbox_nosym, "RCP")])
    plot_sym_neg(ax1,ax2,gl[:3])
    fig1.savefig("pta_sym_neg_g.pdf")
    fig2.savefig("pta_sym_neg_PN.pdf")
    ax1.cla()
    ax2.cla()
    plot_sym_neg(ax1,ax2,gl[3:])
    fig1.savefig("pta_sym_neg_g2.pdf")
    fig2.savefig("pta_sym_neg_PN2.pdf")
    ax1.cla()
    plot_sym_neg_around_zero(ax1,gl[:3])
    fig1.savefig("pta_sym_neg_PN_zero.pdf")
    ax1.cla()
    plot_sym_neg_around_center(ax1,gl[3:5])
    fig1.savefig("pta_sym_neg_PN_center.pdf")
if __name__ ==  "__main__":
    plotf_sym_neg()

