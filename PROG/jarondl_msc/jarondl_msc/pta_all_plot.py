#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for box distribution.  Anderson localization length focus
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
import pta_all_models


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
########################################################################
###########   code   ###################################################
########################################################################

and_theory = (lambda x,sigma : 6*(4-x**2)/(sigma**2))
and_theory_cons = (lambda x,sigma : 6*(4-(x/2)**2)/(sigma**2))

def get_ev_PN_for_models(models,count,number_of_sites):
    res_type =  np.dtype([("ev",(np.float64,number_of_sites)),
                          ("PN",(np.float64,number_of_sites)),
                          ("sigma",np.float64),
                          ("b",np.float64)
                          ])
    res = np.zeros(count, dtype = res_type) # preallocation is faster..        
    for n,mod in enumerate(models):
        debug('eigenvals ,PN number {0}'.format(n))
        res[n] = (  mod.eigvals,
                    mod.PN_N,
                    mod.epsilon,
                    mod.bandwidth1d)
    return res
    

def plot_anderson(ax, ev_pn,color_seq):
    
    for (mod,color) in zip(ev_pn,color_seq):
        ax.plot(mod['ev'],mod['PN'], '.', color=color, label=r"{0}".format(mod['sigma']))
    ax.legend(loc='upper right')
    
def plot_anderson1d_theory(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        xs = np.linspace(-2,-1)
        ys = and_theory(xs, mod['sigma'])
        ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
        ax.plot(xs,ys,color=color,linewidth=0.8)
        
def plot_anderson1d_theory_conserv(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        xs = np.linspace(-4,-3)
        ys = and_theory_cons(xs, mod['sigma'])
        ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
        ax.plot(xs,ys,color=color,linewidth=0.8)

def plotf_anderson(rates=False,b=1):
    fname_base = 'pta_anderson{0}_b{1}_'.format(("_rates" if rates else ""), b)
    # See if we have numerics:
    try:
        f = np.load(fname_base + '.npz')
        nums_and = f['nums']
    except (OSError, IOError):
        # otherwise, get the data:
        sigmas=(0.01,0.1,0.2)
        sam = ptsplot.create_bloch_sample_1d(2000)
        if rates:
            models = (pta_all_models.Model_Anderson_rates_banded(sam, sigma, bandwidth1d=b) for sigma in sigmas)
        else:
            models = (pta_all_models.Model_Anderson_banded(sam, sigma, bandwidth1d=b) for sigma in sigmas)
        nums_and = get_ev_PN_for_models(models, len(sigmas), 2000)
        np.savez(fname_base + ".npz", nums = nums_and)
        
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson(ax, nums_and,color_seq)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson1d_theory(ax, nums_and,color_seq)
    
    ax.set_ylim(0, nums_and[0]['ev'].size)
    
    ax.set_xlim(-2.5,2.5)
    fig.savefig(fname_base + "1.pdf")
    
    ax.set_xlim(-2.1,-1.5)
    fig.savefig(fname_base + "2.pdf")
    
    
def plotf_anderson_rates_conserved(b=1):
    fname_base = 'pta_anderson_rates_conserv_b{0}_'.format( b)
    # See if we have numerics:
    try:
        f = np.load(fname_base + '.npz')
        nums_and = f['nums']
    except (OSError, IOError):
        # otherwise, get the data:
        sigmas=(0.01,0.1,0.2)
        sam = ptsplot.create_bloch_sample_1d(2000)

        models = (pta_all_models.Model_Anderson_rates_conserv_banded(sam, sigma, bandwidth1d=b) for sigma in sigmas)

        nums_and = get_ev_PN_for_models(models, len(sigmas), 2000)
        np.savez(fname_base + ".npz", nums = nums_and)
        
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson(ax, nums_and, color_seq)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson1d_theory_conserv(ax, nums_and,color_seq)

    
    ax.set_ylim(0, nums_and[0]['ev'].size)
    
    ax.set_xlim(-4.5,0.5)
    fig.savefig(fname_base + "1.pdf")
    
    ax.set_xlim(-4.2,-3.5)
    fig.savefig(fname_base + "2.pdf")
