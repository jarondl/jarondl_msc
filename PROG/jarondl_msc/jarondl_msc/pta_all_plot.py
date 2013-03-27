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
from ptaplot import theor_banded_ev, theor_banded_dev, theor_banded_ev_k, theor_banded_dossum_k, find_ks


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

and_theory = (lambda x,sigma,b : 6*(4*(b**2)-x**2)/(sigma**2))
and_theory_cons = (lambda x,sigma,b : 6*(4-(x/2)**2)/(sigma**2))

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
        ax.plot(-mod['ev'],mod['PN'], '.', color=color, label=r"{0}".format(mod['sigma']))
    ax.legend(loc='upper right')
    
def plot_anderson1d_theory(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['b']
        xs = np.linspace(-2*b,2*b)
        ys = and_theory(xs, mod['sigma'],b)
        ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
        ax.plot(xs,ys,color=color,linewidth=0.8)

def plot_anderson1d_theory_vv(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['b']
        N = 2000
        #xs = np.linspace(-2*b,2*b,N//2)
        #lam = theor_banded_ev(b,N)[:N//2] - 2*b ## There it is conserving and (0,2pi)
        #dev = -theor_banded_dev(b,N)[:N//2]
        lam,dev = sum_dos(b)
        ### the six only works for b=1 !!
        ys = 6 * dev**2 / (mod['sigma'])**2
        #ys = and_theory(xs, mod['sigma'],b)
        ax.plot(lam,ys,color="b",linewidth=1)# gives a white "border"
        ax.plot(lam,ys,color=color,linewidth=0.8)
        
        
def sum_dos(b):
    k = np.linspace(0,pi,2000)
    evs = theor_banded_ev_k(k,b,0)
    lams = np.linspace(evs.min(), evs.max(),2000)
    k_of_ev = [find_ks(b,ev) for ev in lams]
    #debug("k_of_ev.shape = {} ".format(k_of_ev.shape))
    doses = np.array([theor_banded_dossum_k(k_of,b) for k_of in k_of_ev])
    return (lams, doses )

        
def plot_anderson1d_theory_conserv(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['b']
        if b==1:
            xs = np.linspace(-2*b,2*b)
            ys = and_theory_cons(xs, mod['sigma'],b)
            ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
            ax.plot(xs,ys,color=color,linewidth=0.8)

def plotf_anderson(rates=False,ddonly=False,b=1):
    fname_base = 'pta_anderson{0}{1}_b{2}'.format(("_rates" if rates else ""),("_ddonly" if ddonly else "") ,b)
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
        elif ddonly:
            models = (pta_all_models.Model_Anderson_rates_diagonal_disorder_only(sam, sigma, bandwidth1d=b) for sigma in sigmas)
        else:
            models = (pta_all_models.Model_Anderson_banded(sam, sigma, bandwidth1d=b) for sigma in sigmas)
        nums_and = get_ev_PN_for_models(models, len(sigmas), 2000)
        np.savez(fname_base + ".npz", nums = nums_and)
        
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    ax.axhline((nums_and[0]['ev'].size)*2/3, ls='--', color='black')
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson(ax, nums_and,color_seq)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson1d_theory_vv(ax, nums_and,color_seq)
    
    b= nums_and[0]['b']
    
    ax.set_ylim(0, nums_and[0]['ev'].size)
    
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
    fig.savefig(fname_base + ".pdf")
    
    ax.set_xlim(-(2*b+0.1),-2*b+1)
    fig.savefig(fname_base + "_zoom.pdf")
    
    
def plotf_anderson_rates_conserved(b=1):
    fname_base = 'pta_anderson_rates_conserv_b{0}'.format( b)
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
    
    ax.axhline((nums_and[0]['ev'].size)*2/3, ls='--', color='black')
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson(ax, nums_and, color_seq)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson1d_theory_conserv(ax, nums_and,color_seq)

    
    ax.set_ylim(0, nums_and[0]['ev'].size)
    
    

    
    ax.set_xlim(-0.5,3*b+0.5)
    fig.savefig(fname_base + ".pdf")
    
    ax.set_xlim(2*b-0.5,2*b+0.1)
    fig.savefig(fname_base + "_zoom.pdf")
    
    
def plotf_theor_banded_ev(bs=6,N=2000):
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch+0.2])
    fig.subplots_adjust(left=0.1,right=0.95)
    pi_labels = ["0",r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$",]
    pi_locs = [0,pi/2,pi,3*pi/2,2*pi]
    
    xs = np.linspace(0,2*pi, N)
    for b in range(1,bs+1):
        y = theor_banded_ev(b,N)
        ax1.plot(xs,y,label=str(b))
        ax2.plot(xs,y-2*b, label=str(b))
    for ax in (ax1,ax2):
        ax.legend(loc='upper right')
        ax.set_xlabel('k')
        ax.axvline(pi, color='black')
        ax.set_xlim(0,2*pi)
        ax.set_xticks(pi_locs)
        ax.set_xticklabels(pi_labels)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    ax1.set_ylabel(r'$\lambda - 2b$')
    ax2.set_ylabel(r'$\lambda$')
    
    fig.savefig("pta_theor_banded_ev.pdf")
    
    

def plotf_theor_banded_dos(b=5,N=2000):
    
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    
    #for b in range(1,bs+1):
    #    y = theor_banded_dev(b,N)
    #    ax.plot(xs,y**(-1),label=str(b))
    lam = theor_banded_ev(b,N)[:N//2] - 2*b ## There it is conserving and (0,2pi)
    dev = -theor_banded_dev(b,N)[:N//2]
    #ax.plot(lam,abs(DEV)**(-1))
    ax.plot(lam,dev**2)
    #ax.legend(loc='upper right')
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel(r'$\frac{d\lambda}{dN}$')
    #ax.axvline(pi, color='black')
    #ax.set_xlim(0,2*pi)
    #ax.set_ylim(-1,1)
    
    
    
    fig.savefig("pta_theor_banded_dos.pdf")
    
