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
from sparsedl import cached_get_key
from plotdl import plt, tight_layout, cummulative_plot
from ptaplot import theor_banded_ev, theor_banded_dev, theor_banded_ev_k, theor_banded_dossum_k, find_ks


import numpy as np
import scipy as sp
import matplotlib as mpl

import ptsplot

import pta_models
import pta_all_models
from models import Model_Anderson_DD_1d, Model_Anderson_ROD_1d



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
        res[n] = (  mod.eig_vals,
                    mod.PN,
                    mod.dis_param,
                    mod.bandwidth)
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
        N = mod['ev'].size
        #xs = np.linspace(-2*b,2*b,N//2)
        #lam = theor_banded_ev(b,N)[:N//2] - 2*b ## There it is conserving and (0,2pi)
        #dev = -theor_banded_dev(b,N)[:N//2]
        lam,dev = sum_dos(b)
        ### the six only works for b=1 !!
        ys = 6 * dev**2 / (mod['sigma'])**2
        #ys = and_theory(xs, mod['sigma'],b)
        ax.plot(lam,ys,color="w",linewidth=1.4)# gives a white "border"
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

def getf_anderson(b=1, N=2000,sigmas=(0.01,0.1,0.2), ROD=False, semiconserving=False):
    fname = 'pta_anderson_b{}{}{}.npz'.format(b, 
                                            ("_ROD" if ROD else ""), 
                                            ("_SC" if semiconserving else ""))
    if ROD:
        models = (Model_Anderson_ROD_1d(N, sigma, bandwidth=b, conserving=False, semiconserving=semiconserving) for sigma in sigmas)
    else:
        models = (Model_Anderson_DD_1d(N, sigma, bandwidth=b, conserving=False) for sigma in sigmas)
    key = str((N, frozenset(sigmas)))
    return cached_get_key(get_ev_PN_for_models, fname, key, models, len(sigmas), N)
    
def plotf_anderson(nums_and,figfilename="pta_anderson", dont_touch_ylim=False):
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    if not dont_touch_ylim:
        ax.axhline((nums_and[0]['ev'].size)*2/3, ls='--', color='black')
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson(ax, nums_and,color_seq)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson1d_theory_vv(ax, nums_and,color_seq)
    
    b= nums_and[0]['b']
    
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel("PN")
    
    if not dont_touch_ylim:
        ax.set_ylim(0, nums_and[0]['ev'].size)
    
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
    fig.savefig(figfilename + ".pdf")
    
    ax.set_xlim(-(2*b+0.1),-2*b+1)
    fig.savefig(figfilename + "_zoom.pdf")
    
def plotf_anderson_byN(nums_and,figfilename="pta_anderson_byN"):
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for (mod,color) in zip(nums_and,color_seq):
        N = (mod['ev'].size)
        debug("N = {} ".format(N))
        debug("mod['ev'].shape = {}".format(mod['ev'].shape))
        ax.plot(-mod['ev'],mod['PN'], '.', color=color, label=r"{0}".format(N))
        ax.axhline(N*2/3, ls='--', color='black')
        debug("plot")
    ax.legend(loc='upper right')
  

    b= nums_and[0]['b']
    maxN = max(mod['ev'].size for mod in nums_and)
    
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel("PN")
    
    ax.set_ylim(0, maxN)
    
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
    fig.savefig(figfilename + ".pdf")

    
def plotf_anderson_forptatex():
    runs1000_low = getf_anderson(b=5, N=1000,sigmas=(0.01,0.1,0.2))
    run2000 = getf_anderson(b=5, N=2000, sigmas=(0.1,))
    run3000 = getf_anderson(b=5, N=3000, sigmas=(0.1,))
    
    plotf_anderson(runs1000_low, figfilename="pta_anderson_b5_low")
    plotf_anderson_byN([runs1000_low[1], run2000[0], run3000[0]])
    
    runs1000_ROD = getf_anderson(b=5, N=1000,sigmas=(0.1,), ROD=True)
    runs1000_ROD_SC = getf_anderson(b=5, N=1000,sigmas=(0.1,), ROD=True, semiconserving=True)
    plotf_anderson_byN([runs1000_ROD[0],runs1000_ROD_SC[0]], figfilename="pta_anderson_b5_ROD_VS_SC")
    
    #stronger
    runs_ROD_strong = getf_anderson(b=5, N=1000,sigmas=(10,), ROD=True)
    runs_ROD_strong_SC = getf_anderson(b=5, N=1000,sigmas=(10,), ROD=True, semiconserving=True)
    plotf_anderson_byN([runs_ROD_strong[0],runs_ROD_strong_SC[0]], figfilename="pta_anderson_b5_s10_ROD_VS_SC")


    run_strong = getf_anderson(b=5, N=1000, sigmas=(1,2,3))
    plotf_anderson(run_strong, figfilename="pta_anderson_strong")
    
    run_strong2 = getf_anderson(b=5, N=2000, sigmas=(1,2,3))
    plotf_anderson(run_strong2, figfilename="pta_anderson_strong2")
    
    run_very_strong = getf_anderson(b=5, N=1000, sigmas=(10,20,30))
    plotf_anderson(run_very_strong, figfilename="pta_anderson_very_strong", dont_touch_ylim=True)

    

    
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
    
    lam,idos = sum_dos(b)
    
    lam2 = theor_banded_ev(b,N)[:N//2] - 2*b ## There it is conserving and (0,2pi)
    dev = -theor_banded_dev(b,N)[:N//2]


    ax.plot(lam,idos, label=r"$DOS^{-1}$")
    ax.plot(lam2,dev, label=r"$v$")

    ax.set_xlim(-(2*b+0.5),(2*b+0.5))

    ax.legend(loc='upper right')

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel(r'$\frac{d\lambda}{dk}$')
    ax.yaxis.set_major_locator(MaxNLocator(5))

    
    
    fig.savefig("pta_theor_banded_dos.pdf")
    
def all_plots():
    plotf_theor_banded_ev(bs=6,N=2000)
    plotf_theor_banded_dos(b=5,N=2000)
    plotf_anderson(rates=True,ddonly=True,b=5)
    plotf_anderson_rates_conserved(b=5)
