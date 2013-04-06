#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for box distribution.  Anderson localization length focus
"""
# make python2 behave more like python3
# still need to worry about absolute import..
from __future__ import division, print_function

###################
####  imports  ####
###################
# standard library
import itertools
import logging
import os

# global packages
from numpy import random, pi, log10, sqrt,  exp, expm1, sort, eye, nanmin, nanmax, log, cos, sinc, ma
from scipy.special import gamma
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator
import numpy as np
import scipy as sp
import matplotlib as mpl
import h5py
import tables

# relative imports
from .libdl import plotdl
from .libdl import sparsedl
from .libdl.tools import cached_get_key, h5_create_if_missing, h5_get_first_rownum_by_args
from .libdl.tools import Ev_and_PN_1000, Ev_and_PN_2000, Ev_and_PN_3000
from .libdl.plotdl import plt, tight_layout, cummulative_plot
from ptaplot import theor_banded_ev, theor_banded_dev, theor_banded_ev_k, theor_banded_dossum_k, find_ks
import pta_models
import pta_all_models
from models import Model_Anderson_DD_1d, Model_Anderson_ROD_1d


### Raise all float errors
np.seterr(all='warn')

#set up logging:
logging.basicConfig(format='%(asctime)s - %(name) - %(message)s')
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

########################################################################
########### get functions create data ##################################
########################################################################

def get_ev_PN_for_models(models,count,number_of_sites):
    res_type =  np.dtype([("eig_vals",(np.float64,number_of_sites)),
                          ("PN",(np.float64,number_of_sites)),
                          ("dis_param",np.float64),
                          ("bandwidth",np.float64)
                          ])
    res = np.zeros(count, dtype = res_type) # preallocation is faster..        
    for n,mod in enumerate(models):
        debug('eigenvals ,PN number {0}'.format(n))
        res[n] = (  mod.eig_vals,
                    mod.PN,
                    mod.dis_param,
                    mod.bandwidth)
    return res
    
    
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


def h5_get_anderson(h5file, model_factory, num = 1000,model_args = dict(model_name = "anderson dd")
                                        , bandwidths=(5,), dis_params=(1,)):
    """ very dirty at the moment, including this num buisness """
    try:
        if num == 1000:
            cls = Ev_and_PN_1000
            h5table = h5file.root.ev_and_pn1000
        elif num == 2000:
            cls = Ev_and_PN_2000
            h5table = h5file.root.ev_and_pn2000
        elif num == 3000:
            cls = Ev_and_PN_3000
            h5table = h5file.root.ev_and_pn3000
    except tables.exceptions.NoSuchNodeError:
        h5table = h5file.createTable('/', 'ev_and_pn{}'.format(num), cls, "Eigenvalues and PN")
        
    nrows = []
    for b,dis in itertools.product(bandwidths, dis_params):
        model_args.update( number_of_points = num, bandwidth =b, dis_param =dis)
        h5_create_if_missing(h5table, model_factory, model_args)
        nrows.append(h5_get_first_rownum_by_args(h5table, model_args))
    return h5table[nrows]
    
def h5_get_anderson_by_type(model_type, h5file,  **kwargs):
    if model_type == "DD":
        # Diagonal disorder
        anderson_dd_factory = lambda args : Model_Anderson_DD_1d(conserving=False, **args) 
        return h5_get_anderson(h5file, anderson_dd_factory, model_args = dict(model_name = "anderson dd"), **kwargs)
    elif model_type == "ROD":
        anderson_rod_factory = lambda args: Model_Anderson_ROD_1d(conserving =False, semiconserving=False, **args)
        return h5_get_anderson(h5file, anderson_rod_factory, model_args = dict(model_name = "anderson rod"), **kwargs)
    elif model_type == "ROD SC":
        anderson_rodsc_factory = lambda args: Model_Anderson_ROD_1d(conserving =False, semiconserving=True, **args)
        return h5_get_anderson(h5file, anderson_rodsc_factory, model_args = dict(model_name = "anderson rod sc"), **kwargs)


def get_sum_dos(b):
    k = np.linspace(0,pi,2000)
    evs = theor_banded_ev_k(k,b,0)
    lams = np.linspace(evs.min(), evs.max(),2000)
    k_of_ev = [find_ks(b,ev) for ev in lams]
    #debug("k_of_ev.shape = {} ".format(k_of_ev.shape))
    doses = np.array([theor_banded_dossum_k(k_of,b) for k_of in k_of_ev])
    return (lams, doses )
    
def cached_get_sum_dos(b):
    with h5py.File('banded_dos.hdf5') as f:
        if str(b) in f:
            lams = np.array(f[str(b)]['eig_vals'])
            dos = np.array(f[str(b)]['dos'])
        if str(b) not in f:
            fb = f.create_group(str(b))
            lams, dos= get_sum_dos(b)
            f[str(b)].create_dataset('eig_vals', data=lams)
            f[str(b)].create_dataset('dos', data=dos)
    return lams, dos

def plot_anderson(ax, ev_pn,color_seq):
    
    for (mod,color) in zip(ev_pn,color_seq):
        ax.plot(-mod['eig_vals'],mod['PN'], '.', color=color, label=r"{0}".format(mod['dis_param']))
    ax.legend(loc='upper right')
    
def plot_anderson1d_theory(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['bandwidth']
        xs = np.linspace(-2*b,2*b)
        ys = and_theory(xs, mod['dis_param'],b)
        ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
        ax.plot(xs,ys,color=color,linewidth=0.8)

def plot_anderson1d_theory_vv(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['bandwidth']
        N = mod['eig_vals'].size
        #xs = np.linspace(-2*b,2*b,N//2)
        #lam = theor_banded_ev(b,N)[:N//2] - 2*b ## There it is conserving and (0,2pi)
        #dev = -theor_banded_dev(b,N)[:N//2]
        lam,dev = cached_get_sum_dos(b)
        ### the six only works for b=1 !!
        ys = 6 * dev**2 / (mod['dis_param'])**2
        #ys = and_theory(xs, mod['dis_param'],b)
        ax.plot(lam,ys,color="w",linewidth=1.4)# gives a white "border"
        ax.plot(lam,ys,color=color,linewidth=0.8)
        


        
def plot_anderson1d_theory_conserv(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['bandwidth']
        if b==1:
            xs = np.linspace(-2*b,2*b)
            ys = and_theory_cons(xs, mod['dis_param'],b)
            ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
            ax.plot(xs,ys,color=color,linewidth=0.8)


    
def plotf_anderson(nums_and,figfilename="pta_anderson", dont_touch_ylim=False):
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    if not dont_touch_ylim:
        ax.axhline((nums_and[0]['eig_vals'].size)*2/3, ls='--', color='black')
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson(ax, nums_and,color_seq)
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_anderson1d_theory_vv(ax, nums_and,color_seq)
    
    b= nums_and[0]['bandwidth']
    
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel("PN")
    
    if not dont_touch_ylim:
        ax.set_ylim(0, nums_and[0]['eig_vals'].size)
    
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
    fig.savefig(figfilename + ".pdf")
    
    ax.set_xlim(-(2*b+0.1),-2*b+1)
    fig.savefig(figfilename + "_zoom.pdf")
    
def plotf_anderson_byN(nums_and,figfilename="pta_anderson_byN"):
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for (mod,color) in zip(nums_and,color_seq):
        N = (mod['eig_vals'].size)
        ax.plot(-mod['eig_vals'],mod['PN'], '.', color=color, label=r"{0}".format(N))
        ax.axhline(N*2/3, ls='--', color='black')
    ax.legend(loc='upper right')
  

    b= nums_and[0]['bandwidth']
    maxN = max(mod['eig_vals'].size for mod in nums_and)
    
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel("PN")
    
    ax.set_ylim(0, maxN)
    
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
    fig.savefig(figfilename + ".pdf")

    
def plotf_anderson_forptatex():
    with tables.openFile("ev_and_pn.hdf5", mode = "a", title = "Eigenvalues and PN") as h5file:
        
        runs1000_low = h5_get_anderson_by_type("DD", h5file, bandwidths=(5,), dis_params= (0.01,0.1,0.2))
        
        run2000 = h5_get_anderson_by_type("DD", h5file, num=2000, bandwidths=(5,), dis_params= (0.1,))
        
        run3000 = h5_get_anderson_by_type("DD", h5file, num=3000, bandwidths=(5,), dis_params= (0.1,))      
        
        plotf_anderson(runs1000_low, figfilename="pta_anderson_b5_low")
        plotf_anderson_byN([runs1000_low[1], run2000[0], run3000[0]])
        
        
        runs1000_ROD = h5_get_anderson_by_type("ROD", h5file,bandwidths=(5,), num=1000,dis_params=(0.1,))
        runs1000_ROD_SC = h5_get_anderson_by_type("ROD SC", h5file,bandwidths=(5,), num=1000,dis_params=(0.1,))
        plotf_anderson_byN([runs1000_ROD[0],runs1000_ROD_SC[0]], figfilename="pta_anderson_b5_ROD_VS_SC")
        
        runs_ROD_strong = h5_get_anderson_by_type("ROD", h5file,bandwidths=(5,), num=1000,dis_params=(10,))
        runs_ROD_strong_SC = h5_get_anderson_by_type("ROD SC", h5file,bandwidths=(5,), num=1000,dis_params=(10,))
        plotf_anderson_byN([runs_ROD_strong[0],runs_ROD_strong_SC[0]], figfilename="pta_anderson_b5_s10_ROD_VS_SC")


        run_strong= h5_get_anderson_by_type("DD", h5file, bandwidths=(5,), dis_params= (1,2,3))
        plotf_anderson(run_strong, figfilename="pta_anderson_strong")

        run_very_strong= h5_get_anderson_by_type("DD", h5file, bandwidths=(5,), dis_params= (20,30,40))
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
    
    lam,idos = cached_get_sum_dos(b)
    
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
    

