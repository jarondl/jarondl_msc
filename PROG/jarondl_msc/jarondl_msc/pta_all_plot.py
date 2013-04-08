#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for box distribution.  Anderson localization length focus
"""
# make python2 behave more like python3
from __future__ import division, print_function, absolute_import

###################
####  imports  ####
###################
# standard library
import itertools
import logging
import os

# global packages
from numpy import pi
from matplotlib.ticker import MaxNLocator
import numpy as np
import tables

# relative (intra-package) imports
from .libdl import plotdl
from .libdl import sparsedl
from .libdl.tools import h5_create_if_missing, h5_get_first_rownum_by_args
from .libdl.tools import ev_and_pn_class, ev_pn_g_class
from .libdl.plotdl import plt, cummulative_plot, get_LogNLocator
from .banded_bloch_ev import theor_banded_ev, theor_banded_dev
from .banded_bloch_ev import cached_get_sum_dos

from .models import Model_Anderson_DD_1d, Model_Anderson_ROD_1d, Model_Anderson_BD_1d
from . import models


### Warn about all float errors
np.seterr(all='warn')

#set up logging:
logger = logging.getLogger(__name__)

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

def h5_get_data(h5file, data_factory, factory_args, bandwidths,
                dis_params, tbl_grp ='/ev_and_pn', tbl_cls = None):
    """ h"""

    num = factory_args['number_of_points']
    if tbl_cls is None:
        tbl_cls = ev_and_pn_class(num)
    try:
        h5table = h5file.getNode(tbl_grp + '/' + str(num))
    except tables.exceptions.NoSuchNodeError:
        h5table = h5file.createTable(tbl_grp, str(num), tbl_cls, "Eigenvalues and PN", createparents=True)
        
    nrows = []
    for b,dis in itertools.product(bandwidths, dis_params):
        factory_args.update( bandwidth =b, dis_param =dis)
        h5_create_if_missing(h5table, data_factory, factory_args)
        nrows.append(h5_get_first_rownum_by_args(h5table, factory_args))
    return h5table[nrows]
    
    
    
def anderson_data_factory(model_name, number_of_points, bandwidth, dis_param, _dis_band=None):
    # parameters that start with _ aren't kept
    if model_name == "Banded diagonal disorder":
        m = Model_Anderson_DD_1d(conserving=False,model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_param = dis_param)

    elif model_name == "Banded tri-diagonal disorder":
        m = Model_Anderson_ROD_1d(conserving =False, semiconserving=False,  model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_param = dis_param)
    elif model_name == "Banded tri-diagonal disorder sc":
        m = Model_Anderson_ROD_1d(conserving =False, semiconserving=True,  model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_param = dis_param)
                                    
    elif model_name.startswith("Banded band-disorder"): # might have a following number.
        m = Model_Anderson_BD_1d(conserving =False, semiconserving=False,  model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_band = _dis_band, dis_param = dis_param)
    elif model_name.startswith("Banded band-disorder sc"):
        m = Model_Anderson_BD_1d(conserving =False, semiconserving=True,  model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_band = _dis_band, dis_param = dis_param)
    else: raise Exception("unknown model name?")
    return dict( eig_vals = m.eig_vals,
                     PN = m.PN)

def thouless_data_factory(model_name, number_of_points, bandwidth, dis_param):
    if model_name == "Banded band-disorder":
        m1 = Model_Anderson_BD_1d(conserving =False, semiconserving=False,  model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_param = dis_param)
        
        m2 = Model_Anderson_BD_1d(conserving =False, semiconserving=False,  model_name=model_name, number_of_points= number_of_points,
                                    bandwidth=bandwidth, dis_param = dis_param, phi=pi)
        
    else: raise Exception("unknown model name?")
    res = dict()
    res['eig_vals'] = m1.eig_vals
    res['PN']       = m1.PN
    res['g'], res['precision'] = sparsedl.thouless_g(m1.eig_vals, m2.eig_vals, 0.01)
    return res

###################################################################
#############  plot functions, recieve ax to draw on ##############
###################################################################

    
def plot_ev(ax,nums):
    for num in nums:
        cummulative_plot(ax,-num['eig_vals'], label = str(num['dis_param']))
    ax.set(ylabel=r'$\mathcal{N}(\lambda)$', xlabel=r"$\lambda$")
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(get_LogNLocator())
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc='lower right')
    
    
def plot_banded_ev_bconst_theor(ax,bandwidth,N):
    cummulative_plot(ax,sorted(theor_banded_ev(5, np.linspace(0,pi,N))),marker=None, linestyle='--', color='black')

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
            
            
            

########################################################################
##   plotf functions always create their own figure and save to file  ##
########################################################################



def plotf_ev(nums, figfilename, xlim=None):
    fig, ax = plt.subplots()
    plot_ev(ax, nums)
    plot_banded_ev_bconst_theor(ax, nums[0]['bandwidth'],nums[0]['number_of_points'])
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.savefig(figfilename + ".pdf")
    plt.close()
    
def plotf_anderson(nums_and,figfilename="pta_anderson", dont_touch_ylim=False, ylogscale=False):
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
        ax.set_ylim(1, nums_and[0]['eig_vals'].size)
    if ylogscale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(get_LogNLocator())
    ax.set_ylim(1,None)
    
    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
    fig.savefig(figfilename + ".pdf")
    
    ax.set_xlim(-(2*b+0.1),-2*b+0.5)
    fig.savefig(figfilename + "_zoom.pdf")
    plt.close()

    
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
    plt.close()

    
def all_plots_forptatex():
    """ Generate all the plots that are in pta.tex. 
        We try to read data from the hdf5 file, but create it if needed"""
    with tables.openFile("ev_and_pn.hdf5", mode = "a", title = "Eigenvalues and PN") as h5file:
        
        #runs1000_low = h5_get_anderson_by_type("DD", h5file, bandwidths=(5,), dis_params= (0.01,0.1,0.2))
        
        runs1000_low = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded diagonal disorder"),
                                            bandwidths=(5,), dis_params= (0.01,0.1,0.2))

        run2000 = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=2000, model_name="Banded diagonal disorder"),
                                             bandwidths=(5,), dis_params= (0.1,))

        run3000 = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=3000, model_name="Banded diagonal disorder"),
                                             bandwidths=(5,), dis_params= (0.1,))
        
        plotf_anderson(runs1000_low, figfilename="pta_anderson_b5_low")
        plotf_anderson_byN([runs1000_low[1], run2000[0], run3000[0]])

        
        
        runs1000_ROD = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded band-disorder"),
                                            bandwidths=(5,), dis_params= (0.1,))
        runs1000_ROD_SC = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded band-disorder sc"),
                                            bandwidths=(5,), dis_params= (0.1,))
        
        
    
        plotf_anderson([runs1000_ROD[0],runs1000_ROD_SC[0]], figfilename="pta_anderson_b5_ROD_VS_SC")
        runs_ROD_strong = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded tri-diagonal disorder"),
                                            bandwidths=(5,), dis_params= (10,))
        runs_ROD_strong_SC = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded tri-diagonal disorder sc"),
                                            bandwidths=(5,), dis_params= (10,))

        plotf_anderson([runs_ROD_strong[0],runs_ROD_strong_SC[0]], figfilename="pta_anderson_b5_s10_ROD_VS_SC")


        run_strong= h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=2000, model_name="Banded diagonal disorder"),
                                            bandwidths=(5,), dis_params= (1,2,3))
        plotf_ev(run_strong, "pta_ev_strong")
        
        plotf_anderson(run_strong, figfilename="pta_anderson_strong")
        run_very_strong= h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=2000, model_name="Banded band-disorder"),
                                             bandwidths=(5,), dis_params= (20,30,40))
        plotf_anderson(run_very_strong, figfilename="pta_anderson_very_strong", dont_touch_ylim=True)
        plotf_anderson(run_very_strong, figfilename="pta_anderson_very_strong_log", dont_touch_ylim=True,ylogscale=True)
        
        run_very_strong_sc= h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=2000, model_name="Banded band-disorder sc"),
                                             bandwidths=(5,), dis_params= (20,30,40))
        plotf_anderson(run_very_strong_sc, figfilename="pta_anderson_very_strong_sc", dont_touch_ylim=True)
        plotf_anderson(run_very_strong_sc, figfilename="pta_anderson_very_strong_sc_log", dont_touch_ylim=True,ylogscale=True)
        
        runs_BD_SC = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded band-disorder sc"),
                                            bandwidths=(5,), dis_params= (0.1,1,2,3))
        plotf_ev(runs_BD_SC, figfilename="pta_B_BC_ev", xlim=(-20,30))
        plotf_anderson(runs_BD_SC, figfilename="pta_B_BC_pn")
    
        runs_BD = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded band-disorder"),
                                            bandwidths=(5,), dis_params= (0.1,1,2,3))
        plotf_ev(runs_BD, figfilename="pta_B_B_ev", xlim=(-20,30))
        plotf_anderson(runs_BD, figfilename="pta_B_B_pn")
        
                
        run = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded tri-diagonal disorder sc"),
                                            bandwidths=(5,), dis_params= (0.1,1,2,3))
        plotf_ev(run, figfilename="pta_B_TC_ev", xlim=(-15,10))
        plotf_anderson(run, figfilename="pta_B_TC_pn")
    
        run = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded tri-diagonal disorder"),
                                            bandwidths=(5,), dis_params= (0.1,1,2,3))
        plotf_ev(run, figfilename="pta_B_T_ev", xlim=(-15,10))
        plotf_anderson(run, figfilename="pta_B_T_pn")
        
        # different b's
        run1 = h5_get_data(h5file, anderson_data_factory, 
                            factory_args = dict(number_of_points=1000, _dis_band = 1,model_name="Banded band-disorder 1"),
                                            bandwidths=(5,), dis_params= (0.5,))
                                            
        run2 = h5_get_data(h5file, anderson_data_factory, 
                            factory_args = dict(number_of_points=1000, _dis_band = 2,model_name="Banded band-disorder 2"),
                                            bandwidths=(5,), dis_params= (0.5,))
                                            
        run3 = h5_get_data(h5file, anderson_data_factory, 
                            factory_args = dict(number_of_points=1000, _dis_band = 10,model_name="Banded band-disorder 10"),
                                            bandwidths=(5,), dis_params= (0.5,))
        runs_dd = h5_get_data(h5file, anderson_data_factory, factory_args = dict(number_of_points=1000, model_name="Banded diagonal disorder"),
                                            bandwidths=(5,), dis_params= (0.5,))
        plotf_ev((runs_dd[0],run1[0],run2[0],run3[0]), figfilename="pta_B_BB_ev", xlim=(-15,10))
        plotf_anderson((runs_dd[0],run1[0],run2[0],run3[0]), figfilename="pta_B_BB_pn")    
        
        plotf_theor_banded_ev()
        plotf_theor_banded_dos()
        
    
def plotf_theor_banded_ev(bs=6,N=2000):
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch+0.2])
    fig.subplots_adjust(left=0.1,right=0.95)
    pi_labels = ["0",r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$",]
    pi_locs = [0,pi/2,pi,3*pi/2,2*pi]
    
    xs = np.linspace(0,2*pi, N)
    for b in range(1,bs+1):
        y = theor_banded_ev(b,xs)
        ax1.plot(xs,y,label=str(b))
        ax2.plot(xs,y+2*b, label=str(b))
    for ax in (ax1,ax2):
        ax.legend(loc='upper right')
        ax.set_xlabel('k')
        ax.axvline(pi, color='black')
        ax.set_xlim(0,2*pi)
        ax.set_xticks(pi_locs)
        ax.set_xticklabels(pi_labels)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    ax1.set_ylabel(r'$\lambda$')
    ax2.set_ylabel(r'$\lambda + 2b$')
    
    fig.savefig("pta_theor_banded_ev.pdf")
    plt.close()
    
    

def plotf_theor_banded_dos(b=5,N=2000):
    
    fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    fig.subplots_adjust(left=0.1,right=0.95)
    
    lam,idos = cached_get_sum_dos(b)
    
    ks = np.linspace(0,pi,N)
    lam2 = theor_banded_ev(b,ks)
    dev = -theor_banded_dev(b,ks)


    ax.plot(lam,idos, label=r"$DOS^{-1}$")
    ax.plot(lam2,dev, label=r"$v$")

    ax.set_xlim(-(2*b+0.5),(2*b+0.5))

    ax.legend(loc='upper right')

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel(r'$\frac{d\lambda}{dk}$')
    ax.yaxis.set_major_locator(MaxNLocator(5))

    
    
    fig.savefig("pta_theor_banded_dos.pdf")
    plt.close()
    
