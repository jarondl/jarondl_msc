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

    
def anderson_data_factory(Model):
    def f(*args, **kwargs):
        m = Model(*args, **kwargs)
        return dict( eig_vals = m.eig_vals, PN = m.PN, model_name=m.model_name)
    return f

def h5_get_data(h5file, model, factory_args, bandwidths,
                dis_params, tbl_grp ='/ev_and_pn', tbl_cls = None, data_factory=anderson_data_factory):
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
        factory_args.setdefault('model_name', model.__name__) # set model name if not specifically defined
        h5_create_if_missing(h5table, data_factory(model), factory_args)
        nrows.append(h5_get_first_rownum_by_args(h5table, factory_args))
    return h5table[nrows]
    
    

    

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

def decide_labels(ev_pn):
    """ auto decide on labels. If one field changes use it. """
    if any(ev_pn['dis_param']!=ev_pn['dis_param'][0]): # if dis_param are not all equal
        return ev_pn['dis_param']
    elif any(ev_pn['bandwidth']!=ev_pn['bandwidth'][0]):
        return ev_pn['bandwidth']
    elif any(ev_pn['number_of_points']!=ev_pn['number_of_points'][0]):
        return ev_pn['number_of_points']
    else:
        info("Could not decide on labels, please provide your own")
        return itertools.repeat('')
    
def plot_ev(ax,nums):
    for num in nums:
        cummulative_plot(ax,-num['eig_vals'], label = str(num['dis_param']), markersize=2)
    ax.set(ylabel=r'$\mathcal{N}(\lambda)$', xlabel=r"$\lambda$")
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(get_LogNLocator())
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc='lower right')
    
    
def plot_banded_ev_bconst_theor(ax,bandwidth,N):
    cummulative_plot(ax,sorted(theor_banded_ev(5, np.linspace(0,pi,N))),marker=None, linestyle='--', color='black')

def plot_ev_pn(ax, ev_pn,color_seq,labels=None, shift_ev=False):
    if labels is None:
        labels = decide_labels(ev_pn)
    for (mod,color,label) in zip(ev_pn,color_seq,labels):
        if shift_ev : 
            shift =  -(2*mod['bandwidth']+1)*(np.expm1(-mod['dis_param']))/mod['dis_param']
        else: shift = 0
        ax.plot(-mod['eig_vals']+shift,mod['PN'], '.', markersize=2, color=color, label=r"{0}".format(label))
    #ax.legend(loc='upper right')
    ax.legend(loc='best')
  
def plot_anderson1d_theory(ax, ev_pn, color_seq):
    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['bandwidth']
        xs = np.linspace(-2*b,2*b)
        ys = and_theory(xs, mod['dis_param'],b)
        ax.plot(xs,ys,color="b",linewidth=1)# gives a white "border"
        ax.plot(xs,ys,color=color,linewidth=0.8)

def plot_anderson1d_theory_vv(ax, ev_pn, color_seq, add_const=0, shift_ev =False):

    for (mod,color) in zip(ev_pn,color_seq):
        b=mod['bandwidth']
        N = mod['eig_vals'].size
        debug("dos sum for b = {}".format(b))

        #xs = np.linspace(-2*b,2*b,N//2)
        #lam = theor_banded_ev(b,N)[:N//2] - 2*b ## There it is conserving and (0,2pi)
        #dev = -theor_banded_dev(b,N)[:N//2]
        
        cgsd = cached_get_sum_dos(b)
        lam, dev = cgsd['eig_vals'], cgsd['inverse_dos']
        if shift_ev:
            shift =-(2*b+1)*(np.expm1(-mod['dis_param']))/mod['dis_param']
        else:
            shift =0
        ### the six only works for b=1 !!
        ys = 6 * dev**2 / (mod['dis_param'])**2
        #ys = and_theory(xs, mod['dis_param'],b)
        ax.plot(lam+add_const+shift,ys,color="w",linewidth=0.8)# gives a white "border"
        ax.plot(lam+add_const+shift,ys,color=color,linewidth=0.4)
        


        
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

def plotf_anderson(nums_and,figfilename="pta_anderson", 
                   dont_touch_ylim=False, ylogscale=False, zoom=False, 
                   labels=None,xlim = None, plot_theory=True, add_const=-1,shift_ev=0, text_kwargs=None):
    if zoom:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch]) 
   
    plot_anderson_complete(nums_and,fig,ax, 
                   dont_touch_ylim, ylogscale, zoom, 
                   labels,xlim, plot_theory, add_const,shift_ev,text_kwargs)
    
    fig.savefig(figfilename + ".pdf")
    plt.close() 

def plot_anderson_complete(nums_and,fig,ax, 
                   dont_touch_ylim=False, ylogscale=False, zoom=False, 
                   labels=None,xlim = None, plot_theory=True, add_const=-1,shift_ev=0, text_kwargs=None):
    """  This is a general pn vs ev plotter. 
         Most of the time we keep change one parameter and keep rest constant.
    """

    fig.subplots_adjust(left=0.1,right=0.95)
    if not dont_touch_ylim:
        ax.axhline((nums_and[0]['eig_vals'].size)*2/3, ls='--', color='black')
    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    plot_ev_pn(ax, nums_and,color_seq, labels,shift_ev)
    if plot_theory:
        color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        plot_anderson1d_theory_vv(ax, nums_and,color_seq, add_const, shift_ev)
    
    b= max(nums_and['bandwidth'])
    
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel("PN")
    
    if not dont_touch_ylim:
        ax.set_ylim(1, nums_and[0]['eig_vals'].size)
    if ylogscale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(get_LogNLocator())
    ax.set_ylim(1,None)
    
    if zoom:
        ax.set_xlim(-2*b-0.01,-2*b+0.1)
    elif xlim is not None:
        ax.set_xlim(xlim)
        
    if text_kwargs is not None:
        ax.text(**text_kwargs)
    #else:
    #    ax.set_xlim(-(2*b+0.5),(2*b+0.5))
        


    
def plotf_anderson_byN(nums_and,figfilename="pta_anderson_byN", text_kwargs=None):
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
    
    if text_kwargs is not None:
        ax.text(**text_kwargs)
    fig.savefig(figfilename + ".pdf")
    plt.close()

def get_ev_and_pn_hdf_file():
    return tables.openFile("ev_and_pn.hdf5", mode = "a", title = "Eigenvalues and PN")
    

            

def all_plots_forptatex():
    """ Generate all the plots that are in pta.tex. 
        We try to read data from the hdf5 file, but create it if needed"""
    with tables.openFile("ev_and_pn.hdf5", mode = "a", title = "Eigenvalues and PN") as h5file:

        
            
        #### Diagonal disorder, our fabulous band structure:
        run  = h5_get_data(h5file, model = models.Model_Anderson_DD_1d,
            factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,), dis_params= (0.1,))
        plotf_anderson(run, figfilename="pta_B_DD_low",add_const=0, 
                text_kwargs = {"x":-8,"y":700,"s":"Anderson", "fontsize":18})
        #plotf_anderson(run, figfilename="pta_B_DD_low_zoom", zoom=True,add_const=0)
        plotf_ev(run, figfilename="pta_B_DD_low_ev")

        # anderson by N:
        run1000 = run
        run2000 = h5_get_data(h5file, model = models.Model_Anderson_DD_1d,
            factory_args = dict(number_of_points=2000),
                                            bandwidths=(5,), dis_params= (0.1,))
        run3000 = h5_get_data(h5file, model = models.Model_Anderson_DD_1d,
            factory_args = dict(number_of_points=3000),
                                            bandwidths=(5,), dis_params= (0.1,))
        plotf_anderson_byN([run1000[0],run2000[0],run3000[0]], 
                text_kwargs = {"x":-8,"y":2200,"s":"Anderson", "fontsize":18})
    
        #### Exponential (potentially sparse)
        run  = h5_get_data(h5file, model = models.Model_Positive_Exp_banded_1d, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,), dis_params= (0.1,2,10))
                                            
                
        plotf_anderson(run,figfilename="pta_exp_shift", 
                ylogscale=True, shift_ev=True,xlim=(-2,14), 
                text_kwargs = {"x":4,"y":140,"s":"Sparse / P S", "fontsize":18})
        
        plotf_anderson(run,figfilename="pta_exp_regular", ylogscale=True, 
                shift_ev=False, xlim=(-11,5), 
                text_kwargs={"x":-10,"y":80,"s":"Sparse / P", "fontsize":18})
        
        plotf_ev(run, figfilename="pta_exp_low_ev")    
        
                
        #### Exponential conservative (potentially sparse)
        run  = h5_get_data(h5file, model = models.Model_Positive_Exp_banded_1d_conservative, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,), dis_params= (0.1,2,10))
                                            
                                            
        plotf_anderson(run, figfilename="pta_exp_cons", ylogscale=True, xlim=(-2,14),
                plot_theory=False, text_kwargs= {"x":4, "y":140, "s":"Sparse / P C", "fontsize":18})
        
        
    
    

        #### Box
        run  = h5_get_data(h5file, model = models.Model_Positive_Box_banded_1d, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,), dis_params= (0.1,2,10))
        plotf_anderson(run, figfilename="pta_box_low", text_kwargs={"x":-25,"y":80,"s":"Wigner", "fontsize":18})
        #plotf_anderson(run, figfilename="pta_box_low_zoom", zoom=True)
        #plotf_ev(run, figfilename="pta_box_low_ev")
        
        #### Box2 -positive
        run  = h5_get_data(h5file, model = models.Model_Positive_Box_banded_1d, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,15), dis_params= (2,))
        plotf_anderson(run, figfilename="pta_box2_positive", ylogscale=True, 
                text_kwargs={"x":-25,"y":80,"s":"Wigner / P", "fontsize":18})
        #plotf_ev(run, figfilename="pta_box2_positive_ev")
        
        #### Box2 - symmetric
        run  = h5_get_data(h5file, model = models.Model_Symmetric_Box_banded_1d, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,15), dis_params= (2,))
        plotf_anderson(run, figfilename="pta_box2_symmetric", ylogscale=True,plot_theory=False, 
                text_kwargs={"x":-14,"y":120,"s":"Wigner", "fontsize":18})
        #plotf_ev(run, figfilename="pta_box2_symmetric_ev")
        
        #### Box2  -positive cons
        run  = h5_get_data(h5file, model = models.Model_Positive_Box_banded_1d_conservative, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,15), dis_params= (2,))
        plotf_anderson(run, figfilename="pta_box2_pos_cons", ylogscale=True,plot_theory=False, 
                text_kwargs={"x":5,"y":100,"s":"Wigner / P C", "fontsize":18})
        #plotf_ev(run, figfilename="pta_box2_pos_cons_ev")
        
        #### Box2  -symmetric cons
        run  = h5_get_data(h5file, model = models.Model_Symmetric_Box_banded_1d_conservative, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,15), dis_params= (2,))
        plotf_anderson(run, figfilename="pta_box2_sym_cons", ylogscale=True,plot_theory=False, 
                text_kwargs={"x":-25,"y":80,"s":"Wigner / C", "fontsize":18})
        #plotf_ev(run, figfilename="pta_box2_sym_cons_ev")
        
                
        #### Box2 - around 1
        run  = h5_get_data(h5file, model = models.Model_Positive_Box_around1_banded_1d, factory_args = dict(number_of_points=1000),
                                            bandwidths=(5,), dis_params= (0.1,0.5,1))
        plotf_anderson(run, figfilename="pta_box_around1_positive", ylogscale=True)
        #plotf_ev(run, figfilename="pta_box_around1_positive_ev")

        

        
        
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
    
    cgsd = cached_get_sum_dos(b)
    lam, idos = cgsd['eig_vals'], cgsd['inverse_dos']
        
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
    
if __name__== "__main__":
    all_plots_forptatex()
    
