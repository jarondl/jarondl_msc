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
from collections import OrderedDict

# global packages
from numpy import pi
from matplotlib.ticker import MaxNLocator
import numpy as np
import tables
import yaml

# relative (intra-package) imports
from .libdl import plotdl
from .libdl import h5_dl
from .libdl import sparsedl
from .libdl.tools import h5_create_if_missing, h5_get_first_rownum_by_args
from .libdl.tools import ev_and_pn_class, ev_pn_g_class, c_k_g_class
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

    
def plot_1d_chain(N=2000, b=1, dis_param=0.):
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    
    m1 = models.Model_Anderson_BD_1d(N, dis_param=dis_param, bandwidth=b, periodic=False)
    ax.plot(-m1.eig_vals, m1.PN)
    fig.savefig("pta_ver2_chain_PN.pdf")
    ax.cla()
    
    g = sparsedl.chain_g(m1.eig_matrix, b=b)
    
    ax.plot(-m1.eig_vals, g)
    ax.text(x=0, y=0, s = "sum of g : {} ".format(np.nansum(g)))
    
    fig.savefig("pta_ver2_chain_g.pdf")
    plt.close()

def g_data_factory(model_name, number_of_points, bandwidth,dis_param, c,k):
    if model_name != "Anderson":
        raise Error("NotImpelmented")
    m = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
             bandwidth=bandwidth, dis_param=dis_param, periodic=False)
    g = sparsedl.A_matrix_inv(m.rate_matrix,c,k)
    return {'g': g}
    
def calculate_and_store(h5file, table_group, table_class, data_factory_class):
    """ This """
    if tbl_cls is None:
        tbl_cls = c_k_g_class()
    try:
        h5table = h5file.getNode(tbl_grp + '/ckg')
    except tables.exceptions.NoSuchNodeError:
        h5table = h5file.createTable(tbl_grp, 'ckg', tbl_cls, "c k g", createparents=True)
        
    factory_args = dict(const_args)
    ovar_args = OrderedDict(var_args)
    nrows = []
    for val_set in itertools.product(*ovar_args.values()):
        factory_args.update( zip(ovar_args.keys(), val_set))
        h5_create_if_missing(h5table, g_data_factory, factory_args)
        nrows.append(h5_get_first_rownum_by_args(h5table, factory_args))
    return h5table[nrows]        
    
    
def parse_and_calc(yaml_file):
    f = yaml.load_all(open(yaml_file,"r"))
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

    for run in f:
        ax.cla()
        ckg = calc_g(run['fig_name'], run['args'])
        g = abs(ckg['g'])
        ax.plot(run['args'][run['variable']], g)
        ax.set_xlabel(run['variable'])
        ax.set_ylabel('g')
        #ax.set_yscale('log')
        #ax.yaxis.set_major_locator(get_LogNLocator())
        
        fig.savefig(run['fig_name'].format(**run['args']))
        return ckg
    
def calc_g(fig_name, args = dict(model_name = ('Anderson',), number_of_points=(100,), bandwidth=(1,),
            dis_param=(0,), k= np.linspace(0,pi,10), c = np.arange(1,10))):
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

        r = h5_dl.Factory_Transmission_g(h5file)
        ckg = r.create_if_missing(args)
    return ckg

    g = ckg['g'].reshape([cr.size,-1])
    im = ax.imshow(abs(g), interpolation='nearest',origin='lower',aspect='auto', extent=[kr.min(),kr.max(),cr.min(),cr.max()])
    fig.colorbar(im,ax=ax)
    ax.set_xlabel('k')
    ax.set_ylabel('c')
    fig.savefig(fig_name.format(**args))
    return ckg

    r = h5_dl.Factory_Transmission_g(h5file)

def calc_and_plot_g(dis_param=0, kr= np.linspace(0,pi,10), cr = np.arange(1,10)):
    """ calculate and plot (should be separated)
    g for all kind of matrices """
    # ordered stuff:
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])


        args = dict(model_name = 'Anderson', number_of_points=1000, 
                    bandwidth=1, dis_param=dis_param)
        var_args = {'k' : kr, 'c' : cr}
        ckg = calc_g_to_h5(h5file, args, var_args)
        g = ckg['g'].reshape([cr.size,-1])
        im = ax.imshow(abs(g), interpolation='nearest',origin='lower',aspect='auto', extent=[kr.min(),kr.max(),cr.min(),cr.max()])
        fig.colorbar(im,ax=ax)
        ax.set_xlabel('k')
        ax.set_ylabel('c')
        fig.savefig('pta_ordered_s{0}.png'.format(dis_param))
        return ckg


def calc_and_plot_g_over_s(dis_range=np.linspace(0,1,20), kr= pi/2, cr = 1):
    """ calculate and plot (should be separated)
    g for all kind of matrices """
    # ordered stuff:
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

        #debug
        #import pdb; pdb.set_trace()
        args = dict(model_name = 'Anderson', number_of_points=1000, 
                    bandwidth=1, k = kr, c = cr)
        var_args = {'dis_param':dis_range}
                    
        ckg = calc_g_to_h5(h5file, args, var_args)
        g = abs(ckg['g'])
        ax.plot(dis_range, g)
        ax.set_xlabel('s')
        ax.set_ylabel('g')
        #ax.set_yscale('log')
        #ax.yaxis.set_major_locator(get_LogNLocator())
        
        fig.savefig('pta_disorder.png')
        return ckg

def calc_and_plot_g_over_N(dis_param=0.1, kr= pi/2, cr = 1, N_range=np.arange(100,101)):
    """ calculate and plot (should be separated)
    g for all kind of matrices """
    # ordered stuff:
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

        #debug
        #import pdb; pdb.set_trace()
        args = dict(model_name = 'Anderson',  
                    bandwidth=1, k = kr, c = cr , dis_param=dis_param)
        var_args = {'number_of_points':N_range}
                    
        ckg = calc_g_to_h5(h5file, args, var_args)
        g = abs(ckg['g'])
        ax.plot(N_range, g,'.')
        ax.set_xlabel('N')
        ax.set_ylabel('g')
        ax.set_ylim([0,1])
        #ax.set_yscale('log')
        #ax.yaxis.set_major_locator(get_LogNLocator())
        
        fig.savefig('pta_disorder_byN_s{}.png'.format(dis_param))
        return ckg
        
def new_calc_over_N(dis_param=0.1, kr= pi/2, cr = 1, N_range=np.arange(100,101)):
    """ calculate and plot (should be separated)
    g for all kind of matrices """
    # ordered stuff:
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

        #debug
        #import pdb; pdb.set_trace()
        args = dict(model_name = ('Anderson',),  
                    bandwidth=(1,), k = (kr,), c = (cr,) , dis_param=(dis_param,),
                    number_of_points = N_range)
                    
        r = h5_dl.Factory_Transmission_g(h5file)
                    
        ckg = r.create_if_missing(args)
        g = abs(ckg['g'])
        ax.plot(N_range, g)
        ax.set_xlabel('N')
        ax.set_ylabel('g')
        ax.set_ylim([0,1])
        #ax.set_yscale('log')
        #ax.yaxis.set_major_locator(get_LogNLocator())
        fig.savefig('pta_disorder_byN_s{}.png'.format(dis_param))
        return ckg
        
def plot_all_g_plots():
    calc_and_plot_g_over_N(dis_param=0.4, N_range=np.arange(2,500))
    calc_and_plot_g_over_N(dis_param=0.1, N_range=np.arange(2,500))
    calc_and_plot_g_over_k(kr=linspace(0,pi,1000), dis_param=0.4, N=400)
    calc_and_plot_g_over_k(kr=linspace(0,pi,1000), dis_param=0.4, N=100)
    calc_and_plot_g_over_k(kr=linspace(0,pi,1000), dis_param=0.1, N=400)
    calc_and_plot_g_over_k(kr=linspace(0,pi,1000), dis_param=0.1, N=100)


    
if __name__== "__main__":
    all_plots_forptatex()
    
