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

        
class Factory_Transmission_g(h5_dl.DataFactory):
    table_group   = '/ckg'
    table_name = 'ckg'
    table_class = c_k_g_class()
    table_description = ' ckg ' 
    def calculate(self, model_name, number_of_points, bandwidth,dis_param, c,k, seed):
        if model_name != "Anderson":
            raise Error("NotImpelmented")
        prng = np.random.RandomState((None if seed == 0  else seed))
        m = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=False, prng = prng)
        g = sparsedl.A_matrix_inv(m.rate_matrix,c,k)
        return {'g': g}
        
    
def parse_and_calc(yaml_file):
    f = yaml.load_all(open(yaml_file,"r"))
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

    for run in f:
        ax.cla()
        ckg = calc_g(run['fig_name'], run['args'])
        
        ### each run is a plot, but it could have multiple lines.
        # this requires some magic, in seperating our data by the second var.
        if 'second_variable' in run.keys():
            second_vars = run['args'][run['second_variable']]
            for s_var in second_vars:
                relevant_idxs = (ckg[run['second_variable']] == s_var)
                g  = abs(ckg['g'][relevant_idxs])
                ax.plot(ckg[run['variable']][relevant_idxs], g, '.')
        else:
            g = abs(ckg['g'])
            ax.plot(run['args'][run['variable']], g, '.')
        ax.set_xlabel(run['variable'])
        ax.set_ylabel('g')
           
        fig.savefig(run['fig_name'].format(**run['args']))
        plt.close(fig)
        
        #return ckg
        
def plot_psi1_psi2(seed=0):
    fig, axes  = plt.subplots(3,2,figsize=[2*plotdl.latex_width_inch, 3*plotdl.latex_height_inch],
                    sharex=True, sharey=True)
    for seed, ax in enumerate(itertools.chain(*axes)):
        prng = np.random.RandomState(seed)
        m = models.Model_Anderson_DD_1d(number_of_points=400, bandwidth=1, dis_param=0.3, periodic=False, prng=prng)
        g = sparsedl.A_matrix_inv(m.rate_matrix,1,pi/2)
        ax.plot((m.eig_matrix[0,:]), (m.eig_matrix[-1,:]),'.', label=str(abs(g)))
        #ax.legend()
        ax.set_title(str(abs(g)))
    fig.savefig("psi1_psi2_0.3.png")
        
def plot_psi1_psi2_abs(seed=0):
    fig, axes  = plt.subplots(3,2,figsize=[2*plotdl.latex_width_inch, 3*plotdl.latex_height_inch],
                    sharex=True, sharey=True)
    for seed, ax in enumerate(itertools.chain(*axes)):
        prng = np.random.RandomState(seed)
        m = models.Model_Anderson_DD_1d(number_of_points=400, bandwidth=1, dis_param=0.3, periodic=False, prng=prng)
        g = sparsedl.A_matrix_inv(m.rate_matrix,1,pi/2)
        ax.plot(abs(m.eig_matrix[0,:]), abs(m.eig_matrix[-1,:]),'.', label=str(abs(g)))
        #ax.legend()
        ax.set_title(str(abs(g)))
    fig.savefig("psi1_psi2abs_0.3.png")
    
def calc_g(fig_name, args = dict(model_name = ('Anderson',), number_of_points=(100,), bandwidth=(1,),
            dis_param=(0,), k= np.linspace(0,pi,10), c = np.arange(1,10))):
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

        r = Factory_Transmission_g(h5file)
        ckg = r.create_if_missing(args)
    return ckg


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

    
if __name__== "__main__":
    parse_and_calc(yaml_file)
    
