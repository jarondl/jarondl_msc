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
from .libdl.tools import ev_and_pn_class, ev_pn_g_class, c_k_g_class, ckg_dtype, ckg_psis_dtyper
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
    dtype = ckg_dtype
    def calculate(self, model_name, number_of_points, bandwidth,dis_param, c,k, seed):
        if model_name != "Anderson":
            raise Error("NotImpelmented")
        prng = np.random.RandomState((None if seed == 0  else seed))
        m = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=False, prng = prng)
        g = sparsedl.A_matrix_inv(m.rate_matrix,c,k)
        return {'g': g}
        
class Factory_psi1_psiN(h5_dl.DataFactory):
    def __init__(self, *args, **kwargs):
        self._N = kwargs.pop("N", None)
        self.dtype = ckg_psis_dtyper(self._N)
        super(Factory_psi1_psiN,self).__init__(*args, **kwargs)

    def calculate(self, model_name, number_of_points, bandwidth,dis_param, c,k, seed):
        if model_name != "Anderson":
            raise Error("NotImpelmented")
        prng = np.random.RandomState((None if seed == 0  else seed))
        m = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=False, prng = prng)
    
        g = sparsedl.A_matrix_inv(m.rate_matrix, c, k)
        psi_1, psi_N = (m.eig_matrix[0,:]), (m.eig_matrix[-1,:])
        return {'g': g, 'psi_1': psi_1, 'psi_N': psi_N}
        
def loc_length(c,s):
    return 20*(np.array(c)/np.array(s))**2
def parse_and_calc(yaml_file = 'pta_chain_def.yaml'):
    f = yaml.load(open(yaml_file,"r"))
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

    for run in f:
        ax.cla()
        #ckg = calc_g(run['fig_name'], run['npz_fname'].format(**run['args']), run['args'])
        r = Factory_Transmission_g(run['npz_fname'].format(**run['args']))
        ckg = r.create_if_missing(run['args'])
        ### each run is a plot, but it could have multiple lines.
        # this requires some magic, in seperating our data by the second var.
        if 'second_variable' in run:
            second_vars = run['args'][run['second_variable']]
            for s_var in second_vars:
                relevant_idxs = (ckg[run['second_variable']] == s_var)
                g  = abs(ckg['g'][relevant_idxs])
                ax.plot(ckg[run['variable']][relevant_idxs], g, '.')
        elif 'average_over' in run:
            avgg=0
            avglng = 0
            for a_var in run['args'][run['average_over']]:
                relevant_idxs = (ckg[run['average_over']] == a_var)
                #debug(len(relevant_idxs)
                g = abs(ckg['g'][relevant_idxs])
                avgg += g
                avglng += np.log(g)
                #ax.plot(ckg[run['variable']][relevant_idxs], g, '.')
            avgg /= len(run['args'][run['average_over']])
            avglng /= len(run['args'][run['average_over']])
            
            ## accidentaly use last relevant_idxs. hope this works.
            ax.plot(ckg[run['variable']][relevant_idxs], avgg)
            ax.plot(ckg[run['variable']][relevant_idxs], np.exp(avglng))
            x =ckg[run['variable']][relevant_idxs]
            lloc = loc_length(run['args']['c'],run['args']['dis_param'])
            n = (run['args']['number_of_points'])
            
            #ax.plot(n, 2*(1+np.exp(n/lloc))**(-1))
            ax.plot(x, 2*(1+np.exp(n/lloc))**(-1))
        else:
            g = abs(ckg['g'])
            x =ckg[run['variable']]
            debug("NOTICE - taking first c and dis_param")
            #lloc = 104*run['args']['c'][0]**2 / (3*((run['args']['dis_param'][0])**2))
            lloc = loc_length(run['args']['c'],run['args']['dis_param'])

            n = run['args']['number_of_points']
            print((len(x), len(n), len(lloc),len(run['args']['dis_param'])))
            
            ax.plot(x, 2*(1+np.exp(n/lloc))**(-1))
            ax.plot(run['args'][run['variable']], g, '.')
        ax.set_xlabel(run['variable'])
        ax.set_ylabel('g')
           
        fig.savefig(run['fig_name'].format(**run['args']))
        plt.close(fig)
        
        #return ckg
        
def plot_psi1_psi2(seed=0, abs_value=False):
    fig1, axes1  = plt.subplots(3,2,figsize=[2*plotdl.latex_width_inch, 3*plotdl.latex_height_inch],
                    sharex=True, sharey=True)
    fig2, axes2  = plt.subplots(3,2,figsize=[2*plotdl.latex_width_inch-2, 3*plotdl.latex_height_inch],
                    sharex=True, sharey=True)
    fig3, axes3  = plt.subplots(3,2,figsize=[2*plotdl.latex_width_inch-2, 3*plotdl.latex_height_inch],
                    sharex=True, sharey=True)
    r = Factory_psi1_psiN("psi_1_psi_N.npz", N=400)
    ckg = r.create_if_missing(dict(model_name= ["Anderson",], 
                        number_of_points=[400,], bandwidth=[1,],
                         dis_param=[0.3,],c=[1,], k=[1.57,], seed=np.arange(1,7)))

    for seed, (ax1,ax2,ax3,ck) in enumerate(zip(axes1.flat, axes2.flat, axes3.flat,ckg)):
        
        
        g, psi_1, psi_N = ck['g'], ck['psi_N'], ck['psi_1']
        if abs_value: 
            psi_1, psi_N = abs(psi_1), abs(psi_N)
        ax1.plot(psi_1, psi_N,'.', label=str(abs(g)))
        #ax.legend()
        ax1.set_title(str(abs(g)))
        psi_times_psi = abs(psi_1*psi_N)
        ax2.plot(psi_times_psi,'.')
        ax2.set_title((sum(psi_times_psi)))
        psi_heat = (abs(psi_1)**2)*(abs(psi_N)**2) / ((abs(psi_1)**2) + (abs(psi_N)**2))
        ax3.plot(psi_heat,'.')
        ax3.set_title((np.nansum(psi_heat)))
    fig1.suptitle(r"$\psi_1, \psi_N$")
    fig2.suptitle(r"$ |\psi_1\psi_N|$")
    fig3.suptitle(r"$ \frac{{|\psi_1|^2\psi_N|^2}}{{|\psi_1|^2+\psi_N|^2}}$")
    fig1.savefig("psi1_psi2{}_0.3.png".format('_abs' if abs else ''))
    fig2.savefig("psi_times_psi2{}_0.3.png".format('_abs' if abs else ''))
    fig3.savefig("psi_heat_psi2{}_0.3.png".format('_abs' if abs else ''))
    
def disperssion_g(fig_name='pta_disperse_s_{dis_param[0]}{log}.png', args = dict(model_name = ('Anderson',), number_of_points=(100,), bandwidth=(1,),
            dis_param=(0.8,), k= (1.57,) , c = (1,), seed=np.arange(1000))):
    
        
    r = Factory_Transmission_g("pta_dispersion.npz")
    ckg = r.create_if_missing(args)
        
        
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    ax.hist(abs(ckg['g']),bins=np.sqrt(len(ckg['g'])))
    fig.savefig(fig_name.format(log="", **args))
    ax.cla()
    ax.hist(np.log(abs(ckg['g'])),bins=np.sqrt(len(ckg['g'])))
    ax.set_xlabel('log(g)')
    fig.savefig(fig_name.format(log="log", **args))
    plt.close(fig)


    
def calc_g(fig_name,npz_fname, args = dict(model_name = ('Anderson',), number_of_points=(100,), bandwidth=(1,),
            dis_param=(0,), k= np.linspace(0,pi,10), c = np.arange(1,10))):
    with tables.openFile("trans_g.hdf5", mode = "a", title = "Transmission g") as h5file:
        
        #r = Factory_Transmission_g(h5file)
        r = Factory_Transmission_g(npz_fname)
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
    parse_and_calc()
    
