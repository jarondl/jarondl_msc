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
import pdb 

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
from .libdl import phys_functions

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
        g = phys_functions.A_matrix_inv(m.rate_matrix,c,k)
        psi_1, psi_N = (m.eig_matrix[0,:]), (m.eig_matrix[-1,:])
        heat_g = phys_functions.heat_g(psi_1, psi_N)

        psi1psiN = np.nansum(abs(psi_1*psi_N))
        return {'g': g, 'psi1psiN':psi1psiN, 'heat_g': heat_g }
        
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
    
        g = phys_functions.A_matrix_inv(m.rate_matrix, c, k)
        psi_1, psi_N = (m.eig_matrix[0,:]), (m.eig_matrix[-1,:])
        return {'g': g, 'psi_1': psi_1, 'psi_N': psi_N}
        
        
class Factory_thouless_psi(Factory_psi1_psiN):
    def __init__(self, *args, **kwargs):
        self._phase = kwargs.pop("phase", 0.001)
    
        super(Factory_thouless_psi, self).__init__(*args, **kwargs)

    def calculate(self, model_name, number_of_points, bandwidth,dis_param, c,k, seed, phi=0):
        """ Notice the periodicity !!!!! only works with anderson! """
        if model_name != "Anderson":
            raise Error("NotImpelmented")
        prng = np.random.RandomState((None if seed == 0  else seed))
        
        m = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=False, prng = prng)
        prng = np.random.RandomState((None if seed == 0  else seed))
        
        m1 = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=True, prng = prng, phi=0)
        prng = np.random.RandomState((None if seed == 0  else seed))
        
        m2 = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=True, prng = prng, phi=phi)
    
        g = phys_functions.A_matrix_inv(m.rate_matrix, c, k)
        psi_1, psi_N = (m1.eig_matrix[0,:]), (m1.eig_matrix[-1,:])
        thouless, prec = phys_functions.pure_thouless_g(m1.eig_vals, m2.eig_vals, phi)
        heat_g = phys_functions.heat_g(psi_1, psi_N)
        psi1psiN = np.nansum(abs(psi_1*psi_N))
        #############  UGLY HACK : Append zeros to fit sizes
        vals_with_N_size = [('psi_1', psi_1),
                                ('psi_N', psi_N),('thouless_g', abs(thouless))]
        if self._N == number_of_points :
            appended = vals_with_N_size
        else:
            myzeros = np.zeros(self._N - number_of_points)
        
            appended = [(key, np.append(val,myzeros)) for key,val in vals_with_N_size]
        
        print (prec*number_of_points, np.nansum(thouless))
        return dict([('g', g), ('psi1psiN',psi1psiN), 
                 ('thouless_sum', abs(np.nansum(abs(thouless)))), ('phi',phi),('heat_g', heat_g)] +appended)
        
        
class Factory_evs_phi(Factory_psi1_psiN):
    def calculate(self, model_name, number_of_points, bandwidth,dis_param, c,k, seed, phi=0):
        """ Notice the periodicity !!!!! only works with anderson! """
        if model_name != "Anderson":
            raise Error("NotImpelmented")

        prng = np.random.RandomState((None if seed == 0  else seed))
        
        m1 = models.Model_Anderson_DD_1d(number_of_points=number_of_points,
                 bandwidth=bandwidth, dis_param=dis_param, periodic=True, prng = prng, phi =phi)


        return {'phi':phi , 'eig_vals': m1.eig_vals}
        
def lyap_gamma(c,s,E=0):
    c,s = np.asarray(c), np.asarray(s)
    return 2*(s)**2/(24*(4*c**2-E**2))
    
    #return 20*(np.array(c)/np.array(s))**2

def parse_and_calc(yaml_file = 'pta_chain_def.yaml'):
    f = yaml.load(open(yaml_file,"r"))
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])

    for run in f:
        ax.cla()
        plot_a_run(run, ax)
        fig.savefig(run['fig_name'].format(**run['args']))
    plt.close(fig)


def plot_a_run(run, ax):
    """ plot a single run in my yaml format 
    """

    
    if run.get('N_dependance',False): ## light version's size is independent of N.
        #pdb.set_trace()
        # if all N are equal, we should use N, otherwise use largest.
        N = np.max(run['args']['number_of_points'])
        r = Factory_thouless_psi(run['npz_fname'].format(**run['args']), N= N)
    else:
        #pdb.set_trace()
        r = Factory_Transmission_g(run['npz_fname'].format(**run['args']))
    
    ckg = r.create_if_missing(run['args'])
    y_var = run.get('y_variable', 'g')
    y = ckg[y_var]

    ### each run is a plot, but it could have multiple lines.
    # this requires some magic, in seperating our data by the second var.
    if 'second_variable' in run:
        second_vars = run['args'][run['second_variable']]
        for s_var in second_vars:
            relevant_idxs = (ckg[run['second_variable']] == s_var)
            ax.plot(ckg[run['variable']][relevant_idxs],y[relevant_idxs] , '.')
        ## Plot red line - THIS should be consolidated soon
        x =ckg[run['variable']][relevant_idxs]
        E = 2*np.asarray(run['args']['c'])*np.cos(run['args']['k'])
        lloc = lyap_gamma(run['args']['c'],run['args']['dis_param'], E)
        n = np.asarray(run['args']['number_of_points'])
        if x.size == (n/lloc).size :
            #pdb.set_trace()
            #ax.plot(n, 2*(1+np.exp(n/lloc))**(-1))
            ax.plot(x, (np.exp(-2*n*lloc)),color='red', ls='--')

    elif 'average_over' in run:
        avgy=0
        avglny = 0
        for a_var in run['args'][run['average_over']]:
            relevant_idxs = (ckg[run['average_over']] == a_var)
            #debug(len(relevant_idxs)
            this_y = abs(y[relevant_idxs])
            avgy += this_y
            avglny += np.log(this_y)
            #ax.plot(ckg[run['variable']][relevant_idxs], g, '.')
        avgy /= len(run['args'][run['average_over']])
        avglny /= len(run['args'][run['average_over']])
        
        ## accidentaly use last relevant_idxs. hope this works..
        
        x =ckg[run['variable']][relevant_idxs]

        
        ax.plot(x, avgy)
        ax.plot(x, np.exp(avglny))
        lloc = lyap_gamma(run['args']['c'],run['args']['dis_param'])
        n = np.asarray(run['args']['number_of_points'])
        if x.size == (n/lloc).size :
            #pdb.set_trace()
            #ax.plot(n, 2*(1+np.exp(n/lloc))**(-1))
            ax.plot(x, (np.exp(-2*n*lloc)),color='red', ls='--')
    else:
        x =ckg[run['variable']]
        debug("NOTICE - taking first c and dis_param")
        #lloc = 104*run['args']['c'][0]**2 / (3*((run['args']['dis_param'][0])**2))
        lloc = lyap_gamma(run['args']['c'],run['args']['dis_param'])

        n = np.asarray(run['args']['number_of_points'])
        #print((len(x), len(n), len(lloc),len(run['args']['dis_param'])))
        if x.size == (n/lloc).size:
            ax.plot(x, (np.exp(-2*n*lloc)),color='red', ls='--')

        ax.plot(x, abs(y), '.')
    ax.set_xlabel(run['variable'])
    ax.set_ylabel(y_var)
       
       
def plot_gheat_g(seed=1):
    """ important since compares different g values """
    fig, ax = plt.subplots(figsize=[2.5*plotdl.latex_width_inch, 3*plotdl.latex_height_inch])
    
    r = Factory_psi1_psiN( "aapta_of_s_N{number_of_points[0]}.npz", N=400)
    ckg = r.create_if_missing(dict(model_name= ["Anderson",], 
                        number_of_points=[400,], bandwidth=[1,],
                         dis_param=np.linspace(0,1,100),c=[1,], k=[1.57,], seed=np.arange(1,6)))    
    color_seq = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for (seed,c) in zip(np.arange(1,6),color_seq):
        ck = ckg[ckg['seed']==seed]
        g, psi_1, psi_N = ck['g'], ck['psi_N'], ck['psi_1']

        psi_heat = 2*(abs(psi_1)**2)*(abs(psi_N)**2) / ((abs(psi_1)**2) + (abs(psi_N)**2))
        
        phs = np.nansum(psi_heat,axis=1)
        
        psi1psiN = np.nansum(abs(psi_1*psi_N), axis=1)
        #print(ckg['dis_param'], phs)
        ax.plot(ck['dis_param'], phs,'.', color=c)
        ax.plot(ck['dis_param'], abs(g),'+', color=c)
        ax.plot(ck['dis_param'], psi1psiN,'d', color=c)
    ax.set_xlabel('dis_param')
    fig.savefig('pta_comparison_of_s_N400.png')
    plt.close(fig)
    ## use last ck
    fig1, axes1  = plt.subplots(3,2,figsize=[2*plotdl.latex_width_inch, 3*plotdl.latex_height_inch],
                    sharex=True, sharey=True)
    axes1.flat[0].xaxis.set_major_locator(MaxNLocator(4))
    axes1.flat[0].yaxis.set_major_locator(MaxNLocator(4))
    for n, ax1 in zip(range(1,20,3), axes1.flat):
        ax1.plot(abs(ck['psi_1'][n]), abs(ck['psi_N'][n]), '.')  
        ax1.set_title("W = {:0.2}".format(ck['dis_param'][n]))
    fig1.savefig('pta_psi_1_psi_2_N400.png')
    
    ax.cla()
    ax.plot(ck['dis_param'], np.real(g), label='real')
    ax.plot(ck['dis_param'], np.imag(g), label='imag')
    ax.plot(ck['dis_param'], np.abs(g), label='abs')
    ax.legend(loc = 'upper right')
    ax.set_xlabel('dis_param')
    ax.set_ylabel('g')
    fig.savefig('pta_real_imag_g_s_N400')

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
        ax1.set_title("{:0.3}".format(abs(g)))
        psi_times_psi = abs(psi_1*psi_N)
        ax2.plot(psi_times_psi,'.')
        ax2.set_title("{:0.3}".format(sum(psi_times_psi)))
        psi_heat = (abs(psi_1)**2)*(abs(psi_N)**2) / ((abs(psi_1)**2) + (abs(psi_N)**2))
        ax3.plot(psi_heat,'.')
        ax3.set_title("{:0.3}".format(np.nansum(psi_heat)))
        for ax in (ax1,ax2,ax3):
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
    fig1.suptitle(r"$\psi_1, \psi_N$")
    fig2.suptitle(r"$ |\psi_1\psi_N|$")
    fig3.suptitle(r"$ \frac{{|\psi_1|^2\psi_N|^2}}{{|\psi_1|^2+\psi_N|^2}}$")
    fig1.savefig("psi1_psi2{}_0.3.png".format('_abs' if abs else ''))
    fig2.savefig("psi_times_psi2{}_0.3.png".format('_abs' if abs else ''))
    fig3.savefig("psi_heat_psi2{}_0.3.png".format('_abs' if abs else ''))
    
def disperssion_g(fig_name='pta_disperse_s_{dis_param[0]}{log}.png', args = dict(model_name = ('Anderson',), number_of_points=(200,), bandwidth=(1,),
            dis_param=(0.4,), k= (1.57,) , c = (1,), seed=np.arange(10000))):
    
        
    r = Factory_Transmission_g("pta_dispersion{}.npz".format(args['dis_param'][0]))
    ckg = r.create_if_missing(args)
    lloc = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0]
    N = ckg['number_of_points'][0]
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    ax.hist(abs(ckg['g']),bins=np.sqrt(len(ckg['g'])))
    fig.savefig(fig_name.format(log="", **args))
    ax.cla()
    
    gamma =  -np.log(abs(ckg['g']))/(2*N)
    lloc = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0]
    sig = lloc/(2*N)
    ax.hist(gamma,bins=np.sqrt(len(ckg['g'])), normed=True)
    
    g_space = np.linspace(min(gamma), max(gamma))
    gauss = np.exp(-(g_space-lloc)**2/(2*sig))/(np.sqrt(2*pi*sig))
    gaussf = np.exp(-(g_space-lloc)**2/(2*sig))/(np.sqrt(2*pi*sig)) + np.exp(-(-g_space-lloc)**2/(2*sig))/(np.sqrt(2*pi*sig))
    ax.plot(g_space, gaussf, '-.' , color='red')
    ax.set_xlabel(r'$\gamma$')
    fig.savefig(fig_name.format(log="log", **args))
    return ckg
    #plt.close(fig)
    
def plot_dispression_of_N(fig_name='pta_disperse_s_of_N_{dis_param[0]}.png', 
            args = dict(model_name = ('Anderson',), number_of_points=np.arange(2,500), 
            bandwidth=(1,), dis_param=(0.8,), k= (1.57,) , c = (1,), seed=np.arange(30))):
    
    r = Factory_Transmission_g("pta_dispersion_of_N_{}.npz".format(args['dis_param'][0]))
    ckg = r.create_if_missing(args)
    N = ckg['number_of_points'].reshape([len(args['number_of_points']), len(args['seed'])])
    g = ckg['g'].reshape([len(args['number_of_points']), len(args['seed'])])
    #gamma = np.log(1+1.0/abs(g))/(2*N)
    gamma = -np.log(abs(g))/(2*N)
    lloc = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0]

    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    ax.plot(N[:,0], np.average(gamma,axis=1),'.', color='blue')
    ax.axhline(lloc, ls="--", color='red')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$\gamma , 2N\langle\gamma^2\rangle$')
    #ax.plot(N[:,0], np.average(abs(g),axis=1),'.', color='cyan')
    #ax2 = ax.twinx()
    ax2=ax
    ax2.plot(N[:,0], np.var(gamma,axis=1)*2*N[:,0],'.', color='green')
    #ax2.axhline(lloc, ls="--", color='red')
    #ax2.set_ylabel(r'$2N\langle\gamma^2\rangle$')
    #ax2.plot(N[:,0], np.var(abs(g),axis=1),'.', color='magenta')
    fig.savefig(fig_name.format(**args))



    
if __name__== "__main__":
    parse_and_calc()
    
