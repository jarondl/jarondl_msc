#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" PTA chain conductance.

Mainly calculating and plotting all kinds of conductances,
for 1d Anderson models. The definitions for the data
creation and plotting are hosted on separate yaml files for
easy configuration. 
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
import argparse

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

from .libdl.phys_functions import lyap_gamma
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
        g = abs(phys_functions.A_matrix_inv(m.rate_matrix,c,k))**2
        ######################### TEMP #######################
        #g = phys_functions.alternative_A_matrix_inv(m.rate_matrix,c,k)
        #g = phys_functions.diag_approx_A_matrix_inv(m.rate_matrix,c,k, m.eig_matrix)
        #g_diag_approx = abs(phys_functions.alter_diag_approx(m.eig_vals,c,k, m.eig_matrix))
        abs_g_diag_approx = phys_functions.diag_approx_abs(m.eig_vals,c,k, m.eig_matrix)
        g_diag_approx = abs(phys_functions.alter_diag_approx(m.eig_vals,c,k, m.eig_matrix))**2
        psi_1, psi_N = (m.eig_matrix[0,:]), (m.eig_matrix[-1,:])
        heat_g = phys_functions.heat_g(psi_1, psi_N)

        psi1psiN = np.nansum(abs(psi_1*psi_N))
        return {'g': g, 'psi1psiN':psi1psiN, 'heat_g': heat_g , 'g_diag_approx': g_diag_approx, 'abs_g_diag_approx': abs_g_diag_approx}
        
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
    
        g = abs(phys_functions.A_matrix_inv(m.rate_matrix, c, k))**2
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
    
        g = abs(phys_functions.A_matrix_inv(m.rate_matrix, c, k))**2
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
        

    #return 20*(np.array(c)/np.array(s))**2


def plot_localization_length(ax, c, k, dis_param, n, x):
    """ Add the theoretical expected line for localization length"""
    E = 2*np.asarray(c)*np.cos(k)
    gamma_inf = lyap_gamma(c,dis_param, E)
    loc_length1 = 1/((np.cosh(n*gamma_inf))**2)
    loc_length2 = 1/(np.exp(2*n*gamma_inf)+1)
    loc_length3 = 1/(np.exp(2*n*gamma_inf))
    if x.size == loc_length1.size :
        ax.plot(x, loc_length1,color='black', ls='--')
        ax.plot(x, loc_length2,color='purple', ls='--')
        ax.plot(x, loc_length3,color='cyan', ls='--')

def calc_a_run(run):
    """ create data for a run ( a'run' is a yaml configuration segment)"""
    if run.get('N_dependance',False): ## light version's size is independent of N.
        #pdb.set_trace()
        # if all N are equal, we should use N, otherwise use largest.
        N = np.max(run['args']['number_of_points'])
        r = Factory_thouless_psi(run['npz_fname'].format(**run['args']), N= N)
    else:
        #pdb.set_trace()
        r = Factory_Transmission_g(run['npz_fname'].format(**run['args']))
    
    return r.create_if_missing(run['args'])

def plot_a_run(run, ax):
    """ plot a single run in my yaml format 
    """
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    y_var = run['y_variable']
    full_y = ckg[y_var]
    x_var = run['x_variable']
    full_x = ckg[x_var]
    ### each run is a plot, but it could have multiple lines.
    # this requires some magic, in seperating our data by the second var.
    ## I ASSUME, and this is important, that only two variables change
    x_to_plot = full_x
    x_to_calc = full_x
    y_to_plot = full_y
    ckg_fc = ckg
    if 'second_var' in run:     
        ckg_fc = ckg[:,0]
        x_to_calc = full_x[:,0]
    elif ('average_over' in run):#### always do log average
        #y_to_plot = np.average(full_y, axis=1)
        y_to_plot = np.exp(np.average(np.log(full_y), axis=1))
        
        ckg_fc = ckg[:,0]
        x_to_plot = x_to_calc = full_x[:,0]
        #pdb.set_trace()
    ax.plot(x_to_plot, y_to_plot,".")
    plot_localization_length(ax, ckg_fc['c'],ckg_fc['k'], ckg_fc['dis_param'], ckg_fc['number_of_points'] , x_to_calc)
    ax.set_xlabel(x_var)
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
    
def plot_dispersion_g(run):
    
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    N = ckg['number_of_points'][0]
    g = ckg['g']

    gamma = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0]
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    gamma2N = gamma*2*N
    x = 2*np.arccosh(1/np.sqrt(g))
    #sig = lloc/(2*N)
    ax.hist(x,bins=np.sqrt(len(ckg['g'])), normed=True, edgecolor='none')
    sig = gamma2N
    g_space = np.linspace(min(x), max(x))
    #gauss = np.exp(-(g_space-gamma)**2/(2*sig))/(np.sqrt(2*pi*sig))
    gauss = lambda y: np.exp(-(y-gamma2N)**2/(2*sig))/(np.sqrt(2*pi*sig))
    gaussf = gauss(g_space) + gauss(-g_space)
    gaussx = g_space * gauss(g_space)
    ax.autoscale(False)
    ax.plot(g_space, gauss(g_space), ':' , color='red')
    ax.plot(g_space, gaussx, '--' , color='red')
    ax.set_xlabel(r'$x$')
    fig.savefig(run['fig_name'])

    
def plot_dispersion_of_N(run):
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    N = ckg['number_of_points']
    g = ckg['g']
    #gamma = np.log(abs(g))/(N)
    #gamma = np.cosh(abs(g))**2/(N)
    #x = 2*np.arccosh(1/np.sqrt(g))
    x = 2*np.arccosh(1/np.sqrt(g))
    #x = np.log(g)
    gamma = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0,0]
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    ax.plot(N[:,0], np.average(x,axis=1),'.', color='blue')
    #ax.axhline(lloc, ls="--", color='red')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$x$, $\langle x^2 \rangle$')
    #ax.plot(N[:,0], np.average(abs(g),axis=1),'.', color='cyan')
    #ax2 = ax.twinx()
    ax2=ax
    ax2.plot(N[:,0], np.var(x,axis=1),'.', color='green')
    ax2.plot(N[:,0], 2*gamma*N[:,0],'--')
    #ax2.plot(N[:,0], gamma*N[:,0],'--')
    #ax2.axhline(lloc, ls="--", color='red')
    #ax2.set_ylabel(r'$2N\langle\gamma^2\rangle$')
    #ax2.plot(N[:,0], np.var(abs(g),axis=1),'.', color='magenta')
    fig.savefig(run['fig_name'])
    
def plot_compare_g_of_N(run):
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    N = ckg['number_of_points'][:,0]
    avg = np.average(ckg['g'], axis=1)
    lanavg = np.exp(np.average(np.log(ckg['g']), axis=1))
    heat_g = np.average(ckg['heat_g'], axis=1)
    psi1psiN = np.average(ckg['psi1psiN'], axis=1)
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, 1.5*plotdl.latex_height_inch])
    ax.plot(N, avg, label="$g$")
    ax.plot(N, lanavg, label=r"$g_{TYP}$")
    ax.plot(N, heat_g, label=r"$g_H$")
    ax.plot(N, psi1psiN, label=r"$\psi_1\psi_N$")
    ckg0 = ckg[:,0]
    ax.set_yscale('log')
    ax.set_xlabel('N')
    ax.autoscale(False)
    plot_localization_length(ax,ckg0['c'],ckg0['k'],ckg0['dis_param'],N,N)

    ax.legend(loc='lower left')
    fig.savefig(run['fig_name'])
    ax.autoscale(True)
    
    

def plot_special_plot(run):
    options =  {'dispersion_of_N': plot_dispersion_of_N,
                'dispersion_g'   : plot_dispersion_g,
                'compare_g_of_N' : plot_compare_g_of_N}
    options.get(run['special_plot'])(run)

    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_yaml','-d', type=file)
    parser.add_argument('--plots_yaml','-p', type=file)
    args = parser.parse_args()
    
    if args.data_yaml is not None:
        f = yaml.load(args.data_yaml)
        for run in f:
            calc_a_run(run)
    
    if args.plots_yaml is not None:
        f = yaml.load(args.plots_yaml)
        
        fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
        for run in f:
            if 'special_plot' in run:
                # special plot means any non regular one.
                plot_special_plot(run)
            else:
                plot_a_run(run, ax)
                fig.savefig(run['fig_name'])
                ax.cla()
        plt.close()
    
if __name__== "__main__":
    main()
