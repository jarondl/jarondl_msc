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
from scipy.stats import norm, lognorm
from matplotlib.ticker import MaxNLocator, LogLocator
import numpy as np
import tables
import yaml

# relative (intra-package) imports
from .libdl import plotdl
from .libdl import h5_dl
from .libdl import sparsedl
from .libdl import phys_functions

from .libdl.phys_functions import lyap_gamma
from .libdl.sparsedl import logavg
from .libdl.tools import h5_create_if_missing, h5_get_first_rownum_by_args
from .libdl.tools import ev_and_pn_class, ev_pn_g_class, c_k_g_class, ckg_dtype, ckg_psis_dtyper
from .libdl.plotdl import plt, cummulative_plot, get_LogNLocator, s_cummulative_plot
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
        abs_g_diag_approx = phys_functions.abs_g_diag_approx(m.eig_vals,c,k, m.eig_matrix)
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
        return {'g': g, 'psi_1': psi_1, 'psi_N': psi_N , 'eig_vals' : m.eig_vals}
        
        
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
    loc_length3 = np.exp(-2*n*gamma_inf)
    if x.size == loc_length3.size :
        ax.autoscale(False)
        ax.plot(x, loc_length3,color='black', ls='--')

def calc_a_run(run):
    """ create data for a run ( a'run' is a yaml configuration segment)"""
    if run.get('N_dependance',False): ## light version's size is independent of N.
        # if all N are equal, we should use N, otherwise use largest.
        N = np.max(run['args']['number_of_points'])
        r = Factory_thouless_psi(run['npz_fname'].format(**run['args']), N= N)
    else:
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
       
def plot_gh_sum(N=800):
    
    fig, ax = plt.subplots(figsize=[2.5*plotdl.latex_width_inch, 3*plotdl.latex_height_inch])
    
    r = Factory_psi1_psiN( "pta_of_s_N{number_of_points[0]}.npz", N=N)
    ckg = r.create_if_missing(dict(model_name= ["Anderson",], 
                        number_of_points=[N,], bandwidth=[1,],
                         dis_param=[0.4,],c=[1,], k=[1.57,], seed=np.arange(1,101)))
                         
    a1 = abs(ckg['psi_1'])**2
    aN = abs(ckg['psi_N'])**2
    ev = ckg['eig_vals']
    v = 2
    aa = (a1+aN)/2
    ga = 2*N*a1*aN/(a1+aN)
    gda = 4 * aa * ga / ( ev**2 + (2*aa)**2 )

    avg_ga = np.nansum(ga, axis=1)
    avg_gda = np.nansum(gda, axis=1)
    #print(np.nansum(gda))
    minx = min(min(avg_ga), min(avg_gda))
    maxx = max(max(avg_ga), max(avg_gda))
    xs = np.linspace(minx,maxx)
    ax.plot(avg_ga, avg_gda, '.')
    ax.set_xlabel(r'${  \sum N g_\alpha (E=0)   }$')
    ax.set_ylabel(r'${  \sum g_{DA}(E=0) }$')
    #plt.draw()
    #ax.autoscale(False)
    ax.plot(xs,xs)
    fig.savefig('pta_ga_gda.png') 
    #fig.close()
    fig, ax = plt.subplots(figsize=[2.5*plotdl.latex_width_inch, 3*plotdl.latex_height_inch])
    ax.plot(ev[0], ga[0])
    ax.plot(ev[0], gda[0])
    ax.set_yscale('log')
    ax.set_ylim(1e-40,10)
    ax.set_xlabel(r'${ \varepsilon_\alpha  }$')
    ax.set_ylabel(r'${ g  }$')
    fig.savefig('pta_ga_gda_of_e.png')
    ax.cla()
    ax.plot(ev.flat, ga.flat, '.')
    ax.plot(ev.flat, gda.flat , '.')
    ax.set_ylim(1e-40,10)
    fig.savefig('pta_many_ga_gda_of_e.png')
    
    return ckg, ga, gda
    
       
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
    
def plot_histogram_with_norm_fit(ax, data, expected=None,color='blue'):
    loc, scale = norm.fit(data)
    xspace = np.linspace(min(data), max(data), 100)
    ax.hist(data, bins=np.sqrt(data.size), normed=True, edgecolor='none', color=color)
    
    ax.plot(xspace, norm.pdf(xspace, loc, scale), 
            color = 'black', ls="--", label=' $\mu = {:0.3}$,  $\sigma^2 = {:0.3}$'.format(loc,scale**2))
    if expected is not None:
        ax.plot(xspace, norm.pdf(xspace, expected[0], np.sqrt(expected[1])), 
            color = 'black', ls='-', label=' $\mu = {:0.3}$,  $\sigma^2 = {:0.3}$'.format(expected[0],expected[1]))

def plot_compare_dispersions(run):
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    N = ckg['number_of_points'][0]
    dp = ckg['dis_param'][0]
    gam = lyap_gamma(1,dp)
    fig, ((ax1,ax2),(ax3,ax4))  = plt.subplots(nrows=2, ncols=2, 
                figsize=[2*plotdl.latex_width_inch, 2*plotdl.latex_height_inch])
                

    lng = -np.log(abs(ckg['g']))
    plot_histogram_with_norm_fit(ax1, lng, expected=(2*gam*N, 2*gam*N),color='blue')
    #ax1.legend()
    ax1.axvline((2*gam*N),color='green')
    ln_heatg = -np.log(abs(ckg['heat_g']))
    plot_histogram_with_norm_fit(ax2, ln_heatg,color='blue')
    #ax2.legend()
    ax2.autoscale(False)
    ax2.axvline((gam*N),color='red')
    ax2.axvline((2*gam*N),color='green')
    ax2.axvline((0.5*gam*N),color='purple')

    g = (abs(ckg['g']))
    plot_histogram_with_norm_fit(ax3, g, color='blue')
    
    #ax3.legend()
    heatg = (abs(ckg['heat_g']))
    plot_histogram_with_norm_fit(ax4, heatg,color='blue')
    ax4.autoscale(False)
    ax4.axvline(np.exp(-gam*N),color='red')
    ax4.axvline(np.exp(-2*gam*N),color='green')
    #ax3.legend()
    fig.suptitle("E = 0,  $W={}$  $N={}$    $\gamma^{{-1}}$ = {:5}    #={}".format(dp, N, gam**(-1), ckg.size))
    ax1.set_xlabel('$-\ln(g)$')
    ax2.set_xlabel('$-\ln(g_h)$')
    ax3.set_xlabel('$g$')
    ax4.set_xlabel('$g_h$')
    for ax in ax1, ax2, ax3, ax4:
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ### accomodate suptitle
    fig.tight_layout(rect = (0,0,1,0.9))
    fig.savefig(run['fig_name'])
    plt.close() 
    

    
def plot_dispersion_g(run): ##(histograms)
    
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    N = ckg['number_of_points'][0]
    g = ckg['g']

    gamma = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0]
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    gamma2N = gamma*2*N
    #gamma2N = gamma*N
    lng = -np.log(g)
    #lng = -np.log(ckg['heat_g'])
    #sig = lloc/(2*N)
    ax.hist(lng, bins=np.sqrt(len(ckg['g'])), normed=True, edgecolor='none')
    sig = gamma2N
    g_space = np.linspace(min(lng), max(lng))
    #gauss = np.exp(-(g_space-gamma)**2/(2*sig))/(np.sqrt(2*pi*sig))
    gauss = lambda y: np.exp(-(y-gamma2N)**2/(2*sig))/(np.sqrt(2*pi*sig))
    gaussf = gauss(g_space) + gauss(-g_space)
    gaussx = g_space * gauss(g_space)
    ax.autoscale(False)
    ax.plot(g_space, gauss(g_space), ':' , color='red')
    ax.plot(g_space, gaussf, '--' , color='red')
    ax.set_xlabel(r'$x$')
    fig.suptitle("E = 0,  $W={}$  $N={}$    $\gamma^{{-1}}$ = {:5}".format(ckg['dis_param'][0], N, gamma**(-1)))
    fig.savefig(run['fig_name'])
    plt.close()
    
def plot_dispersion_of_N(run):
    npz = np.load(run['npz_fname'])
    ckg = npz['nums']
    N = ckg['number_of_points']
    g = abs(ckg['g'])
    da = abs(ckg['abs_g_diag_approx'])
    gh = abs(ckg['heat_g'])    
    gamma = lyap_gamma(ckg['c'],ckg['dis_param'],E=0)[0,0]
    dp = ckg['dis_param'][0,0]
    
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, 2*plotdl.latex_height_inch])
    ax.plot(N[:,0], np.average(np.log(g),axis=1), label= r"$\langle \ln(g)\rangle$")
    #ax.plot(N[:,0], np.var(x,axis=1),  label=r"var$(\ln(g))$")
    #ax.plot(N[:,0], np.average(x2,axis=1), label=r"$-2\langle \ln(\psi_1\psi_N)\rangle$")
    ax.plot(N[:,0], np.log(np.average(g,axis=1)), label=r"$\ln(\langle g\rangle)$")
    ax.plot(N[:,0], np.average(np.log(da),axis=1), label=r"$\langle\ln( g_{DA})\rangle$")
    ax.plot(N[:,0], np.log(np.average(da,axis=1)), label=r"$\ln(\langle g_{DA}\rangle)$")
    ax.plot(N[:,0], np.log(np.average(gh,axis=1)), label=r"$\ln(\langle g_h\rangle )$")

    ax.set_xlabel('N')
    ax.set_ylabel('${\ln(g)}$')
    ax.autoscale(False)
    ax.plot(N[:,0], -2*gamma*N[:,0],'--', color='black',label='$-2\gamma N$')
    ax.plot(N[:,0], -gamma*N[:,0],'-.', color='black',label='$-\gamma N$')
    ax.plot(N[:,0], -0.5*gamma*N[:,0],':', color='black',label='$-0.5\gamma N$')
    if gamma*N.max() > 1:
        ax.axvline(gamma**(-1), color='gray')
    ax.legend(loc='lower left')
    fig.suptitle("E = 0,   $W$ = {},   $\gamma^{{-1}}$ = {:5}".format(dp, gamma**(-1)))

    fig.savefig(run['fig_name'])
    plt.close()
    


def plot_da(N=800):
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    

    mw = lambda w : models.Model_Anderson_DD_1d(number_of_points=N, bandwidth=1, periodic=False,
                        prng  = np.random.RandomState(2), dis_param=w)
    w = 0.4

    m = mw(w)
    e_space = np.linspace(m.eig_vals.min(), m.eig_vals.max(), 10000)
    ada = [phys_functions.abs_g_diag_approx_of_E(m.eig_vals, 1, e, m.eig_matrix) for e in e_space]
    ax.plot(e_space, ada, label='g_{DA}')
    mga = phys_functions.ga(m.eig_matrix)
    ax.plot(m.eig_vals, N*mga, label='$g_a$')
    ax.set_xlabel('E')
    fig.savefig('plots/pta_da800.png')
    ax.plot(m.eig_vals, N*mga, 'g.',label='$g_a$')

    ax.set_xlim(-0.5,+0.5)
    fig.savefig('plots/pta_da800z.png')
    ax.set_xlim(-0.2,+0.2)
    fig.savefig('plots/pta_da800zz.png')


def plot_special_plot(run):
    options =  {'dispersion_of_N': plot_dispersion_of_N,
                #'dispersion_g'   : plot_dispersion_g}
                'dispersion_g'  : plot_compare_dispersions}
    options.get(run['special_plot'])(run)



def plot_gh_distribution(N=100, dp = 2.0, sds = np.arange(1000)):
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    cbandwidth = 20 #N #20
    tga = np.zeros([len(sds), cbandwidth])
    gg = np.zeros([len(sds)])
    for n, (sd, tg) in enumerate(zip(sds, tga)):
        m = models.Model_Anderson_DD_1d(number_of_points=N, dis_param=dp, 
                    periodic=False, bandwidth=1, prng=np.random.RandomState(sd))
        ## TAKE ONLY CENTER OF BAND:
        crange = np.arange(N//2 -cbandwidth//2, N//2 + cbandwidth//2)
        #crange = (abs(m.eig_vals).copy().argsort())[:4]
        #debug((crange, np.arange(N//2 -cbandwidth//2, N//2 + cbandwidth//2)))
        ga = N*phys_functions.ga(m.eig_matrix[:,[crange]])
        tg[:] = ga
        gg[n] = abs(phys_functions.A_matrix_inv(m.rate_matrix,1,1.57))**2
        #p = s_cummulative_plot(ax, ga)

    mi, ma, theo = phys_functions.ga_cdf(lyap_gamma(1,dp), N)
    ghs = np.logspace(np.log10(mi),np.log10(ma)+1, 1000)
    logg = logavg(tga,axis=0)
    s_cummulative_plot(ax, logg)
    debug((logg.shape, tga.shape))
    p = s_cummulative_plot(ax, tga)
    sh, lo, sc = lognorm.fit(tga.flat)
    mu = np.log(sc)
    stdv = sh
    debug((sh,lo,sc))
    #lg = lognorm()
    xspace = np.logspace(np.log10(tga.min()), np.log10(tga.max()))
    ax.plot(xspace, norm.cdf(np.log(xspace), [stdv],loc=mu, scale=stdv), '-.', color='cyan')
    ax.plot(ghs, theo(ghs) , color='black')    
    #return ghs, theo
    #tga_sum = np.nansum(tga,axis=1)
    tga_avg = np.average(tga, axis=1)
    ax.axvline(logavg(gg))
    ax.axvline(np.average(gg), ls='--')
    ax.axvline(logavg(tga_avg), ls='-', color='red')
    ax.axvline(np.average(tga), ls='-', color='magenta')
    ax.axvline(logavg(tga[tga>1e-100]), ls=':', color='red')
    ax.axvline(np.exp(-2*N*lyap_gamma(1,dp)), ls='--', color='green')
    #ax.axvline(4*(lyap_gamma(1,dp)**2)*N*np.exp(-2*N*lyap_gamma(1,dp)), ls='--', color='green')
    print(2*N*lyap_gamma(1,dp))
    ax.set_xscale('log')
    #ax.set_xlim(1e-10, 10)
    ax.set_ylim(1e-20, 1)
    ax.set_title("W = {}, $\gamma$ = {:.2}, N = {}".format(dp, lyap_gamma(1,dp), N))
    ax.xaxis.set_major_locator(LogLocator(numdecs=10))
    fig.savefig('plots/pta_gh_dist.png')
    #return ghs, theo

def plot_gh_of_x(N=100, dp = 2.0, sds = np.arange(1000)):
    fig, ax  = plt.subplots(figsize=[2*plotdl.latex_width_inch, plotdl.latex_height_inch])
    cbandwidth = 20 #N #20
    tga = np.zeros([len(sds), cbandwidth])
    locs = np.zeros([len(sds), cbandwidth])
    energies = np.zeros([len(sds), cbandwidth])
    for n, (sd, tg, loc, ene) in enumerate(zip(sds, tga, locs, energies)):
        m = models.Model_Anderson_DD_1d(number_of_points=N, dis_param=dp, 
                    periodic=False, bandwidth=1, prng=np.random.RandomState(sd))
        ## TAKE ONLY CENTER OF BAND:
        crange = np.arange(N//2 -cbandwidth//2, N//2 + cbandwidth//2)
        ga = N*phys_functions.ga(m.eig_matrix[:,[crange]])
        lo = sparsedl.mode_center(m.eig_matrix[:,crange])
        order = lo.argsort()
        tg[:] = ga[0][order]
        loc[:] = lo[order]- N//2
        ene[:] = (m.eig_vals[crange])[order]
    
    xr = np.arange(N)-N//2

    ax.plot(locs.flat,tga.flat,'.')
    ax.plot(xr, phys_functions.ander_ga(N, lyap_gamma(1,dp, 0), xr), color='red')
    ax.plot(xr, N*phys_functions.ander_ga(N, lyap_gamma(1,dp, 0), xr), ':',color='red')
    ax.plot(xr, (1/lyap_gamma(1,dp))*phys_functions.ander_ga(N, lyap_gamma(1,dp, 0), xr),'--', color='red')
    ax.set_yscale('log')
    ax.set_title("W = {}, $\gamma$ = {:.2}, N = {}, $\gamma N$ = {:.2}".format(dp, lyap_gamma(1,dp), N, N*lyap_gamma(1,dp) ))
    fig.savefig('plots/pta_gh_of_x_W{}.png'.format(dp))
        
def relevant_alm(mode, dis_param):
    N = mode.size
    x0 = np.abs(mode).argmax()-N/2.0+0.5
    return anderson_localized_mode(N, dis_param, x0)
    
def anderson_localized_mode(N, dis_param, x0):
    gamma = lyap_gamma(1,dis_param)
    
    x = np.linspace(-N/2, N/2)
    y = np.sqrt(gamma)*np.exp(-gamma*np.abs(x-x0))
    return y




def calc_all(open_file):
    f = yaml.load(open_file)
    for run in f:
        calc_a_run(run)

def plot_all(open_file):
    f = yaml.load(open_file)
    
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_yaml','-d', type=file)
    parser.add_argument('--plots_yaml','-p', type=file)
    args = parser.parse_args()
    if args.data_yaml is not None:
        calc_all(args.data_yaml)
    if args.plots_yaml is not None:
        plot_all(args.plots_yaml)
    
if __name__== "__main__":
    main()
