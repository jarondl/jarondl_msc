#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
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
from plotdl import plt, tight_layout


import numpy as np
import scipy as sp
import matplotlib as mpl

import ptsplot

import pta_models


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



##### Dirty hack, should be fixed by matplotlib 1.2.0
def get_LogNLocator(N = 6):
    try:
        return LogLocator(numticks=6)
    except TypeError:
        warning('using undocumented hack for log ticks')
        Log6Locator = LogLocator()
        Log6Locator.numticks = 6
        return Log6Locator



def theor_banded_ev(b,N):
    k = 2*pi*np.arange(N)/N
    n = np.arange(1,b+1)
    km,nm = np.meshgrid(k,n)
    return -2*(np.cos(km*nm)-1).sum(axis=0)




##########################################################
################  Plotting functions (on axes) ###########
##########################################################
def plot_banded_pn(ax, b, s_values, number_of_sites=1000, pinning=False):
    """ Banded 1d """
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    for s in s_values:
        if pinning:
            model = pta_models.ExpModel_Banded_Logbox_pinning(sample, epsilon=s, bandwidth1d=b)
        else:
            model = pta_models.ExpModel_Banded_Logbox(sample, epsilon=s, bandwidth1d=b)
        model.plot_PN(ax, label=r"$\sigma={0}$".format(s))


def get_ev_thoules_g(b_space, s_space, number_of_sites = 1000, phi = pi):
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    win_avg_mtrx = sparsedl.window_avg_mtrx(number_of_sites - 1)
    res_type =  np.dtype([("ev",(np.float64,number_of_sites)),
                          ("PN",(np.float64,number_of_sites)),
                          ("thoules_g",(np.float64,number_of_sites)),
                          ("s",np.float64),
                          ("b",np.float64)
                          ])
    
    s_grid, b_grid = np.meshgrid(np.asarray(s_space), np.asarray(b_space))
    
    res = np.zeros(s_grid.size, dtype = res_type) # preallocation is faster..
    for n,(s,b) in enumerate(zip(s_grid.flat, b_grid.flat)):
        debug('eigenvals ,PN and thouless for s = {0}, b= {1}, n={2}'.format(s,b,n))
        model = pta_models.ExpModel_Banded_Logbox(sample, epsilon=s, bandwidth1d=b, rseed=n)
        model_phi = pta_models.ExpModel_Banded_Logbox_phase(sample, epsilon=s, bandwidth1d=b,rseed=n,phi=phi)
        g = abs(model.eigvals - model_phi.eigvals) / (phi**2)
        # Approximation of  the minimal precision:
        prec = FLOAT_EPS * max(abs(model.eigvals))* number_of_sites  
        debug("precision = {0}, minimal g  = {1}".format(prec, min(g)))
        #g = ma.masked_less(g,prec)
        avg_spacing = win_avg_mtrx.dot(-model.eigvals[1:]+model.eigvals[:-1])
        # avg_spacing is now smaller than eigvals. We duplicate the last value to accomodate (quite  hackish)
        avg_spacing = np.append(avg_spacing,avg_spacing[-1])
        ga = ma.masked_less(g/avg_spacing,prec)
        res[n] = ( model.eigvals, model.PN_N, ma.filled(ga, fill_value=-0) , s, b)
        
    return res

def get_ev_for_phases(b,s,phases,number_of_sites=1000, rseed=1):
    """ return the complete eigenvalue list for every phase """
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)

    evs = np.zeros([len(phases), number_of_sites], dtype=np.float64)
    for n,phase in enumerate(phases):
        debug('calculating eigenvalues for phase number {0}  : {1}'.format(n,phase))
        model = pta_models.ExpModel_Banded_Logbox_phase(sample,
                    epsilon=s, bandwidth1d=b,rseed=rseed,phi=phase)
        evs[n] = -model.eigvals
    return evs
    
    
def plot_scatter_g(ax):
    g = get_ev_thoules_g([5],[0.2,1,10],phi=pi)
    log_g = -log(g['thoules_g'])
    log_IPN = -log(g['PN'])
    ax.plot(log_g[0], log_IPN[0], '.', label=r"$\sigma = 0.2 ({0})$".format(np.isinf(log_g[0]).sum()))
    ax.plot(log_g[1], log_IPN[1], '.', label=r"$\sigma = 1   ({0})$".format(np.isinf(log_g[1]).sum()))
    ax.plot(log_g[2], log_IPN[2], '.', label=r"$\sigma = 10  ({0})$".format(np.isinf(log_g[2]).sum()))
    ax.legend(loc='lower right')
    ax.set_xlabel(r"$-\log(g)$")
    ax.set_ylabel(r"$-\log(PN)$")

def plot_thoules_g(ax, b, s_values, number_of_sites=1000,phi=0.1):
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    win_avg_mtrx = sparsedl.window_avg_mtrx(number_of_sites - 2)
    for s in s_values:
        model = pta_models.ExpModel_Banded_Logbox(sample, epsilon=s, bandwidth1d=b, rseed=1)
        #model_pi = pta_models.ExpModel_Banded_Logbox_pi(sample, epsilon=s, bandwidth1d=b)
        model_phi = pta_models.ExpModel_Banded_Logbox_phase(sample, epsilon=s, bandwidth1d=b,rseed=1,phi=phi)
        g = abs(model.eigvals[1:] - model_phi.eigvals[1:]) / (phi**2)
        avg_spacing = win_avg_mtrx.dot(-model.eigvals[2:]+model.eigvals[1:-1])
        ax.plot(-model.eigvals[1:-1], g[:-1]/avg_spacing, '.', markersize=7,label=r"$\sigma={0}$".format(s))
    ax.legend(loc='best')
    ax.set(xlabel=r"$\lambda$", ylabel=r"$g$", yscale = 'log')
    ax.yaxis.set_major_locator(get_LogNLocator())


def plot_represntive_vectors(fig_vecs, ax_PN, number_of_sites=1000):
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    b = 5
    s = 0.1
    rseed = 1
    mob0 = pta_models.ExpModel_Banded_Logbox_phase(sample,
                    epsilon=s, bandwidth1d=b,rseed=rseed,phi=0)
    mobpi = pta_models.ExpModel_Banded_Logbox_phase(sample,
                    epsilon=s, bandwidth1d=b,rseed=rseed,phi=pi)
    ev0 = -mob0.eigvals
    evpi = -mobpi.eigvals
    g = abs(ev0-evpi)
    ax_PN.plot(mob0.PN_N,".")
    rep_vecs = [100,200,500,800,900,995]
    ylabels = map(lambda n: r"$\lambda = {ev}  PN = {PN}  g = {g}$".format(ev=ev0[n], PN = mob0.PN_N[n],
                                                                               g = g[n]), rep_vecs)
    for vec_n in rep_vecs:
        ax_PN.vlines(vec_n, 0,1000)
        
    plotdl.plot_several_vectors_dense(fig_vecs, mob0.eig_matrix[:,rep_vecs], ylabels)
    


#################################################################
###################  Plotting to files (using previous funcs) ###
#################################################################

def plotf_scatter_g():
    fig, ax = plt.subplots()
    plot_scatter_g(ax)
    tight_layout(fig)
    fig.savefig('pta_scatter_g.pdf')

def plotf_banded_pn(pinning=False):
    """  This plots the two relevant files from the `plot_banded_pn_nopinning`
         function """
    pin = 'pin' if pinning else 'nopin'
    fig, ax = plt.subplots()
    plot_banded_pn(ax, 5, [0.1,0.2,0.4,0.6],pinning=pinning)
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(get_LogNLocator())
    tight_layout(fig)
    fig.savefig('pta_low_s_{}.pdf'.format(pin))
    ax.set_xscale('log')
    ax.set_xlim([1e-3,2e1]) ## That is to allow comparison between pin an nopin
    fig.savefig('pta_low_s_log_{}.pdf'.format(pin))
    ax.cla()
    plot_banded_pn(ax, 5, [1,2,3,4],pinning=pinning)
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
    tight_layout(fig)
    fig.savefig('pta_higher_s_{}.pdf'.format(pin))
    ax.cla()
    plot_banded_pn(ax, 5, [10,20,30,40],pinning=pinning)
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
    tight_layout(fig)
    fig.savefig('pta_highest_s_{}.pdf'.format(pin))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(get_LogNLocator())
    tight_layout(fig)
    fig.savefig('pta_highest_s_log_{}.pdf'.format(pin))

def plotf_thouless_g():
    fig, ax = plt.subplots()
    plot_thoules_g(ax, 5, [0.1,0.2,0.4,0.6], phi=0.01)
    tight_layout(fig)
    fig.savefig('pta_thouless_low_s.pdf')
    ax.cla()
    
    plot_thoules_g(ax, 5, [1,2,3,4], phi=0.01)
    tight_layout(fig)
    fig.savefig('pta_thouless_higher_s.pdf')
    ax.cla()
    
    plot_thoules_g(ax, 5, [10,20,30,40], phi=0.01)
    tight_layout(fig)
    fig.savefig('pta_thouless_highest_s.pdf')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(get_LogNLocator())
    fig.savefig('pta_thouless_highest_s_log.pdf')
    ax.cla()
    

if __name__ ==  "__main__":

    #print("Not Implemented")
    plotf_banded_pn(pinning=False)
    plotf_banded_pn(pinning=True)
    plotf_thouless_g()
    

