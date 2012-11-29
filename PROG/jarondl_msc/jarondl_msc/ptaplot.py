#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

import itertools
import logging
import os

#from scipy.sparse import linalg as splinalg
from numpy import random, pi, log10, sqrt,  exp, expm1, sort, eye, nanmin, nanmax, log, cos, sinc
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

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# show warnings (remove for production)
logger.setLevel("DEBUG")

info = logger.info
warning = logger.warning
debug = logger.debug







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


#################################################################
###################  Plotting to files (using previous funcs) ###
#################################################################


def plotf_banded_pn(pinning=False):
    """  This plots the two relevant files from the `plot_banded_pn_nopinning`
         function """
    pin = 'pin' if pinning else 'nopin'
    fig, ax = plt.subplots()
    plot_banded_pn(ax, 5, [0.1,0.2,0.4,0.6],pinning=pinning)
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
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
    fig.savefig('pta_thouless_highest_s_log.pdf')
    ax.cla()
    

if __name__ ==  "__main__":

    #print("Not Implemented")
    plotf_banded_pn(pinning=False)
    plotf_banded_pn(pinning=True)
    

