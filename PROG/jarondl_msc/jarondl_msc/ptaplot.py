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
from matplotlib import use
use('cairo.pdf')
from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
import matplotlib as mpl

import ptsplot
import plotdl
import pta_models


mpl.rc('figure', autolayout=True)

### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug







##########################################################
################  Plotting functions  ####################
##########################################################
def plot_banded_pn_nopinning(ax, b, s_values, number_of_sites=1000):
    """ Banded 1d """
    sample = ptsplot.create_bloch_sample_1d(number_of_sites)
    for s in s_values:
        model = pta_models.ExpModel_Banded_Logbox_pinning(sample, epsilon=s, bandwidth1d=b)
        model.plot_PN(ax, label=r"$\sigma={0}$".format(s))

def plotf_banded_pn_nopinning():
    """  This plots the two relevant files from the `plot_banded_pn_nopinning`
         function """
    fig, ax = plt.subplots()
    plot_banded_pn_nopinning(ax, 5, [0.1,0.2,0.4,0.6])
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('pta_low_s.pdf')
    ax.set_xscale('log')
    fig.savefig('pta_low_s_log.pdf')
    ax.cla()
    plot_banded_pn_nopinning(ax, 5, [1,2,3,4])
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('pta_higher_s.pdf')
    ax.cla()
    plot_banded_pn_nopinning(ax, 5, [10,20,30,40])
    plotdl.set_all(ax, xlabel = r"$\lambda$", ylabel = "PN", legend_loc='best')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('pta_highest_s.pdf')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig('pta_highest_s_log.pdf')

if __name__ ==  "__main__":
    print("Not Implemented")
    
