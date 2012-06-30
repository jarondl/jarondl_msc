#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division

import itertools
import logging

import pylab 
from matplotlib import rc
import numpy as np
import scipy as sp

#import sparsedl
#import plotdl
#from geometry import Sample
#from sparsedl import sorted_eigvalsh, banded_ones, periodic_banded_ones, zero_sum, lazyprop, omega_d
#from plotdl import cummulative_plot

latex_width_inch = 3.4 ## (aps single column)
latex_height_inch = latex_width_inch * (np.sqrt(5)-1.0)/2.0 # golden ratio
rc('figure', figsize=[latex_width_inch*2, latex_height_inch*np.sqrt(3)])
rc('legend', fontsize='smaller')
rc('xtick', labelsize='smaller')
rc('ytick', labelsize='smaller')


### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug


#RMS = lambda sig : np.sqrt( -np.expm1(-2*sig)/sig)
RMS = lambda sig : np.sqrt( -np.expm1(-2*sig)/(2*sig))
gs_ERH = lambda nc : lambda sig, b : ((1+nc/(2*b))*np.exp(-nc*sig/(2*b)) - np.exp(-2*sig))/(-np.expm1(-2*sig))
gs_ERH2 = gs_ERH(2)


def read_data():
    return np.genfromtxt("data.dat")

def time_rescaler(sig_in):
    sig = np.asarray(sig_in, dtype=np.float64)
    #powers = np.array([0,0,-1,0,-2,1,-1,-1])
    powers = np.array([0,0,-1,0,-2,-1,-1,-1])
    pg, sg = np.meshgrid(powers, RMS(sig))
    return sg**(pg)


def plot_D():
    f = read_data()
    #f = f

    sig = f[:,1]
    D1_scaled = f[:,2] / ( RMS( f[:,1] ) * f[:,0]**2.5 )
    for (n, (marker, mec)) in zip((0,5,10) , (('o','b'), ('s','r'), ('D', 'g'))) :

        pylab.plot(sig[n:n+5] ,  D1_scaled[n:n+5], ' ', mfc='none', marker=marker, mec=mec, label=" b = {0}".format(f[n,0]))
        #pylab.plot(sig[n:n+5] ,  gs_ERH(2)(sig[n:n+5], f[n,0]), '--', color= mec)
        #pylab.plot((nf[n:n+5, 1]),  nf[n:n+5,6]/ nf[n:n+5,0]**2.5, ls =ls)
    pylab.xlim(0,6)
    pylab.xlabel(r"$\sigma$")
    pylab.ylabel("$D_1$ - scaled")
    pylab.legend()


def plot_D2():
    f = read_data()
    #f = f
    sig = f[:,1]
    D2_scaled = f[:,6] / ( RMS( f[:,1] ) * f[:,0]**2.5 )
    for (n, (marker, mec)) in zip((0,5,10) , (('o','b'), ('s','r'), ('D', 'g'))) :

        pylab.plot(sig[n:n+5] ,  D2_scaled[n:n+5], ' ', mfc='none', marker=marker, mec=mec, label=" b = {0}".format(f[n,0]))
        #pylab.plot(sig[n:n+5] ,  gs_ERH(2)(sig[n:n+5], f[n,0]), '--', color= mec)
        #pylab.plot((nf[n:n+5, 1]),  nf[n:n+5,6]/ nf[n:n+5,0]**2.5, ls =ls)
    pylab.xlim(0,6)
    pylab.xlabel(r"$\sigma$")
    pylab.ylabel("$D_2$ - scaled)")
    pylab.legend()

def plot_D2_vs_gs():
    f = read_data()
    #f = f
    sig = f[:,1]
    D2_scaled = f[:,6] / ( RMS( f[:,1] ) * f[:,0]**2.5 )
    for (n, (marker, mec)) in zip((0,5,10) , (('o','b'), ('s','r'), ('D', 'g'))) :

        pylab.plot(gs_ERH(2)(sig[n:n+5], f[n,0]) ,  D2_scaled[n:n+5], ' ', mfc='none', marker=marker, mec=mec, label=" b = {0}".format(f[n,0]))
        #pylab.plot(sig[n:n+5] ,  gs_ERH(2)(sig[n:n+5], f[n,0]), '--', color= mec)
        #pylab.plot((nf[n:n+5, 1]),  nf[n:n+5,6]/ nf[n:n+5,0]**2.5, ls =ls)
    #pylab.xlim(0,6)
    xspace1 = np.linspace(0.65,0.79)
    pylab.plot( xspace1, xspace1, "k--")
    xspace2 = np.linspace(0.65,0.9)
    pylab.plot( xspace2, 0.5*xspace2+0.33, ":", color='0.5')
    pylab.xlabel(r"$g_s(\sigma)$")
    pylab.ylabel("$D_2$ - scaled")
    pylab.legend(loc="best")


def plot_v2():
    f = read_data()
    sig = f[:,1]
    V2_scaled = f[:,4] / ( RMS( f[:,1] )**2 * f[:,0]**3 )

    for (n, (marker, mec)) in zip((0,5,10) , (('o','b'), ('s','r'), ('D', 'g'))) :

        pylab.plot(sig[n:n+5] ,  V2_scaled[n:n+5], ' ', mfc='none', marker=marker, mec=mec, label=" b = {0}".format(f[n,0]))


    pylab.xlim(0,6)
    pylab.xlabel(r"$\sigma$")
    pylab.ylabel("$V^2$ - scaled")
    pylab.legend()



def plot_tball():
    f = read_data()
    sig = f[:,1]
    tball_scaled = f[:,5] * RMS( f[:,1] ) * np.sqrt(f[:,0])

    for (n, (marker, mec)) in zip((0,5,10) , (('o','b'), ('s','r'), ('D', 'g'))) :

        pylab.plot(sig[n:n+5] ,  tball_scaled[n:n+5], ' ', mfc='none', marker=marker, mec=mec, label=" b = {0}".format(f[n,0]))
    pylab.xlim(0,6)
    pylab.xlabel(r"$\sigma$")
    pylab.ylabel(r"$t_{bal}$ - scaled")
    pylab.legend()


def plots_for_dtstex():
    pylab.close()
    pylab.clf()
    plot_D()
    pylab.savefig("new/D1.pdf")
    pylab.clf()
    plot_D2()
    pylab.savefig("new/D2.pdf")
    pylab.clf()
    plot_D2_vs_gs()
    pylab.savefig("new/D2_vs_gs.pdf")
    pylab.clf()
    plot_v2()
    pylab.savefig("new/v_square.pdf")
    pylab.clf()
    plot_tball()
    pylab.savefig("new/tball.pdf")
    pylab.clf()
    pylab.close()



