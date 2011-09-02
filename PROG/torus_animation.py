#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
#from scipy.sparse import linalg as splinalg
from numpy import linalg, random, pi, log10, sqrt
from matplotlib.colors import LogNorm
from matplotlib.cm import summer
from matplotlib.widgets import Slider
from copy import deepcopy

import numpy

import sparsedl
import plotdl
import geometry
from geometry import Sample
from eigenvalue_plots import eigenvalues_cummulative
from plotdl import cummulative_plot

### Raise all float errors 
numpy.seterr(all='warn')
EXP_MAX_NEG = numpy.log(numpy.finfo( numpy.float).tiny)




#########  Torus animation #####
def torus_show_state(ax, time, torus ,xi=1):
    """
    """
    N = torus.number_of_points
    # Create initial condition rho
    rho0 = numpy.zeros(N)
    rho0[0] = 1
        
    # Create rate matrix W
    dis =  geometry.fast_periodic_distance_matrix(torus.points, torus.dimensions)
    W = numpy.exp(-dis/xi)
    sparsedl.zero_sum(W)

    # 
    rho = sparsedl.rho(time, rho0, W) 
    ax.scatter( torus.xpoints, torus.ypoints, c=rho)
    

def torus_plot_rho(ax, rho, torus, colorbar=False ):
    """
    """
    sct = ax.scatter(torus.xpoints, torus.ypoints, edgecolors='none',
            c=rho, norm=LogNorm( vmin=(1/torus.number_of_points)/1000, vmax =1))
    
    if colorbar:
        ax.get_figure().colorbar(sct)

def replot_rho_factory(ax, rhos, torus, eigvals):
    """
    """
    def replot_rho(slider_position):
        ax.clear()
        pos = int(slider_position)
        eigval = eigvals[pos]
        participation_number = ((rhos[:,pos]**2).sum(axis = 0))**(-1)
        torus_plot_rho(ax, rhos[:,pos], torus, colorbar=False)
        ax.set_title(r"\#${0}, \lambda={1}, PN = {2}$".format(pos, eigval, participation_number))
        plotdl.draw()
    return replot_rho

def torus_rhos_slider(fig,rhos, torus, eigvals):
    """
    """
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    torus_plot_rho(ax, rhos[:,0], torus, colorbar=True)
    replot_rho = replot_rho_factory(ax, rhos, torus, eigvals)
    axsl = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    sl = Slider(axsl, "eigenmode", 0,rhos.shape[1],0)
    sl.on_changed(replot_rho)


def torus_list_of_rhos(torus, times, xi=1):
    """
    """
    N = torus.number_of_points
    rho0 = numpy.zeros(N)
    rho0[0] = 1
        
    # Create rate matrix W
    dis =  geometry.fast_periodic_distance_matrix(torus.points, torus.dimensions)
    W = numpy.exp(-dis/xi)
    sparsedl.zero_sum(W)

    # 
    rhos = []
    for t in times:
        rhos += [sparsedl.rho(t, rho0, W)]
    return rhos

def torus_time():
    """
    """
    times = numpy.linspace(0,1,100)
    torus = Sample((10,10),100)
    rhos = torus_list_of_rhos(torus, times)
    plotdl.animate(torus_plot_rho, "test", rhos, torus=torus)




if __name__ ==  "__main__":
    all_plots()
