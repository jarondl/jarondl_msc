#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several plot functions.
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg  
from matplotlib.figure import Figure
from matplotlib import rc
import numpy


## I use some latex info to derive optimial default sizes
latex_width_pt = 460
latex_height_pt = latex_width_pt * (numpy.sqrt(5)-1.0)/2.0  #Golden mean. idea by:http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
latex_dpi = 72.27
latex_width_inch = latex_width_pt/latex_dpi
latex_height_inch = latex_height_pt/latex_dpi

params = {'axes.labelsize': 10, 'text.fontsize': 10,
          'legend.fontsize': 10,
          'text.usetex': True, 'figure.figsize': [latex_width_inch,latex_height_inch]}
rc('text', usetex=True, fontsize=10)
rc('figure', figsize=[latex_width_inch,latex_height_inch])
rc('legend', fontsize=10)


def wexp_plot():
    fig = Figure()
    fig.subplots_adjust(hspace=0.5)
    ax_b3 = fig.add_subplot(2,2,1)
    ax_b3_log = fig.add_subplot(2,2,3)
    ax_b09 = fig.add_subplot(2,2,2)
    ax_b09_log = fig.add_subplot(2,2,4)
    
    for ax in [ax_b3, ax_b3_log, ax_b09, ax_b09_log]:
        ax.set_xlabel(r"$w$")   
        ax.set_ylabel(r"$w^{\beta-1}e^{-w}$") 
    
    w = numpy.linspace(0.001,5,100)
    beta = 3
    y_b3 = numpy.multiply(pow(w,beta-1)  , numpy.exp(-w) )

    ax_b3.plot(w,y_b3)
    ax_b3.set_title( r"$\beta = {0}$".format(beta))
    ax_b3_log.semilogx(w,y_b3)
    ax_b3_log.set_title( r"$\beta = {0}$".format(beta))

    beta = 0.9
    y_b09 = numpy.multiply(pow(w,beta-1)  , numpy.exp(-w) )

    ax_b09.plot(w,y_b09)
    ax_b09.set_title( r"$\beta = {0}$".format(beta))
    ax_b09_log.semilogx(w,y_b09)
    ax_b09_log.set_title( r"$\beta = {0}$".format(beta))
    
    savefig(fig, "expw.png")


def savefig(fig, fname, size=[latex_width_inch,latex_height_inch]):
    fig.set_size_inches(size)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fname+".pdf")
    canvas.print_figure(fname+".eps")
    #canvas.print_figure(fname+".png",dpi=100)


