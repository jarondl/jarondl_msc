#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several plot functions.
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg  
from matplotlib.figure import Figure
from matplotlib import rc
import numpy
rc('text', usetex=True)

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


def savefig(fig, fname, size=[6,4]):
    fig.set_size_inches(size)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fname+".pdf")
    canvas.print_figure(fname+".eps")
    #canvas.print_figure(fname+".png",dpi=100)


