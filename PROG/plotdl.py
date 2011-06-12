#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several plot functions.
"""
from __future__ import division  # makes true division instead of integer division
from matplotlib.backends.backend_agg import FigureCanvasAgg  # For raster rendering, e.g. png
#from matplotlib.backends.backend_cairo import FigureCanvasCairo # For vector rendering e.g. pdf, eps  ###Doesn't have tex
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.backends.backend_ps import FigureCanvasPS
from matplotlib.figure import Figure
from matplotlib import rc
try:
    from matplotlib import pyplot
    X_AVAILABLE = True
except RuntimeError:
    X_AVAILABLE = False
    print("X is not available, non interactive use only")

import numpy
import os


## I use some latex info to derive optimial default sizes
latex_width_pt = 460
latex_height_pt = latex_width_pt * (numpy.sqrt(5) - 1.0) / 2.0  # Golden mean. idea by:http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
latex_dpi = 72.27
latex_width_inch = latex_width_pt / latex_dpi
latex_height_inch = latex_height_pt / latex_dpi


rc('text', usetex=True)
rc('font', size=10)
rc('figure', figsize=[latex_width_inch, latex_height_inch])
rc('legend', fontsize=10)


def set_all(ax, title=None, xlabel=None, ylabel=None, legend_loc=False):
    """ Set several attributes for an ax at once
        :param:`legend_loc` sets the location of the legend. Use "best" for defualt location
    """
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend_loc: ax.legend(loc=legend_loc)


def plot_to_file(plot_func, filename, **kwargs):
    """
    """
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_func(ax, **kwargs)
    savefig(fig, filename)
    return fig


def plot_2subplots_to_file(plot_func1, plot_func2, filename, suptitle=None, **kwargs):
    """
    """
    fig = Figure()
    if suptitle:
        fig.suptitle(suptitle)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot_func1(ax1, **kwargs)
    plot_func2(ax2, **kwargs)
    savefig(fig, filename)
    return fig


def plot_twin_subplots_to_file(plot_func_twin, filename, suptitle=None, **kwargs):
    """  plot_func_twin should accept two axes to draw on.
    """
    fig = Figure()
    if suptitle:
        fig.suptitle(suptitle)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    savefig(fig, filename)
    return fig


def plot_to_screen(plot_func, **kwargs):
    """
    """
    if X_AVAILABLE:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot_func(ax, **kwargs)
        #pyplot.show()
    else:
        print("X is not available")


def savefig(fig, fname, size=[latex_width_inch, latex_height_inch], size_factor=(1, 1)):
    """ Save figure to pdf and eps
    """

    fig.set_size_inches((size[0] * size_factor[0], size[1] * size_factor[1]))
    canvas_pdf = FigureCanvasPdf(fig)
    canvas_ps = FigureCanvasPS(fig)
    pdfname = os.path.join("figures", fname + ".pdf")
    epsname = os.path.join("figures", fname + ".eps")
    canvas_pdf.print_figure(pdfname)
    canvas_ps.print_figure(epsname)
    print("Created:\n\t {0} \n\t {1}".format(pdfname, epsname))
