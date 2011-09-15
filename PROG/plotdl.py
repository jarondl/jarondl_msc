#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  This module provides my version of the matplotlib api. The main
    purpose is to allow usage of the same plotting functions both for interactive
    and non interactive work modes, in an Object-Oriented way.

    The recommended usage for interactive mode is::

        ax = plotdl.new_ax_for_screen()
        # do something with the ax, e.g. ax.plot(x, y) or call a function to plot on it.
        plotdl.draw()

    The recommended usage for non interactive work is::

        ax = plotdl.new_ax_for_file()
        # do something with the ax, e.g. ax.plot(x, y) or call a function to plot on it.
        plotdl.save_ax(ax, "filename")

    You can also use :func:`set_all` to set multiple plot parameters for a axis,
    for example the title and legend location.
"""
from __future__ import division  # makes true division instead of integer division

import tempfile
import os
import shutil
import subprocess

from matplotlib.backends.backend_agg import FigureCanvasAgg  # For raster rendering, e.g. png
#from matplotlib.backends.backend_cairo import FigureCanvasCairo # For vector rendering e.g. pdf, eps  ###Doesn't have tex
from matplotlib.backends.backend_pdf import FigureCanvasPdf
#from matplotlib.backends.backend_ps import FigureCanvasPS
from matplotlib.figure import Figure
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.transforms import Bbox
try:
    from matplotlib import pyplot
    from matplotlib.pyplot import draw, draw_if_interactive
    X_AVAILABLE = True
    pyplot.ion()
except RuntimeError:
    X_AVAILABLE = False
    print("X is not available, non interactive use only")
    def draw_if_interactive():
        pass

import numpy



### Raise all float errors
numpy.seterr(all='warn')

## I use some latex info to derive optimial default sizes
latex_width_pt = 460
latex_height_pt = latex_width_pt * (numpy.sqrt(5) - 1.0) / 2.0  # Golden mean. idea by:http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
latex_dpi = 72.27
latex_width_inch = latex_width_pt / latex_dpi
latex_height_inch = latex_height_pt / latex_dpi


#rc('text', usetex=True)
rc('font', size=10)
rc('figure', figsize=[latex_width_inch, latex_height_inch])
rc('legend', fontsize=10)


def set_all(ax, title=None, xlabel=None, ylabel=None, legend_loc=False):
    """ Set several attributes for an ax at once

        :param legend_loc: sets the location of the legend. Use "best" for defualt location
    """
    ## REMOVE all titles temporarily for "production"
    #if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend_loc: ax.legend(loc=legend_loc)


def new_ax_for_file():
    """  Create a new :class:`matplotlib.Figure` and a new :class:`matplotlib.Axes`.

        :return: A :class:`matplotlib.Axes` instance you can draw on.
    """
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    return ax

def new_ax_for_screen():
    """  Create a new :class:`matplotlib.Figure` and a new :class:`matplotlib.Axes`.

        :return: A :class:`matplotlib.Axes` instance you can draw on.
    """
    if X_AVAILABLE:
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        return ax
    else:
        print("X is not available")


def save_ax(ax, fname, **kwargs):
    """  Save an axis to a pdf file.
    """
    save_fig(ax.get_figure(), fname, **kwargs)

def save_fig_to_png(fig, fname):
    """ """
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fname + ".png")
    print("Created:\n\t {0} ".format(fname + ".png"))

def save_fig(fig, fname, size=[latex_width_inch, latex_height_inch], size_factor=(1, 1)):
    """ Save figure to pdf and eps
    """

    fig.set_size_inches((size[0] * size_factor[0], size[1] * size_factor[1]))
    canvas_pdf = FigureCanvasPdf(fig)
    #canvas_ps = FigureCanvasPS(fig)
    pdfname = os.path.join("figures", fname + ".pdf")
    #epsname = os.path.join("figures", fname + ".eps")
    canvas_pdf.print_figure(pdfname)
    #canvas_ps.print_figure(epsname)
    print("Created:\n\t {0} ".format(pdfname))


def animate(plot_function, filename, variable_range, **kwargs):
    """
    """
    # Create temporary dir:
    tempdir = tempfile.mkdtemp()

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    for num, var in enumerate(variable_range):
        plot_function(ax, var, num=num, **kwargs)
        tempname = os.path.join(tempdir, "img{0:04}".format(num))
        save_fig_to_png(fig, tempname)
        ax.clear()

    # make a movie
    command = ("mencoder", "mf://{0}/img*.png".format(tempdir), "-ovc", "lavc", "-speed", "0.2", "-o", filename + ".mpg")
    try:
        subprocess.check_call(command)
    except OSError:
        print( "Movie creation failed. Make sure you have mencoder installed")

    shutil.rmtree(tempdir)

def cummulative_plot(ax, values, label=None, **kwargs):
    """  Plot cummulative values.
    """
    N = len(values)
    #ax.plot(numpy.sort(values), numpy.linspace(1/N, 1, N), marker=".", linestyle='', label=label, **kwargs)
    line_return = ax.plot(values, numpy.linspace(1/N, 1, N), marker=".", linestyle='', label=label, **kwargs)
    draw_if_interactive()
    return line_return

def matshow_cb(ax, matrix, vmin=10**(-10), colorbar=True):
    """
    """
    #vals, vecs = sparsedl.sorted_eigh(matrix)
    ms = ax.matshow(matrix, norm=LogNorm(vmin=vmin ))
    if colorbar:
        ax.figure.colorbar(ms)


def autoscale_based_on(ax, lines):
    """ http://stackoverflow.com/questions/7386872/make-matplotlib-autoscaling-ignore-some-of-the-plots/7396313#7396313 """
    ax.dataLim = Bbox.unit()
    xy = numpy.vstack(lines[0].get_data()).T
    ax.dataLim.update_from_data_xy(xy, ignore=True)
    for line in lines[1:]:
        xy = numpy.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    print ("limits changed to ", ax.dataLim)
    ax.autoscale_view()
