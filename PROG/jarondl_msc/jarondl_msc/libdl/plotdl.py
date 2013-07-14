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
import logging
from functools import wraps

from matplotlib import use
if 'DISPLAY' in os.environ.keys() :
    #use('gtk')
    X_AVAILABLE = True
else:
    X_AVAILABLE = False
    use('cairo')
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import draw, draw_if_interactive
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.transforms import Bbox

from matplotlib.widgets import Slider
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator

import numpy
import numpy as np

from handle_tight import tight_layout

# set up logging
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug

### Raise all float errors
numpy.seterr(all='warn')

## I use some latex info to derive optimial default sizes
#latex_width_pt = 460
#latex_height_pt = latex_width_pt * (numpy.sqrt(5) - 1.0) / 2.0  # Golden mean. idea by:http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
#latex_dpi = 72.27
#latex_width_inch = latex_width_pt / latex_dpi
#latex_height_inch = latex_height_pt / latex_dpi
latex_width_inch = 3.4 ## (aps single column)
latex_height_inch = latex_width_inch * (numpy.sqrt(5)-1.0)/2.0 # golden ratio

#rc('text', usetex=True)
#rc('font', size=10)
rc('figure', figsize=[latex_width_inch, latex_height_inch])
rc('figure.subplot', left=0.2, right=0.9, top=0.9, bottom=0.2)
rc('legend', fontsize='smaller')
rc('xtick', labelsize='smaller')
rc('ytick', labelsize='smaller')



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
    """  OBSOLETE Create a new :class:`matplotlib.Figure` and a new :class:`matplotlib.Axes`.

        :return: A :class:`matplotlib.Axes` instance you can draw on.
        
        This is obsolete. Instead use plt.subplots
    """
    f, ax = plt.subplots()
    return ax

def save_ax(ax, fname, **kwargs):
    """  Save an axis to a pdf file.
    """
    save_fig(ax.get_figure(), fname, **kwargs)

def save_fig(fig, fname, size=[latex_width_inch, latex_height_inch], size_factor=(1, 1),pad=1.2, h_pad=None, w_pad=None, tight=True):
    """ Save figure to pdf and eps
    """

    fig.set_size_inches((size[0] * size_factor[0], size[1] * size_factor[1]))
    if tight:
        tight_layout(fig, pad=pad, h_pad=h_pad, w_pad=w_pad)
    pdfname = os.path.join("figures", fname + ".pdf")
    fig.savefig(pdfname)
    print("Created:\n\t {0} ".format(pdfname))
    

def animate(plot_function, filename, variable_range, **kwargs):
    """ matplotlib can now make animations
    """
    # Create temporary dir:
    tempdir = tempfile.mkdtemp()

    fig, ax = plt.subplots()
    
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
    # set default kwargs:
    kwargs.setdefault("marker",".") # only matters if "marker" doesn't exist yet.
    kwargs.setdefault("linestyle","")
    line_return = ax.plot(values, numpy.linspace(1/N, 1, N), label=label, **kwargs)
    draw_if_interactive()
    return line_return
    
def s_cummulative_plot(ax, values, **kwargs):
    """ sorted cummulative plot """
    svalues = np.sort(values.flat)
    return cummulative_plot(ax, (svalues), **kwargs)

def matshow_cb(ax, matrix, vmin=10**(-10), colorbar=True):
    """
    """
    #vals, vecs = sparsedl.sorted_eigh(matrix)
    ms = ax.matshow(matrix, norm=LogNorm(vmin=vmin ))
    if colorbar:
        ax.figure.colorbar(ms)
        
def plot_several_vectors_dense(fig, vectors, ylabels):
    """  Plots vectors 
    """
    num_of_vectors = len(vectors)
    axes = {} # empty_dict

    for n,(vec,ylabel) in enumerate(zip(vectors,ylabels)):
        if n==0:
            axes[n] = fig.add_subplot(num_of_vectors,1,n+1)
        else:
            axes[n] = fig.add_subplot(num_of_vectors,1,n+1, sharex=axes[0])
        axes[n].plot(vec)
        axes[n].set_ylabel(ylabel)
        axes[n].set_yticks([])
        axes[n].set_xticks([])
    fig.subplots_adjust(hspace=0)

    

def autoscale_based_on(ax, lines):
    """ http://stackoverflow.com/questions/7386872/make-matplotlib-autoscaling-ignore-some-of-the-plots/7396313#7396313 """
    ax.dataLim = Bbox.unit()
    xy = numpy.vstack(lines[0].get_data()).T
    ax.dataLim.update_from_data_xy(xy, ignore=True)
    for line in lines[1:]:
        xy = numpy.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    #print ("limits changed to ", ax.dataLim)
    ax.autoscale_view()

class DictScalarFormatter(ticker.ScalarFormatter):
    """ Same as scalarFormatter, but uses a dictionary for values
    https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/ticker.py
    """
    def __init__(self, vals_dict, **kwargs):
        ticker.ScalarFormatter.__init__(self,**kwargs)
        self.vals_dict = vals_dict
    def __call__(self, x, pos=None):
        val = self.vals_dict.get(x,0)
        s = self.pprint_val(val)
        debug((val,s))
        print((val,s))
        self.fix_minus(s)
        return ticker.ScalarFormatter.__call__(self, self.vals_dict.get(x, 0), pos)
        
        
def draw_if(f):
    """ adds a call to draw_if_interactive after a plotting function """
    @wraps(f)
    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        draw_if_interactive()
    return wrapper
    
##### Dirty hack, should be fixed by matplotlib 1.2.0
def get_LogNLocator(N = 6):
    try:
        return LogLocator(numticks=N)
    except TypeError:
        warning('using undocumented hack for log ticks')
        LogNLocator = LogLocator()
        LogNLocator.numticks = N
        return LogNLocator
        
def close10():
    for n in range(10):
        plt.close()

