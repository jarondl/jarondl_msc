#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several plot functions.
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg  # For raster rendering, e.g. png
#from matplotlib.backends.backend_cairo import FigureCanvasCairo # For vector rendering e.g. pdf, eps  ###Doesn't have tex
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.backends.backend_ps import FigureCanvasPS
from matplotlib.figure import Figure
from matplotlib import rc
import numpy
import os


## I use some latex info to derive optimial default sizes
latex_width_pt = 460
latex_height_pt = latex_width_pt * (numpy.sqrt(5)-1.0)/2.0  #Golden mean. idea by:http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
latex_dpi = 72.27
latex_width_inch = latex_width_pt/latex_dpi
latex_height_inch = latex_height_pt/latex_dpi


rc('text', usetex=True)
rc('font', size=10)
rc('figure', figsize=[latex_width_inch,latex_height_inch])
rc('legend', fontsize=10)


def set_all(ax, title=None, xlabel=None, ylabel=None, legend=False):
    """ Set several attributes for an ax at once
    """
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend: ax.legend()


def savefig(fig, fname, size=[latex_width_inch,latex_height_inch]):
    """ Save figure to pdf and eps 
    """
    fig.set_size_inches(size)
    canvas_pdf = FigureCanvasPdf(fig)
    canvas_ps = FigureCanvasPS(fig)
    pdfname = os.path.join("figures" , fname + ".pdf")
    epsname = os.path.join("figures" , fname + ".eps")
    canvas_pdf.print_figure(pdfname)
    canvas_ps.print_figure(epsname)
    print("Created:\n\t {0} \n\t {1}".format(pdfname,epsname))
