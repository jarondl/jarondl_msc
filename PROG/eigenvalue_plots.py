#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  Plotting eigenvalues for different systems.
     The first argument is always a matplotlib axes instance, to plot on.
"""
from __future__ import division

import numpy
from numpy import pi, sqrt, linalg, random

import sparsedl
import plotdl


### Raise all float errors 
numpy.seterr(all='warn')

def eigenvalues_lognormal(ax, number_of_points=200, band_width_list=(1, ),
            **kwargs):
    """  Plot the eigenvalues for a lognormal sparse banded matrix
    """
    for band_width in band_width_list:
        rates = sparsedl.lognormal_construction(number_of_points * band_width, **kwargs)
        rate_matrix     = sparsedl.create_sparse_matrix(number_of_points, rates, band_width).todense()
        diff_coef = sparsedl.resnet(rate_matrix, band_width)
        eigvals = eigenvalues_cummulative(ax, rate_matrix,
                label = "b = {0}, D = {1}".format(band_width, diff_coef))

        diffusion_plot(ax, diff_coef, eigvals)
    plotdl.set_all(ax, title="lognormal, N={N}".format(N=number_of_points),
            legend_loc="upper left")


def eigenvalues_ones(ax, number_of_points=200, band_width_list=(1, )):
    """  Plot the eigenvalues for a banded matrix with ones only, for each b.
    """
    for band_width in band_width_list:
        rates = numpy.ones(number_of_points*band_width)
        rate_matrix = sparsedl.create_sparse_matrix(number_of_points, rates, band_width).todense()
        if not sparsedl.zero_sum(rate_matrix):
            raise Exception("The rate_matrix matrix is not zero summed. How did you make it?")
        diff_coef = sparsedl.resnet(rate_matrix, band_width)
        label = "b = {0}, D = {1}".format(band_width, diff_coef)
        eigvals = eigenvalues_cummulative(ax, rate_matrix, label)
        diffusion_plot(ax, diff_coef, eigvals)
        #ones_analytic_plot(ax, number_of_points)
    #ax.set_xscale('linear')
    #ax.set_yscale('linear')
    plotdl.set_all(ax, title="All ones, N = {N}".format(N=number_of_points),
            legend_loc="best")


def eigenvalues_alter(ax, number_of_points=200, lower_w = 3, higher_w = 8,
        band_width_list=(1, )):
    """  Plot the eigenvalues for a banded matrix with alternating 3 and 8, for each b.
    """
    for band_width in band_width_list:
        rates = numpy.zeros(number_of_points*band_width)
        rates[::2] = lower_w
        rates[1::2] = higher_w
        rate_matrix = sparsedl.create_sparse_matrix(number_of_points, rates, band_width).todense()
        if not sparsedl.zero_sum(rate_matrix):
            raise Exception("The rate_matrix matrix is not zero summed. How did you make it?")
        diff_coef = sparsedl.resnet(rate_matrix, band_width)
        label = "b = {0}, D = {1}".format(band_width, diff_coef)
        eigvals = eigenvalues_cummulative(ax, rate_matrix, label)
        diffusion_plot(ax, diff_coef, eigvals)
    alter_analytic_plot(ax, lower_w, higher_w, number_of_points)
    plotdl.set_all(ax, legend_loc="best",
            title="Alternating {l}-{h}, N = {N}".format(N=number_of_points, l=lower_w, h=higher_w))


def eigenvalues_box(ax, number_of_points=200, lower_w = 3, higher_w = 8, band_width_list=(1, )):
    """  Plot the eigenvalues for a banded matrix with alternating 3 and 8, for each b.
    """
    for band_width in band_width_list:
        rates = numpy.random.uniform(lower_w, higher_w, number_of_points*band_width)
        rate_matrix = sparsedl.create_sparse_matrix(number_of_points, rates, band_width).todense()
        if not sparsedl.zero_sum(rate_matrix):
            raise Exception("The rate_matrix matrix is not zero summed. How did you make it?")
        diff_coef = sparsedl.resnet(rate_matrix, band_width)
        label = "b = {0}, D = {1}".format(band_width, diff_coef)
        eigvals = eigenvalues_cummulative(ax, rate_matrix, label)
        diffusion_plot(ax, diff_coef, eigvals)
    plotdl.set_all(ax, title="Box distibution {l}-{h}, N = {N}".format(N=number_of_points, l=lower_w, h=higher_w), legend_loc="best")


def eigenvalues_uniform(ax, number_of_points=100):
    """  Plot the eigenvalues for a uniform random matrix
    """
    rate_matrix = numpy.random.uniform(-1, 1, number_of_points**2).reshape([number_of_points, number_of_points])
    eigvals = linalg.eigvalsh(rate_matrix)  #  eigvalsh works for real symmetric matrices
    eigvals.sort()
    ax.plot(eigvals, numpy.linspace(0, number_of_points, number_of_points),
            label="Cummulative eigenvalue distribution", marker='.', linestyle='')

    radius = numpy.max(eigvals) 
    #R=2.0
    semicircle = numpy.sqrt(numpy.ones(number_of_points)*radius**2 - numpy.linspace(-radius, radius, number_of_points)**2)#/(2*pi)
    cum_semicircle = numpy.cumsum(semicircle)
    cum_semicircle = cum_semicircle / numpy.max(cum_semicircle)*number_of_points
    ax.plot(numpy.linspace(-radius, radius, number_of_points),
            semicircle, linestyle="--",
            label=r"Semi circle, with $R \approx {0:.2}$".format(radius))
    ax.plot(numpy.linspace(-radius, radius, number_of_points), cum_semicircle,
            linestyle="--",
            label = r"Cummulative semicircle, with $R \approx {0:.2}$".format(radius))

    plotdl.set_all(ax, title=r"uniform, $[-1, 1]$", legend_loc="upper left")


###############  Meta-eigenvalue #########
def eigenvalues_cummulative(ax, matrix, label):
    """  Plot the cummulative density of the eigenvalues
    """
    number_of_points = matrix.shape[0]
    eigvals = -linalg.eigvalsh(matrix)
    eigvals.sort()
    eigvals = eigvals[1:]  ## The zero (or nearly zero) is a problematic eigenvalue.
    assert  eigvals[0] > 0, ("All eigenvalues [except the first] should be positive" + str(eigvals))
    ax.loglog(eigvals, numpy.linspace(1/(number_of_points-1), 1, number_of_points-1), marker=".", linestyle='', label=label)
    return eigvals
    


################ Plots related to the eigenvalue plots ############3

def diffusion_plot(ax, diff_coef, eigvals):
    """ """
    max_log_value = numpy.log10(numpy.min((numpy.max(eigvals), diff_coef*pi**2)))
    diffusion_space = numpy.logspace(numpy.log10(numpy.min(eigvals)), max_log_value, 100)
    diffusion = numpy.sqrt(diffusion_space/(diff_coef))/pi
    ax.loglog(diffusion_space, diffusion, linestyle='--', label="")


def alter_analytic_plot(ax, a, band_width, number_of_points):
    """
    """
    space = numpy.linspace(1/number_of_points, 0.5, number_of_points // 2 )  # removed -1
    alter = sparsedl.analytic_alter(a, band_width, space)
    alter.sort()
    ax.loglog(alter, space, linestyle='', marker='+', label = r"Analytic alternating model")


def ones_analytic_plot(ax, number_of_points):
    """
    """
    space = numpy.linspace(1/number_of_points, 1, number_of_points)
    analytic1 = 2*(1-numpy.cos(pi*space))/ number_of_points
    approx_space = numpy.linspace(0, 1/number_of_points)
    analytic2 = numpy.arccos(1-number_of_points*approx_space/2)/pi
    approx = sqrt(number_of_points*approx_space)/pi
    #analytic.sort()
    ax.loglog(analytic1, space, linestyle='', marker='+', label = r"Analytic : $2(1-\cos(\pi n))$")
    ax.plot(approx_space, analytic2, linestyle='--', label=r"Analytic, $\cos^{-1}$")
    ax.plot(approx_space, approx, linestyle='', marker = '+', label=r"Approximation, $\sqrt{N*n}/\pi$")

def all_plots(seed=1):
    """ Plot all of the eigenvalue plots """
    ax = plotdl.new_ax_for_file()

    random.seed(seed)
    eigenvalues_lognormal(ax, band_width_list=(1, 5, 10, 20, 40))
    plotdl.save_ax(ax, "eigvals_lognormal_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_lognormal_normal")
    ax.clear()

    eigenvalues_ones(ax, number_of_points=200, band_width_list=(1, 5, 10, 20, 30))
    plotdl.save_ax(ax, "eigvals_ones_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_ones_normal")
    ax.clear()

    eigenvalues_alter(ax, number_of_points=200, band_width_list=(1, 5, 10, 20, 30))
    plotdl.save_ax(ax, "eigvals_alter_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_alter_normal")
    ax.clear()

    random.seed(seed)
    eigenvalues_box(ax, number_of_points=200, band_width_list=(1, 5, 10, 20, 30))
    plotdl.save_ax(ax, "eigvals_box_loglog")
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plotdl.save_ax(ax, "eigvals_box_normal")
    ax.clear()

    eigenvalues_uniform(ax)
    plotdl.save_ax(ax, "eigvals_uniform")


if __name__ ==  "__main__":
    all_plots()


