#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several geometry related functions.
"""
from __future__ import division  # makes true division instead of integer division

import numpy as np, numpy



### Raise all float errors 
numpy.seterr(all='warn')

class Sample(object):
    def __init__(self, dimensions, number_of_points=None):
        """
        :param dimensions:  A tuple of the dimensions
        """

        # one int/float doesn't have length
        try:
            self.d = len(dimensions)
            self.dimensions = dimensions
        except TypeError:
            self.d = 1
            self.dimensions = (dimensions,) # a one-tuple
        self.volume = np.prod(self.dimensions)
        self.D = None # placeholder for diffusion coef

        if number_of_points is not None:
            self.generate_points(number_of_points)

    def generate_points(self, number_of_points=None):
        if number_of_points is None:
            number_of_points = self.number_of_points()
        d_points = []
        for dimension in range(self.d):
            d_points += [numpy.random.uniform(0, self.dimensions[dimension], number_of_points)]
        if self.d ==1:
            self.points = d_points[0]
            self.points.sort()
        else:
            self.points = numpy.vstack(d_points).T
    def number_of_points(self):
        return self.points.shape[0]
    def n(self):
        return self.number_of_points() / self.volume
        self.n = self.number_of_points() / self.volume
    def r_0(self):
        return  self.n()**(-1/self.d)

    def epsilon_to_xi(self, epsilon):
        """ Returns xi, using the current sample density. Should raise an error if no points were created."""
        return epsilon * self.r_0()

    def periodic_distance_matrix(self):
        return fast_periodic_distance_matrix(self.points, self.dimensions)

    def non_periodic_distance_matrix(self):
        return fast_distance_matrix(self.points)


def euclid(point1, point2):
    """  Calculate the euclidean distance (norm) between any two points. Points
        can be given as a numpy array or simple list
    """
    point1, point2 = numpy.array(point1), numpy.array(point2)

    return np.sqrt( np.dot((point2-point1), (point2-point1)) )

def distance_matrix(points, distance_function=euclid):
    """  Create a distance matrix, where A_ij is |r_i-r_j|. ** SLOW **
         Points should be given as N dublets, or (Nx2) array.
         If no distance function is give, euclidean distance is used
    """
    N = len(points)
    S = numpy.zeros([N, N])
    for i in range(N):
        for j in range(N):
            S[i, j] = distance_function(points[i], points[j])
    return S

def old_fast_distance_matrix(points):
    """
    """
    try:
        n,m = points.shape
    except ValueError:
        n = points.shape[0]
        m = 1
    if m == 1:
        data = points
    else:
        data = points[:,0]
    delta = (data - data[:,np.newaxis])**2
    for d in range(1,m):
        data = points[:,d]
        delta += (data - data[:,np.newaxis])**2
    #weird 1d problem solving:
    if m==1:
        return numpy.sqrt(delta[:,:,0])
    #  Check for symmetry
    assert (delta == delta.T).all()
    return numpy.sqrt(delta)


def fast_distance_matrix(points, dimensions=(1,1)):
    """ based on http://stackoverflow.com/questions/3518574/why-does-looping-beat-indexing-here
    """
    try:
        n,m = points.shape
    except ValueError:
        n = points.shape[0]
        m = 1
    dims = np.tile(dimensions, (n,m))
    if m == 1:
        data = points
    else:
        data = points[:,0]
    delta = (data - data[:,np.newaxis])**2
    for d in range(1,m):
        data = points[:,d]
        delta += (data - data[:,np.newaxis])**2
        #  Check for symmetry
    assert (delta == delta.T).any()
    return numpy.sqrt(delta)


def fast_periodic_distance_matrix(points, dimensions=(1,1)):
    """ based on http://stackoverflow.com/questions/3518574/why-does-looping-beat-indexing-here
    """
    try:
        n,m = points.shape
    except ValueError:
        n = points.shape[0]
        m = 1
    dims = np.tile(dimensions, (n,m))
    if m == 1:
        data = points
    else:
        data = points[:,0]
    temp_delta = []
    temp_delta += [(data - data[:,np.newaxis])**2]
    temp_delta += [(dims[:,0] + data - data[:,np.newaxis])**2]
    temp_delta += [(-dims[:,0] + data - data[:,np.newaxis])**2]
    delta = numpy.dstack(temp_delta).min(axis=2)
    for d in range(1,m):
        data = points[:,d]
        temp_delta = []
        temp_delta += [(data - data[:,np.newaxis])**2]
        temp_delta += [(dims[:,d] + data - data[:,np.newaxis])**2]
        temp_delta += [(-dims[:,d] + data - data[:,np.newaxis])**2]
        delta += numpy.dstack(temp_delta).min(axis=2)

    #  Check for symmetry
    assert (delta == delta.T).any()
    return numpy.sqrt(delta)



