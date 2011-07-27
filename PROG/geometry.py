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
        self.dimensions = dimensions
        self.number_of_points = number_of_points
        self.d = len(dimensions)
        self.volume = np.prod(self.dimensions)

        if number_of_points is not None:
            self.generate_points(number_of_points)

    def generate_points(self, number_of_points):
        d_points = []
        for dimension in range(self.d):
            d_points += [numpy.random.uniform(0, self.dimensions[dimension], number_of_points)]
        self.points = numpy.vstack(d_points).T
        self.number_of_points = number_of_points
        self.n = self.number_of_points / self.volume

    def epsilon_to_xi(self, epsilon):
        """ Returns xi, using the current sample density. Should raise an error if no points were created."""
        return epsilon * self.n**(-1/self.d)

    def periodic_distance_matrix(self):
        return fast_periodic_distance_matrix(self.points, self.dimensions)

    def non_periodic_distance_matrix(self):
        return fast_distance_matrix(self.points)


class Torus(object):
    def __init__(self, (a, b), number_of_points=None):
        """ ARGUMENTS:
                a - length of torus (x dim)
                b - width of torus ( y dim)
        """
        self.a = a
        self.b = b
        self.d = 2
        self.dimensions = (a,b)
        self.volume = a*b
        self.description = "2d periodic torus"
        self.short_name="torus"
        if number_of_points is not None:
            self.number_of_points = number_of_points
            self.generate_points(number_of_points)

    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        ydiff = np.min( (np.abs(y2-y1), self.b - np.abs(y2-y1))  )
        xdiff = np.min( (np.abs(x2-x1), self.a - np.abs(x2-x1))  )
        return np.sqrt(xdiff**2 + ydiff**2)

    def generate_points(self, number_of_points):
        self.xpoints = numpy.random.uniform(0, self.a, number_of_points)
        self.ypoints = numpy.random.uniform(0, self.b, number_of_points)
        self.points = numpy.vstack((self.xpoints, self.ypoints)).T
        self.number_of_points = number_of_points
        #return self.points
    
    def epsilon_to_xi(self, epsilon):
        n = self.number_of_points/ self.volume
        return epsilon * n**(-1/self.d)

class PeriodicLine(object):
    def __init__(self, a, number_of_points=None):
        self.a = a
        self.d = 1
        self.dimensions = (a,)
        self.volume = a

        self.description = "1d periodic line"
        self.short_name="line"
        if number_of_points is not None:
            self.number_of_points = number_of_points
            self.generate_points(number_of_points)

    def distance(self, point1,point2):
        return np.min(( np.abs(point2 - point1) , (self.a - np.abs(point2 - point1))  ))

    def generate_points(self, number_of_points):
        self.points = numpy.random.uniform(0,self.a,number_of_points)
        self.points.sort()
        self.number_of_points = number_of_points

    def epsilon_to_xi(self, epsilon):
        n = self.number_of_points/ self.volume
        return epsilon * n**(-1/self.d)

def euclid(point1, point2):
    """  Calculate the euclidean distance (norm) between any two points. Points
        can be given as a numpy array or simple list
    """
    point1, point2 = numpy.array(point1), numpy.array(point2)

    return np.sqrt( np.dot((point2-point1), (point2-point1)) )

def distance_matrix(points, distance_function=euclid):
    """  Create a distance matrix, where A_ij is |r_i-r_j|.
         Points should be given as N dublets, or (Nx2) array.
         If no distance function is give, euclidean distance is used
    """
    N = len(points)
    S = numpy.zeros([N, N])
    for i in range(N):
        for j in range(N):
            S[i, j] = distance_function(points[i], points[j])
    return S

def fast_distance_matrix(points):
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

    return numpy.sqrt(delta)



