#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several geometry related functions.
"""

import numpy as np, numpy



class Torus(object):
    def __init__(self,(a,b)):
        """ ARGUMENTS:
                a - length of torus (x dim)
                b - width of torus ( y dim)
        """
        self.a = a
        self.b = b
    def distance(self, point1, point2):
        x1,y1 = point1
        x2,y2 = point2
        ydiff = np.min( (np.abs(y2-y1), self.b - np.abs(y2-y1))  )
        xdiff = np.min( (np.abs(x2-x1), self.a - np.abs(x2-x1))  )
        return np.sqrt(xdiff**2 + ydiff**2)
    
    def generate_points(self, N):
        self.xpoints = numpy.random.uniform(0, self.a, N)
        self.ypoints = numpy.random.uniform(0, self.b, N)
        self.points = numpy.vstack((self.xpoints,self.ypoints)).T
        return self.points

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
    S = numpy.zeros([N,N])
    for i in range(N):
        for j in range(N):
            S[i,j] = distance_function(points[i], points[j])
    return S
        
