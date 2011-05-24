#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  A module containing several geometry related functions.
"""

import numpy as np, numpy



class Torus(object):
    def __init__(self,a,b):
        """ ARGUMENTS:
                a - length of torus (x dim)
                b - width of torus ( y dim)
        """
        self.a = a
        self.b = b
    def distance(self, point1, point2):
        x1,y1 = point1
        x2,y2 = point2
        ydiff = np.min( (np.abs(y2-y1), np.abs(y2+self.b-y1))  )
        xdiff = np.min( (np.abs(x2-x1), np.abs(x2+self.a-x1))  )
        return np.sqrt(xdiff**2 + ydiff**2)
    
    def generate_points(self, N):
        self.xpoints = numpy.random.uniform(0, self.a, N)
        self.ypoints = numpy.random.uniform(0, self.b, N)
        self.points = numpy.vstack((self.xpoints,self.ypoints)).T
        return self.points

    def distance_matrix(self):
        N = len(self.xpoints)
        S = numpy.zeros([N,N])
        for i in range(N):
            for j in range(N):
                S[i,j] = self.distance(self.points[i], self.points[j])
        return S
    
    def exp_matrix(self,xi=1):
        return np.exp(-self.distance_matrix() /xi)
            
