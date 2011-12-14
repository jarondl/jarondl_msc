#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Survival and spreading for log normal distribution.
"""
from __future__ import division
import sys

import ptsplot
import plotdl

plotdl.pyplot.ioff()

if len(sys.argv)  >1:
    eps = float(sys.argv[1])
else:
    eps = 4

f = plotdl.pyplot.figure()
s = ptsplot.Sample((1,1),900)
m = ptsplot.ExpModel_2d(s, 4)
ptsplot.scatter_eigmode_slider(f,m)
plotdl.pyplot.show()
