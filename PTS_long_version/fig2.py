#!/usr/bin/python
"""
"""
from __future__ import division
from pylab import *

dd = arange(-0.5, 0.6, 0.1)
for d in dd:
    (x,y) = (2*exp(d)-2, 2 ) if  2*exp(d)-2 >0 else (0, 2*exp(d))
    plot([x,y], [ y,x ],"r-")
    x = linspace(-log(2)-d,log(2)+d,20)
    plot(exp(-x-d), exp(x-d),"b--")
show()
