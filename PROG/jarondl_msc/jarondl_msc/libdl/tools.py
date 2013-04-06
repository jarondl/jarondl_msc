#!/usr/bin/env python
"""  

"""
from __future__ import division  # makes true division instead of integer division


import numpy as np
import logging
import time


import tables


### Raise all float errors
np.seterr(all='warn')
EXP_MAX_NEG = np.log(np.finfo( np.float).tiny)

#set up logging:
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

info = logger.info
warning = logger.warning
debug = logger.debug
     



def lazyprop_old(fn):
    """ based on http://stackoverflow.com/questions/3012421/python-lazy-property-decorator"""
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    @_lazyprop.setter
    def _lazyprop(self, value):
        setattr(self, attr_name, value)
    return _lazyprop
    
class lazyprop(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self,obj,cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value
    



def cached_get_key(function, filename, key='numerics', *args, **kwargs):
    """ Cached data getter.
        The idea is that we have a function producing numpy arrays
        that can be stored in a npz file. 
        If the file exists and the key exists, this data is returned.
        If the file exists but key doesn't, create data, add to file, and return data.
        If both file doesn't exist it will be created.
    """

    try:
        f = np.load(filename)
        fd = dict(f)
        f.close()
        
    except (OSError, IOError):
        fd = dict()
    debug("current keys in {} are {}".format(filename, fd.keys()))
    debug("We seek {}".format(key))
    if key not in fd:
        fd[key]  = function(*args, **kwargs)
        np.savez(filename, **fd)
    return fd[key]


class Ev_and_PN_1000(tables.IsDescription):
    
    model_name      = tables.StringCol(100)
    date            = tables.StringCol(20)
          
    number_of_points= tables.Int64Col()
    bandwidth       = tables.Int64Col()
    dis_param       = tables.Float64Col()
    eig_vals        = tables.Float64Col(1000)
    PN              = tables.Float64Col(1000)

class Ev_and_PN_2000(tables.IsDescription):
    
    model_name      = tables.StringCol(100)
    date            = tables.StringCol(20)
          
    number_of_points= tables.Int64Col()
    bandwidth       = tables.Int64Col()
    dis_param       = tables.Float64Col()
    eig_vals        = tables.Float64Col(2000)
    PN              = tables.Float64Col(2000)

class Ev_and_PN_3000(tables.IsDescription):
    
    model_name      = tables.StringCol(100)
    date            = tables.StringCol(20)
          
    number_of_points= tables.Int64Col()
    bandwidth       = tables.Int64Col()
    dis_param       = tables.Float64Col()
    eig_vals        = tables.Float64Col(3000)
    PN              = tables.Float64Col(3000)


def check_args_in_row(row, args_dict):
    return all(row[key]==val for (key, val) in args_dict.items())
    
def h5_get_first_rownum_by_args(h5table, args_dict):
    for row in h5table.iterrows():
        if check_args_in_row(row,args_dict):
            return row.nrow
    
def fill_args_in_row(row,args_dict):
    for (key,val) in args_dict.items():
        row[key] = val
        
    
def h5_create_if_missing(h5table, model_factory, model_args):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    exists =  any(check_args_in_row(x,model_args) for x in h5table.iterrows())
    if not exists:
        debug("creating new data for {}".format(model_args))
        m= model_factory(model_args)
        ev = m.eig_vals
        pn= m.PN
        nr = h5table.row
        fill_args_in_row(nr, model_args)
        nr['date'] = t
        nr['eig_vals'] =ev
        nr['PN'] = pn
        nr.append()
        h5table.flush()

