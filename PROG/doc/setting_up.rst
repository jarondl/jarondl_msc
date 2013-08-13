How to run the code
=====================================

Setting up your python environment
-------------------------------------
Almost every Linux system has python installed. However, the
nice thing about python is the additional available modules.
To run the ``jarondl_msc`` package code, you'll need (in Fedora):

    yum install numpy scipy python-matplotlib python-yaml python-tables
    
This will allow the code to run, but the interactivity will be limited.
I recommend using ipython  ( from the ``python-ipython`` package),
but if you prefer a more graphic experience (Matlab-like) use ``spyder``.
    
Adding the package to the path
-------------------------------------
Of-course the code can be run straight from its library,
but that is considered a bad practice. One option is to add
it to your python search path. In the bgu physics cluster, this means:

    setenv PYTHONPATH /users/physics/jarondl/PROJ/PROG/jarondl_msc
    
An **alternative** is to install the package with pip. This will be explained later.

Using the code interactively 
------------------------------------
In order to plot the PN as function of eigenvalue for a banded anderson model 
:class:`Model_Anderson_DD_1d` ::

    ### either run 
    ipython --pylab
    ### or 
    python
    from pylab import *
    
followed by::

    from jarondl_msc import models
    m1 = models.Model_Anderson_DD_1d(number_of_points=1000,
                  bandwidth=1, dis_param=0.4, periodic=False)
    plot(m1.eig_vals, m1.PN)


Running all of the pta plots
----------------------------------
Running 1,2 or three realizations is instantaneous. 
But since running many realizations takes a long time, and in order to
separate configuration from code, a different system was devised. 
To run the numerics a data configuration file is needed. To create the
plots a plot configuration file is needed. Both are written in ``yaml``. 
Be advised that the numerics take long time, but they can be re used
if only the plotting changes. To run them both use::

    python -m jarondl_msc.pta_chain_g  \
           -d /users/physics/jarondl/PROJ/PROG/jarondl_msc/pta_chain_data_def.yaml \
           -p /users/physics/jarondl/PROJ/PROG/jarondl_msc/pta_chain_plots_def.yaml 
    
This will create a directory called data and fill it with ``npz`` files,
and a directory called plots with the plots. 
By making your own version of the files every aspect of the plots can be changed
(data ranges, plot format, etc.)


