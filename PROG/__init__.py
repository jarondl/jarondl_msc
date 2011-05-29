"""
There is one module acting as a "library" containing all of the shared functions
sparsedl.py
This module is always imported into other python scripts, and does nothing when run stand-alone.

The rest of the files are scripts that create data and figures.

The basic workflow is to create numpy data using create_data.py:
./create_data.py 300 examplefile.npz

Then, create the rho as a function of t figures and movie:
./figs_rho_of_t.py examplefile.npz

Now create survival and second moment as a function of t figures:
./survival_figs.py examplefile.npz

"""
