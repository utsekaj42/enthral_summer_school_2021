# ipython magic
%matplotlib notebook
%load_ext autoreload
%autoreload 2
import os, sys, inspect
import matplotlib
import matplotlib.pyplot as plt
# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_source")))
if cmd_subfolder not in sys.path:
     sys.path.insert(0, cmd_subfolder)


from present_output import print_vectors_relerror, print_3vectors_relerror
