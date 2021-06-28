This is STARFiSh a program for simulation of 1D blood vessels with physiologically motivated
boundary conditions while accounting for input uncertainty.



# Running your first simulation
If the software environment has been installed correctly, you should be able to navigate to the
folder `UnitTesting` and run any of the python scripts therein e.g. `python unitTest_singleVessel.py`.
If any errors were reported something is wrong.

The typical usage of STARFiSh assumes one has a working directory where the configuration and
results for simulating a specific vascular network may be stored.

The topology, geometry, boundary conditions and simulation parameters are all stored in an XML file
in a folder under the name of the network (typically referring to a topolgy+geometry) though this is
not enforced.
For each specific set of simulation parameters a simulation number is assigned and a subfolder with
this number will be created where the exact network XML file will be saved along with all simulation
output data.

# Basic interface
```
python -m starfish
```
Should bring up the main Menu and show which modules could be imported.

