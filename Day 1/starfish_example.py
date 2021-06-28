import sys
import os
import gc
import logging
import matplotlib.pyplot as plt
import h5py

cur = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'starfish')))

import starfish.UtilityLib.moduleXML as mXML
import starfish.SolverLib.class1DflowSolver as c1dFS

OUTPUT_PATH = os.path.join(cur, 'starfish_outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)
m3_to_ml = 1e6

bifurcation_template = "starfish/starfish/TemplateNetworks/singleBifurcation_template/singleBifurcation_template.xml"
single_vessel_template = "starfish/starfish/TemplateNetworks/singleVessel_template/singleVessel_template.xml"

network_template_path = single_vessel_template
init_method='Auto'
dt=-2
total_time=15
networkXmlFileLoad = os.path.join(cur, network_template_path)
networkName = os.path.basename(networkXmlFileLoad)
networkName = os.path.splitext(networkName)[0]

# Define output name
case_str = init_method + "first"
networkXmlFileSave = os.path.join(OUTPUT_PATH, networkName + "." + case_str + ".xml")
pathSolutionDataFilename = os.path.join(OUTPUT_PATH, networkName + "." + case_str + ".hdf5")
dataNumber = 'tst'

# Load template network
vascularNetworkNew = mXML.loadNetworkFromXML(networkName,
                                             dataNumber,
                                             networkXmlFile = networkXmlFileLoad,
                                             pathSolutionDataFilename = pathSolutionDataFilename,
                                             )

# Modify parameters
vascularNetworkNew.dt = dt
vascularNetworkNew.initialsationMethod = method
vascularNetworkNew.quiet = False
vascularNetworkNew.totalTime = total_time

# Modify Inlet
bc = vascularNetworkNew.boundaryConditions[0][0]
bc.Npulse = 2
bc.amp = 300/m3_to_ml
bc.freq = 5/3
bc.T_space = 0.2
bc.prescribe = 'influx' #'influx' or 'total'
# Modify Vessel
vessel = vascularNetworkNew.vessels[0]
vessel.radiusProximal = 0.0294
vessel.length =0.5
vessel.betaHayashi = 1.8315018315 # Stiffness parameter

# Modify Outlet Windkessel
bc = vascularNetworkNew.boundaryConditions[0][-1]
bc.Rtotal = 133000000.0
bc.C = 3.52355288029e-08


# Run simulation
flowSolver = c1dFS.FlowSolver(vascularNetworkNew, quiet=False)
flowSolver.solve()
vascularNetworkNew.saveSolutionData()
mXML.writeNetworkToXML(vascularNetworkNew, dataNumber, networkXmlFileSave)
del flowSolver

# Load solution results
solutionNew = h5py.File(pathSolutionDataFilename, "r")
netNew = solutionNew['VascularNetwork']
tNew = netNew['simulationTime']
vesselsNew = solutionNew['vessels']

#
vesselId_to_plot = 0
for subGroupName, subGroup in vesselsNew.items():
    vesselId = int(subGroupName.split(' - ')[-1])
    if vesselId_to_plot == vesselId:
        Qnew = subGroup['Qsol']
        plt.figure(num='flow')
        plt.plot(tNew, Qnew[:, 0]*m3_to_ml)
        plt.xlabel('t')
        plt.ylabel('Q')
        plt.legend()

        plt.figure(num='pressure')
        Pnew = subGroup['Psol']
        plt.plot(tNew, Pnew[:, 0])
        plt.xlabel('t')
        plt.ylabel('P')
        plt.legend()