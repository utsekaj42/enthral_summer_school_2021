##!/usr/bin/env python
# -*- coding: utf-8 -*- ####################################################################################### STARFiSh v0.4 
########################################################################################
## 
# http://www.ntnu.no/starfish
#
# Contributors:
# Leif Rune Hellevik, Vinzenz Gregor Eck, Jacob Sturdy, Fredrik Eikeland Fossan, 
# Einar Nyberg Karlsen, Yvan Gugler, Yapi Donatien Achou, Hallvard Moian Nydal, 
# Knut Petter Maråk, Paul Roger Leinan
#
# TODO: ADD LICENSE (MIT) and COPYRIGHT
#
#Copyright (c) <2012-> <NTNU>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this 
#software and associated documentation files (the "Software"), to deal in the Software
#without restriction, including without limitation the rights to use, copy, modify,
#merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
#permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or 
#substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
#PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
#HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
#CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
##

#---------------------------------------------------------------------------------------#
from __future__ import print_function, absolute_import
from future.utils import iteritems, iterkeys, viewkeys, viewitems, itervalues, viewvalues
from builtins import input as input3
import time 
import sys,os
# set the path relative to THIS file not the executing file!
cur = os.path.dirname( os.path.realpath('__file__') )
import logging
logger = logging.getLogger(__name__)
import starfish
import starfish.SolverLib.class1DflowSolver as c1DFlowSolv
import starfish.UtilityLib.moduleXML as mXML
import starfish.UtilityLib.moduleStartUp as mStartUp #import parseOptions
import starfish.UtilityLib.moduleFilePathHandler as mFPH
import starfish.UtilityLib.moduleLogFile as mLOG
import matplotlib.pyplot as plt
import gc
import subprocess

def main():
    optionsDict = mStartUp.parseOptions(['f','n','d','s','v','r','w','p'])
    
    networkName           = optionsDict['networkName']
    save                  = optionsDict['save']
    dataNumber            = optionsDict['dataNumber']
    simulationDescription = optionsDict['simulationDescription']
    vizOutput             = optionsDict['vizOutput']
    resimulate            = optionsDict['resimulate']
    
    filename = str(networkName+'.xml')
        
    logger.info('____________Simulation_______________')
    logger.info('%-20s %s' % ('Network name',networkName))
    logger.info('%-20s %s' % ('Data number', dataNumber))
    logger.info('%-20s %s' % ('Save simulation', save))
    logger.info('%-20s %s' % ('Case description', simulationDescription))
    logger.info('%-20s %s' % ('Resimulate', resimulate))
    logger.info('%-20s %s' % ('Visualisationmode', vizOutput))
    
    ## check if template
    if '_template' in networkName:
        networkName = mFPH.createWorkingCopyOfTemplateNetwork(networkName)
    
    # load network from the path!
    if resimulate == False:
        vascularNetwork = mXML.loadNetworkFromXML(networkName) # moved to vascularNetowrk constror
    else:
        # resimulate network
        vascularNetwork = mXML.loadNetworkFromXML(networkName, dataNumber = dataNumber)        
        if simulationDescription == '':
            simulationDescription = vascularNetwork.description
    
    if vascularNetwork == None: exit()
    
    
    vascularNetwork.update({'description':simulationDescription,
                            'dataNumber' :dataNumber})
    
    timeSolverInitStart = time.time()
    #initialize Solver
    flowSolver = c1DFlowSolv.FlowSolver(vascularNetwork)
    timeSolverInit = time.time()-timeSolverInitStart
    timeSolverSolveStart = time.time()
    #solve the system
    flowSolver.solve()
    timeSolverSolve = time.time()-timeSolverSolveStart
    
    minutesInit = int(timeSolverInit/60.)
    secsInit = timeSolverInit-minutesInit*60.
    minutesSolve = int(timeSolverSolve/60.)
    secsSolve = timeSolverSolve-minutesSolve*60.

    #mLog2 = mLOG.NetworkLogFile(vascularNetwork, dataNumber=dataNumber, dataNumberCompare=None, dt=flowSolver.dt, CpuTimeInit=[minutesInit, secsInit], CpuTimeSolve=[minutesSolve, secsSolve])
    #mLog2.writeNetworkLogfile(compileLogFile=True, deleteAuxiliary=True)
    
    
    logger.info('____________ Solver time _____________')
    logger.info('Initialisation: {} min {} sec'.format(minutesInit,secsInit))
    logger.info('Solving:        {} min {} sec'.format(minutesSolve,secsSolve))
    logger.info('=====================================')
    print('Initialisation: {} min {} sec'.format(minutesInit,secsInit))
    print('Solving:        {} min {} sec'.format(minutesSolve,secsSolve))
    vascularNetwork.saveSolutionData()
    mXML.writeNetworkToXML(vascularNetwork, dataNumber = dataNumber) # needs to be moved to vascularNetwork
    del flowSolver
    gc.collect()
    mFPH.updateSimulationDescriptions(networkName, dataNumber, simulationDescription)
    gc.collect()
    
    string2d = ' '.join([sys.executable, '-c' 
        '"import starfish.VisualisationLib.class2dVisualisation as viz;', 
        "viz.main()", '"', 
        '-f', vascularNetwork.name, 
        '-n',str(dataNumber)])

    string3d = ' '.join([sys.executable, '-c' 
        '"import starfish.VisualisationLib.class3dVisualisation as viz;', 
        "viz.main()", '"', 
        '-f', vascularNetwork.name, 
        '-n',str(dataNumber)])
    
    if vizOutput == "2D":
            subprocess.Popen(string2d, shell=True)
    elif vizOutput == "3D":
        subprocess.Popen(string3d, shell=True)
    elif vizOutput == "2D+3D":
        viz2d = subprocess.Popen(string2d, shell = True)
        viz3d = subprocess.Popen(string3d, shell = True)
        
        while True:
            
            if viz2d.poll() is not None:
                viz2d.terminate()
                exit()
                
            if viz3d.poll() is not None:
                viz3d.terminate()
                exit()
        
if __name__ == '__main__':
    main()
