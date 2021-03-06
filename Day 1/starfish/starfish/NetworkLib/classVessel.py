from __future__ import print_function, absolute_import
from builtins import range
from future.utils import iteritems, iterkeys, viewkeys, viewitems, itervalues, viewvalues
from builtins import input as input3
import pprint
from copy import deepcopy
import sys, os
from math import pi, cos, sin
import numpy as np
import scipy
import scipy.signal
from scipy.integrate import simps
from scipy import interpolate
import matplotlib.pyplot as plt
import time as timeModule
import shutil

## TODO: needsto be imported as modules
from starfish.NetworkLib.moduleGrids import *
from starfish.NetworkLib.classCompliance import *
# TODO: Can't change import without crashing simulation.
# moduleGrids: ERROR: classVessel.initialize(): Grid calculation of vessel 1
# classCompliance: ERROR: classVessel.initialize(): init compliance of vessel 1

from starfish.UtilityLib import classStarfishBaseObject as cSBO
from starfish.UtilityLib.constants import newestNetworkXml as nxml

class Vessel(cSBO.StarfishBaseObject):
    """
    Class representing a vessel in a vascular Network
    """
    number = 0
    quiet = False


    solutionMemoryFields = ["Psol", "Asol", "Qsol", "positionStart", "positionEnd", "rotToGlobalSys", "netGravity"]
    solutionMemoryFieldsToSave = ["Psol", "Asol", "Qsol", "positionStart", "positionEnd", "rotToGlobalSys", "netGravity"]

    def __init__(self, Id= None, name = None, geometryPath=None):
        """
        Constructor
        """
        ### Input properties

        ## attributes
        self.Id     = Id                            # id of the vessel
        self.name   = name                          # name of the vessel
        self.geometryPath = geometryPath        # path to folder with length and radius for each vessel, used in fromFile

        ## topology properties
        self.startNode          = None              # node id of starting node
        self.endNode            = None              # node id of ending node
        self.leftDaughter       = None              # id of left daughter vessel
        self.rightDaughter      = None              # id of right daughter vessel
        self.leftMother         = None              # id of left mother
        self.rightMother        = None              # id of right mother

        self.daughterList       = []                # list of ids of daughters
        self.motherList         = []                # list of ids of mothers

        # Coordinate system is RHS with Z along vessel axis. Origin is at the center of area of
        # the vessel, with Z+ pointing towards the distal end
        self.angleXMother       = 0                 # rotation angle in rad around x-axis relative to mother vessel
        self.angleYMother       = 0                 # rotation angle in rad around y-axis relative to mother vessel
        self.angleZMother       = 0                 # rotation angle in rad around z-axis relative to mother vessel

        ## geometry properties
        self.geometryType       = 'uniform'         # #TODO: fix this geometry type: 'uniform', 'cone', 'Cons'
        self.radiusProximal     = 0.01              # radius at vessel begin // in hole Vessel if radiusB = 0
        self.radiusDistal       = 0.01              # radius at vessel end
        self.length             = 0.1               # length of the vessel
        self.N                  = 5                 # number of gridpoints

        ## compliance properties
        self.complianceType     = 'Hayashi'         # type of the compliance see classCompliance
        self.constantCompliance = False             # use constant compliance C0 calculated with Ps for hole simulation
        self.externalPressure   = 0.                # external pressure of the vessel usually 0 for head arteries 2000 Pa
        self.Ps                 = 0.                # pressure at state S
        self.As                 = None              # Area state S input from XML if None it is calculate from given radius

        self.wallThickness      = None              # wall thickness
        self.youngModulus       = None              # youngModulus
        self.betaExponential    = None              # material Parameter of Exponential compliance model
        self.betaHayashi        = 1.                # material Parameter of Hayashi compliance model
        self.betaLaplace        = None              # material Parameter of Laplace compliance model
        # reymond model
        self.Cs                 = None              # arterial area compliance at reference pressure for Reymond compliance model
        self.PmaxC              = None
        self.Pwidth             = None
        self.a1                 = None
        self.b1                 = None

        # FLUID properties TODO: annotate units
        self.applyGlobalFluid   = True              # bool: apply global fluid properties or the ones stored in vessel XML
        self.my                 = 1.e-6             # blood viscosity
        self.rho                = 1050.             # blood density
        self.gamma              = 2.0              # velocity profile gamma correction

        # vascular Polynomial chaos data dictionary with uncertainty variables

        ### Calculated properties
        # positions in space
        # realtive to global system
        self.positionStart  = np.zeros((1,3))         # instantaneous position of vessel start point in the global system
        self.positionEnd    = np.zeros((1,3))         # instantaneous position of vessel end point in the global system
        self.rotToGlobalSys = np.zeros((1,3,3))
        self.rotToGlobalSys[0] = np.array([np.eye(3)])   # rotational matrix to the global system #TODO should this be initialized like this here?



        ## motion
        self.angleXMotherTime   = []                # describes angular motion over time; set up by flow solver
        self.angleYMotherTime   = []                # describes angular motion over time; set up by flow solver
        self.angleZMotherTime   = []                # describes angular motion over time; set up by flow solver

        # gravity
        self.netGravity = np.zeros(1)
        self.gravityConstant    = -9.81             # gravity constant of the gravitational field

        # GRID properties
        self.z                  = None              # axial coordinates of the vessel
        self.dz                 = None              # axial grid spacing
        self.A0                 = None              # initial Area
        self.AsVector           = None              # vector for the total grid containing area at state S

        # SOLID properties
        self.compliance         = None              # the compliance class reference
        self.P_Cexp             = None              # pressuer P0 for the ComplianceType exponential
        self.C                  = None              # Compliance function C(P) for all nodes
        self.C_nID              = None              # Compliance function C(P,NodeId) for specific Node with NodeID
        self.A                  = None              # Area function A(P) for all nodes
        self.A_nID              = None              # Area function A(P,NodeId) for specific Node with NodeID
        self.c                  = self.waveSpeed    # wave speed function c(rho,A,C) for all nodes /single if given single values
        
        self.cd_in              = None              # wavespeed at inlet at Ps
        self.cd_out             = None              # wavespeed at outlet at Ps
        self.Cv                 = None              # vessel compliance at Ps
        self.Lv                 = None              # vessel Inertance at Ps
        
        self.resistance         = None              # vessel resistance
        self.womersleyNumber    = None              # Womersley number


        ## flag to indicate vessel data should be saved
        self.save         = True
        self.dsetGroup    = None

        # pointers to solutionData objects
        self.Psol         = np.zeros((0,self.N))  # None  # pressure
        self.Qsol         = np.zeros((0,self.N))  # None  # flow
        self.Asol         = np.zeros((0,self.N))  # None  # area
        # calculated values for post processing
        self.csol         = None # wave speed
        self.vsol         = None # mean velocity
        self.Csol         = None # compliance solution

        self.quiet = False
        Vessel.number += 1

    def initialize(self, globalFluid):
        """
        Initialisation of the vessel.
        This method calculates and set up, the compliance, grid, fluid and resistance of the Vessel.
        """
        # initialize fluid
        for key,value in iteritems(globalFluid):

            try:
                if self.applyGlobalFluid == True:
                    self.__getattribute__(key)
                    self.__setattr__(key,value)
            except Exception:
                if key != 'venousPressure':
                    self.exception("classVessel.initialize(): Fluid initialisation could not update variable {} of vessel {}!".format(key,self.Id))

        # initialize grid
        try:
            if self.geometryType == 'fromFile':
                # TODO add this functionality to grids?
                self.z,self.dz,self.A0 = self.loadVesselGeometryFromFile(N=self.N)
                # self.z,self.dz,self.A0 = cone(self.length, self.radiusProximal, self.radiusDistal, self.N)
            else:
                self.z,self.dz,self.A0 = eval(self.geometryType)(self.length, self.radiusProximal, self.radiusDistal, self.N)
            
        except Exception:
            self.exception("classVessel.initialize(): Grid calculation of vessel {} failed!".format(self.Id))


        ## calculate compliance reference area vector As
        try:
            if self.geometryType == 'uniform':
                
                
                if self.As == None:
                    AsProximal = self.radiusProximal**2.0*np.pi
                    As_distal = AsProximal
                    self.AsVector = np.linspace(AsProximal,As_distal,self.N)
                    #print "WARNING: no reference Area for the Compliance As is given for vessel {} ! \n         As is now calculatet with given radiusA".format(self.Id)
                else:
                    self.AsVector = np.ones(self.N)*self.As
            elif self.geometryType == 'cone':
                AsProximal = self.radiusProximal**2.0*np.pi
                As_distal = self.radiusDistal**2.0*np.pi
                self.AsVector = np.linspace(AsProximal,As_distal,self.N)
            elif self.geometryType == 'constriction':
                AsProximal = self.radiusProximal**2.0*np.pi
                As_distal = self.radiusDistal**2.0*np.pi
                self.AsVector = self.A0.copy() 
                
            elif self.geometryType == 'fromFile':
                _, _, self.AsVector = self.loadVesselGeometryFromFile()
                
            elif self.geometryType == 'mtStig2016':
                self.AsVector = self.A0.copy() 
            else:
                raise ValueError("classVessel.initialize(): no grid geometry not proper defined for vessel {} ! ".format(self.Id))
        except Exception:
            self.exception("classVessel.initialize(): As calculation of vessel {}!".format(self.Id))

        ## initialize compliance
        try:
            self.compliance = eval(self.complianceType)(self.rho,self.AsVector)
            # initialize compliance element
            complianceDict = {}
            for variable in nxml.vesselComplianceElements[self.complianceType]:
                if variable not in ['As','constantCompliance']:
                    complianceDict[variable] = self.getVariableValue(variable)

            self.compliance.initialize(complianceDict)

            # apply functionals
            self.A  = self.compliance.A
            self.A_nID = self.compliance.A_Node

            if self.constantCompliance == False:
                self.C     = self.compliance.C
                self.C_nID = self.compliance.C_Node
            else:
                self.C     = self.compliance.C0
                self.C_nID = self.compliance.C0_Node

        except Exception:
            self.exception("classVessel.initialize(): init Compliance of vessel {}!".format(self.Id))

        # set up Resistance
        try:
            # calculate peuisell resistance R = (8*mu*L) / (pi*r**4)
            if self.geometryType == "uniform":
                self.resistance = 2*(self.gamma + 2)*self.my*self.length / (pi*self.radiusProximal**4)
            elif self.geometryType == "cone":
                self.resistance = self.calcVesselResistance()
            elif self.geometryType == "fromFile":
                self.resistance = self.calcVesselResistance(Nintegration=-1)
            elif self.geometryType == "constriction":
                #TODO: Not sure if this is important for anything.
                self.resistance = 8*self.my*self.length / (pi*self.radiusProximal**4)
        except Exception:
            self.exception("classVessel.initialize(): in calculating resistance of vessel {}!".format(self.Id))
            
        # calculate vessel compliance (CV) at Ps 
        try:
            self.Cv = self.calcVesselCompliance()
            self.cd_in, self.cd_out = self.calcVesselWavespeed_in_out()
            self.Lv = self.calcVesselInertance()
        except Exception:
            self.exception("classVessel.initialize(): in calculating compliance Cv of vessel {}!".format(self.Id))
        # calculate Womersley number

        #set hooks waveSpeed function
        if self.c == None: self.c = self.waveSpeed

    def initializeForSimulation(self, initialValues, runtimeMemoryManager, nTsteps, vesselsDataGroup):
        """
        Initialize the solution data and allocates memory for it

        Input:
            memoryArraySize := number of time points of one array in memory
        """
        # TODO: THESE MUST BE CORRECTED TO THE RIGHT SHAPE AS THE ADAPTIVE GRID DOESN'T RESIZE WHEN CHANGING N
        # TODO: The memory estimate is therefore off as well
        numberOfGridPoints = self.N
        self.Psol = np.ones((0,numberOfGridPoints))
        self.Qsol = np.zeros((0,numberOfGridPoints))
        self.Asol = np.zeros((0,numberOfGridPoints))
        # init these should have a value for each time point not each time step
        self.positionStart    = np.zeros((0,3))
        self.positionEnd      = np.zeros((0,3))
        self.rotToGlobalSys   = np.zeros((0,3,3))
        self.netGravity       = np.zeros(1)

        # create a new group in the data file
        self.dsetGroup = vesselsDataGroup.create_group(' '.join([self.name, ' - ', str(self.Id)]))
        self.dsetGroup.attrs['dz'] = self.length/self.N
        self.dsetGroup.attrs['N'] = self.N
        self.dsetGroup.attrs['length'] = self.length
        self.allocate(runtimeMemoryManager)
        # set initial values
        try:
            p0, p1 = initialValues["Pressure"]
            Q0, Q1    = initialValues["Flow"]
            self.Psol[0] = np.linspace(p0, p1, self.N)
            self.Qsol[0] = np.linspace(Q0, Q1, self.N) #np.ones((1,self.N))*Qm
        except Exception:
            self.warning("vessel could not use initial values from network")

        self.Asol[0] = np.ones((1, self.N))*self.A(self.Psol[0])


    
    #### function for loading vessel geometry from file with two columns; x, r in SI units
    def loadVesselGeometryFromFile(self, smoothingFilter=True, windowLength=5, polyOrder=3, genFigures=False, N=None):
        
        #t_start = timeModule.time()
        
        topDir = self.geometryPath
        fileName = self.geometryPath + '/vessel_{0}.txt'.format(self.Id)
        data = np.genfromtxt(fileName)
        X = data[:,0]
        R = data[:,1]
        # X = []
        # R = []
        # f = open(fileName, 'r')
        # for line in f:
        #     line_split = line.split('    ')
        #     X.append(float(line_split[0]))
        #     R.append(float(line_split[1]))
        #
        # f.close()
        
        # X = np.array(X)
        
        X = X  - X[0]
        # R = np.array(R)
        if N is None:
            N = len(X)
        
        X_new = np.linspace(X[0], X[-1], self.N)
        dx = np.diff(X_new)
        #print (N)
        if N < 4:
            R_new = np.interp(X_new, X, R)
            R_filter = R_new
        else:
            tck = interpolate.splrep(X, R, s=0)
            
            R_new = interpolate.splev(X_new, tck, der=0)
            if self.N < windowLength:
                if self.N % 2 == 0:
                    windowLength = self.N - 1
                else:
                    windowLength = self.N
            if polyOrder >= windowLength:
                polyOrder = windowLength - 1
            
            if smoothingFilter:
                R_filter = scipy.signal.savgol_filter(R_new, windowLength, polyOrder)
            else:
                R_filter = R_new
        A = np.pi*R_filter**2
        
        if genFigures:
            if os.path.isdir(topDir + '/fig'):
                pass
            else:
                os.mkdir(topDir + '/fig')
                
            plt.figure()
            plt.plot(X*100, R*100, '-o')
            plt.plot(X_new*100, R_new*100, '-*')
            plt.plot(X_new*100, R_filter*100, '-^')
            plt.legend(['original', 'interp', 'filter'])
            plt.xlabel('x [cm]')
            plt.ylabel('r [cm]')
            plt.savefig(topDir + '/fig/vessel_sol_{0}_{1}.png'.format(self.Id, self.N))
            plt.close()
        
        #os.remove(fileNameNew)
        return X_new, dx, A
    #### Functions for to calculate dependent solution variables (also used for simulations)

    def loadSolutionDataRange(self, tSliceToLoad,
                                  values=["All",
                                  "Pressure",
                                  "Flow",
                                  "Area",
                                  "WaveSpeed",
                                  'Compliance',
                                  "MeanVelocity",
                                  "Gravity",
                                  "Position",
                                  "Rotation"]
                              ):
        """
        loads the solution data of the vessels specified into memory for the times
            specified and drops any other previously loaded data.
        Inputs:
            tSliceToLoad - a numpy slice object of the form np.s_[nBegin:nEnd:nSteps]
            values = a dictionary specifying which quantities to load entries keys are booleans and may be 'loadAll',
                'loadPressure', 'loadArea', 'loadFlow', 'loadWaveSpeed', and 'loadMeanVelocity'. If 'All'
                is in the list all quantities are loaded. Inputs are case insensitive.
        Effects and Usage:
            loads the specified values into memory such that they may be accessed as
            vascularNetwork.vessels[vesselId].Pressure, etc, returning a matrix of
            solution values corresponding to the time points in vascularNetwork.tsol.
            Accessing vessels and values not set to be loaded will produce errors.
        """
        # Update loaded data tracking if inputs are valid
        # We could do this value = d.get(key, False) returns the value or False if it doesn't exist
        validValues = ["All", "Pressure", "Flow", "Area", "WaveSpeed",
                       "Compliance", "MeanVelocity", "Gravity", "Position",
                       "Rotation", "linearWavesplit"]
        values = set(values)
        if 'All' in values:
            values.update(validValues)
        else:
            if "WaveSpeed" in values:
                values.update(["Pressure", "Area"])
            elif "MeanVelocity" in values:
                values.update(["Pressure","Flow"])
            elif "linearWavesplit" in values:
                values.update(["Pressure","Flow","Area","WaveSpeed"])
            elif "Compliance" in values:
                values.update(["Pressure", "Compliance"])

        dsetGroup = self.dsetGroup

        # TODO Implement h5py direct_read method to improve speed
        # i.e. dsetGroup["Psol"].read_direct(self.Psol,
        # np.s_[nselectedBegin:nSelectedEnd:nTStepSpace], None)
        # reads the selected values into the already allocated array Psol.
        # Psol probably needs to be allocated already though.
        if 'Pressure' in values:
            self.Psol = dsetGroup['Psol'][tSliceToLoad]
        if 'Flow' in values:
            self.Qsol = dsetGroup['Qsol'][tSliceToLoad]
        if  'Area' in values:
            self.Asol = dsetGroup['Asol'][tSliceToLoad]
        if 'WaveSpeed' in values:
            #self.csol = self.waveSpeed(self.Asol,self.C(self.Psol))
            self.postProcessing(['WaveSpeed'])
        if 'MeanVelocity' in values:
            #self.vsol = self.Qsol/self.Asol
            self.postProcessing(["MeanVelocity"])
        if 'Compliance' in values:
            self.postProcessing(['Compliance'])
        if "linearWavesplit" in values:
            self.postProcessing(["linearWavesplit"])
        if 'Gravity' in values:
            try: self.netGravity = dsetGroup['NetGravity'][tSliceToLoad]
            except Exception: self.warning("vessel.loadSolutionDataRange():  no netGravity stored in solutiondata file for vessel {}".format(self.Id))
        if 'Rotation' in values:
            try: self.rotToGlobalSys = dsetGroup['RotationToGlobal'][tSliceToLoad]
            except Exception: self.warning("vascularNetwork.loadSolutionDataRange():  no rotation matrices stored in solutiondata file for vessel {}".format(self.Id))
        if 'Position' in values:
            try: self.positionStart = dsetGroup['PositionStart'][tSliceToLoad]
            except Exception: self.warning("vascularNetwork.loadSolutionDataRange():  no positionStart stored in solutiondata file for vessel {}".format(self.Id))

    def postProcessing(self, variablesToProcess):
        """
        Input:
            variablesToProcess <list>: [ <str>, ...] variables to process
        """
        for variableToProcess in variablesToProcess:
            if variableToProcess == "WaveSpeed":
                self.csol = self.waveSpeed(self.Asol,self.C(self.Psol))
            elif variableToProcess == "MeanVelocity":
                self.vsol = self.Qsol/self.Asol
            elif variableToProcess == "linearWavesplit":
                self.linearWaveSplitting()
            elif variableToProcess == "Compliance":
                self.Csol = self.C(self.Psol)

    def waveSpeed(self,Area,Compliance,sqrt= np.sqrt):
        """
        Input:
            Area: Area array or float at one ID
            Compliance: Compliance array or float at one ID
            NOTE: it does not check if len(Area) == len(Compliance)
        Output:
            waveSpeed : sqrt(A/(rho*C))
        """
        try : return sqrt(Area/(self.rho*Compliance))
        except Exception:
            errorMessage = "ERROR: classVessel.waveSpeed(): wave speed calculation of vessel id {}!".format(self.Id)
            if (Area < 0).any()  : errorMessage = errorMessage.join(["\n     Area < 0 !!!"])
            if (Compliance < 0).any() : errorMessage = errorMessage.join(["\n    Compliance < 0 !!!"])
            raise ValueError(errorMessage)


    def Impedance(self,Pressure):
        """
        Input:
            Pressure array for the hole grid
        Output:
            Impedance : 1.0/(c*C))
        """
        Compliance = self.C(Pressure)
        Area = self.A(Pressure)
        c = self.waveSpeed(Area, Compliance)
        return 1.0/(c*Compliance)
    
    def updateDaughterList(self, vesselId):
        """Check if vesselId is in self.daughterList, and add if not"""
        
        if vesselId not in self.daughterList:
            self.daughterList.append(vesselId)

    def updateMotherList(self, vesselId):
        """Check if vesselId is in self.daughterList, and add if not"""
        
        if vesselId not in self.motherList:
            self.motherList.append(vesselId)


    def calcVesselResistance(self, P=None, Nintegration=101):
        """ calculate the vessel resistance:
            Rv = 2(gamma + 2)*pi*my*K3, where
            K3 = int(1/Ad**2)dx
            
            P is a list with [P_prox, P_dist]
            """
        

        
        if Nintegration <0:
            # only use gridPoints in integration
            x = np.linspace(0, self.length, self.N)
            Ad = self.AsVector
        else:
            
            if P == None:
                areaProx, areaDist = pi*self.radiusProximal**2, pi*self.radiusDistal**2
            else:
                areaProx = self.A_nID(P, 0)
                areaDist = self.A_nID(P, -1)
            x = np.linspace(0, self.length, self.N)
            Ad = np.linspace(areaProx, areaDist, self.N)
        
        f = 1./(Ad**2)
        K3 = simps(f, x)
        Rv = 2*(self.gamma + 2)*pi*self.my*K3
        
        return Rv

    def calcVesselResistanceVector(self, P):

        x = np.linspace(0, self.length, self.N)
        A = self.A(P)
        R = np.zeros_like(x)
        for n in range(1, len(x)):
            x_tmp = x[:n + 1]
            A_tmp = A[:n + 1]
            N_tmp = len(x_tmp)
            x_new = np.linspace(x_tmp[0], x_tmp[-1], N_tmp)
            A_new = np.interp(x_new, x_tmp, A_tmp)
            
            Ad = A_new
            
            f = 1./(Ad**2)
            K3 = simps(f, x_new)
            Rv_tmp = 2*(self.gamma + 2)*np.pi*self.my*K3
            R[n] = Rv_tmp
        
        return R
    
    
    def calcVesselInertance(self, P=None, Nintegration=101):
        """ calculate the vessel Compliance:
            Lv = rho*K2, where
            K2 = int(1/Ad)dx
            
            P is a list with [P_prox, P_dist]
            """
        
        x = np.linspace(0, self.length, Nintegration)
        if P == None:
            areaProx, areaDist = pi*self.radiusProximal**2, pi*self.radiusDistal**2
        else:
            areaProx = self.A_nID(P, 0)
            areaDist = self.A_nID(P, -1)
        Ad = np.linspace(areaProx, areaDist, Nintegration)
        
        f = 1./(Ad)
        K2 = simps(f, x)
        Lv = self.rho*K2 
        
        
        return Lv
    
    
    def calcVesselCompliance(self, P=None, Nintegration=101):
        """ calculate the vessel compliance at Ps :
            Cv = K1/rho, where
            K1 = int(Ad/cd**2)dx
            """
        
        # TODO: make it possible to have more integrations than self.N
        x = np.linspace(0, self.length, self.N)
        if P == None:
            areaProx, areaDist = pi*self.radiusProximal**2, pi*self.radiusDistal**2
            Pd = self.Ps*np.ones(self.N)
            
        else:
            areaProx = self.A_nID(P, 0)
            areaDist = self.A_nID(P, -1)
            Pd = np.linspace(P[0], P[-1], self.N)
            
        Ad = np.linspace(areaProx, areaDist, self.N)
        
        
        C = self.C(Pd)
        cd = self.waveSpeed(Ad, C)
        
        
        f = Ad/(cd**2)
        K1 = simps(f, x)
        Cv = K1/self.rho
        #Cv = simps(C, x)
        
        
        return Cv
    
    
    def calcVesselWavespeed_in_out(self, P=None):

        if P == None:
            areaProx, areaDist = pi*self.radiusProximal**2, pi*self.radiusDistal**2
            P = self.Ps*np.ones(self.N)
        else:
            areaProx = self.A_nID(P, 0)
            areaDist = self.A_nID(P, -1)
        A = np.linspace(areaProx, areaDist, self.N)
        P = np.linspace(P[0], P[-1], self.N)

        C = self.C(P)
        c = self.waveSpeed(A, C)
        
        c_in, c_out = c[0], c[-1]
        
        return c_in, c_out
        
        
    def linearWaveSplitting(self):
        """
        calculates the linear wave splitting for the hole vessel
        """

        numberOfTimeSteps = len(self.Psol)

        self.PsolF = np.zeros_like(self.Psol)
        self.PsolB = np.zeros_like(self.Psol)
        self.QsolF = np.zeros_like(self.Psol)
        self.QsolB = np.zeros_like(self.Psol)

        for gridNode in range(int(self.N)):
            pf,pb,qf,qb =  self.linearWaveSplittingGridNode(gridNode)
            self.PsolF[1::,[gridNode]] = pf.reshape(numberOfTimeSteps-1,1)
            self.PsolB[1::,[gridNode]] = pb.reshape(numberOfTimeSteps-1,1)
            self.QsolF[1::,[gridNode]] = qf.reshape(numberOfTimeSteps-1,1)
            self.QsolB[1::,[gridNode]] = qb.reshape(numberOfTimeSteps-1,1)

    def linearWaveSplittingGridNode(self,gridNode):
        """
        calculates the linear wave splitting for a given grid node

        return
            PsolF <np.array>
            PsolB <np.array>
            QsolF <np.array>
            QsolB <np.array>
        """

        ## calculate Zo and recast// delete last element
        Zo = self.rho*self.csol[:,[gridNode]]/self.Asol[:,[gridNode]]
        Zo = np.ones_like(Zo)*np.mean(Zo)

        ##calculateing dP and dQ
        dP = self.Psol[:,[gridNode]][1::] - self.Psol[:,[gridNode]][0:-1]
        dQ = self.Qsol[:,[gridNode]][1::] - self.Qsol[:,[gridNode]][0:-1]

        dP_div_Z = self.Psol[:,[gridNode]][1::]/Zo[1::] - self.Psol[:,[gridNode]][0:-1]/Zo[0:-1]
        dQ_multi_Z = self.Qsol[:,[gridNode]][1::]*Zo[1::] - self.Qsol[:,[gridNode]][0:-1]*Zo[0:-1]

        ## calculate dp_f, dp_b and dQ_f, dq_b
        dp_f = (dP + dQ_multi_Z)/2.0
        dp_b = (dP - dQ_multi_Z)/2.0
        dQ_f = (dQ + dP_div_Z)/2.0
        dQ_b = (dQ - dP_div_Z)/2.0

        pf = np.cumsum(dp_f)
        pb = np.cumsum(dp_b)
        qf = np.cumsum(dQ_f)
        qb = np.cumsum(dQ_b)

        return pf,pb,qf,qb

    def linearWaveSplittingGivenArray(self,pressureArray,flowArray,areaArray,waveSpeedArray):
        """
        calculates the linear wave splitting for a given P,Q,A,c (np.arrays) and rho (float)
        return values are: P_forward, P_backward, Q_forward, Q_backward
        """

        ## calculate Zo and recast// delete last element
        Zo = self.rho*waveSpeedArray/areaArray
        Zo = np.ones_like(Zo)*np.mean(Zo)

        ##calculateing dP and dQ
        dP = pressureArray[1::] - pressureArray[0:-1]
        dQ = flowArray[1::] - flowArray[0:-1]

        dP_div_Z = pressureArray[1::]/Zo[1::] - pressureArray[0:-1]/Zo[0:-1]
        dQ_multi_Z = flowArray[1::]*Zo[1::] - flowArray[0:-1]*Zo[0:-1]

        ## calculate dp_f, dp_b and dQ_f, dq_b
        dp_f = (dP + dQ_multi_Z)/2.0
        dp_b = (dP - dQ_multi_Z)/2.0
        dQ_f = (dQ + dP_div_Z)/2.0
        dQ_b = (dQ - dP_div_Z)/2.0

        pf = np.cumsum(dp_f)
        pb = np.cumsum(dp_b)
        qf = np.cumsum(dQ_f)
        qb = np.cumsum(dQ_b)

        return pf,pb,qf,qb

    def update(self, Dict):
        """
            updates the vessel data using a dictionary in from of
            dataDict = {'variableName': value}
        """
        for key,value in iteritems(Dict):
            try:
                self.__getattribute__(key)
                self.__setattr__(key,value)
            except Exception:
                self.warning("Vessel.updateData (vesselId {}) Wrong key: {}, could not update varibale".format(self.Id, key))

    def getVariableValue(self,variableName):
        """
        Returns value of variable with name : variableName
        States Error if not such variable
        """
        try:
            return self.__getattribute__(variableName)
        except Exception:
            self.warning("Vessel.getVariable() : vessel has no variable {}".format(variableName))
            #print "ERROR Vessel.getVariable() : vessel has no variable {}".format(variableName)

    def getVariableDict(self):
        """
        Returns a deep copy of the class variable dict
        """
        return self.__dict__

    def printToConsole(self):
        """
        writes all variables and their values to the console
        """
        print("----------------")
        print("    Vessel %d"%self.Id,"\n")
        for variable,value in iteritems(self.__dict__):
            try:
                print(" {:<20} {:>8}".format(variable,value))
            except Exception: print(" {:<20} {:>8}".format(variable,'None'))

    def caculatePositionAndGravity(self, n, positionEndMother, rotToGlobalSysMother):
        """
        calculates the position and gravity for given time point n and inserts these
        in the corresponding vectors

        Initializing net gravity on the vessels.
        """
        if hasattr(self, "angleXMotherTime") and len(self.angleXMotherTime)>0:
            angleXMother = self.angleXMotherTime[n]
        else:
            angleXMother = self.angleXMother

        try:    angleYMother = self.angleYMotherTime[n]
        except Exception: angleYMother = self.angleYMother

        try:    angleZMother = self.angleZMotherTime[n]
        except Exception: angleZMother = self.angleZMother

        rotToGlobalSys = rotToGlobalSysMother

        # 2. assemble rotation matrices,
        ## 2.1 calculate local rotation matrix to mother depending on
        ### x axis, angleXMother
        if angleXMother != 0:
            rotDaughterMotherX = np.array([[1, 0                     , 0                     ],
                                           [0, cos(angleXMother), sin(angleXMother)],
                                           [0,-sin(angleXMother), cos(angleXMother)]])

            rotToGlobalSys = np.dot(rotDaughterMotherX,rotToGlobalSys)

        ## y axis, angleYMother
        if angleYMother != 0:
            rotDaughterMotherY = np.array ([[ cos(angleYMother), 0,  -sin(angleYMother)],
                                            [ 0                     , 1,  0                      ],
                                            [ sin(angleYMother), 0,  cos(angleYMother) ]])

            rotToGlobalSys = np.dot(rotDaughterMotherY,rotToGlobalSys)

        ## z axis, angleZMother
        if angleZMother != 0:
            rotDaughterMotherZ = np.array ([[  cos(angleZMother), sin(angleZMother),  0],
                                            [ -sin(angleZMother), cos(angleZMother),  0],
                                            [  0,                      0,                       1]])

            rotToGlobalSys = np.dot(rotDaughterMotherZ,rotToGlobalSys)

        # 3. calulate pos end
        self.positionEnd[n][2] = self.length
        self.positionEnd[n] = np.dot(self.positionEnd[n], rotToGlobalSys) + positionEndMother

        gravityVector = np.array([0,0,self.gravityConstant])
        netGravity = np.dot(gravityVector, rotToGlobalSys)[2]

        self.positionStart[n]  = positionEndMother ## positionStart
        self.rotToGlobalSys[n] = rotToGlobalSys
        self.netGravity[n]     = netGravity




