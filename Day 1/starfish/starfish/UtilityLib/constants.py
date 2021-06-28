## Units
from __future__ import print_function, absolute_import
from future.utils import iteritems, iterkeys, viewkeys, viewitems, itervalues, viewvalues
from builtins import input as input3
import numpy as np

# define current network xml description variable
"""
network xml version definition
"""
newestNetworkXmlVersion = '4.2'
from starfish.UtilityLib import networkXml042 as newestNetworkXml
##########################################################################################
#----------------------------------------------------------------------------------------#
#convert unit to SI system
unitsDictSI = {'m':1.0,
            'cm':1.0E-2,
            'mm':1.0E-3,
            'm-1':1.0,
            'cm-1':1.0E2,
            'mm-1':1.0E3,
            'm2':1.0,
            'cm2':1.E-4,
            'mm2':1.E-6,
            'm-2':1.0,
            'cm-2':1.E4,
            'mm-2':1.E6,
            'm3':1.0,
            'cm3':1E-6,
            'mm3':1E-9,
            'm-3':1.0,
            'cm-3':1.0E6,
            'mm-3':1.0E9,
            'Pa':1.0,
            'Pa-1':1.0,
            'mmHg':133.32,
            'mmHg-1':1.0/133.32,
            'ml': 1E-6,
            'ml-1':1E6,
            'N':  1.0,
            'N-1':1.0,
            'dyne':1.e-5,
            'dyne-1':1.e-5,
            's':  1.0,
            's2': 1.0,
            's-2':1.0,
            's-1':1.0,
            'min': 60.0,
            'min-1': 1.0/60.0,
            'min2': 3600.0,
            'min-2': 1.0/3600.0,
            'e6':1.E6,
            'e5':1.E5,
            'e4':1.E4,
            'e3':1.0E3,
            'e2':1.0E2,
            'e-6':1.E-6,
            'e-5':1.E-5,
            'e-4':1.E-4,
            'e-3':1.0E-3,
            'e-2':1.0E-2,
            '' : 1.0,
            ' ':1.0,
            'kg': 1.0,
            'g': 1.E-3,
            'mg': 1.E-6,
            'rad':1.0,
            'deg':np.pi/180.,
            'MB': 1.0,
            'KB': 1./1024.,
            'GB': 1024.,
            }

#----------------------------------------------------------------------------------------#
#convert unit to medical system  Medical
unitsDictMed = {'m':1.0E2,
                'cm':1.0,
                'mm':1.0E-1,
                'm-1':1.0E-2,
                'cm-1':1.0,
                'mm-1':1.0E1,
                'm2':1.0E4,
                'cm2':1.0,
                'mm2':1.E-2,
                'm-2':1.E-4,
                'cm-2':1.0,
                'mm-2':1.E2,
                'm3':1.E6,
                'cm3':1.0,
                'mm3':1.E-3,
                'm-3':1.E-6,
                'cm-3':1.0,
                'mm-3':1.E3,
                'Pa':1.0/133.32,
                'Pa-1':133.32,
                'mmHg':1.0,
                'mmHg-1':1.0,
                'ml': 1.0,
                'ml-1':1.0,
                'N':  1.0,
                'N-1':1.0,
                's':  1.0,
                's2': 1.0,
                's-2':1.0,
                's-1':1.0,
                'min': 60.0,
                'min-1': 1.0/60.0,
                'min2': 3600.0,
                'min-2': 1.0/3600.0,
                'e6':1.E6,
                'e5':1.E5,
                'e4':1.E4,
                'e3':1.0E3,
                'e2':1.0E2,
                'e-6':1.E-6,
                'e-5':1.E-5,
                'e-4':1.E-4,
                'e-3':1.0E-3,
                'e-2':1.0E-2,
                '' : 1.0,
                ' ':1.0,
                'kg': 1.0E3,
                'g': 1.0,
                'mg': 1.E-3,
                'rad':1.0,
                'deg':np.pi/180.0,
                'MB': 1.0,
                'KB': 1./1024.,
                'GB': 1024.,
                }

#----------------------------------------------------------------------------------------#
# units dict for all variables in the system
# based in SI units
variableUnitsSI = { # SimulationContext
                    'CFL': '',
                    'totalTime': 's',
                    'dt': 's',
                    'NumScheme': '',
                    'EqSystem': '',
                    # initialisationControls
                    'initMeanFlow':'m3 s-1',
                    'initMeanPressure':'Pa',
                    'compTotalSys':'m3 Pa-1',
                    'centralVenousPressure': 'Pa',
                    'minimumVenousPressure': 'Pa',
                    'geometryPath' :'',
                    'gravityConstant': 'm s-2',
                    'angleXSystem': '',
                    'gravitationalField': '',
                   # BoundaryConditions
                    'Rt': '',
                    'RtOpen': '',
                    'Topen1': 's',
                    'Topen2': 's',
                    'RtClosed': '',
                    'Tclosed1': 's',
                    'Tclosed2':'s',
                    'Rc':'Pa s m-3',
                    'L':'Pa m-3',
                    'C':'m3 Pa-1',
                    'Ctotal':'m3 Pa-1',
                    'Ca':'m3 Pa-1',
                    'Cm':'m3 Pa-1',
                    'R':'Pa s m-3',
                    'RT':'Pa s m-3',
                    'Z':'Pa s m-3',
                    'Rtotal':'Pa s m-3',
                    'Rp':'Pa s m-3',
                    'Rm':'Pa s m-3',
                    'Rd':'Pa s m-3',
                    'amp':'m3 s-1',
                    'ampConst':'m3 s-1',
                    'Tpulse': 's',
                    'Tspace': 's',
                    'Npulse': '',
                    'scale': '',
                    'freq': 's-1',
                    'runtimeEvaluation': '',
                    'T' : 's',
                    'Emax' : 'Pa m-3',
                    'Emin' : 'Pa m-3',
                    'Tpeak' : 's',
                    'V0' : 'm3',
                    'K'  : 's m-3',
                    'filePathName' :'',
                   # Vessel
                    'id': '',
                    'name': '',
                    'start': '',
                    'end': '',
                    'leftDaughter': '',
                    'rightDaughter': '',
                    'angleYMother': '',
                    'geom': '',
                    'length': 'm',
                    'radiusA': 'm',
                    'radiusB': 'm',
                    'N': '',
                    'complianceType': '',
                    'Pfunc': '',
                    'Ps': 'Pa',
                    'As': 'm2',
                    'wallThickness': 'm',
                    'youngModulus': 'Pa',
                    #'beta': 'Pa m-1',
                    'beta': '',
                    # Fluid properties
                    'my': 'Pa s',
                    'rho': 'kg m-3',
                    'gamma': '',
                    # Plot Variables,
                    'Flow': 'm3 s-1',
                    'Pressure':'Pa'}

#----------------------------------------------------------------------------------------#
# based on medical units
variableUnitsMed  = {
                    # SimulationContext
                    'CFL': '',
                    'totalTime': 's',
                    'dt': 's',
                    'NumScheme': '',
                    'EqSystem': '',
                    # initialisationControls
                    'initMeanFlow':'ml s-1',
                    'initMeanPressure':'mmHg',
                    'compTotalSys':'cm3 mmHg-1',
                    'centralVenousPressure': 'Pa',
                    'minimumVenousPressure': '',
                    'geometryPath': '',
                    'gravityConstant': 'm s-2',
                    'angleXSystem': '',
                    'gravitationalField': '',
                    # BoundaryConditions
                    'Rt': '',
                    'RtOpen': '',
                    'Topen1': 's',
                    'Topen2': 's',
                    'RtClosed': '',
                    'Tclosed1': 's',
                    'Tclosed2':'s',
                    'Rc':'mmHg s cm-3',
                    'L':'mmHg cm-3',
                    'C':'cm3 mmHg-1',
                    'Ctotal':'cm3 mmHg-1',
                    'Ca':'cm3 mmHg-1',
                    'Cm':'cm3 mmHg-1',
                    'R':'mmHg s cm-3',
                    'RT':'mmHg s cm-3',
                    'Z':'mmHg s cm-3',
                    'Rtotal':'mmHg s cm-3',
                    'Rp':'mmHg s cm-3',
                    'Rm':'mmHg s cm-3',
                    'Rd':'mmHg s cm-3',
                    'amp':'ml s',
                    'ampConst':'ml s',
                    'Tpulse': 's',
                    'Tspace': 's',
                    'Npulse': '',
                    'scale': '',
                    'freq': 's-1',
                    'runtimeEvaluation': '',
                    'T' : 's',
                    'Emax' : 'mmHg cm-3',
                    'Emin' : 'mmHg cm-3',
                    'Tpeak' : 's',
                    'V0' : 'cm3',
                    'K'  : 's cm-3',
                    'filePathName' :'',
                   # Vessel
                    'id': '',
                    'name': '',
                    'start': '',
                    'end': '',
                    'leftDaughter': '',
                    'rightDaughter': '',
                    'angleYMother': '',
                    'geom': '',
                    'length': 'cm',
                    'radiusA': 'cm',
                    'radiusB': 'cm',
                    'N': '',
                    'complianceType': '',
                    'Pfunc': '',
                    'Ps': 'mmHg',
                    'As': 'cm2',
                    'wallThickness': 'cm',
                    'youngModulus': 'mmHg',
                    'beta': 'mmHg cm-1',
                    # Fluid properties
                    'my': 'mmHg s',
                    'rho': 'g cm-3',
                    'gamma': '',
                    'centralVenousPressure': 'mmHg',
                    'minimumVenousPressure': 'mmHg',
                    # Plot Variables,
                    'Flow': 'ml s-1',
                    'Pressure':'mmHg'}

##########################################################################################
#### definition of all input/outut variables


variablesDict = {## class Vascular Network
                 # simulation context
                 'description'              : {'type':'str',   'unitSI': None,  'strCases': ['anything'], 'multiVar': False},
                 'totalTime'                : {'type':'float', 'unitSI': 's',  'strCases': None, 'multiVar': False},
                 'dt'                : {'type':'float', 'unitSI': 's',  'strCases': None, 'multiVar': False},
                 'timeSaveBegin'            : {'type':'float', 'unitSI': 's',  'strCases': None, 'multiVar': False},
                 'timeSaveEnd'              : {'type':'float', 'unitSI': 's',  'strCases': None, 'multiVar': False},
                 'minSaveDt'               : {'type':'float', 'unitSI': 's',  'strCases': None, 'multiVar': False},
                 'maxMemory'                : {'type':'float', 'unitSI': 'MB',  'strCases': None, 'multiVar': False},
                 'saveInitialisationPhase'  : {'type':'bool',  'unitSI': None,      'strCases': None, 'multiVar': False},
                 'CFL'                      : {'type':'float', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 'gravitationalField'       : {'type':'bool',  'unitSI': None,      'strCases': None, 'multiVar': False},
                 'gravityConstant'          : {'type':'float', 'unitSI': 'm s-2',   'strCases': None, 'multiVar': False},
                 'centralVenousPressure'    : {'type':'float',      'unitSI': 'Pa',      'strCases': None, 'multiVar': False},
                 'minimumVenousPressure'    : {'type':'float None', 'unitSI': 'Pa',      'strCases': None, 'multiVar': False},
                 # solver calibration
                 'solvingSchemeField'       : {'type':'str',   'unitSI': None, 'strCases': ['MacCormack_Matrix', 'MacCormack_Flux', 'MacCormack_TwoStep'], 'multiVar': False},
                 'solvingSchemeConnections' : {'type':'str',   'unitSI': None, 'strCases': ['Linear'], 'multiVar': False},
                 'rigidAreas'               : {'type':'bool',  'unitSI': None, 'strCases': None, 'multiVar': False},
                 'simplifyEigenvalues'      : {'type':'bool',  'unitSI': None, 'strCases': None, 'multiVar': False},
                 'riemannInvariantUnitBase' : {'type':'str',   'unitSI': None, 'strCases': ['Flow', 'Pressure'], 'multiVar': False},
                 'automaticGridAdaptation'  : {'type':'bool',  'unitSI': None, 'strCases': None, 'multiVar': False},
                 # initialisationControl
                 'initialsationMethod'          : {'type':'str',        'unitSI': None,      'strCases': ['Auto', 'AutoLinearSystem', 'MeanFlow','MeanPressure','ConstantPressure','FromSolution'], 'multiVar': False},
                 'initMeanFlow'                 : {'type':'float',      'unitSI': 'm3 s-1',  'strCases': None, 'multiVar': False},
                 'initMeanPressure'             : {'type':'float',      'unitSI': 'Pa',      'strCases': None, 'multiVar': False},
                 'geometryPath'                 : {'type':'str',    'unitSI': None,        'strCases': ['anything'], 'multiVar': False},
                 'estimateWindkesselCompliance' : {'type':'str',        'unitSI': None,      'strCases': ['Tree','Wk3','System','No'], 'multiVar': False},
                 'compPercentageWK3'            : {'type':'float',      'unitSI': None,      'strCases': None, 'multiVar': False},
                 'compPercentageTree'           : {'type':'float',      'unitSI': None,      'strCases': None, 'multiVar': False},
                 'compTotalSys'                 : {'type':'float',      'unitSI': 'm3 Pa-1', 'strCases': None, 'multiVar': False},
                 # global fluid
                 'my'                : {'type':'float',      'unitSI': 'Pa s',   'strCases': None, 'multiVar': False},
                 'rho'               : {'type':'float',      'unitSI': 'kg m-3', 'strCases': None, 'multiVar': False},
                 'gamma'             : {'type':'float None', 'unitSI':  None,    'strCases': None, 'multiVar': False},
                 # external stimuli
                 'startAngle' : {'type':'float', 'unitSI':'rad', 'strCases': None, 'multiVar': False},
                 'stopAngle'  : {'type':'float', 'unitSI':'rad', 'strCases': None, 'multiVar': False},
                 'startTime'  : {'type':'float', 'unitSI':'s', 'strCases': None, 'multiVar': False},
                 'duration'   : {'type':'float', 'unitSI':'s', 'strCases': None, 'multiVar': False},
                 ## class Communicators
                 'comType'           : {'type':'str', 'unitSI': None, 'strCases': ['CommunicatorRealTimeViz','CommunicatorBaroreceptor'], 'multiVar': False},
                 'comId'             : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 'vesselId'          : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 'node'              : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 'dn'                : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 'quantitiesToPlot'  : {'type':'str', 'unitSI': None, 'strCases': ['Pressure','Area','Flow','elastance'], 'multiVar': True},
                 ## boundary conditions
                 # flux bc
                 'amp'               : {'type':'float',  'unitSI': 'm3 s-1',    'strCases': None, 'multiVar': False},
                 'ampConst'          : {'type':'float',  'unitSI': 'm3 s-1',    'strCases': None, 'multiVar': False},
                 'Npulse'            : {'type':'float',  'unitSI': None,        'strCases': None, 'multiVar': False},
                 'Tpulse'            : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'freq'              : {'type':'float',  'unitSI': 's-1',       'strCases': None, 'multiVar': False},
                 'Tspace'            : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'runtimeEvaluation' : {'type':'bool',   'unitSI': None,        'strCases': None, 'multiVar': False},
                 'filePathName'      : {'type':'str',    'unitSI': None,        'strCases': ['anything'], 'multiVar': False},
                 'prescribe'         : {'type':'str',    'unitSI': None,        'strCases': ['influx','total'], 'multiVar': False},
                 'gaussC'            : {'type':'float',  'unitSI': None,        'strCases': None, 'multiVar': False},
                 # reflection coefficients
                 'Rt'                : {'type':'float',  'unitSI': None,        'strCases': None, 'multiVar': False},
                 'RtOpen'            : {'type':'float',  'unitSI': None,        'strCases': None, 'multiVar': False},
                 'Topen1'            : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'Topen2'            : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'RtClosed'          : {'type':'float',  'unitSI': None,        'strCases': None, 'multiVar': False},
                 'Tclosed1'          : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'Tclosed2'          : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 # windkessel
                 'Rtotal'            : {'type':'float None',      'unitSI': 'Pa s m-3', 'strCases': None, 'multiVar': False},
                 'Rc'                : {'type':'float str None',  'unitSI': 'Pa s m-3', 'strCases': ['VesselImpedance'],  'multiVar': False},
                 'Z'                 : {'type':'float str',       'unitSI': 'Pa s m-3', 'strCases': ['VesselImpedance'], 'multiVar': False},
                 'L'                 : {'type':'float',       'unitSI': 'Pa m-3', 'strCases': None, 'multiVar': False},
                 'C'                 : {'type':'float',           'unitSI': 'm3 Pa-1',  'strCases': None, 'multiVar': False},
                 'Ctotal'            : {'type':'float None',           'unitSI': 'm3 Pa-1',  'strCases': None, 'multiVar': False},
                 'Rp'                : {'type':'float None',      'unitSI': 'Pa s m-3', 'strCases': None, 'multiVar': False},
                 'Rm'                : {'type':'float None',      'unitSI': 'Pa s m-3', 'strCases': None, 'multiVar': False},
                 'Rd'                : {'type':'float None',      'unitSI': 'Pa s m-3', 'strCases': None, 'multiVar': False},
                 'Ca'                : {'type':'float None',           'unitSI': 'm3 Pa-1',  'strCases': None, 'multiVar': False},
                 'Cm'                : {'type':'float None',           'unitSI': 'm3 Pa-1',  'strCases': None, 'multiVar': False},
                 # heart model
                 'T'                 : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'Emax'              : {'type':'float',  'unitSI': 'Pa m-3',    'strCases': None, 'multiVar': False},
                 'Emin'              : {'type':'float',  'unitSI': 'Pa m-3',    'strCases': None, 'multiVar': False},
                 'Tpeak'             : {'type':'float',  'unitSI': 's',         'strCases': None, 'multiVar': False},
                 'V0'                : {'type':'float',  'unitSI': 'm3',        'strCases': None, 'multiVar': False},
                 'K'                 : {'type':'float',  'unitSI': 's m-3',     'strCases': None, 'multiVar': False},
                 'residualName'      : {'type':'str',  'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 ## class vessel
                 # attributes
                 'name'              : {'type':'str',  'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'Id'                : {'type':'int',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 # topology
                 'leftDaughter'      : {'type':'int None list',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 'rightDaughter'     : {'type':'int None',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 'angleYMother'      : {'type':'float None', 'unitSI': 'rad',    'strCases': None, 'multiVar': False},
                 # geometry
                 'geometryType'      : {'type':'str',    'unitSI': None,    'strCases': ['uniform','cone','constriction','mtStig2016', 'fromFile'], 'multiVar': False},
                 'length'            : {'type':'float',  'unitSI': 'm',     'strCases': None, 'multiVar': False},
                 'radiusProximal'    : {'type':'float',  'unitSI': 'm',     'strCases': None, 'multiVar': False},
                 'radiusDistal'      : {'type':'float',  'unitSI': 'm',     'strCases': None, 'multiVar': False},
                 'N'                 : {'type':'int',    'unitSI': None,    'strCases': None, 'multiVar': False},
                 # compliances
                 'complianceType'    : {'type':'str',    'unitSI': None,   'strCases': ['Laplace', 'Laplace2', 'LaplaceAdan', 'Laplace3', 'Exponential', 'Hayashi', 'Reymond'], 'multiVar': False},
                 'constantCompliance': {'type':'bool',   'unitSI': None,   'strCases': None, 'multiVar': False},
                 'externalPressure'  : {'type':'float',  'unitSI': 'Pa',   'strCases': None, 'multiVar': False},
                 'Ps'                : {'type':'float',  'unitSI': 'Pa',   'strCases': None, 'multiVar': False},
                 'As'                : {'type':'float None','unitSI': 'm2','strCases': None, 'multiVar': False},
                 'wallThickness'     : {'type':'float',  'unitSI': 'm',    'strCases': None, 'multiVar': False},
                 'youngModulus'      : {'type':'float',  'unitSI': 'm2',   'strCases': None, 'multiVar': False},
                 'beta'              : {'type':'float',  'unitSI': 'Pa',   'strCases': None, 'multiVar': False},
                 'betaLaplace'       : {'type':'float',  'unitSI': 'Pa m-1',   'strCases': None, 'multiVar': False},
                 'betaExponential'   : {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'betaHayashi'       : {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'Cs'                : {'type':'float',  'unitSI': 'm2 Pa-1', 'strCases': None, 'multiVar': False},
                 'PmaxC'             : {'type':'float',  'unitSI':  'Pa',     'strCases': None, 'multiVar': False},
                 'Pwidth'            : {'type':'float',  'unitSI':  'Pa',     'strCases': None, 'multiVar': False},
                 'a1'                : {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'b1'                : {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 # fluid
                 'applyGlobalFluid'  : {'type':'bool',   'unitSI': None,   'strCases': None, 'multiVar': False},
                 # baroreceptors
                 'baroId'            : {'type':'int',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 'cellMLBaroreceptorModel'    : {'type':'bool',  'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'receptorType'      : {'type':'str',  'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'modelName'         : {'type':'str',  'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'vesselIds'         : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': True},
                 'vesselIdLeft'      : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 'vesselIdRight'     : {'type':'int', 'unitSI': None, 'strCases': None, 'multiVar': False},
                 ## Pettersen pars
                 'modelName' : {'type':'str',  'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'L0': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'n0': {'type':'float',  'unitSI':  's-1',     'strCases': None, 'multiVar': False},
                 'g': {'type':'float',  'unitSI':  's-1',     'strCases': None, 'multiVar': False},
                 'tau1': {'type':'float',  'unitSI':  's-1',     'strCases': None, 'multiVar': False},
                 'tau2': {'type':'float',  'unitSI':  's-1',     'strCases': None, 'multiVar': False},
                 'Gp': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'Gs': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'HR0': {'type':'float',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 'HRmax': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'HRmin': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 ## Bugenhagen pars
                 'bgh': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 ## Combined pars
                 'pn':  {'type':'float',  'unitSI':  "Pa",     'strCases': None, 'multiVar': False},
                 'ka':  {'type':'float',  'unitSI':  "Pa",     'strCases': None, 'multiVar': False},
                 'tau_z':  {'type':'float',  'unitSI':  "s",     'strCases': None, 'multiVar': False},
                 'cR':  {'type':'float',  'unitSI':  "Pa s m-3",     'strCases': None, 'multiVar': False},
                 'cT':  {'type':'float',  'unitSI':  "s",     'strCases': None, 'multiVar': False},
                 'cE':  {'type':'float',  'unitSI':  "Pa m-3",     'strCases': None, 'multiVar': False},
                 'cVusv': {'type':'float',  'unitSI':  "m3",     'strCases': None, 'multiVar': False},
                 'carotid_G_R':  {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'carotid_G_T':  {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'carotid_G_Emax':  {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'carotid_G_Vusv': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'aortic_G_R':  {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'aortic_G_T':  {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'aortic_G_Emax':  {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 'aortic_G_Vusv': {'type':'float',  'unitSI':  None,     'strCases': None, 'multiVar': False},
                 ### random variables
                 'distributionType': {'type':'str',    'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'a'               : {'type':'float',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 'b'               : {'type':'float',  'unitSI': None,     'strCases': None, 'multiVar': False},
                 'location'        : {'type':'str',    'unitSI': None,     'strCases': ['anything'], 'multiVar': False},
                 'randomInputType' : {'type':'str',    'unitSI': None,     'strCases': ['generalRandomInput','parametricRandomInput'], 'multiVar': False},
                 }

##########################################################################################
## Vascular Polynomial Chaos


polynomialChaosDistributions = ['Uniform','Normal']

