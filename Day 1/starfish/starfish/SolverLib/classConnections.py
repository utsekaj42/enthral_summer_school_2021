from __future__ import print_function, absolute_import
from future.utils import iteritems, iterkeys, viewkeys, viewitems, itervalues, viewvalues
from builtins import input as input3
import sys, os
import numpy as np
from numpy.linalg import solve
from scipy.optimize import fsolve

from pprint import pprint as pp
#cur = os.path.dirname( os.path.realpath( __file__ ) )
#sys.path.append(cur+'/../')
from starfish.UtilityLib import classStarfishBaseObject as cSBO

from copy import copy as copy
import sympy
from sympy.matrices import Matrix



class Junction():
    """
    General junction object with Nm mother vessels, and Nd daughter vessels
    """
    
    def __init__(self, listOfMothers, listOfMotherSys,
                    listOfDaughters, listOfDaughterSys,
                    currentMemoryIndex, dt, rigidAreas, solvingScheme):
        
        self.listOfMothers = listOfMothers
        self.listOfDaughters = listOfDaughters
        
        self.listOfMotherIds = []
        self.listOfDaughterIds = []

        self.type = 'Junction'
        
        self.name = 'Junction'
        self.names = []
        
        self.vesselDict = {} # dictionary containing necessary variables etc
        # Todo: check if memory usage is too high (saving Psol etc in dicts)
        #===================================================
        # init variables for mother vessels
        for n, vessel_m in enumerate(listOfMothers):
            tmpDict = {}
            
            self.name += '_' + str(vessel_m)
            
            tmpDict['rho'] = vessel_m.rho
            tmpDict['my'] = vessel_m.my
            tmpDict['systemEquations'] = listOfMotherSys[n]
            tmpDict['z'] = vessel_m.z
            tmpDict['position'] = -1
            tmpDict['A_func'] = vessel_m.A_nID
            tmpDict['Psol'] = vessel_m.Psol
            tmpDict['Qsol'] = vessel_m.Qsol
            tmpDict['Asol'] = vessel_m.Asol
            tmpDict['domega_2'] = 0 # unknown
            
            self.vesselDict[vessel_m.Id] = tmpDict
            

            self.listOfMotherIds.append(vessel_m.Id)
            self.names.append(str(vessel_m.Id))

        #===================================================
        # init variables for daughter vessels
        for n, vessel_d in enumerate(listOfDaughters):
            
            tmpDict = {}
            self.name += '_' + str(vessel_d)
            
            tmpDict['rho'] = vessel_d.rho
            tmpDict['my'] = vessel_d.my
            tmpDict['systemEquations'] = listOfDaughterSys[n]
            tmpDict['z'] = vessel_d.z
            tmpDict['position'] = 0
            tmpDict['A_func'] = vessel_d.A_nID
            tmpDict['Psol'] = vessel_d.Psol
            tmpDict['Qsol'] = vessel_d.Qsol
            tmpDict['Asol'] = vessel_d.Asol
            tmpDict['domega_1'] = 0 # unknown
            
            self.vesselDict[vessel_d.Id] = tmpDict
            
            self.listOfDaughterIds.append(vessel_d.Id)
            self.names.append(str(vessel_d.Id))
            
            

        self.dt = dt
        self.currentMemoryIndex  = currentMemoryIndex
        self.rigidAreas = rigidAreas
        
        # equations to solve in f solve
        self.J_py, self.F_py, self.variablesList_sym, self.X_sym = self.initializeResidualFuncsAndJacobian()
        
        self._callfcn = self.callNonLinear

    def __call__(self):
        return self._callfcn()
        

    
    def initializeResidualFuncsAndJacobian(self):
        """
        compute residual functions and (inverse) Jacobian for given junction using sympy, and convert/lambdify
        into numerical python functions. 
        
        1) create sympy symbols (rho, Q, P, A, dw1, dw2, R11, R12, R21, R22) for all mothers
        2) add unknown (dw2) to X and symbols from 1) to variablesList
        3) create sympy symbols (rho, Q, P, A, dw1, dw2, R11, R12, R21, R22) for all daughters
        4) add unknown (dw1) to X and symbols from 2) to variablesList
        5) create symbolic residual functions associated with continuity eq (f1) and pressure eqs. (f2-fN) and add to F (list of residual functions)
        6) compute symbolic matrix/expression of the Jacobian and inverse of the Jacobian (Jinv)
        7) convert/lambdify J and F into J_py and F_py (will take numerical values associated with variablesList as input)
        """
        listOfMotherIds = self.listOfMotherIds
        listOfDaughterIds = self.listOfDaughterIds
        vesselDict = {} # dict of vesselIds with associated dict of sympy symbols
        variablesList = [] # list of variables that will be inputs for converted/lambdified function
        X = [] # list of unknowns
        
        #===================================================
        # create symbols for all mother vessels
        for Id_m in listOfMotherIds:
            tmpDict = {}
            tmpDict['rho_m{0}'.format(Id_m)] = sympy.symbols('rho_m{0}'.format(Id_m))
            tmpDict['Q_m{0}'.format(Id_m)] = sympy.symbols('Q_m{0}'.format(Id_m))
            tmpDict['P_m{0}'.format(Id_m)] = sympy.symbols('P_m{0}'.format(Id_m))
            tmpDict['A_m{0}'.format(Id_m)] = sympy.symbols('A_m{0}'.format(Id_m))
            tmpDict['dw1_m{0}'.format(Id_m)], tmpDict['dw2_m{0}'.format(Id_m)] = sympy.symbols('dw1_m{0}, dw2_m{0}'.format(Id_m))
            tmpDict['R11_m{0}'.format(Id_m)], tmpDict['R12_m{0}'.format(Id_m)] = sympy.symbols('R11_m{0}, R12_m{0}'.format(Id_m))
            tmpDict['R21_m{0}'.format(Id_m)], tmpDict['R22_m{0}'.format(Id_m)] = sympy.symbols('R21_m{0}, R22_m{0}'.format(Id_m))
            
            vesselDict[Id_m] = tmpDict
            
            X.append('dw2_m{0}'.format(Id_m)) # add unknown to X(backward traveling characteristic variable)
            
            variablesList.append('rho_m{0}'.format(Id_m))
            variablesList.append('Q_m{0}'.format(Id_m))
            variablesList.append('P_m{0}'.format(Id_m))
            variablesList.append( 'A_m{0}'.format(Id_m))
            variablesList.append('dw1_m{0}'.format(Id_m))
            variablesList.append('dw2_m{0}'.format(Id_m))
            variablesList.append('R11_m{0}'.format(Id_m))
            variablesList.append('R12_m{0}'.format(Id_m))
            variablesList.append('R21_m{0}'.format(Id_m))
            variablesList.append('R22_m{0}'.format(Id_m))
            

        #===================================================
        # create symbols for all daughter vessels
        for Id_d in listOfDaughterIds:
            tmpDict = {}
            tmpDict['rho_d{0}'.format(Id_d)] = sympy.symbols('rho_d{0}'.format(Id_d))
            tmpDict['Q_d{0}'.format(Id_d)] = sympy.symbols('Q_d{0}'.format(Id_d))
            tmpDict['P_d{0}'.format(Id_d)] = sympy.symbols('P_d{0}'.format(Id_d))
            tmpDict['A_d{0}'.format(Id_d)] = sympy.symbols('A_d{0}'.format(Id_d))
            tmpDict['dw1_d{0}'.format(Id_d)], tmpDict['dw2_d{0}'.format(Id_d)] = sympy.symbols('dw1_d{0}, dw2_d{0}'.format(Id_d))
            tmpDict['R11_d{0}'.format(Id_d)], tmpDict['R12_d{0}'.format(Id_d)] = sympy.symbols('R11_d{0}, R12_d{0}'.format(Id_d))
            tmpDict['R21_d{0}'.format(Id_d)], tmpDict['R22_d{0}'.format(Id_d)] = sympy.symbols('R21_d{0}, R22_d{0}'.format(Id_d))
            
            vesselDict[Id_d] = tmpDict
            
            X.append('dw1_d{0}'.format(Id_d)) # add unknown to X(forward traveling characteristic variable)
            
            variablesList.append('rho_d{0}'.format(Id_d))
            variablesList.append('Q_d{0}'.format(Id_d))
            variablesList.append('P_d{0}'.format(Id_d))
            variablesList.append( 'A_d{0}'.format(Id_d))
            variablesList.append('dw1_d{0}'.format(Id_d))
            variablesList.append('dw2_d{0}'.format(Id_d))
            variablesList.append('R11_d{0}'.format(Id_d))
            variablesList.append('R12_d{0}'.format(Id_d))
            variablesList.append('R21_d{0}'.format(Id_d))
            variablesList.append('R22_d{0}'.format(Id_d))
            

        F = [] # list of residualfunctions
        f1 = 0 # residualfunction associated with continuity equation
        
        ###################################
        # continuity eq.
        ###################################
        
        # add flow into all junctions
        for Id_m in listOfMotherIds:
            
            Q_prev = vesselDict[Id_m]['Q_m{0}'.format(Id_m)]
            dw1 = vesselDict[Id_m]['dw1_m{0}'.format(Id_m)]
            dw2 = vesselDict[Id_m]['dw2_m{0}'.format(Id_m)]
            R21, R22 = vesselDict[Id_m]['R21_m{0}'.format(Id_m)], vesselDict[Id_m]['R22_m{0}'.format(Id_m)]
            
            Q_m_tmp =  Q_prev + dw1*R21 + dw2*R22 # substitution: Q written in terms of characteristic variables
            f1 += Q_m_tmp
        
        # subtract flow out of all junctions
        for Id_d in listOfDaughterIds:
            
            Q_prev = vesselDict[Id_d]['Q_d{0}'.format(Id_d)]
            dw1 = vesselDict[Id_d]['dw1_d{0}'.format(Id_d)]
            dw2 = vesselDict[Id_d]['dw2_d{0}'.format(Id_d)]
            R21, R22 = vesselDict[Id_d]['R21_d{0}'.format(Id_d)], vesselDict[Id_d]['R22_d{0}'.format(Id_d)]
            
            Q_d_tmp =  Q_prev + dw1*R21 + dw2*R22 # substitution: Q written in terms of characteristic variables
            f1 -= Q_d_tmp
        
        F.append(f1)
        

        ###################################
        # pressure eqs.
        ###################################
        
        Id_m = listOfMotherIds[0]
            
        rho_m = vesselDict[Id_m]['rho_m{0}'.format(Id_m)]
        
        Q_prev = vesselDict[Id_m]['Q_m{0}'.format(Id_m)]
        P_prev = vesselDict[Id_m]['P_m{0}'.format(Id_m)]
        A_m = vesselDict[Id_m]['A_m{0}'.format(Id_m)]
        
        dw1 = vesselDict[Id_m]['dw1_m{0}'.format(Id_m)]
        dw2 = vesselDict[Id_m]['dw2_m{0}'.format(Id_m)]
        R21, R22 = vesselDict[Id_m]['R21_m{0}'.format(Id_m)], vesselDict[Id_m]['R22_m{0}'.format(Id_m)]
        R11, R12 = vesselDict[Id_m]['R11_m{0}'.format(Id_m)], vesselDict[Id_m]['R12_m{0}'.format(Id_m)]
        
        Q_m_tmp =  Q_prev + dw1*R21 + dw2*R22
        P_m_tmp =  P_prev + dw1*R11 + dw2*R12
        
        if len(listOfMotherIds) > 1:
            for Id_m2 in listOfMotherIds[1:]:
                
                rho_m2 = vesselDict[Id_m2]['rho_m{0}'.format(Id_m2)]
                
                Q_prev = vesselDict[Id_m2]['Q_m{0}'.format(Id_m2)]
                P_prev = vesselDict[Id_m2]['P_m{0}'.format(Id_m2)]
                A_m2 = vesselDict[Id_m2]['A_m{0}'.format(Id_m2)]
                
                dw1 = vesselDict[Id_m2]['dw1_m{0}'.format(Id_m2)]
                dw2 = vesselDict[Id_m2]['dw2_m{0}'.format(Id_m2)]
                R21, R22 = vesselDict[Id_m2]['R21_m{0}'.format(Id_m2)], vesselDict[Id_m2]['R22_m{0}'.format(Id_m2)]
                R11, R12 = vesselDict[Id_m2]['R11_m{0}'.format(Id_m2)], vesselDict[Id_m2]['R12_m{0}'.format(Id_m2)]
                
                Q_m2_tmp =  Q_prev + dw1*R21 + dw2*R22
                P_m2_tmp =  P_prev + dw1*R11 + dw2*R12
                
                F.append(P_m_tmp + 0.5*rho_m*(Q_m_tmp/A_m)**2 - P_m2_tmp - 0.5*rho_m2*(Q_m2_tmp/A_m2)**2)
        
        for Id_d in listOfDaughterIds:
            
            rho_d = vesselDict[Id_d]['rho_d{0}'.format(Id_d)]
            
            Q_prev = vesselDict[Id_d]['Q_d{0}'.format(Id_d)]
            P_prev = vesselDict[Id_d]['P_d{0}'.format(Id_d)]
            A_d = vesselDict[Id_d]['A_d{0}'.format(Id_d)]
            
            dw1 = vesselDict[Id_d]['dw1_d{0}'.format(Id_d)]
            dw2 = vesselDict[Id_d]['dw2_d{0}'.format(Id_d)]
            R21, R22 = vesselDict[Id_d]['R21_d{0}'.format(Id_d)], vesselDict[Id_d]['R22_d{0}'.format(Id_d)]
            R11, R12 = vesselDict[Id_d]['R11_d{0}'.format(Id_d)], vesselDict[Id_d]['R12_d{0}'.format(Id_d)]
            
            Q_d_tmp =  Q_prev + dw1*R21 + dw2*R22
            P_d_tmp =  P_prev + dw1*R11 + dw2*R12
            
            F.append(P_m_tmp + 0.5*rho_m*(Q_m_tmp/A_m)**2 - P_d_tmp - 0.5*rho_d*(Q_d_tmp/A_d)**2)

        
        F = Matrix([F]).T # convert list of residualfunctions into sympy Matrix/expression
        
        def jacobi(i,j):
            
            return sympy.diff(F[i], X[j])
        
        N = len(X)
        
        J = Matrix(N, N, jacobi) # create jacobian. sympy Matrix/expression
        #Jinv = J.inv()
        #Jinv_py = sympy.lambdify(variablesList, Jinv)
        J_py = sympy.lambdify(variablesList, J)
        F_py = sympy.lambdify(variablesList, F)
        
        
        return J_py, F_py, variablesList, X

    
    def callNonLinear(self):

        """
        Call function for vessel-vessel junction
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        
        listOfMotherIds = self.listOfMotherIds
        listOfDaughterIds = self.listOfDaughterIds
        Nunknown = len(listOfMotherIds) + len(listOfDaughterIds)
        Xold = np.ones((Nunknown, 1))
        x_indx = 0
        variablesList = []

        #===================================================
        # Preprocess
        #===================================================
        
        #===================================================
        # update variables for mother vessels from previous time step
        for vesselId_m in listOfMotherIds:

            # get variables from previous timestep
            P = self.vesselDict[vesselId_m]['Psol'][n]
            Q = self.vesselDict[vesselId_m]['Qsol'][n]
            A = self.vesselDict[vesselId_m]['Asol'][n]
            pos_m = -1
            
            self.vesselDict[vesselId_m]['P0'] = P[pos_m]
            self.vesselDict[vesselId_m]['Q0'] = Q[pos_m]
            self.vesselDict[vesselId_m]['A0'] = A[pos_m]
            
            L, R, LMBD, Z1, Z2, domega_1 = self.vesselDict[vesselId_m]['systemEquations'].updateLARL(P, Q, A, pos_m)
            domega_2 = self.vesselDict[vesselId_m]['domega_2']

            R_11 = R[0][0]
            R_12 = R[0][1]
            R_21 = R[1][0]
            R_22 = R[1][1]

            self.vesselDict[vesselId_m]['domega_1'] = domega_1
            self.vesselDict[vesselId_m]['R_11'] = R_11
            self.vesselDict[vesselId_m]['R_12'] = R_12
            self.vesselDict[vesselId_m]['R_21'] = R_21
            self.vesselDict[vesselId_m]['R_22'] = R_22
            
            Xold[x_indx, 0] = domega_2 # use previous value as initial guess
            x_indx_variableList = x_indx*10 + 5
            A_indx_variableList = x_indx*10 + 3
            self.vesselDict[vesselId_m]['x_indx'] = x_indx
            self.vesselDict[vesselId_m]['x_indx_variableList'] = x_indx_variableList
            self.vesselDict[vesselId_m]['A_indx_variableList'] = A_indx_variableList
            
            
            variablesList.append(self.vesselDict[vesselId_m]['rho'])
            variablesList.append(Q[pos_m])
            variablesList.append(P[pos_m])
            variablesList.append(A[pos_m])
            variablesList.append(domega_1)
            variablesList.append(domega_2)
            variablesList.append(R_11)
            variablesList.append(R_12)
            variablesList.append(R_21)
            variablesList.append(R_22)
            x_indx += 1

        #===================================================
        # update variables for daughter vessels from previous time step
        for vesselId_d in listOfDaughterIds:

            # get variables from previous timestep
            P = self.vesselDict[vesselId_d]['Psol'][n]
            Q = self.vesselDict[vesselId_d]['Qsol'][n]
            A = self.vesselDict[vesselId_d]['Asol'][n]
            pos_d = 0
            
            self.vesselDict[vesselId_d]['P0'] = P[pos_d]
            self.vesselDict[vesselId_d]['Q0'] = Q[pos_d]
            self.vesselDict[vesselId_d]['A0'] = A[pos_d]
            
            L, R, LMBD, Z1, Z2, domega_2 = self.vesselDict[vesselId_d]['systemEquations'].updateLARL(P, Q, A, pos_d)
            domega_1 = self.vesselDict[vesselId_d]['domega_1']

            R_11 = R[0][0]
            R_12 = R[0][1]
            R_21 = R[1][0]
            R_22 = R[1][1]

            self.vesselDict[vesselId_d]['domega_2'] = domega_2
            self.vesselDict[vesselId_d]['R_11'] = R_11
            self.vesselDict[vesselId_d]['R_12'] = R_12
            self.vesselDict[vesselId_d]['R_21'] = R_21
            self.vesselDict[vesselId_d]['R_22'] = R_22
            
            Xold[x_indx, 0] = domega_1 # use previous value as initial guess
            x_indx_variableList = x_indx*10 + 4
            A_indx_variableList = x_indx*10 + 3
            self.vesselDict[vesselId_d]['x_indx'] = x_indx
            self.vesselDict[vesselId_d]['x_indx_variableList'] = x_indx_variableList
            self.vesselDict[vesselId_d]['A_indx_variableList'] = A_indx_variableList
            
            
            variablesList.append(self.vesselDict[vesselId_d]['rho'])
            variablesList.append(Q[pos_d])
            variablesList.append(P[pos_d])
            variablesList.append(A[pos_d])
            variablesList.append(domega_1)
            variablesList.append(domega_2)
            variablesList.append(R_11)
            variablesList.append(R_12)
            variablesList.append(R_21)
            variablesList.append(R_22)
            x_indx += 1
        


        epsilonvalues = np.ones_like(Xold)
        epsilonP = 1
        epsilonQ = 1
        epsilonlimitQ = 1e-10 # Todo: link to error of the scheme
        epsilonlimitP = 1e-5  # Todo: link to error of the scheme

        Niterations = 0

        #===================================================
        # Solve using iterative newton Rhapson solver
        #===================================================
        while epsilonQ>epsilonlimitQ or epsilonP > epsilonlimitP:

            Jinv = np.array(np.linalg.inv(self.J_py(*variablesList)))
            #Jinv = np.array(self.J_py(*variablesList))
            F = np.array(self.F_py(*variablesList))


            Xnew = Xold - np.dot(Jinv, F)
            #exit()

            epsilonvalues = np.abs(F)
            epsilonQ = epsilonvalues[0, 0]
            epsilonP = np.amax(epsilonvalues[1:, 0])
            Niterations += 1
            
            #===================================================
            # update  domega_2 and A in variablesList for all mothers for current iteration
            for vesselId_m in listOfMotherIds:
                
                P0 = self.vesselDict[vesselId_m]['P0']
                domega_1 = self.vesselDict[vesselId_m]['domega_1']
                R_11 = self.vesselDict[vesselId_m]['R_11']
                R_12 = self.vesselDict[vesselId_m]['R_12']
                
                x_indx = self.vesselDict[vesselId_m]['x_indx']
                domega_2 = Xnew[x_indx, 0]
                
                P = P0 + R_11*domega_1 + R_12*domega_2
                A = self.vesselDict[vesselId_m]['A_func']([P], -1)
                
                x_indx_variableList = self.vesselDict[vesselId_m]['x_indx_variableList']
                A_indx_variableList = self.vesselDict[vesselId_m]['A_indx_variableList']
                variablesList[x_indx_variableList] = domega_2
                variablesList[A_indx_variableList] = A

            #===================================================
            # update  domega_1 and A in variablesList for all daughters for current iteration
            for vesselId_d in listOfDaughterIds:
                
                P0 = self.vesselDict[vesselId_d]['P0']
                domega_2 = self.vesselDict[vesselId_d]['domega_2']
                R_11 = self.vesselDict[vesselId_d]['R_11']
                R_12 = self.vesselDict[vesselId_d]['R_12']
                
                x_indx = self.vesselDict[vesselId_d]['x_indx']
                domega_1 = Xnew[x_indx, 0]
                
                P = P0 + R_11*domega_1 + R_12*domega_2
                A = self.vesselDict[vesselId_d]['A_func']([P], 0)
                
                x_indx_variableList = self.vesselDict[vesselId_d]['x_indx_variableList']
                A_indx_variableList = self.vesselDict[vesselId_d]['A_indx_variableList']
                variablesList[x_indx_variableList] = domega_1
                variablesList[A_indx_variableList] = A
                
            
            Xold = Xnew
            
            if Niterations > 60:
                print('its exceeded')
                print ('\n', epsilonvalues)
                break
                

        #===================================================
        #Postprocess solutions
        #===================================================
        for vesselId_m in listOfMotherIds:
            
            P0 = self.vesselDict[vesselId_m]['P0']
            Q0 = self.vesselDict[vesselId_m]['Q0']
            domega_1 = self.vesselDict[vesselId_m]['domega_1']
            pos_m = -1
            
            R_11 = self.vesselDict[vesselId_m]['R_11']
            R_12 = self.vesselDict[vesselId_m]['R_12']
            R_21 = self.vesselDict[vesselId_m]['R_21']
            R_22 = self.vesselDict[vesselId_m]['R_22']
            
            x_indx = self.vesselDict[vesselId_m]['x_indx']
            domega_2 = Xnew[x_indx, 0]
            
            Q = Q0 + R_21*domega_1 + R_22*domega_2
            P = P0 + R_11*domega_1 + R_12*domega_2
            A = self.vesselDict[vesselId_m]['A_func']([P], pos_m)

            self.vesselDict[vesselId_m]['Qsol'][n + 1][pos_m] = Q
            self.vesselDict[vesselId_m]['Psol'][n + 1][pos_m] = P
            self.vesselDict[vesselId_m]['Asol'][n + 1][pos_m] = A
            
            
        for vesselId_d in listOfDaughterIds:
            
            P0 = self.vesselDict[vesselId_d]['P0']
            Q0 = self.vesselDict[vesselId_d]['Q0']
            domega_2 = self.vesselDict[vesselId_d]['domega_2']
            pos_d = 0
            
            R_11 = self.vesselDict[vesselId_d]['R_11']
            R_12 = self.vesselDict[vesselId_d]['R_12']
            R_21 = self.vesselDict[vesselId_d]['R_21']
            R_22 = self.vesselDict[vesselId_d]['R_22']
            
            x_indx = self.vesselDict[vesselId_d]['x_indx']
            domega_1 = Xnew[x_indx, 0]
            
            Q = Q0 + R_21*domega_1 + R_22*domega_2
            P = P0 + R_11*domega_1 + R_12*domega_2
            A = self.vesselDict[vesselId_d]['A_func']([P], pos_d)

            self.vesselDict[vesselId_d]['Qsol'][n + 1][pos_d] = Q
            self.vesselDict[vesselId_d]['Psol'][n + 1][pos_d] = P
            self.vesselDict[vesselId_d]['Asol'][n + 1][pos_d] = A
            

class Link():
    """
    Link object represends the connection between two vessels
    """
    def __init__(self, mother, motherSys, 
                     daughter, daughterSys,
                     currentMemoryIndex, dt, rigidAreas, solvingScheme):
        self.type = 'Link'
        
        self.name = ' '.join(['Link',str(mother.Id),str(daughter.Id)])
        
        self.rho             = []
        self.my              = []
        self.systemEquations = []
        self.z               = []
        self.A_func          = []
        self.positions       = []
        self.names           = []

        self.dt = dt
        
        self.currentMemoryIndex  = currentMemoryIndex
        
        # equations to solve in f solve
        self.fsolveFunction = None
        self.jacobiMatrix = None
        
        #initialize Vessels
        # mother vessel
        self.rho.append(mother.rho)
        self.my.append(mother.my)
        self.z.append(mother.z)
        self.systemEquations.append(motherSys)
        self.positions.append(-1)
        self.names.append(mother.Id)
        self.A_func.append(mother.A_nID)
        # SolutionVariables
        self.P_mother = mother.Psol
        self.Q_mother = mother.Qsol
        self.A_mother = mother.Asol
               
        # daughter vessel
        self.rho.append(daughter.rho)
        self.my.append(daughter.my)
        self.z.append(daughter.z)
        self.systemEquations.append(daughterSys)
        self.positions.append(0)
        self.names.append(daughter.Id)
        self.A_func.append(daughter.A_nID)
        # SolutionVariables
        self.P_daughter = daughter.Psol
        self.Q_daughter = daughter.Qsol
        self.A_daughter = daughter.Asol
            


        self.rigidAreas = rigidAreas
        #solvingScheme = "Stenosis"
        solvingScheme = "NonLinear"
        # Define the call function depending on the solving Scheme
        if solvingScheme == "Linear": 
            self._callfcn = self.callLinear
        elif solvingScheme == "NonLinear":
            self._callfcn = self.callNonLinear
        elif solvingScheme == "Stenosis":
            self._callfcn = self.callStenosisYoungAndTsai
        else:
            raise ImportError("Connections wrong solving scheme! {}".format(solvingScheme))
    
        ## benchamark Test variables
        self.nonLin = False
        self.updateL = False
        self.sumQErrorCount = 0
        self.maxQError = 0
        self.maxPErrorNonLin = 0 
        self.maxPError = 0
        self.sumPErrorCount = 0
        self.sumPErrorNonLinCount = 0
    
    def __call__(self):
        return self._callfcn()

    def callLinear(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        
        P1 = self.P_mother[n]
        Q1 = self.Q_mother[n]
        A1 = self.A_mother[n]
        
        P2 = self.P_daughter[n]
        Q2 = self.Q_daughter[n]
        A2 = self.A_daughter[n]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        
        # update system equation and store L1
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_2 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
            
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
                
        ###### Linear approach
        denom = (R1_12*R2_21-R1_22*R2_11)
        # -1 reflectionCoeff mother->daugther
        alpha1 = -(R1_11*R2_21-R1_21*R2_11)/denom
        # +1 transmission daugther->mother
        alpha2 = -(R2_11*R2_22-R2_21*R2_12)/denom 
        # +1 transmission daugther->mother
        beta1 = -(R1_11*R1_22-R1_12*R1_21)/denom 
        # -1 reflectionCoeff daugther->mother
        beta2 = -(R1_12*R2_22-R2_12*R1_22)/denom
        
        #print 'cC153 alphas',alpha1,alpha2,beta1,beta2
                
        domega1_2 = alpha1 * domega1_1 + alpha2 * domega2_2
        domega2_1 = beta1  * domega1_1 + beta2  * domega2_2
                
        P1_new = P1o + R1_11*domega1_1 + R1_12*domega1_2
        Q1_new = Q1o + R1_21*domega1_1 + R1_22*domega1_2
    
        P2_new = P2o + R2_11*domega2_1 + R2_12*domega2_2
        Q2_new = Q2o + R2_21*domega2_1 + R2_22*domega2_2
        
        # apply calculated values to next time step
        self.P_mother[n+1][pos1]   = P1_new
        self.Q_mother[n+1][pos1]   = Q1_new
        self.P_daughter[n+1][pos2] = P2_new
        self.Q_daughter[n+1][pos2] = Q2_new
        
        if P1_new < -500*133.32 or P2_new < -500*133.32:
            raise ValueError("Connection: {} calculated negative pressure, P1_new = {}, P2_new = {}, at time {} (n {},dt {})".format(self.name,P1_new, P2_new,n*dt,n,dt))
            #print P1_new, P2_new
            #exit()
                
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)          
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
        # apply Areas
        self.A_mother[n+1][pos1]   = A1n
        self.A_daughter[n+1][pos2] = A2n
              
        # Error estimation
        try: sumQError = abs(abs(Q1_new)-abs(Q2_new))/abs(Q1_new)
        except: sumQError = 0.0
        if sumQError > 0.0: 
            self.sumQErrorCount = self.sumQErrorCount+1
        if sumQError > self.maxQError:
            self.maxQError  = sumQError
        #print self.name, ' Error cons mass',  sumQError, self.maxQError ,' - ', n, self.sumQErrorCount
        if sumQError > 1.e-5:
            raise ValueError("Connection: {} too high error, sumQError = {}, in conservation of mass at time {} (n {},dt {})".format(self.name,sumQError,n*dt,n,dt))
            #print sumQError
            #exit()
            
        # Linear presssure equation
        sumPError = abs(abs(P1_new)-abs(P2_new))/abs(P1_new)
        if sumPError > 0.0: 
            self.sumPErrorCount = self.sumPErrorCount+1
        if sumPError > self.maxPError:
            self.maxPError  = sumPError
        if sumPError > 1.e-5:
            raise ValueError("Connection: {} too high error, sumPError = {}, in conservation of pressure at time {} (n {},dt {})".format(self.name,sumPError,n*dt,n,dt))
            #print sumPError
            #exit()
                                               
        ## Non linear pressure equation
        sumPErrorNonLin = abs(abs(P1_new+1000*0.5*(Q1_new/A1n)**2)-abs(abs(P2_new+1000*0.5*(Q2_new/A2n)**2)))/abs(P1_new+0.5*(Q1_new/A1n)**2)
        if sumPErrorNonLin > 0.0: 
            self.sumPErrorNonLinCount = self.sumPErrorNonLinCount+1
        if sumPErrorNonLin > self.maxPErrorNonLin:
            self.maxPErrorNonLin  = sumPErrorNonLin
        # TODO: This was set to test sumPError, not sumPErrorNonLin
        # TODO: When set correctly, program wouldn't run. Commented out.
#        if sumPErrorNonLin > 1.e-10:
#           raise ValueError("Connection: {} too high error, sumPErrorNonLin = {} in conservation of pressure at time {} (n {},dt {})".format(self.name,sumPErrorNonLin,n*dt,n,dt))
            #print sumPError
            #exit()
            
        

    def callNonLinear(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        #if n == 1:
            #print "using nonlinear link model"
        P1 = self.P_mother[n]
        Q1 = self.Q_mother[n]
        A1 = self.A_mother[n]
        
        P2 = self.P_daughter[n]
        Q2 = self.Q_daughter[n]
        A2 = self.A_daughter[n]
        
        rho1 = self.rho[0]
        rho2 = self.rho[1]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        A1o = A1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        A2o = A2[pos2]
        # update system equation and store L1
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_2 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
            
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
                

        
        # apply calculated values to next time step
                
        domega1_2_init = -R1_11*domega1_1/R1_12 #result if pressure is constant- could be optimized
        domega2_1_init = -R2_12*domega2_2/R2_11 #result if pressure is constant
        
 
        epsilonvalues = np.array([1,1])
        epsilonlimit = 1e-10
        domega1_2_last = domega1_2_init
        domega2_1_last = domega2_1_init

        Niterations = 0
        Xold = np.array([domega1_2_last, domega2_1_last])

        while epsilonvalues[0]>1e-14 or epsilonvalues[1]>0.0001 :
            """iterative Newton Rahpson solver
                domega1_2, domega2_1, domega3_1 are the unknowns that are changing from
            each iteration. A1, A2, A3 are also changing, but the value from the previous iteration is used. (should actually be for the next timestep, but this should converge to the correct solution)
            R1_11, R1_12, R1_21, R1_22, ,R2_11, R2_12, R2_21, R2_22, ,R3_11, R3_12, R3_21, R3_22, domega1_1, domega2_2, domega3_2, Q1o, Q2o, Q2o 
            are constant for each timestep, and are taken from previous timestep. domega1_1, domega2_2, domega3_2 are the field domegas.
            """
            domega1_2_last = Xold[0]
            domega2_1_last = Xold[1]

            
            Q1discretize = Q1o + R1_21*domega1_1 + R1_22*domega1_2_last
            Q2discretize = Q2o + R2_21*domega2_1_last + R2_22*domega2_2

            
            P1discretize = P1o + R1_11*domega1_1 + R1_12*domega1_2_last
            P2discretize = P2o + R2_11*domega2_1_last + R2_12*domega2_2
            
            if self.rigidAreas == False:
                A1_last = self.A_func[0]([P1discretize],pos1)
                A2_last = self.A_func[1]([P2discretize],pos2)         
            else:
                A1_last = A1o
                A2_last = A2o
            
            f1 = Q1discretize - Q2discretize#R1_21*domega1_1 + R1_22*domega1_2_last  - R2_21*domega2_1_last - R2_22*domega2_2 - R3_21*domega3_1_last - R3_22*domega3_2
            
            f2 = P1discretize + 0.5*rho1*(Q1discretize/A1_last)**2 - P2discretize - 0.5*rho2*(Q2discretize/A2_last)**2
            

            
            F = np.array([f1, f2])
            """Inverse Jacobi elements: """
            a = R1_22
            b = -R2_21

            d = R1_12 + rho1*R1_22*(Q1discretize)/(A1_last**2)
            e = - R2_11 - rho2*R2_21*(Q2discretize)/(A2_last**2)
            
            Determinant = a*e -b*d
            
            J_inv = np.array([[ e, -b],
                          [ -d, a]]) / (Determinant)
            
            Xnew = Xold - np.dot(J_inv,F)

            epsilonvalues = np.abs(F)
            Niterations = Niterations + 1
            
            
            if Niterations > 30:
                
                print("Niterations excedded in link calculation in vessel, Niterations: ", self.names[0], Niterations)
                print("f1,f2: ", f1, f2)
                
                break
            Xold = Xnew
            #exit()
        
        
        Q1_new = Q1discretize
        Q2_new = Q2discretize
        
        
        P1_new = P1discretize
        P2_new = P2discretize
        
        self.P_mother[n+1][pos1]   = P1_new
        self.Q_mother[n+1][pos1]   = Q1_new
        self.P_daughter[n+1][pos2] = P2_new
        self.Q_daughter[n+1][pos2] = Q2_new
        
        if P1_new < -500*133.32 or P2_new < -500*133.32:
            raise ValueError("Connection: {} calculated negative pressure, P1_new = {}, P2_new = {}, at time {} (n {},dt {})".format(self.name,P1_new,P2_new,n*dt,n,dt))
            #print P1_new, P2_new
            #exit()
                 
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)          
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
        # apply Areas
        self.A_mother[n+1][pos1]   = A1n
        self.A_daughter[n+1][pos2] = A2n

        

    def callStenosisYoungAndTsai(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        #if n == 1:
            #print "using nonlinear link model"
        P1 = self.P_mother[n]
        Q1 = self.Q_mother[n]
        A1 = self.A_mother[n]
        
        P2 = self.P_daughter[n]
        Q2 = self.Q_daughter[n]
        A2 = self.A_daughter[n]
        
        rho1 = self.rho[0]
        rho2 = self.rho[1]
        
        my1 = self.my[0]
        my2 = self.my[1]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        A1o = A1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        A2o = A2[pos2]
        # update system equation and store L1
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_2 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
            
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
        
        
        # apply calculated values to next time step
        if n>0:
            P1o2 = self.P_mother[n-1][pos1]
            P2o2 = self.P_daughter[n-1][pos2]
            
            deltaP1_last = P1o - P1o2
            deltaP2_last = P2o - P2o2
            
        else:
            deltaP1_last = 0
            deltaP2_last = 0
                
        domega1_2_init = deltaP1_last - R1_11*domega1_1/R1_12 #result if pressure is constant- could be optimized
        domega2_1_init = deltaP2_last - R2_12*domega2_2/R2_11 #result if pressure is constant
        
 
        epsilonvalues = np.array([1,1])
        epsilonlimit = 1e-10
        domega1_2_last = domega1_2_init
        domega2_1_last = domega2_1_init

        Niterations = 0
        Xold = np.array([domega1_2_last, domega2_1_last])
#         print("\n")
#         print("domega1_2_last, domega2_1_last, domega3_1_last, domega1_1, domega2_2, domega3_2 : ", domega1_2_last, domega2_1_last, domega3_1_last, domega1_1, domega2_2, domega3_2)
#         print("\n")
        while epsilonvalues[0]>1e-14 or epsilonvalues[1]>0.0001 :
            """iterative Newton Rahpson solver
                domega1_2, domega2_1, domega3_1 are the unknowns that are changing from
            each iteration. A1, A2, A3 are also changing, but the value from the previous iteration is used. (should actually be for the next timestep, but this should converge to the correct solution)
            R1_11, R1_12, R1_21, R1_22, ,R2_11, R2_12, R2_21, R2_22, ,R3_11, R3_12, R3_21, R3_22, domega1_1, domega2_2, domega3_2, Q1o, Q2o, Q2o 
            are constant for each timestep, and are taken from previous timestep. domega1_1, domega2_2, domega3_2 are the field domegas.
            """
            domega1_2_last = Xold[0]
            domega2_1_last = Xold[1]

            
            Q1discretize = Q1o + R1_21*domega1_1 + R1_22*domega1_2_last
            Q2discretize = Q2o + R2_21*domega2_1_last + R2_22*domega2_2

            
            P1discretize = P1o + R1_11*domega1_1 + R1_12*domega1_2_last
            P2discretize = P2o + R2_11*domega2_1_last + R2_12*domega2_2
            
            if self.rigidAreas == False:
                A1_last = self.A_func[0]([P1discretize],pos1)
                A2_last = self.A_func[1]([P2discretize],pos2)         
            else:
                A1_last = A1o
                A2_last = A2o
            
            f1 = Q1discretize - Q2discretize#R1_21*domega1_1 + R1_22*domega1_2_last  - R2_21*domega2_1_last - R2_22*domega2_2 - R3_21*domega3_1_last - R3_22*domega3_2
            
            U1 = Q1discretize/A1_last
            U2 = Q2discretize/A2_last
            D1 = np.sqrt(4*A1_last/np.pi)
            Re1 = rho1*U1*D1/my1
            
            Astenosis = A1_last*0.05
            
            Kv = 4500. # pouseille friction correction coeficient
            Kt = 0.9 # expansion coefficient
            if abs(Re1<0.01):
                deltaPstenosis = 0
            else:
                deltaPstenosis = rho1*U1**2*(Kv/Re1 + (Kt/2.)*(A1_last/Astenosis - 1)**2)
            
            
            f2 = P1discretize + 0.5*rho1*(Q1discretize/A1_last)**2 - P2discretize - 0.5*rho2*(Q2discretize/A2_last)**2 - deltaPstenosis
            #f2 = P1discretize - P2discretize  - deltaPstenosis
            

            
            F = np.array([f1, f2])
            """Inverse Jacobi elements: """
            a = R1_22
            b = -R2_21

            d = R1_12 + rho1*R1_22*(Q1discretize)/(A1_last**2)
            e = - R2_11 - rho2*R2_21*(Q2discretize)/(A2_last**2)
            
            Determinant = a*e -b*d
            
            J_inv = np.array([[ e, -b],
                          [ -d, a]]) / (Determinant)
            
            Xnew = Xold - np.dot(J_inv,F)

            epsilonvalues = np.abs(F)
            Niterations = Niterations + 1
            
            
            if Niterations > 60:
                
                print("Niterations excedded in link calculation in vessel, Niterations: ", self.names[0], Niterations)
                print("f1,f2: ", f1, f2/133.32, P1discretize/133.32, P2discretize/133.32, deltaPstenosis/133.32)
                #P2discretize = P1discretize - deltaPstenosis
                
                break
            Xold = Xnew
            #exit()

        
        
        Q1_new = Q1discretize
        Q2_new = Q2discretize
        
        
        P1_new = P1discretize
        P2_new = P2discretize
        
        self.P_mother[n+1][pos1]   = P1_new
        self.Q_mother[n+1][pos1]   = Q1_new
        self.P_daughter[n+1][pos2] = P2_new
        self.Q_daughter[n+1][pos2] = Q2_new
        
        if P1_new < -500*133.32 or P2_new < -500*133.32:
            raise ValueError("Connection: {} calculated negative pressure, P1_new = {}, P2_new = {}, at time {} (n {},dt {})".format(self.name,P1_new,P2_new,n*dt,n,dt))
            #print P1_new, P2_new
            #exit()
                 
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)          
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
        # apply Areas
        self.A_mother[n+1][pos1]   = A1n
        self.A_daughter[n+1][pos2] = A2n

class Stenoses():
    """
    Link object represends the connection between two vessels
    """
    def __init__(self, mother, motherSys, 
                     daughter, daughterSys,
                     Kv, Kt, Ku, A0, As, Ls, 
                     currentMemoryIndex, dt, rigidAreas, solvingScheme, ):
        self.type = 'Stenoses'
        
        self.name = ' '.join(['Stenoses',str(mother.Id),str(daughter.Id)])
        
        self.rho             = []
        self.my              = []
        self.systemEquations = []
        self.z               = []
        self.A_func          = []
        self.positions       = []
        self.names           = []

        self.dt = dt
        
        self.currentMemoryIndex  = currentMemoryIndex
        
        # equations to solve in f solve
        self.fsolveFunction = None
        self.jacobiMatrix = None
        
        #initialize Vessels
        # mother vessel
        self.rho.append(mother.rho)
        self.my.append(mother.my)
        self.z.append(mother.z)
        self.systemEquations.append(motherSys)
        self.positions.append(-1)
        self.names.append(mother.Id)
        self.A_func.append(mother.A_nID)
        # stenoses variables
        self.Kv = Kv
        self.Kt = Kt
        self.Ku = Ku
        self.A0 = A0
        self.As = As
        self.Ls = Ls
        self.D0 = 2*np.sqrt(A0/np.pi)
        # SolutionVariables
        self.P_mother = mother.Psol
        self.Q_mother = mother.Qsol
        self.A_mother = mother.Asol
        
        self.Q_prev_dt = None
        self.Q_prev_prev_dt = None
        self.Q_der_last = 0
        
        self.domega1_2_init = 0
        self.domega2_1_init = 0
               
        # daughter vessel
        self.rho.append(daughter.rho)
        self.my.append(daughter.my)
        self.z.append(daughter.z)
        self.systemEquations.append(daughterSys)
        self.positions.append(0)
        self.names.append(daughter.Id)
        self.A_func.append(daughter.A_nID)
        # SolutionVariables
        self.P_daughter = daughter.Psol
        self.Q_daughter = daughter.Qsol
        self.A_daughter = daughter.Asol
        
        self.J_py, self.Jinv_py, self.F_py = self.initializeResidualFuncsAndJacobian()
            


        self.rigidAreas = rigidAreas
        #solvingScheme = "Stenosis"
        solvingScheme = "Stenoses"
        # Define the call function depending on the solving Scheme
        if solvingScheme == "Stenoses":
            self._callfcn = self.callStenosisYoungAndTsai
            print('Tsai')
        else:
            raise ImportError("Connections wrong solving scheme! {}".format(solvingScheme))
    
        ## benchamark Test variables
        self.nonLin = False
        self.updateL = False
        self.sumQErrorCount = 0
        self.maxQError = 0
        self.maxPErrorNonLin = 0 
        self.maxPError = 0
        self.sumPErrorCount = 0
        self.sumPErrorNonLinCount = 0
    
    def __call__(self):
        return self._callfcn()
    
    def initializeResidualFuncsAndJacobian(self):
        print('start sympy')
        rho = sympy.symbols('rho')
        P0_m = sympy.symbols('P0_m')
        Q0_m = sympy.symbols('Q0_m')
        dw1_m = sympy.symbols('dw1_m')
        dw2_m = sympy.symbols('dw2_m')
        R11_m = sympy.symbols('R11_m')
        R12_m = sympy.symbols('R12_m')
        R21_m = sympy.symbols('R21_m')
        R22_m = sympy.symbols('R22_m')
        A_m = sympy.symbols('A_m')
        
        P0_d = sympy.symbols('P0_d')
        Q0_d = sympy.symbols('Q0_d')
        dw1_d = sympy.symbols('dw1_d')
        dw2_d = sympy.symbols('dw2_d')
        R11_d = sympy.symbols('R11_d')
        R12_d = sympy.symbols('R12_d')
        R21_d = sympy.symbols('R21_d')
        R22_d = sympy.symbols('R22_d')
        A_d = sympy.symbols('A_d')
        
        a = sympy.symbols('a')
        b = sympy.symbols('b')
        c = sympy.symbols('c')
        Q_abs = sympy.symbols('Q_abs')
        Q_der = sympy.symbols('Q_der')
        
        P_m = P0_m + dw1_m*R11_m + dw2_m*R12_m
        Q_m = Q0_m + dw1_m*R21_m + dw2_m*R22_m
        
        P_d = P0_d + dw1_d*R11_d + dw2_d*R12_d
        Q_d = Q0_d + dw1_d*R21_d + dw2_d*R22_d
        
        deltaP = a*Q_m + b*Q_m*Q_abs + c*Q_der
        
        f1 = Q_m - Q_d
        f2 = P_m + 0.5*rho*(Q_m/A_m)**2 - P_d - 0.5*rho*(Q_d/A_d)**2 - deltaP
        
        X = [dw2_m, dw1_d]
        f = [f1, f2]
        
        F = Matrix([f]).T
        def jacobi(i, j):
            
            return sympy.diff(F[i], X[j])
                
        N = len(X)
        
        J = Matrix(2, 2, jacobi) # create jacobian. sympy Matrix/expression
        Jinv = J.inv()

        #print(J, Jinv.shape)
        J_py = sympy.lambdify([rho, P0_m, Q0_m, dw1_m, dw2_m, R11_m, R12_m, R21_m, R22_m, A_m,
                                  P0_d, Q0_d, dw1_d, dw2_d, R11_d, R12_d, R21_d, R22_d, A_d, a, b, c, Q_abs, Q_der], J)
        Jinv_py = sympy.lambdify([rho, P0_m, Q0_m, dw1_m, dw2_m, R11_m, R12_m, R21_m, R22_m, A_m,
                                  P0_d, Q0_d, dw1_d, dw2_d, R11_d, R12_d, R21_d, R22_d, A_d, a, b, c, Q_abs, Q_der], Jinv)
        F_py = sympy.lambdify([rho, P0_m, Q0_m, dw1_m, dw2_m, R11_m, R12_m, R21_m, R22_m, A_m,
                                  P0_d, Q0_d, dw1_d, dw2_d, R11_d, R12_d, R21_d, R22_d, A_d, a, b, c, Q_abs, Q_der], F)
        
        print("end sympy")
        return J_py, Jinv_py, F_py
        

    def callStenosisYoungAndTsai(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        #if n == 1:
            #print "using nonlinear link model"
        P1 = self.P_mother[n]
        Q1 = self.Q_mother[n]
        A1 = self.A_mother[n]
        
        P2 = self.P_daughter[n]
        Q2 = self.Q_daughter[n]
        A2 = self.A_daughter[n]
        
        rho1 = self.rho[0]
        rho2 = self.rho[1]
        
        my1 = self.my[0]
        my2 = self.my[1]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        A1o = A1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        A2o = A2[pos2]
        # update system equation and store L1
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_2 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
        
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
        
        
        # apply calculated values to next time step
        if n>0:
            P1o2 = self.P_mother[n-1][pos1]
            P2o2 = self.P_daughter[n-1][pos2]
            
            deltaP1_last = P1o - P1o2
            deltaP2_last = P2o - P2o2
            
        else:
            deltaP1_last = 0
            deltaP2_last = 0
        
        if n == 1 and self.Q_prev_dt == None:
            self.Q_prev_dt = self.Q_mother[n-1][pos1]
            self.Q_der_last = (Q1o - self.Q_prev_dt)/dt # first order backward difference
            
                
        domega1_2_init = self.domega1_2_init #deltaP1_last - R1_11*domega1_1/R1_12 #result if pressure is constant- could be optimized
        domega2_1_init = self.domega2_1_init #deltaP2_last - R2_12*domega2_2/R2_11 #result if pressure is constant
        
 
        epsilonvalues = np.array([[1],[1]])
        epsilonlimit = 1e-10
        domega1_2_last = domega1_2_init
        domega2_1_last = domega2_1_init
        A1_last = A1o
        A2_last = A2o
        
        a = self.Kv*my1/(self.A0*self.D0)
        b = (self.Kt*rho1/(2*self.A0**2))*(self.A0/self.As - 1)**2
        c = self.Ku*rho1*self.Ls/self.A0
        
        Q_abs_last = abs(Q1o)

        Niterations = 0
        Xold = np.array([[domega1_2_last], [domega2_1_last]])
#         print("\n")
#         print("domega1_2_last, domega2_1_last, domega3_1_last, domega1_1, domega2_2, domega3_2 : ", domega1_2_last, domega2_1_last, domega3_1_last, domega1_1, domega2_2, domega3_2)
#         print("\n")
        while epsilonvalues[0, 0]>1e-10 or epsilonvalues[1, 0]>0.0001 :
            """iterative Newton Rahpson solver
                domega1_2, domega2_1, are the unknowns that are changing from
            each iteration. A1, A2, A3 are also changing, but the value from the previous iteration is used. (should actually be for the next timestep, but this should converge to the correct solution)
            R1_11, R1_12, R1_21, R1_22, ,R2_11, R2_12, R2_21, R2_22, ,R3_11, R3_12, R3_21, R3_22, domega1_1, domega2_2, domega3_2, Q1o, Q2o, Q2o 
            are constant for each timestep, and are taken from previous timestep. domega1_1, domega2_2, domega3_2 are the field domegas.
            """
            domega1_2_last = Xold[0, 0]
            domega2_1_last = Xold[1, 0]
            
             
            args = [rho1, P1o, Q1o, domega1_1, domega1_2_last, R1_11, R1_12, R1_21, R1_22, A1_last,
                    P2o, Q2o, domega2_1_last, domega2_2, R2_11, R2_12, R2_21, R2_22, A2_last, a, b, c, Q_abs_last, self.Q_der_last]
            
            
            Jinv = np.array(self.Jinv_py(*args))
            #Jinv = np.array(np.linalg.inv(self.J_py(*args)))
            F = np.array(self.F_py(*args))
            Xnew = Xold - np.dot(Jinv, F)
            
            #print(Jinv.shape, F.shape, Xold.shape, Xnew.shape)
            epsilonvalues = np.abs(F)
            Niterations = Niterations + 1
            
            domega1_2 = Xnew[0, 0]
            domega2_1 = Xnew[1, 0]

            Q1discretize = Q1o + R1_21*domega1_1 + R1_22*domega1_2
            Q2discretize = Q2o + R2_21*domega2_1 + R2_22*domega2_2
            
            Q_abs_last = abs(Q1discretize)
            
            if self.Q_prev_dt == None:
                self.Q_der_last = (Q1discretize - Q1o)/dt # first order backward difference
            else:
                self.Q_der_last = (3*Q1discretize - 4*Q1o + self.Q_prev_dt)/(2*dt) # second order backward difference

            
            P1discretize = P1o + R1_11*domega1_1 + R1_12*domega1_2
            P2discretize = P2o + R2_11*domega2_1 + R2_12*domega2_2

            # calculate new areas
            if self.rigidAreas == False:
                A1_last = self.A_func[0]([P1discretize],pos1)
                A2_last = self.A_func[1]([P2discretize],pos2)          
            else:
                A1_last = A1o
                A2_last = A2o
            
            
            if Niterations > 60:
                f1 = F[0]
                f2 = F[1]
                print("Niterations excedded in link calculation in vessel, Niterations: ", self.names[0], Niterations)
                print("f1,f2: ", f1, f2/133.32, P1discretize/133.32, P2discretize/133.32)
                #P2discretize = P1discretize - deltaPstenosis
                
                break
            Xold = Xnew
            #exit()

        
#         if n == 0:
#             print("a: {0}, b: {1}".format(a, b))
#             deltaP = a*Q1discretize + b*Q2discretize*abs(Q2discretize)
#             print("Pm - Pd: {0}, deltaP: {1}".format((P1discretize-P2discretize)/133.32, deltaP/133.32))
        
        self.domega1_2_init = domega1_2
        self.domega2_1_init = domega2_1
        
        Q1_new = Q1discretize
        Q2_new = Q2discretize
        self.Q_prev_dt = Q1o
        
        
        P1_new = P1discretize
        P2_new = P2discretize
        
        self.P_mother[n+1][pos1]   = P1_new
        self.Q_mother[n+1][pos1]   = Q1_new
        self.P_daughter[n+1][pos2] = P2_new
        self.Q_daughter[n+1][pos2] = Q2_new
        
        if P1_new < -500*133.32 or P2_new < -500*133.32:
            raise ValueError("Connection: {} calculated negative pressure, P1_new = {}, P2_new = {}, at time {} (n {},dt {})".format(self.name,P1_new,P2_new,n*dt,n,dt))
            #print P1_new, P2_new
            #exit()
                 
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)          
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
        # apply Areas
        self.A_mother[n+1][pos1]   = A1n
        self.A_daughter[n+1][pos2] = A2n
   
    
class Bifurcation():
    
    def __init__(self, mother, motherSys,
                       leftDaughter, leftDaughterSys,
                       rightDaughter, rightDaughterSys, 
                       currentMemoryIndex, dt, rigidAreas, solvingScheme):
        # vessel variables initially set, constant through simulation
        self.type = 'Bifurcation'
        
        self.name = ' '.join(['Bifurcation',str(mother.Id),str(leftDaughter.Id),str(rightDaughter.Id)])
        
        #System Variables
        self.dt = dt
        self.currentMemoryIndex = currentMemoryIndex
        
        self.rho = []
        self.systemEquations = []
        self.z = []
        self.A_func = []
        self.positions =[]
#         self.vz = []
        self.names = []
        
#         # equations to solve in f solve
#         self.fsolveFunction = None
#         self.jacobiMatrix = None
        
        ###initialize
        ##mother branch
        self.rho.append(mother.rho)
        self.z.append(mother.z)
        self.systemEquations.append(motherSys)
        self.positions.append(-1)
        self.names.append(mother.Id)
        self.A_func.append(mother.A_nID)
#        self.vz.append(-1)
        #SolutionVariables
        self.P_leftMother = mother.Psol
        self.Q_leftMother = mother.Qsol
        self.A_leftMother = mother.Asol
              
        
        ##left daughter
        self.rho.append(leftDaughter.rho)
        self.z.append(leftDaughter.z)
        self.systemEquations.append(leftDaughterSys)
        self.positions.append(0)
        self.names.append(leftDaughter.Id)
        self.A_func.append(leftDaughter.A_nID)
        
        #SolutionVariables
        self.P_leftDaughter = leftDaughter.Psol
        self.Q_leftDaughter = leftDaughter.Qsol
        self.A_leftDaughter = leftDaughter.Asol
        
        ##right daughter
        self.rho.append(rightDaughter.rho)
        self.z.append(rightDaughter.z)
        self.systemEquations.append(rightDaughterSys)
        self.positions.append(0)
        self.names.append(rightDaughter.Id)
        self.A_func.append(rightDaughter.A_nID)
        
        #SolutionVariables
        self.P_rightDaughter = rightDaughter.Psol
        self.Q_rightDaughter = rightDaughter.Qsol
        self.A_rightDaughter = rightDaughter.Asol
        
        self.rigidAreas = rigidAreas
        solvingScheme = "NonLinear"
        # Define the call function depending on the solving Scheme
        if solvingScheme == "Linear": 
            self._callfcn = self.callLinear
        elif solvingScheme == "NonLinear":
            self._callfcn = self.callNonLinear

        else:
            raise ImportError("Connections wrong solving scheme! {}".format(solvingScheme))
        
        ## benchamark Test variables
        self.sumQErrorCount = 0
        self.maxQError = 0
        self.maxPErrorNonLin = 0 
        self.maxPError = 0
        self.sumPErrorCount = 0
        self.sumPErrorNonLinCount = 0
    
    def __call__(self):
        return self._callfcn()
    
    def callLinear(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        pos3 = self.positions[2]
        
        P1 = self.P_leftMother[n]
        Q1 = self.Q_leftMother[n]
        A1 = self.A_leftMother[n]
        
        P2 = self.P_leftDaughter[n]
        Q2 = self.Q_leftDaughter[n]
        A2 = self.A_leftDaughter[n]
        
        P3 = self.P_rightDaughter[n]
        Q3 = self.Q_rightDaughter[n]
        A3 = self.A_rightDaughter[n]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        P3o = P3[pos3]
        Q3o = Q3[pos3]
        
        ## update LARL
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_2 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
        L,R3,LMBD,Z1,Z2,domega3_2 = self.systemEquations[2].updateLARL(P3,Q3,A3,pos3)
        
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
        
        R3_11 = R3[0][0]
        R3_12 = R3[0][1]
        R3_21 = R3[1][0]
        R3_22 = R3[1][1]
        
        ###### Linear approach
        denom = R1_12 * R2_11 * R3_21 + R1_12 * R2_21 * R3_11 - R1_22 * R2_11 * R3_11 
        
        alpha1 = -(R1_11 * R2_11 * R3_21 + R1_11 * R2_21 * R3_11 - R1_21 * R2_11 * R3_11)/denom
        alpha2 = -(R2_11 * R2_22 * R3_11 - R2_12 * R2_21 * R3_11)/denom
        alpha3 = -(R2_11 * R3_22 * R3_11 - R2_11 * R3_12 * R3_21)/denom
        
        beta1 = -(R1_11 * R1_22 * R3_11 - R1_12 * R1_21 * R3_11)/denom
        beta2 = -(R1_12 * R2_12 * R3_21 + R1_12 * R2_22 * R3_11 - R1_22 * R2_12 * R3_11)/denom
        beta3 = -(R1_12 * R3_11 * R3_22 - R1_12 * R3_12 * R3_21)/denom
        
        gamma1 = -(R1_11 * R1_22 * R2_11 - R1_12 * R1_21 * R2_11)/denom
        gamma2 = -(R1_12 * R2_11 * R2_22 - R1_12 * R2_12 * R2_21)/denom
        gamma3 = -(R1_12 * R2_11 * R3_22 + R1_12 * R2_21 * R3_12 - R1_22 * R2_11 * R3_12)/denom
        
        domega1_2 = alpha1 * domega1_1  + alpha2 * domega2_2 + alpha3 * domega3_2 
        domega2_1 = beta1  * domega1_1  + beta2  * domega2_2 + beta3  * domega3_2 
        domega3_1 = gamma1 * domega1_1  + gamma2 * domega2_2 + gamma3 * domega3_2 
        
        P1_new = P1o + (R1_11*domega1_1 + R1_12*domega1_2)
        Q1_new = Q1o + (R1_21*domega1_1 + R1_22*domega1_2)
    
        P2_new = P2o + (R2_11*domega2_1 + R2_12*domega2_2)
        Q2_new = Q2o + (R2_21*domega2_1 + R2_22*domega2_2)
        
        P3_new = P3o + (R3_11*domega3_1 + R3_12*domega3_2)
        Q3_new = Q3o + (R3_21*domega3_1 + R3_22*domega3_2)
               
        
        # apply calculated values to next time step
        self.P_leftMother[n+1][pos1]    = P1_new
        self.Q_leftMother[n+1][pos1]    = Q1_new
        self.P_leftDaughter[n+1][pos2]  = P2_new
        self.Q_leftDaughter[n+1][pos2]  = Q2_new
        self.P_rightDaughter[n+1][pos3] = P3_new
        self.Q_rightDaughter[n+1][pos3] = Q3_new
        print("using linear bifurcation model")
        if P1_new < -500*133.32 or P2_new < -500*133.32 or P3_new < -500*133.32:
            raise ValueError("Connection: {} calculated negative pressure, P1_new = {}, P2_new = {}, P3_new = {}, at time {} (n {},dt {})".format(self.name,P1_new, P2_new, P3_new, n*dt,n,dt))
            #print P1_new, P2_new, P3_new
            #exit()
        
        
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)
            A3n = self.A_func[2]([P3_new],pos3)
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
            A3n = A3[pos3] 
           
        self.A_leftMother[n+1][pos1]       = A1n
        self.A_leftDaughter[n+1][pos2]     = A2n       
        self.A_rightDaughter[n+1][pos3]    = A3n  
            
        ## non linear error        
        try: sumQError = abs(Q1_new-Q2_new-Q3_new)/abs(Q1_new)
        except Exception: sumQError = 0.0
        if sumQError > 0.0: 
            self.sumQErrorCount = self.sumQErrorCount+1
        if sumQError > self.maxQError:
            self.maxQError  = sumQError
        #print self.name,' \n Error cons mass',  sumQError, self.maxQError ,' - ', n, self.sumQErrorCount
        if sumQError > 1.e-5:
            raise ValueError("Connection: {} too high error, sumQError = {} in conservation of mass at time {} (n {},dt {})".format(self.name,sumQError,n*dt,n,dt))
            #print sumQError
            #exit()
        
        sumPError = abs(P1_new-P2_new)/abs(P1_new)
        if sumPError > 0.0: 
            self.sumPErrorCount = self.sumPErrorCount+1
        if sumPError > self.maxPError:
            self.maxPError  = sumPError
        #print self.name,' Error P lin    ',  sumPError, self.maxPError ,' - ', n, self.sumPErrorCount
        if sumPError > 1.e-10:
            raise ValueError("Connection: {} too high error, sumPError = {}, in conservation of pressure at time {} (n {},dt {}), exit system".format(self.name,sumPError,n*dt,n,dt))
            #print sumPError
            #exit()
        
        sumPErrorNonLin = abs(P1_new+500*(Q1_new/A1n)**2-(P2_new+500*(Q2_new/A2n)**2))/abs(P1_new+0.5*(Q1_new/A1n)**2)
        if sumPErrorNonLin > 0.0: 
            self.sumPErrorNonLinCount = self.sumPErrorNonLinCount+1
        if sumPErrorNonLin > self.maxPErrorNonLin:
            self.maxPErrorNonLin  = sumPErrorNonLin


    def callNonLinear(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        pos3 = self.positions[2]
        
        P1 = self.P_leftMother[n]
        Q1 = self.Q_leftMother[n]
        A1 = self.A_leftMother[n]
        
        P2 = self.P_leftDaughter[n]
        Q2 = self.Q_leftDaughter[n]
        A2 = self.A_leftDaughter[n]
        
        P3 = self.P_rightDaughter[n]
        Q3 = self.Q_rightDaughter[n]
        A3 = self.A_rightDaughter[n]
        
        rho1 = self.rho[0]
        rho2 = self.rho[1]
        rho3 = self.rho[2]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        A1o = A1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        A2o = A2[pos2]
        P3o = P3[pos3]
        Q3o = Q3[pos3]
        A3o = A3[pos3]

        ## update LARL
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_2 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
        L,R3,LMBD,Z1,Z2,domega3_2 = self.systemEquations[2].updateLARL(P3,Q3,A3,pos3)
        
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
        
        R3_11 = R3[0][0]
        R3_12 = R3[0][1]
        R3_21 = R3[1][0]
        R3_22 = R3[1][1]
        
        if n>0:
            P1o2 = self.P_leftMother[n-1][pos1]
            P2o2 = self.P_leftDaughter[n-1][pos2]
            P3o2 = self.P_rightDaughter[n-1][pos3]
            
            deltaP1_last = P1o - P1o2
            deltaP2_last = P2o - P2o2
            deltaP3_last = P3o - P3o2
            
            domega1_2_init = (deltaP1_last - R1_11*domega1_1)/R1_12 #same omega as previous timestep
            domega2_1_init = (deltaP2_last - R2_12*domega2_2)/R2_11 #same omega as previous timestep
            domega3_1_init = (deltaP3_last - R3_12*domega3_2)/R3_11 #same omega as previous timestep
        else:
            domega1_2_init = 0#-R1_11*domega1_1/R1_12 #result if pressure is constant- could be optimized
            domega2_1_init = 0#-R2_12*domega2_2/R2_11 #result if pressure is constant
            domega3_1_init = 0#-R3_12*domega3_2/R3_11 #result if pressure is constant

        epsilonvalues = np.array([1,1,1])
        epsilonlimit = 1e-10
        domega1_2_last = domega1_2_init
        domega2_1_last = domega2_1_init
        domega3_1_last = domega3_1_init
        A1_last = A1o
        A2_last = A2o
        A3_last = A3o
        Niterations = 0
        Xold = np.array([domega1_2_last, domega2_1_last, domega3_1_last])

        while epsilonvalues[0]>1e-10 or epsilonvalues[1]>1e-5 or epsilonvalues[2]>1e-5:
            """iterative Newton Rahpson solver
                domega1_2, domega2_1, domega3_1 are the unknowns that are changing from
            each iteration. A1, A2, A3 are also changing, but the value from the previous iteration is used. (should actually be for the next timestep, but this should converge to the correct solution)
            R1_11, R1_12, R1_21, R1_22, ,R2_11, R2_12, R2_21, R2_22, ,R3_11, R3_12, R3_21, R3_22, domega1_1, domega2_2, domega3_2, Q1o, Q2o, Q2o 
            are constant for each timestep, and are taken from previous timestep. domega1_1, domega2_2, domega3_2 are the field domegas.
            """
            domega1_2_last = Xold[0]
            domega2_1_last = Xold[1]
            domega3_1_last = Xold[2]
            
            Q1discretize = Q1o + R1_21*domega1_1 + R1_22*domega1_2_last
            Q2discretize = Q2o + R2_21*domega2_1_last + R2_22*domega2_2
            Q3discretize = Q3o + R3_21*domega3_1_last + R3_22*domega3_2
            
            P1discretize = P1o + R1_11*domega1_1 + R1_12*domega1_2_last
            P2discretize = P2o + R2_11*domega2_1_last + R2_12*domega2_2
            P3discretize = P3o + R3_11*domega3_1_last + R3_12*domega3_2
            

            if self.rigidAreas == False:
                try:
                    A1_last = self.A_func[0]([P1discretize],pos1)
                    A2_last = self.A_func[1]([P2discretize],pos2)
                    A3_last = self.A_func[2]([P3discretize],pos3)
                except FloatingPointError as E:
                    print("Floating Point error in Connection {}".format(self.name))
                    raise E
            else:
                A1_last = A1[pos1]
                A2_last = A2[pos2]
                A3_last = A3[pos3] 

            
            f1 = Q1discretize - Q2discretize - Q3discretize#R1_21*domega1_1 + R1_22*domega1_2_last  - R2_21*domega2_1_last - R2_22*domega2_2 - R3_21*domega3_1_last - R3_22*domega3_2
            
            f2 = P1discretize + 0.5*rho1*((Q1discretize/A1_last)**2) - P2discretize - 0.5*rho2*((Q2discretize/A2_last)**2)
            
            f3 = P1discretize + 0.5*rho1*((Q1discretize/A1_last)**2) - P3discretize - 0.5*rho3*((Q3discretize/A3_last)**2)
            
            F = np.array([f1, f2, f3])
            """Inverse Jacobi elements: """
            a = R1_22
            b = -R2_21
            c = -R3_21
            d = R1_12 + rho1*R1_22*(Q1discretize)/(A1_last**2)
            e = - R2_11 - rho2*R2_21*(Q2discretize)/(A2_last**2)
            f = d
            g = - R3_11 - rho3*R3_21*(Q3discretize)/(A3_last**2)
            
            Determinant = a*e*g -b*d*g -c*e*f
            
            J_inv = np.array([[ e*g, -b*g, -c*e  ],
                          [ -d*g, a*g-c*f, c*d ],
                          [  -e*f, b*f, a*e-b*d ]]) / (Determinant)
            
            Xnew = Xold - np.dot(J_inv,F)

            epsilonvalues = np.abs(F)
            Niterations = Niterations + 1
            
            if Niterations > 30:
                print("\n")
                print("Niterations excedded in Bifurcation calculation in vessel, Niterations: ", self.names[0], Niterations)

                print("Xnew: ", Xnew)
                print("Xold: ", Xold)
                
                print("f1: ", f1)
                print("f2: ", f2)
                print("f3: ", f3)
                print("epsilonvalues: ", epsilonvalues)
                print("Q1discretize, Q1o: ", Q1discretize, Q1o)
                print("Q2discretize, Q2o: ", Q2discretize, Q2o)
                print("Q3discretize, Q3o: ", Q3discretize, Q3o)
                print("P1discretize, P1o: ", P1discretize, P1o)
                print("P2discretize, P2o: ", P2discretize, P2o)
                print("P3discretize, P3o: ",P3discretize, P3o)
                
                break
            Xold = Xnew
            #exit()
        
        Q1_new = Q1discretize
        Q2_new = Q2discretize
        Q3_new = Q3discretize
        
        P1_new = P1discretize
        P2_new = P2discretize
        P3_new = P3discretize
        
        # apply calculated values to next time step
        self.P_leftMother[n+1][pos1]    = P1_new
        self.Q_leftMother[n+1][pos1]    = Q1_new
        self.P_leftDaughter[n+1][pos2]  = P2_new
        self.Q_leftDaughter[n+1][pos2]  = Q2_new
        self.P_rightDaughter[n+1][pos3] = P3_new
        self.Q_rightDaughter[n+1][pos3] = Q3_new
        
        if P1_new < -500*133.32 or P2_new < -500*133.32 or P3_new < -500*133.32:
            print("ERROR: Connection: {} calculated negative pressure at time {} (n {},dt {}), exit system".format(self.name,n*dt,n,dt))
            print(P1_new, P2_new, P3_new)
            print("Niterations: ", Niterations)
            
            print("solving nonlinear/total pressure Bifurcation")
            print("domega1_2_init: ", domega1_2_init)
            print("domega1_1: ", domega1_1)
                    
            print("f1: ", f1)
            print("f2: ", f2)
            print("f3: ", f3)
            print("epsilonvalues: ", epsilonvalues)
            print("Q1discretize, Q1o: ", Q1discretize, Q1o)
            print("Q2discretize, Q2o: ", Q2discretize, Q2o)
            print("Q3discretize, Q3o: ", Q3discretize, Q3o)
            print("P1discretize, P1o: ", P1discretize, P1o)
            print("P2discretize, P2o: ", P2discretize, P2o)
            print("P3discretize, P3o: ",P3discretize, P3o)
            exit()
        
        
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)
            A3n = self.A_func[2]([P3_new],pos3)
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
            A3n = A3[pos3] 
           
        self.A_leftMother[n+1][pos1]       = A1n
        self.A_leftDaughter[n+1][pos2]     = A2n       
        self.A_rightDaughter[n+1][pos3]    = A3n

        #print self.name,' Error P non lin',  sumPErrorNonLin, self.maxPErrorNonLin ,' - ', n, self.sumPErrorNonLinCount
        
        
        
                
#     def callMacCormackField2(self):
#         """
#         Call function for a bifurcation
#         """  
#         #print self.counter
#         #self.counter = 1+ self.counter
#         
#         
#         dt = self.dt
#         n = self.currentMemoryIndex[0]  
#         pos1 = self.positions[0]
#         pos2 = self.positions[1]
#         pos3 = self.positions[2]
#         
#         #""" Predictor Step """positions
#         if self.step == "predictor":
#             P1 = self.P_leftMother[n]
#             Q1 = self.Q_leftMother[n]
#             A1 = self.A_leftMother[n]
#             
#             P2 = self.P_rightMother[n]
#             Q2 = self.Q_rightMother[n]
#             A2 = self.A_rightMother[n]
#             
#             P3 = self.P_daughter[n]
#             Q3 = self.Q_daughter[n]
#             A3 = self.A_daughter[n]
#                  
#         #"""Corrector Step"""    
#         elif self.step == "corrector":
#             P1 = self.P_mother_pre
#             Q1 = self.Q_mother_pre
#             A1 = self.A_mother_pre
#             
#             P2 = self.P_leftDaughter_pre
#             Q2 = self.Q_leftDaughter_pre
#             A2 = self.A_leftDaughter_pre
#             
#             P3 = self.P_rightDaughter_pre
#             Q3 = self.Q_rightDaughter_pre
#             A3 = self.A_rightDaughter_pre
#                  
#         P1o = P1[pos1]
#         Q1o = Q1[pos1]
#         P2o = P2[pos2]
#         Q2o = Q2[pos2]
#         P3o = P3[pos3]
#         Q3o = Q3[pos3]
#         
#         # update system equation and store L1o
#         self.systemEquations[0].updateLARL(P1,Q1,A1,idArray=[pos1],update='L') #
#         L1o = self.systemEquations[0].L[pos1][pos1+1]
#         # calculate domega1
#         z1 = self.z[0][pos1] - self.systemEquations[0].LAMBDA[pos1][0] * dt
#         du1 = np.array([np.interp(z1,self.z[0],P1)-P1o,np.interp(z1,self.z[0],Q1)-Q1o])
#         domega1 =  np.dot(L1o,du1)
#         
#         # update system equation and store L2o
#         self.systemEquations[1].updateLARL(P2,Q2,A2,idArray=[pos2],update='L') #
#         L2o = self.systemEquations[1].L[pos2][pos2+1]
#         # calculate domega2
#         z2 = self.z[1][pos2] - self.systemEquations[1].LAMBDA[pos2][1] * dt
#         du2 = np.array([np.interp(z2,self.z[1],P2)-P2o,np.interp(z2,self.z[1],Q2)-Q2o])
#         domega2 =  np.dot(L2o,du2)
#         
#         # update system equation and store L2o
#         self.systemEquations[2].updateLARL(P3,Q3,A3,idArray=[pos3],update='L') #
#         L3o = self.systemEquations[2].L[pos3][pos3+1]
#         # calculate domega2
#         z3 = self.z[2][pos3] - self.systemEquations[2].LAMBDA[pos3][1] * dt
#         du3 = np.array([np.interp(z3,self.z[2],P3)-P3o,np.interp(z3,self.z[2],Q3)-Q3o])
#         domega3 =  np.dot(L3o,du3)
#         
#         # setup solve function
#         args = [A1[pos1],A2[pos2],A3[pos3],pos1,pos2,pos3,self.vz[0],self.vz[1],self.vz[2],P1o,Q1o,P2o,Q2o,P3o,Q3o,domega1,domega2,domega3,self.rho[0],self.rho[1],self.rho[2],L1o,L2o,L3o,du1,du2,du3]      
#         x = [P2o,Q2o,P1o,Q1o,Q3o,P3o]
#         #sol = fsolve(self.fsolveFunction ,x ,args = args, fprime = self.jacobiMatrix)
#         
#         #error = sum([abs(i) for i in self.fsolveFunction(x,args)])
#         
#         #if error < 1.E-4:
#             #print error
#             #return [P1o,Q1o,A[0][pos1],P2o,Q2o,A[1][pos2],P3o,Q3o,A[2][pos3]]
#         # solve system
#         sol,infodict,a,b = fsolve(self.fsolveFunction ,x ,args = args, fprime = self.jacobiMatrix,full_output=True)
#         #print "cC",infodict['nfev'],infodict['njev'], b
#         #if infodict['nfev'] > 2:input3("")
#         #""" Predictor Step """
#         if self.step == "predictor":
#             self.step = "corrector"
#
#             # apply changed values
#             self.P_mother_pre[pos1]        = sol[2]
#             self.Q_mother_pre[pos1]        = sol[3]
#             self.P_leftDaughter_pre[pos2]  = sol[0]
#             self.Q_leftDaughter_pre[pos2]  = sol[1]
#             self.P_rightDaughter_pre[pos3] = sol[5]
#             self.Q_rightDaughter_pre[pos3] = sol[4]
#
#             if self.rigidAreas == False:
#                 # calculate new areas
#                 A1n = self.A_func[0]([sol[2]],pos1)
#                 A2n = self.A_func[1]([sol[0]],pos2)
#                 A3n = self.A_func[2]([sol[5]],pos3)
#                 
#                 self.A_mother_pre[pos1]        = A1n
#                 self.A_leftDaughter_pre[pos2]  = A2n
#                 self.A_rightDaughter_pre[pos3] = A3n  
#             else:
#                 self.A_mother_pre[pos1]        = A1[pos1]
#                 self.A_leftDaughter_pre[pos2]  = A2[pos2]
#                 self.A_rightDaughter_pre[pos3] = A3[pos3]
#         
#         #"""Corrector Step"""    
#         elif self.step == "corrector":
#             self.step = "predictor"
#     
#             # apply changed values
#             self.P_leftMother[n+1][pos1]        = sol[2]
#             self.Q_leftMother[n+1][pos1]        = sol[3]
#             self.P_rightMother[n+1][pos2]  = sol[0]
#             self.Q_rightMother[n+1][pos2]  = sol[1]
#             self.P_daughter[n+1][pos3] = sol[5]
#             self.Q_daughter[n+1][pos3] = sol[4]
#             
#             if self.rigidAreas == False:
#                 # calculate new areas
#                 A1n = self.A_func[0]([sol[2]],pos1)
#                 A2n = self.A_func[1]([sol[0]],pos2)
#                 A3n = self.A_func[2]([sol[5]],pos3)
#                 
#                 self.A_leftMother[n+1][pos1]        = A1n
#                 self.A_rightMother[n+1][pos2]  = A2n
#                 self.A_daughter[n+1][pos3] = A3n  
#             else:
#                 self.A_leftMother[n+1][pos1]        = A1[pos1]
#                 self.A_rightMother[n+1][pos2]  = A2[pos2]
#                 self.A_daughter[n+1][pos3] = A3[pos3]
#         
#             print(  )
#             sumQError = abs(abs(sol[3])-abs(sol[1])-abs(sol[4]))/abs(sol[3])
#             if sumQError > 0.0: 
#                 self.sumQErrorCount = self.sumQErrorCount+1
#             if sumQError > self.maxQError:
#                 self.maxQError  = sumQError
#             print('Error cons mass',  sumQError, self.maxQError ,' - ', n, self.sumQErrorCount)
#             
#             sumPError = abs(abs(sol[2])-abs(sol[0]))/abs(sol[2])
#             if sumPError > 0.0: 
#                 self.sumPErrorCount = self.sumPErrorCount+1
#             if sumPError > self.maxPError:
#                 self.maxPError  = sumPError
#             print('Error P lin    ',  sumPError, self.maxPError ,' - ', n, self.sumPErrorCount)
#             
#             ## non linear error
#             sumPErrorNonLin = abs((sol[2]+0.5*(sol[3]/A1n)**2)-(sol[0]+0.5*(sol[1]/AL12n)**2))/abs(sol[2]+0.5*(sol[3]/A1n)**2)
#             if sumPErrorNonLin > 0.0: 
#                 self.sumPErrorNonLinCount = self.sumPErrorNonLinCount+1
#             if sumPErrorNonLin > self.maxPErrorNonLin:
#                 self.maxPErrorNonLin  = sumPErrorNonLin
#             print('Error P non lin',  sumPErrorNonLin, self.maxPErrorNonLin ,' - ', n, self.sumPErrorNonLinCount)
#             
#     
#     def fsolveBifurcationSys0(self,x,args):
#         """
#         Residual Function with equations to solve for at the bifuraction
#         Using constant areas, i.e. initial areas
#         
#         Input:     x = array [P2,Q2,P1,Q1,Q3,P3]
#                    args = args with local variables
#         Returns array with residuals 
#         """
#         P2,Q2,P1,Q1,Q3,P3 = x
#         A1,A2,A3,pos1,pos2,pos3,vz1,vz2,vz3,P1o,Q1o,P2o,Q2o,P3o,Q3o,domega1,domega2,domega3,rho1,rho2,rho3,L1,L2,L3,du1,du2,du3 = args
#         
#         du1[0] = P1 - P1o    
#         du1[1] = Q1 - Q1o
#         du2[0] = P2 - P2o
#         du2[1] = Q2 - Q2o
#         du3[0] = P3 - P3o
#         du3[1] = Q3 - Q3o
#         
#         self.systemEquations[0].updateLARL([P1],[Q1],[A1],idArray=[pos1],update='L')
#         self.systemEquations[1].updateLARL([P2],[Q2],[A2],idArray=[pos2],update='L')
#         self.systemEquations[2].updateLARL([P3],[Q3],[A3],idArray=[pos3],update='L')
#                 
#         #calculate residuals
#         res1 = vz1*Q1+vz2*Q2+vz3*Q3
#         res2 = vz1*P1+vz1*rho1*0.5*(Q1/A1)**2.+vz2*P2+vz2*rho2*0.5*(Q2/A2)**2.
#         res3 = vz1*P1+vz1*rho1*0.5*(Q1/A1)**2.+vz3*P3+vz3*rho3*0.5*(Q3/A3)**2.    
#         res4 = np.dot(self.systemEquations[0].L[pos1][pos1+1],du1) - domega1
#         res5 = np.dot(self.systemEquations[1].L[pos2][pos2+1],du2) - domega2
#         res6 = np.dot(self.systemEquations[2].L[pos3][pos3+1],du3) - domega3
#         
#         return [res5,res2,res4,res1,res3,res6]
#     
#     def jacobiMatrixBifSys0(self,x, args):
#         """
#         Returns the jabcobi matrix, bifurcation-functions and x; J = dF/dx
#         Using constant areas, i.e. initial areas
#         """
#         P2,Q2,P1,Q1,Q3,P3 = x
#         A1,A2,A3,pos1,pos2,pos3,vz1,vz2,vz3,P1o,Q1o,P2o,Q2o,P3o,Q3o,domega1,domega2,domega3,rho1,rho2,rho3,L1,L2,L3,du1,du2,du3 = args
#                 
#         return np.array([[L2[0], L2[1]            , 0    , 0                , 0                , 0    ],
#                          [vz2  , vz2*rho2*Q2/A2**2, vz1  , vz1*rho1*Q1/A1**2, 0                , 0    ],
#                          [0    , 0                , L1[0], L1[1]            , 0                , 0    ],
#                          [0    , vz2              , 0    , vz1              , vz3              , 0    ],
#                          [0    , 0                , vz1  , vz1*rho1*Q1/A1**2, vz3*rho3*Q3/A3**2, vz3  ],
#                          [0    , 0                , 0    , 0                , L3[1]            , L3[0]]])
#     
#     def fsolveBifurcationSys1(self,x,args):
#         """
#         Residual Function with equations to solve for at the bifuraction
#         Using recalculated areas depending on the new pressure values
#         
#         Input:     x = array [P2,Q2,P1,Q1,Q3,P3]
#                    args = args with local variables
#         Returns array with residuals 
#         """
#         P2,Q2,P1,Q1,Q3,P3 = x
#         A1,A2,A3,pos1,pos2,pos3,vz1,vz2,vz3,P1o,Q1o,P2o,Q2o,P3o,Q3o,domega1,domega2,domega3,rho1,rho2,rho3,L1,L2,L3,du1,du2,du3 = args
#                         
#         A1 = self.A_func[0]([P1],0)
#         A2 = self.A_func[1]([P2],0)
#         A3 = self.A_func[2]([P3],0)
#         
#         du1[0] = P1 - P1o    
#         du1[1] = Q1 - Q1o
#         du2[0] = P2 - P2o
#         du2[1] = Q2 - Q2o
#         du3[0] = P3 - P3o
#         du3[1] = Q3 - Q3o
#         
#         if self.updateL == True:   
#             self.systemEquations[0].updateLARL([P1],[Q1],[A1],idArray=[pos1],update='L')
#             self.systemEquations[1].updateLARL([P2],[Q2],[A2],idArray=[pos2],update='L')
#             self.systemEquations[2].updateLARL([P3],[Q3],[A3],idArray=[pos3],update='L')
#         
#         
#         res1 = vz1*Q1+vz2*Q2+vz3*Q3
#         res4 = np.dot(self.systemEquations[0].L[pos1][pos1+1],du1) - domega1
#         res5 = np.dot(self.systemEquations[1].L[pos2][pos2+1],du2) - domega2
#         res6 = np.dot(self.systemEquations[2].L[pos3][pos3+1],du3) - domega3
#         
#         if self.nonLin == True:
#         #calculate residuals
#             res2 = vz1*P1+vz1*rho1*0.5*(Q1/A1)**2.+vz2*P2+vz2*rho2*0.5*(Q2/A2)**2.
#             res3 = vz1*P1+vz1*rho1*0.5*(Q1/A1)**2.+vz3*P3+vz3*rho3*0.5*(Q3/A3)**2.    
#         else:
#             res2 = vz1*P1+vz2*P2
#             res3 = vz1*P1+vz3*P3  
#         
#         return [res5,res2,res4,res1,res3,res6]
#     
#     def jacobiMatrixBifSys1(self,x, args):
#         """
#         Returns the jabcobi matrix, bifurcation-functions and x; J = dF/dx
#         Using recalculated areas depending on the new pressure values
#         """
#         P2,Q2,P1,Q1,Q3,P3 = x
#         A1,A2,A3,pos1,pos2,pos3,vz1,vz2,vz3,P1o,Q1o,P2o,Q2o,P3o,Q3o,domega1,domega2,domega3,rho1,rho2,rho3,L1,L2,L3,du1,du2,du3 = args
#         
#         A1 = self.A_func[0]([P1],0)
#         A2 = self.A_func[1]([P2],0)
#         A3 = self.A_func[2]([P3],0)
#         
#         
#         if self.nonLin == True:
#             jacobi = np.array([[L2[0], L2[1]            , 0    , 0                , 0                , 0    ],
#                              [vz2  , vz2*rho2*Q2/A2**2, vz1  , vz1*rho1*Q1/A1**2, 0                , 0    ],
#                              [0    , 0                , L1[0], L1[1]            , 0                , 0    ],
#                              [0    , vz2              , 0    , vz1              , vz3              , 0    ],
#                              [0    , 0                , vz1  , vz1*rho1*Q1/A1**2, vz3*rho3*Q3/A3**2, vz3  ],
#                              [0    , 0                , 0    , 0                , L3[1]            , L3[0]]])
#         
#         else:
#             jacobi = np.array([[L2[0], L2[1]          , 0    , 0                , 0                , 0    ],
#                              [vz2  , 0                , vz1  , 0                , 0                , 0    ],
#                              [0    , 0                , L1[0], L1[1]            , 0                , 0    ],
#                              [0    , vz2              , 0    , vz1              , vz3              , 0    ],
#                              [0    , 0                , vz1  , 0                , 0                , vz3  ],
#                              [0    , 0                , 0    , 0                , L3[1]            , L3[0]]])
#         #print max(np.linalg.eigvals(jacobi))
#         
#         return jacobi
#         
        
class Anastomosis():
    
    def __init__(self, leftMother, leftMotherSys,
                       rightMother, rightMotherSys,
                       daughter, daughterSys, 
                       currentMemoryIndex, dt, rigidAreas, solvingScheme):
        
        
        # vessel variables initially set, constant through simulation
        self.type = 'Anastomosis'
        
        self.name = ' '.join(['Anastomosis',str(leftMother.Id),str(rightMother.Id),str(daughter.Id)])
        
        #System Variables
        self.dt = dt
        self.currentMemoryIndex = currentMemoryIndex
        
        self.rho = []
        self.systemEquations = []
        self.z = []
        self.A_func = []
        self.positions =[]
#         self.vz = []
        self.names = []
        
#         # equations to solve in f solve
#         self.fsolveFunction = None
#         self.jacobiMatrix = None
        
        ###initialize
        ##left mother branch
        self.rho.append(leftMother.rho)
        self.z.append(leftMother.z)
        self.systemEquations.append(leftMotherSys)
        self.positions.append(-1)
        self.names.append(leftMother.Id)
        self.A_func.append(leftMother.A_nID)
        #SolutionVariables
        self.P_leftMother = leftMother.Psol
        self.Q_leftMother = leftMother.Qsol
        self.A_leftMother = leftMother.Asol
        
        ##left mother branch
        self.rho.append(rightMother.rho)
        self.z.append(rightMother.z)
        self.systemEquations.append(rightMotherSys)
        self.positions.append(-1)
        self.names.append(rightMother.Id)
        self.A_func.append(rightMother.A_nID)
        #SolutionVariables
        self.P_rightMother = rightMother.Psol
        self.Q_rightMother = rightMother.Qsol
        self.A_rightMother = rightMother.Asol
        
        ##right daughter branch
        self.rho.append(daughter.rho)
        self.z.append(daughter.z)
        self.systemEquations.append(daughterSys)
        self.positions.append(0)
        self.names.append(daughter.Id)
        self.A_func.append(daughter.A_nID)
#        self.vz.append(1)
        #SolutionVariables
        self.P_daughter = daughter.Psol
        self.Q_daughter = daughter.Qsol
        self.A_daughter = daughter.Asol

        self.rigidAreas = rigidAreas
        
        self.quiet = True
    
        # Define the call function depending on the solving Scheme
        solvingScheme = "NonLinear"
        if solvingScheme == "Linear": 
            self._callfcn = self.callLinear
        elif solvingScheme == "NonLinear":
            self._callfcn = self.callNonLinear
        else:
            raise ValueError("Connections; wrong solving scheme! {}".format(solvingScheme))
        
        ## benchamark Test variables
        self.sumQErrorCount = 0
        self.maxQError = 0
        self.maxPErrorNonLin = 0 
        self.maxPError = 0
        self.sumPErrorCount = 0
        self.sumPErrorNonLinCount = 0

        # Internal stuff
        self.J_inv = np.empty((3,3))
        self.F = np.empty(3)
        self.epsilon_values = np.array([1,1,1])
        self.Xold = np.empty(3)
        self.Xnew = np.empty(3)

    def __call__(self):
        return self._callfcn()
    
    def callLinear(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        pos3 = self.positions[2]
        
        P1 = self.P_leftMother[n]
        Q1 = self.Q_leftMother[n]
        A1 = self.A_leftMother[n]
        
        P2 = self.P_rightMother[n]
        Q2 = self.Q_rightMother[n]
        A2 = self.A_rightMother[n]
        
        P3 = self.P_daughter[n]
        Q3 = self.Q_daughter[n]
        A3 = self.A_daughter[n]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        P3o = P3[pos3]
        Q3o = Q3[pos3]
                           
                
        # update LARL
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_1 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
        L,R3,LMBD,Z1,Z2,domega3_2 = self.systemEquations[2].updateLARL(P3,Q3,A3,pos3)
            
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
        
        R3_11 = R3[0][0]
        R3_12 = R3[0][1]
        R3_21 = R3[1][0]
        R3_22 = R3[1][1]
        
        ###### Linear approach
        
        ####### change?!
        denom = R1_12 * R2_12 * R3_21 - R1_12 * R2_22 * R3_11 - R1_22 * R2_12 * R3_11 
        
        alpha1 = -( R1_11 * R2_12 * R3_21 - R1_11 * R2_22 * R3_11 - R1_21 * R2_12 * R3_11)/denom
        alpha2 = -( R2_11 * R2_22 * R3_11 - R2_12 * R2_21 * R3_11)/denom
        alpha3 = -( R2_12 * R3_22 * R3_11 - R2_12 * R3_12 * R3_21)/denom
        
        beta1 = -(R1_11 * R1_22 * R3_11 - R1_12 * R1_21 * R3_11)/denom
        beta2 = -(R1_12 * R2_11 * R3_21 - R1_12 * R2_21 * R3_11 - R1_22 * R2_11 * R3_11)/denom
        beta3 = -(R1_12 * R3_11 * R3_22 - R1_12 * R3_12 * R3_21)/denom
        
        gamma1 = -( R1_11 * R1_22 * R2_12 - R1_12 * R1_21 * R2_12)/denom
        gamma2 = -( R1_12 * R2_11 * R2_22 + R1_12 * R2_12 * R2_21)/denom
        gamma3 = -( R1_12 * R2_12 * R3_22 - R1_12 * R2_22 * R3_12 - R1_22 * R2_12 * R3_12)/denom
        
        ############        
        domega1_2 = (alpha1 * domega1_1  + alpha2 * domega2_1 + alpha3 * domega3_2 )
        domega2_2 = (beta1  * domega1_1  + beta2  * domega2_1 + beta3  * domega3_2 )
        domega3_1 = (gamma1 * domega1_1  + gamma2 * domega2_1 + gamma3 * domega3_2 )
        
        P1_new = P1o + (R1_11*domega1_1 + R1_12*domega1_2)
        Q1_new = Q1o + (R1_21*domega1_1 + R1_22*domega1_2)
    
        P2_new = P2o + (R2_11*domega2_1 + R2_12*domega2_2)
        Q2_new = Q2o + (R2_21*domega2_1 + R2_22*domega2_2)
        
        P3_new = P3o + (R3_11*domega3_1 + R3_12*domega3_2)
        Q3_new = Q3o + (R3_21*domega3_1 + R3_22*domega3_2)
               
        
        # apply calculated values to next time step solution
        self.P_leftMother[n+1][pos1]  = P1_new
        self.Q_leftMother[n+1][pos1]  = Q1_new
        self.P_rightMother[n+1][pos2] = P2_new
        self.Q_rightMother[n+1][pos2] = Q2_new
        self.P_daughter[n+1][pos3]    = P3_new
        self.Q_daughter[n+1][pos3]    = Q3_new
        
        if P1_new < 0 or P2_new < 0 or P3_new < 0:
            tmpstring = "Connection: {} calculated negative pressure at time {} (n {},dt {})".format(self.name,n*dt,n,dt)
            tmpstring = tmpstring + "\n the values were: P1_new = {}, P2_new = {}, P3_new = {}".format(P1_new, P2_new, P3_new)
            raise ValueError(tmpstring)
            #exit()
                
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)
            A3n = self.A_func[2]([P3_new],pos3)
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
            A3n = A3[pos3] 
           
        self.A_leftMother[n+1][pos1]   = A1n
        self.A_rightMother[n+1][pos2]  = A2n       
        self.A_daughter[n+1][pos3]     = A3n  
            
        ## non linear error        
        try: sumQError = abs(Q1_new+Q2_new-Q3_new)#/abs(Q1_new)
        except: sumQError = 0.0
        if sumQError > 0.0: 
            self.sumQErrorCount = self.sumQErrorCount+1
        if sumQError > self.maxQError:
            self.maxQError  = sumQError
        #print self.name,' \n Error cons mass',  sumQError, self.maxQError ,' - ', n, self.sumQErrorCount
        if sumQError > 1.e-5:
            print("Warning: Connection: {} to high error in conservation of mass at time {} (n {},dt {})".format(self.name,n*dt,n,dt))
            print(sumQError, ' <- Q1,Q2,Q3 ',Q1_new, Q2_new, Q3_new)
            #exit()
        
        sumPError = abs(P1_new-P3_new)/abs(P3_new)
        if sumPError > 0.0: 
            self.sumPErrorCount = self.sumPErrorCount+1
        if sumPError > self.maxPError:
            self.maxPError  = sumPError
        #print self.name,' Error P lin    ',  sumPError, self.maxPError ,' - ', n, self.sumPErrorCount
        if sumPError > 1.e-5:
            print("ERROR: Connection: {} to high error in conservation of pressure at time {} (n {},dt {})".format(self.name,n*dt,n,dt))
            print(sumPError, ' <- P1,P2,P3 ',P1_new, P2_new, P3_new)
            #exit()
        
        sumPErrorNonLin = 1050./2.*(Q1_new/A1n)**2#abs(P1_new+500*(Q1_new/A1n)**2-(P2_new+500*(Q2_new/A2n)**2))/abs(P1_new+0.5*(Q1_new/A1n)**2)
        if sumPErrorNonLin > 0.0: 
            self.sumPErrorNonLinCount = self.sumPErrorNonLinCount+1
        if sumPErrorNonLin > self.maxPErrorNonLin:
            self.maxPErrorNonLin  = sumPErrorNonLin
        print(self.name,' Error P non lin',  sumPErrorNonLin, self.maxPErrorNonLin ,' - ', n, self.sumPErrorNonLinCount)
        
        print(self.name,'dynamic Pressures',1050./2.*(Q1_new/A1n)**2,1050./2.*(Q2_new/A2n)**2,1050./2.*(Q3_new/A3n)**2, '--',1050./2.*(Q1_new/A1n)**2+-1050./2.*(Q3_new/A3n)**2)
        

    def callNonLinear(self):
        """
        Call function for vessel-vessel connection
        """        
        dt = self.dt
        n = self.currentMemoryIndex[0]
        #if n == 1:
            #print "using nonlinear bifurcation model"
        #print "using nonlinear bifurcation model"
        pos1 = self.positions[0]
        pos2 = self.positions[1]
        pos3 = self.positions[2]
        
        P1 = self.P_leftMother[n]
        Q1 = self.Q_leftMother[n]
        A1 = self.A_leftMother[n]
        
        P2 = self.P_rightMother[n]
        Q2 = self.Q_rightMother[n]
        A2 = self.A_rightMother[n]
        
        P3 = self.P_daughter[n]
        Q3 = self.Q_daughter[n]
        A3 = self.A_daughter[n]
        
        rho1 = self.rho[0]
        rho2 = self.rho[1]
        rho3 = self.rho[2]
        
        P1o = P1[pos1]
        Q1o = Q1[pos1]
        A1o = A1[pos1]
        P2o = P2[pos2]
        Q2o = Q2[pos2]
        A2o = A2[pos2]
        P3o = P3[pos3]
        Q3o = Q3[pos3]
        A3o = A3[pos3]

        ## update LARL
        L,R1,LMBD,Z1,Z2,domega1_1 = self.systemEquations[0].updateLARL(P1,Q1,A1,pos1)
        L,R2,LMBD,Z1,Z2,domega2_1 = self.systemEquations[1].updateLARL(P2,Q2,A2,pos2)
        L,R3,LMBD,Z1,Z2,domega3_2 = self.systemEquations[2].updateLARL(P3,Q3,A3,pos3)
        
        # local R matrices
        R1_11 = R1[0][0]
        R1_12 = R1[0][1]
        R1_21 = R1[1][0]
        R1_22 = R1[1][1]
        
        R2_11 = R2[0][0]
        R2_12 = R2[0][1]
        R2_21 = R2[1][0]
        R2_22 = R2[1][1]
        
        R3_11 = R3[0][0]
        R3_12 = R3[0][1]
        R3_21 = R3[1][0]
        R3_22 = R3[1][1]
        
        if n>0:
            P1o2 = self.P_leftMother[n-1][pos1]
            P2o2 = self.P_rightMother[n-1][pos2]
            P3o2 = self.P_daughter[n-1][pos3]
            
            deltaP1_last = P1o - P1o2
            deltaP2_last = P2o - P2o2
            deltaP3_last = P3o - P3o2
            
            domega1_2_init = (deltaP1_last - R1_11*domega1_1)/R1_12 #same omega as previous timestep
            domega2_2_init = (deltaP2_last - R2_11*domega2_1)/R2_12 #same omega as previous timestep
            domega3_1_init = (deltaP3_last - R3_12*domega3_2)/R3_11 #same omega as previous timestep
        else:
            domega1_2_init = 0#-R1_11*domega1_1/R1_12 #result if pressure is constant- could be optimized
            domega2_2_init = 0#-R2_12*domega2_1/R2_11 #result if pressure is constant
            domega3_1_init = 0#-R3_12*domega3_2/R3_11 #result if pressure is constant

        epsilonvalues =  self.epsilon_values
        epsilonvalues[:] = 1
        epsilonlimit = 1e-10
        domega1_2_last = domega1_2_init
        domega2_2_last = domega2_2_init
        domega3_1_last = domega3_1_init
        A1_last = A1o
        A2_last = A2o
        A3_last = A3o
        Niterations = 0
        Xold = self.Xold
        Xold[:] = [domega1_2_last, domega2_2_last, domega3_1_last]
        Xnew = self.Xnew
#         print("\n")
#         print("domega1_2_last, domega2_1_last, domega3_1_last, domega1_1, domega2_1, domega3_2 : ", domega1_2_last, domega2_1_last, domega3_1_last, domega1_1, domega2_1, domega3_2)
#         print("\n")
        while epsilonvalues[0]>1e-10 or epsilonvalues[1]>1e-5 or epsilonvalues[2]>1e-5:
            """iterative Newton Rahpson solver
                domega1_2, domega2_1, domega3_1 are the unknowns that are changing from
            each iteration. A1, A2, A3 are also changing, but the value from the previous iteration is used. (should actually be for the next timestep, but this should converge to the correct solution)
            R1_11, R1_12, R1_21, R1_22, ,R2_11, R2_12, R2_21, R2_22, ,R3_11, R3_12, R3_21, R3_22, domega1_1, domega2_1, domega3_2, Q1o, Q2o, Q2o 
            are constant for each timestep, and are taken from previous timestep. domega1_1, domega2_1, domega3_2 are the field domegas.
            """
            domega1_2_last = Xold[0]
            domega2_2_last = Xold[1]
            domega3_1_last = Xold[2]
            
            Q1discretize = Q1o + R1_21*domega1_1 + R1_22*domega1_2_last
            Q2discretize = Q2o + R2_21*domega2_1 + R2_22*domega2_2_last
            Q3discretize = Q3o + R3_21*domega3_1_last + R3_22*domega3_2
            
            P1discretize = P1o + R1_11*domega1_1 + R1_12*domega1_2_last
            P2discretize = P2o + R2_11*domega2_1 + R2_12*domega2_2_last
            P3discretize = P3o + R3_11*domega3_1_last + R3_12*domega3_2
            

            if self.rigidAreas == False:
                try:
                    A1_last = self.A_func[0]([P1discretize],pos1)
                    A2_last = self.A_func[1]([P2discretize],pos2)
                    A3_last = self.A_func[2]([P3discretize],pos3)
                except FloatingPointError as E:
                    print("Floating Point error in Connection {}".format(self.name))
                    raise E
            else:
                A1_last = A1[pos1]
                A2_last = A2[pos2]
                A3_last = A3[pos3] 
            
            f1 = Q1discretize + Q2discretize - Q3discretize#R1_21*domega1_1 + R1_22*domega1_2_last  - R2_21*domega2_1_last - R2_22*domega2_1 - R3_21*domega3_1_last - R3_22*domega3_2
            
            f2 = P1discretize + 0.5*rho1*((Q1discretize/A1_last)**2) - P3discretize - 0.5*rho3*((Q3discretize/A3_last)**2)
            Xold[:] = [domega1_2_last, domega2_2_last, domega3_1_last]
            f3 = P2discretize + 0.5*rho2*((Q2discretize/A2_last)**2) - P3discretize - 0.5*rho3*((Q3discretize/A3_last)**2)
            
            self.F[:] = [f1, f2, f3]
            F = self.F
            """Inverse Jacobi elements: """
            a = R1_22
            b = R2_22
            c = -R3_21
            d = R1_12 + rho1*R1_22*(Q1discretize)/(A1_last**2)
            e = - R3_11 - rho3*R3_21*(Q3discretize)/(A3_last**2)
            f = R2_12 + rho2*R2_22*(Q2discretize)/(A2_last**2)
            
            
            Determinant = a*e*f + b*d*e -c*d*f

            # TODO allocate outside loop
            # J_inv = np.array([[ e*f, b*e - c*f, -b*e  ],
            #               [ -d*e, -a*e, a*e - c*d ],
            #               [  -d*f, a*f, b*d ]]) / (Determinant)
            #
            self.J_inv[:,:] = [[ e*f, b*e - c*f, -b*e  ],
                              [ -d*e, -a*e, a*e - c*d ],
                              [  -d*f, a*f, b*d ]]
            self.J_inv[:] = self.J_inv[:]/Determinant

            Xnew[:] = Xold - np.dot(self.J_inv, F)

            epsilonvalues[:] = np.abs(F)
            Niterations = Niterations + 1
            
            if Niterations > 30:
                if self.quiet:
                    break
                else:
                    print("\n")
                    print("Niterations excedded in anastomosis calculation in vessel, Niterations: ", self.names[0], Niterations)
    
                    print("Xnew: ", Xnew)
                    print("Xold: ", Xold)
                    
                    print("f1: ", f1)
                    print("f2: ", f2)
                    print("f3: ", f3)
                    print("epsilonvalues: ", epsilonvalues)
                    print("Q1discretize, Q1o: ", Q1discretize, Q1o)
                    print("Q2discretize, Q2o: ", Q2discretize, Q2o)
                    print("Q3discretize, Q3o: ", Q3discretize, Q3o)
                    print("P1discretize, P1o: ", P1discretize, P1o)
                    print("P2discretize, P2o: ", P2discretize, P2o)
                    print("P3discretize, P3o: ",P3discretize, P3o)
                    #exit()
                    break
            Xold[:] = Xnew
            #exit()
        

        Q1_new = Q1discretize
        Q2_new = Q2discretize
        Q3_new = Q3discretize
        
        P1_new = P1discretize
        P2_new = P2discretize
        P3_new = P3discretize
        
        # apply calculated values to next time step
        self.P_leftMother[n+1][pos1]    = P1_new
        self.Q_leftMother[n+1][pos1]    = Q1_new
        self.P_rightMother[n+1][pos2]  = P2_new
        self.Q_rightMother[n+1][pos2]  = Q2_new
        self.P_daughter[n+1][pos3] = P3_new
        self.Q_daughter[n+1][pos3] = Q3_new
        
        if P1_new < -500*133.32 or P2_new < -500*133.32 or P3_new < -500*133.32:
            print("ERROR: Connection: {} calculated negative pressure at time {} (n {},dt {}), exit system".format(self.name,n*dt,n,dt))
            print(P1_new, P2_new, P3_new)
            print("Niterations: ", Niterations)
            
            print("solving nonlinear/total pressure anastomosis")
            print("domega1_2_init: ", domega1_2_init)
            print("domega1_1: ", domega1_1)
                    
            print("f1: ", f1)
            print("f2: ", f2)
            print("f3: ", f3)
            print("epsilonvalues: ", epsilonvalues)
            print("Q1discretize, Q1o: ", Q1discretize, Q1o)
            print("Q2discretize, Q2o: ", Q2discretize, Q2o)
            print("Q3discretize, Q3o: ", Q3discretize, Q3o)
            print("P1discretize, P1o: ", P1discretize, P1o)
            print("P2discretize, P2o: ", P2discretize, P2o)
            print("P3discretize, P3o: ",P3discretize, P3o)
            exit()
        
        
        # calculate new areas
        if self.rigidAreas == False:
            A1n = self.A_func[0]([P1_new],pos1)
            A2n = self.A_func[1]([P2_new],pos2)
            A3n = self.A_func[2]([P3_new],pos3)
        else:
            A1n = A1[pos1]
            A2n = A2[pos2]
            A3n = A3[pos3] 
           
        self.A_leftMother[n+1][pos1] = A1n
        self.A_rightMother[n+1][pos2] = A2n
        self.A_daughter[n+1][pos3] = A3n
