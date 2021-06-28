from __future__ import print_function, absolute_import
from future.utils import iteritems, iterkeys, viewkeys, viewitems, itervalues, viewvalues

from starfish.VascularPolynomialChaosLib.testBaseClass import TestBaseClass

class YoungAndTsaiInput(TestBaseClass):

    externVariables      = {'motherId': TestBaseClass.ExtValue(int),
                            'daughterId': TestBaseClass.ExtValue(int),
                            'Kv':TestBaseClass.ExtValue(float),
                            'Kt':TestBaseClass.ExtValue(float),
                            'Ku':TestBaseClass.ExtValue(float),
                            'A0':TestBaseClass.ExtValue(float),
                            'As':TestBaseClass.ExtValue(float),
                            'Ls':TestBaseClass.ExtValue(float)
                            } 
    
    externXmlAttributes  = []
    externXmlElements    = ['motherId',
                            'daughterId',
                            'Kv',
                            'Kt',
                            'Ku',
                            'A0',
                            'As',
                            'Ls'
                            ]

    def __init__(self):        
        '''
        The RandomInputManager class is as in the network xml file
        the container of correlation matrix and random inputs.
        In addition it has methods to sorts out the connection between random inputs
        and connects the random inputs via the update functions to the vascular network variables
        or other random inputs. (see method: linkRandomInputUpdateFunctions)
        '''        
        self.stenosesVariables = {} # randomInput as they stand in xml
        
        self.motherId = None
        self.daughterId = None
        self.Kv = None
        self.Kt = None
        self.Ku = None
        self.A0 = None
        self.As = None
        self.Ls = None

class StenosesInputManager(TestBaseClass):
    
    externVariables      = {'stenoses' : TestBaseClass.ExtDict('stenosesType', 
                                                                   TestBaseClass.ExtObject({'YoungAndTsaiInput':YoungAndTsaiInput}))} 
    externXmlAttributes  = []
    externXmlElements    = ['stenoses']
    
    def __init__(self):        
        '''
        The RandomInputManager class is as in the network xml file
        the container of correlation matrix and random inputs.
        In addition it has methods to sorts out the connection between random inputs
        and connects the random inputs via the update functions to the vascular network variables
        or other random inputs. (see method: linkRandomInputUpdateFunctions)
        '''        
        self.stenoses = {} # randomInput as they stand in xml
        
        
        



                                                

            


