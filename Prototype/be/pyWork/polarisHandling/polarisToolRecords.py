#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np

class polarisToolRecords :
    
    def __init__( self, portDescription, portList, frameList, faceList, stateList, RxList, RyList, RzList, TxList, TyList, TzList, errorList ):
        
        self.port            = int( portList[0] )
        self.portDescription = portDescription
        self.frameList       = frameList
        self.faceList        = faceList
        self.stateList       = stateList
        
        self.RxList          = RxList
        self.RyList          = RyList
        self.RzList          = RzList
        
        self.TxList          = TxList
        self.TyList          = TyList
        self.TzList          = TzList
        
        self.errorList       = errorList
        
        self.meanRx = np.mean( self.RxList )
        self.meanRy = np.mean( self.RyList )
        self.meanRz = np.mean( self.RzList )
        
        self.meanTx = np.mean( self.TxList )
        self.meanTy = np.mean( self.TyList )
        self.meanTz = np.mean( self.TzList )
        
