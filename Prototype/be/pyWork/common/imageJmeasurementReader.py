#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np

def readFileAsArray( strFileNameIn, headerRow=True ):
    
    f       = file(strFileNameIn, 'r')
    lines   = f.readlines()
    numCols = len( lines[1].split() )
    
    if headerRow:
        numRows = len( lines ) - 1
        
        results = np.zeros( (numRows, numCols) )
        header = lines[0].split()
                                
        for i in range( 1, len( lines ) ) :
            results[i-1,:]  = lines[i].split()

    else:
        numRows = len( lines )
        
        results = np.zeros( (numRows, numCols) )
        header = []
                                
        for i in range( 0, len( lines ) ) :
            results[i-1,:]  = lines[i].split()
            
            
    return ( results, header )