#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np

def lowAndHighModelCoordinates( volMeshPoints, deltaX, deltaY, deltaZ ):
    
    # find those nodes of the model, which are close to the sternum... i.e. low x-values
    minXCoordinate = np.min( volMeshPoints[:,0] )
    maxXCoordinate = np.max( volMeshPoints[:,0] )
    
    minYCoordinate = np.min( volMeshPoints[:,1] )
    maxYCoordinate = np.max( volMeshPoints[:,1] )
    
    minZCoordinate = np.min( volMeshPoints[:,2] )
    maxZCoordinate = np.max( volMeshPoints[:,2] )
        
    lowXPoints  = []
    highXPoints = []
    lowXIdx     = []
    highXIdx    = []
    
    lowYPoints  = []
    highYPoints = []
    lowYIdx     = []
    highYIdx    = []
    
    lowZPoints  = []
    highZPoints = []
    lowZIdx     = []
    highZIdx    = []
    
    for i in range( volMeshPoints.shape[0] ):
        # lower x boundary
        if volMeshPoints[i,0] < ( minXCoordinate + deltaX ) :
            lowXIdx.append( i )
            lowXPoints.append( [volMeshPoints[i,0], volMeshPoints[i,1], volMeshPoints[i,2] ] )
        
        # upper x boundary
        if volMeshPoints[i,0] > ( maxXCoordinate - deltaX ) :
            highXIdx.append( i )
            highXPoints.append( [volMeshPoints[i,0], volMeshPoints[i,1], volMeshPoints[i,2] ] )
        
        # lower y boundary
        if volMeshPoints[i,1] < ( minYCoordinate + deltaY ) :
            lowYIdx.append( i )
            lowYPoints.append( [volMeshPoints[i,0], volMeshPoints[i,1], volMeshPoints[i,2] ] )
        
        # higher y boundary
        if volMeshPoints[i,1] > ( maxYCoordinate - deltaY ) :
            highYIdx.append( i )
            highYPoints.append( [volMeshPoints[i,0], volMeshPoints[i,1], volMeshPoints[i,2] ] )
        
        # lower z boundary
        if volMeshPoints[i,2] < ( minZCoordinate + deltaZ ) :
            lowZIdx.append( i )
            lowZPoints.append( [volMeshPoints[i,0], volMeshPoints[i,1], volMeshPoints[i,2] ] )
        
        # higher z boundary
        if volMeshPoints[i,2] > ( maxZCoordinate - deltaZ ) :
            highZIdx.append( i )
            highZPoints.append( [volMeshPoints[i,0], volMeshPoints[i,1], volMeshPoints[i,2] ] )
            
        
        
    lowXPoints  = np.array( lowXPoints  )
    highXPoints = np.array( highXPoints )
    lowXIdx     = np.array( lowXIdx     )
    highXIdx    = np.array( highXIdx    )
    
    lowYPoints  = np.array( lowYPoints  )
    highYPoints = np.array( highYPoints )
    lowYIdx     = np.array( lowYIdx     )
    highYIdx    = np.array( highYIdx    )
    
    lowZPoints  = np.array( lowZPoints  )
    highZPoints = np.array( highZPoints )
    lowZIdx     = np.array( lowZIdx     )
    highZIdx    = np.array( highZIdx    )
    
    print( 'Found %i points within an x range between [ -inf ; %f ]' % (len( lowXIdx ), minXCoordinate + deltaX ) )


    return lowXPoints, lowXIdx, highXPoints, highXIdx, lowYPoints, lowYIdx, highYPoints, highYIdx, lowZPoints, lowZIdx, highZPoints, highZIdx, 
