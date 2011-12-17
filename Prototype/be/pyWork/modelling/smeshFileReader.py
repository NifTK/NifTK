#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import numpy as np

class smeshFileReader :
    
    def __init__( self, smeshFileName ):
        
        # Remember the original file name
        self.smeshFileName = smeshFileName
        
        if not os.path.exists( self.smeshFileName ) :
            return  
        
        self._readAndInterpretFile()
        
        
        
        
    def _readAndInterpretFile( self ):
        
        # read the lines from file
        f     = file( self.smeshFileName, 'r' )
        lines = f.readlines()
        
        # iterate and find the first line which is  not starting with a #
        
        noComments = []
        
        for i in range( len(lines ) ) :
            if lines[i].startswith('#') :
                continue
            else :
                noComments.append( lines[i] )
        
        # First line in file (except form comments) gives node number, dimension, attribute number and boundary marker
        [numNodes, dim, numNodeAttributes, numNodeBoundaryMarkers] = noComments[0].split()
        numNodes               = int(numNodes)
        dim                    = int(dim)
        numNodeAttributes      = int( numNodeAttributes )
        numNodeBoundaryMarkers = int( numNodeBoundaryMarkers )
        
        self.nodes = np.zeros( (numNodes, dim+1 ), dtype = 'float' )
        
        for i in range(1,numNodes+1):
            self.nodes[i-1,:] = noComments[i].split()
            
        [numFacets, numFacetBoundaryMarker] = noComments[ numNodes+1 ].split()
        numFacets              = int(numFacets)
        numFacetBoundaryMarker = int( numFacetBoundaryMarker )
        
        self.facets = np.zeros( (numFacets,4), dtype = 'int' )
        
        for i in range( numNodes+2, numNodes+2+numFacets ):
            self.facets[i - numNodes-2,:] = noComments[i].split()
        
        if not noComments[ numNodes + numFacets + 2 ].startswith('0') :
            print('Warning: Holes are not implemented...')
        
        if not noComments[ numNodes + numFacets + 3 ].startswith('0') :
            print('Warning: Region attributes not implemented...')
            
        print('Reading done.')
            
        
        
if __name__ == '__main__' :
    
    fileName = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/004/chestWall.smesh'
    reader= smeshFileReader(fileName)
    