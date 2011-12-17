#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import numpy as np
import conversions


# Reading VTK files for convenient handling...
class vtkMeshFileReader() :
    
    def __init__( self, vtkMeshFileName ):
        
        if not os.path.exists( vtkMeshFileName ) :
            return 
        
        self.vtkMeshFileName = vtkMeshFileName
        
        self.pointsNum    = 0
        self.pointsType   = ''
        
        self.cellsNum     = 0
        self.cellsEntries = 0
        
        self.cellTypesNum  = 0
        
        # read the file... (currently only ASCII is supported)
        self._readAndInterpretFile()
        
        
        
    def _readAndInterpretFile( self ) :
        ''' Basically split the file into its components. 
            POINTS, CELLS, CELL TYPES  
        '''
        print( 'Reading file from: %s' % self.vtkMeshFileName ) 
        
        # open for reading
        vtkFile = file( self.vtkMeshFileName, 'r' )
        
        # read the lines...
        lines = vtkFile.readlines()
        
        # iterate through the lines and find the "points" key word
        idxPOINTS    = 0
        idxCELLS     = 0 
        idxCELLTYPES = 0
        
        for i in range( len( lines ) ):
            
            # found points?
            if lines[i].startswith( 'POINTS' ) :
                idxPOINTS = i
                [a, self.pointsNum, self.pointsType] = lines[i].split()
                self.pointsNum = int( self.pointsNum )
            
            # found cells?
            if lines[i].startswith( 'CELLS' ) :
                idxCELLS = i
                [a, self.cellsNum, self.cellsEntries] = lines[i].split()
                self.cellsNum     = int( self.cellsNum     )
                self.cellsEntries = int( self.cellsEntries )
                
            
            # found cell types?    
            if lines[i].startswith( 'CELL_TYPES' ) :
                idxCELLTYPES = i
                [a,self.cellTypesNum] = lines[i].split()
                self.cellTypesNum = int( self.cellTypesNum )
            
        print( 'Found %d points in line %d '            % ( self.pointsNum,idxPOINTS       ) )
        print( 'Found %d cells in line %d '             % ( self.cellsNum, idxCELLS        ) )
        print( 'Found %d cell type entries in line %d ' % (self.cellTypesNum, idxCELLTYPES ) )

        self.points    = np.zeros( (self.pointsNum,3), dtype = self.pointsType)
        self.cells     = np.zeros( (self.cellsNum, int( self.cellsEntries/self.cellsNum ) ), dtype = int ) # this might blow up under certain conditions
        self.cellTypes = np.zeros( (self.cellTypesNum, 1), dtype = int )

        # write the points into the array
        for i in range( idxPOINTS + 1, idxPOINTS + 1 + self.pointsNum ):
            self.points[i-idxPOINTS-1,:] = lines[i].split()

        # write the cells into the array
        for i in range( idxCELLS + 1, idxCELLS + 1 + self.cellsNum ):
            self.cells[i-idxCELLS-1,:] = lines[i].split()

        for i in range( idxCELLTYPES + 1, idxCELLTYPES + 1 + self.cellsNum ):
            self.cellTypes[i-idxCELLTYPES-1,:] = lines[i]

        # save what was in the 
        self.vtkHeader       = lines[ 0 : min( idxCELLS, idxCELLTYPES, idxPOINTS ) ]
        self.cellHeader      = lines[ idxCELLS     ]
        self.cellTypesHeader = lines[ idxCELLTYPES ]
        self.pointsHeader    = lines[ idxPOINTS    ]
        print('Reading done.')
        
        
        
        
    def writeToFile( self, outFileName ):
        '''Write the written (and possibly modified) file back to disk...
        '''
        
        print( 'Writing file to: %s' % outFileName )
        
        f = file( outFileName , 'w')

        # Header
        f.writelines( self.vtkHeader )
        f.write('\n')
        
        # Points / nodes
        f.writelines( self.pointsHeader )
        f.writelines( conversions.numpyArrayToStr(self.points, True, '' ) )  
        #writeArrayToFile( reader.points, f, True )
        f.write('\n')
        
        # Cells / elements
        f.writelines( self.cellHeader )
        f.writelines( conversions.numpyArrayToStr(self.cells, False, '' ) )
        #writeArrayToFile( reader.cells, f, False )
        f.write('\n')
        
        # cell types
        f.writelines(self.cellTypesHeader)
        f.writelines(conversions.numpyArrayToStr(self.cellTypes, False, '' ) )
        #writeArrayToFile(reader.cellTypes, f, False )
        f.write('\n')
        
        
        f.close()
        
        print( 'Writing done.' )
        
        


##############################
#
# DEBUG section
#
if __name__ == '__main__':
    
    
    strFileName = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/proneMaskMuscleFatGland.1.vtk'
    
    
    
    vtkMeshFileReader(strFileName)
    