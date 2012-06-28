#! /usr/bin/env python 
# -*- coding: utf-8 -*-



import xmlModelGenerator as xGen
import xmlModelReader as xRead
from glob import glob
import os
import numpy as np
import mayaviPlottingWrap as plotWrap
import xmlModelReader as xRead


class modelDeformationVisualiser :
    
    def __init__ ( self, xmlModelGenerator, deformationFileName=None ) :

        # check if generator is of correct type        
        if not ( isinstance( xmlModelGenerator, xGen.xmlModelGenrator ) or
                 isinstance(xmlModelGenerator, xRead.xmlModelReader   ) ):
            
            if isinstance( xmlModelGenerator, str ):
                self.modelReader = xRead.xmlModelReader( xmlModelGenerator )
                self.xmlGenerator = self.modelReader 
            else:
                print( 'Error Expected an xmlModelGenerator as input...' )
                return
    
        else: 
            self.xmlGenerator = xmlModelGenerator
        
        self.mdlNodes     = self.xmlGenerator.nodes
        self.mldElements  = self.xmlGenerator.elements
    
        self.dim          = 3
        
        if deformationFileName != None :
            self.deformationFileName = deformationFileName
        else :
            self.deformationFileName = 'U.txt'
        
        self._readDeformationFile()
        self._generateDeformedModels()
        self._generateDeformationVectors()
        
        
        
        
    def _readDeformationFile( self ) :
        
        # The deformation file is expected to be in the same directory as the final xml model file. 
        xmlBaseDir = os.path.split( self.xmlGenerator.xmlFileName )[0]
        deformationFileCandidates = glob( xmlBaseDir + '/' + os.path.split(self.deformationFileName)[-1] )
        
        uFile  = file( deformationFileCandidates[0],'r' ) 
        uLines = uFile.readlines()
        uFile.close()
        
        numNodes = self.mdlNodes.shape[0]
        
        if numNodes * 3 != len( uLines ) :
            print( 'Possible error!' )
            return
        
        # peek into the first line and see how many time points were recorded...
        numTimePoints = len( uLines[0].split() )
        self.u = np.zeros( ( numNodes * self.dim, numTimePoints), dtype='d' )
        
        for i in range(len( uLines ) ):
            self.u[i,:] = uLines[i].split()
    
        #scale to mm domain...
        self.u = self.u
        print('Reading done.')
        
        
        
        
    def _generateDeformedModels( self ) :
        
        # try an array of deformed numpyArrays. One element for each time point captured...
        self.deformedNodes = []
        self.displacements = []
        
        for i in range( self.u.shape[1] ) :
            
            self.displacements.append( self.u[:,i].reshape( (-1, self.dim) ) )
            self.deformedNodes.append( self.displacements[-1] + self.mdlNodes )
            
        print( 'Deformed nodes were generated' )
        
        
        
    def _generateDeformationVectors( self ) :
        
        # Get the last deformation from the file
        self.deformVectors = self.u[:,-1].reshape( self.u.shape[0]/self.dim, self.dim )
        
    

    
    def animateDeformation( self ) :
        
        self.plot = plotWrap.plotArrayAs3DPoints( self.mdlNodes * 1000 )
        ms = self.plot.mlab_source
        
        for i in range( len( self.deformedNodes ) ) :
            ms.reset( points = self.deformedNodes[i] * 1000 )
        
        
        
            
    def deformationAsVectors( self ) :
        
        plotWrap.plotVectorsAtPoints( self.deformVectors * 1000, self.mdlNodes * 1000)
        
        pass
        
