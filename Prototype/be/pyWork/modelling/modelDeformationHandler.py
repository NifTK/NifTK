#! /usr/bin/env python 
# -*- coding: utf-8 -*-

# Evaluation: Comparing Tanner simulations with registration results


import xmlModelGenerator as xGen
import xmlModelReader as xRead
from glob import glob
import os
import numpy as np


class modelDeformationHandler :
    
    def __init__ ( self, xmlModelGenerator, deformationFileName=None ) :
        ''' @attention: The xmlModelGenrator can also be an xmlModelReader, provided it has the 
                        member .nodes, .elements and .xmlFileName
        '''

        # check if generator is of correct type        
        if not ( isinstance( xmlModelGenerator, xGen.xmlModelGenrator ) or 
                 isinstance( xmlModelGenerator, xRead.xmlModelReader  ) ):
            print( 'Error Expected an xmlModelGenerator as input...' )
            return
        
        self.mdlNodes     = xmlModelGenerator.nodes
        self.mldElements  = xmlModelGenerator.elements
        self.xmlGenerator = xmlModelGenerator
        self.dim          = 3
        
        if deformationFileName != None :
            self.deformationFileName = deformationFileName
        else :
            self.deformationFileName = 'U.txt'
        
        self._readDeformationFile()
        self._generateDeformedModel()
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
        
        

        
    def _generateDeformedModel( self ) :
        
        # try an array of deformed numpyArrays. One element for each time point captured...
        self.deformedNodes = self.u[:,-1].reshape( (-1, self.dim) ) + self.mdlNodes
        print( 'Deformed nodes were generated' )
        

        
        
    def _generateDeformationVectors( self ) :
        
        # Get the last deformation from the file
        self.deformVectors = self.u[:,-1].reshape( self.u.shape[0]/self.dim, self.dim )
        
        
        
            
    def deformationVectors( self, nodeNumbers=None ) :
        ''' remember, the deformation is given in m
        '''
        
        if nodeNumbers == None :
            return self.deformVectors
        
        defVectsSelection = []
        
        for i in nodeNumbers:
            defVectsSelection.append( self.deformVectors[i,:] )
            
        return np.array( defVectsSelection )
        
        
    def deformedModelNodes(self):
        return self.deformedNodes
