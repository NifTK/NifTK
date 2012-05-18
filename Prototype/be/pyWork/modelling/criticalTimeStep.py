#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import numpy as np
import xmlModelReader as xReader
import elasticParamConversion as eCon
import os



class criticalTimeStep():
    def __init__( self, xReaderIn=None, xmlFileName='' ):
        
        if not isinstance( xReaderIn, xReader.xmlModelReader ):
            
            if os.path.exists(xmlFileName) :
                xReaderIn = xReader.xmlModelReader( xmlFileName )
            else:
                print 'This class needs an xmlModelReader or a valid xmlFileName'
                return
            
        self.xReader = xReaderIn
        self.density = float( self.xReader.modelObject.SystemParams.Density )
        self.numMaterials   = len( self.xReader.materials )
        
        self._getSmallestEdgeLength()
        self._calculatePropagationSpeedAndCriticalTimeStep()
    
    
    
    
    def _getSmallestEdgeLength( self ):
        
        # Calculate all lengths beforehand
        xPts = self.xReader.nodes[:,0]
        yPts = self.xReader.nodes[:,1]
        zPts = self.xReader.nodes[:,2]
        
        # big Matrix with coordinate positions. 
        M = np.array( ( xPts[self.xReader.elements[:,0]], yPts[self.xReader.elements[:,0]], zPts[self.xReader.elements[:,0]], 
                        xPts[self.xReader.elements[:,1]], yPts[self.xReader.elements[:,1]], zPts[self.xReader.elements[:,1]],
                        xPts[self.xReader.elements[:,2]], yPts[self.xReader.elements[:,2]], zPts[self.xReader.elements[:,2]],
                        xPts[self.xReader.elements[:,3]], yPts[self.xReader.elements[:,3]], zPts[self.xReader.elements[:,3]] ) ).T
        
        # Vectors within each element...
        a = M[:,0:3] - M[:,3: 6]
        b = M[:,0:3] - M[:,6: 9]
        c = M[:,0:3] - M[:,9:12]
        d = M[:,3:6] - M[:,6: 9]
        e = M[:,3:6] - M[:,9:12]
        f = M[:,6:9] - M[:,9:12]
        
        #
        # Calculate lengths
        #
        self.Lengths = np.array( np.sqrt( ( np.sum(a * a, 1 ), 
                                            np.sum(b * b, 1 ),
                                            np.sum(c * c, 1 ),
                                            np.sum(d * d, 1 ),
                                            np.sum(e * e, 1 ),
                                            np.sum(f * f, 1 ) ) ) ).T

        self.minEdgeLengths = []
        
        for i in range( self.numMaterials ) :
            matElements = self.xReader.materials[i]['Elements']
            self.minEdgeLengths.append( np.min( self.Lengths[ matElements ] ) )
            print('Smallest edge length material set %i: %.12e' %(i,self.minEdgeLengths[i]) )
            
        
        
        
    def _calculatePropagationSpeedAndCriticalTimeStep( self ):
        
        #
        # Implementaiton based on the niftysim synopsis by Zeike Taylor
        #
        self.propagationSpeeds = []
        self.criticalTimeStep  = []
        
        for i in range( self.numMaterials ) :
            # Neo-Hookean and its visco-elastic counter part
            if (self.xReader.materials[i]['Type'] == 'NH') or  (self.xReader.materials[i]['Type'] == 'NHV') :
                #
                # Assumption: For now the linear elastic material will be approximated by using material parameters 
                # equivalently
                # TODO: Check correctness of this assumption
                #
                shearModulus = float( self.xReader.materials[i]['ElasticParams'][0] ) # lameMu
                bulkModulus  = float( self.xReader.materials[i]['ElasticParams'][1] ) # lameLambda
                
                E = eCon.YoungsModulus( shearModulus, bulkModulus ) # Young's modulus
                P = eCon.PoissonsRatio( shearModulus, bulkModulus ) # Poisson's ratio


                
            # Arruda Boyce
            elif self.xReader.materials[i]['Type'] == 'AB' :
                shearModulus = float( self.xReader.materials[i]['ElasticParams'][0] ) # lameMu
                bulkModulus  = float( self.xReader.materials[i]['ElasticParams'][2] ) # lameLambda
            
                E = eCon.YoungsModulus( shearModulus, bulkModulus ) # Young's modulus
                P = eCon.PoissonsRatio( shearModulus, bulkModulus ) # Poisson's ratio
            elif self.xReader.materials[i]['Type'] == 'LE' :
                E = float( self.xReader.materials[i]['ElasticParams'][0] ) # Young's modulus
                P = float( self.xReader.materials[i]['ElasticParams'][1] ) # Young's modulus
                
            else :
                print('Sorry, no time step estimation for this material: ' + self.xReader.materials[i]['Type'] )
                continue
                
        
            
            self.propagationSpeeds.append( np.sqrt( E * (1.0-P) / ( self.density * (1.0+P) * (1-2.0*P) ) ) )
            self.criticalTimeStep.append( self.minEdgeLengths[i] / self.propagationSpeeds[-1] )
    
    
    
    
if __name__ == '__main__' :
    
    fileName = 'W:/philipsBreastProneSupine/referenceState/01_load/modelFat_prone1G_phi00.xml'
    #R = xReader.xmlModelReader( fileName )
    cts = criticalTimeStep( xmlFileName=fileName )
    print( 'Found the following critical time steps:' )
    print( cts.criticalTimeStep )
    print( 'Done' )
    
    