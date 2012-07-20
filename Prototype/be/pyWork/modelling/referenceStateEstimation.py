#! /usr/bin/env python 
# -*- coding: utf-8 -*-




import xmlModelReader as xReader
import xmlModelGenerator as xGen
import numpy as np
import os
from runSimulation import runNiftySim
import modelDeformationHandler as deformHandler
import mayaviPlottingWrap as mpw




class referenceStateEstimation:
    ''' @summary: Purpose is to estimate the (stress free) reference state from a loaded configuration. 
                  Following the approach that Tim Carter suggested. 
    '''
    
    
    
    def __init__( self, xmlFileName, gravityDirection, gravityMagnitude ):
        ''' @param xmlFileName: String describing the xml model file which describes the loaded configuration. 
                                Note: Gravity constraint needs to be specified in this model already as this 
                                      field will only be replaced. 
            @param gravityDirection: Direction into which the reference state is estimated.  
        '''
        
        self.xmlModelLoaded      = xReader.xmlModelReader( xmlFileName ) 
        self.gravityDirection    = gravityDirection
        self.gravityMagnitude    = gravityMagnitude
        self.simulationFiles     = [] # The first entry is the first guess for the reference state
        self.xmlGenerators       = []
        self.deformationHandlers = []
        self.referenceSateNodes  = []
        self.meanNodalDistance   = []
        self.maxNodalDistance    = []
        self.updateFactors       = []
        self.updateFactor        = 0.7
        self.loadShape           = 'POLY345FLAT'
        
        self.maxErrorAim         = 0.5 # given in mm
        self.maxIterations       = 20 

    
    
    def estimateRefenreceState( self ):
        
        
        # Build a new model with exchanged gravity settings
        self.xmlGenerators.append( xGen.xmlModelGenrator( self.xmlModelLoaded.nodes, 
                                                          self.xmlModelLoaded.elements, 
                                                          self.xmlModelLoaded.modelObject.Elements.Type ) )
       
        # Set all the other model parameters
        self._setStandardModelParameters( self.xmlGenerators[-1] )
        
        # Set the gravity loading for the first guess
        self.xmlGenerators[-1].setGravityConstraint( self.gravityDirection, 
                                                     self.gravityMagnitude, 
                                                     self.xmlGenerators[-1].allNodesArray, 
                                                     self.loadShape )
    
        
        self.simulationFiles.append( os.path.split( self.xmlModelLoaded.xmlFileName )[0] + '/' + 
                                     os.path.splitext( os.path.split( self.xmlModelLoaded.xmlFileName )[1] )[0] +  '_sim-001.xml' )
           
        self.xmlGenerators[-1].writeXML( self.simulationFiles[-1] )
        
        retVal = runNiftySim( os.path.split( self.xmlGenerators[-1].xmlFileName )[-1], os.path.split( self.xmlGenerators[-1].xmlFileName )[0] )
        
        if retVal != 0 :
            return
        
        # Get the first guess for the unloaded configuration        
        self.deformationHandlers.append( deformHandler.modelDeformationHandler( self.xmlGenerators[-1] ) )
        self.referenceSateNodes.append( self.deformationHandlers[-1].deformedNodes )
        
        
        #
        # now do the FROWARD simulations => prone direction
        #   
        its = range( self.maxIterations )
        
        lastImprovedGuess = its[0]
        
        for i in its:
            self.xmlGenerators.append( xGen.xmlModelGenrator( self.referenceSateNodes[-1], 
                                                              self.xmlModelLoaded.elements,
                                                              self.xmlModelLoaded.modelObject.Elements.Type ) )
            
            self._setStandardModelParameters( self.xmlGenerators[ -1 ] )
            
            self.xmlGenerators[-1].setGravityConstraint( -self.gravityDirection, 
                                                         self.gravityMagnitude, 
                                                         self.xmlGenerators[-1].allNodesArray, 
                                                         self.loadShape )
            
            self.simulationFiles.append( os.path.split( self.xmlModelLoaded.xmlFileName )[0] 
                                         + '/' 
                                         + os.path.splitext( os.path.split( self.xmlModelLoaded.xmlFileName )[1] )[0] 
                                         +  '_sim+' + str('%03i' % i) + '.xml' )   
            
            self.xmlGenerators[ -1 ].writeXML( self.simulationFiles[ -1 ] )
            
            retVal = runNiftySim( os.path.split( self.xmlGenerators[-1].xmlFileName )[-1], 
                                  os.path.split( self.xmlGenerators[-1].xmlFileName )[ 0] )
            
            if retVal != 0 :
                return
            
            # Compare the simulation result with the original (loaded) configuration
            self.deformationHandlers.append( deformHandler.modelDeformationHandler( self.xmlGenerators[-1] ) )
            
            # generate the difference vector of the node positions
            nodalDistance = self.deformationHandlers[ -1 ].deformedNodes - self.xmlModelLoaded.nodes
            
            
            # Calculate the mean difference
            meanNodalDistance = np.mean( np.sqrt( nodalDistance[:,0] **2 + 
                                                  nodalDistance[:,1] **2 + 
                                                  nodalDistance[:,2] **2 ) )
            
            maxNodalDistance = np.max(np.sqrt( nodalDistance[:,0] **2 + 
                                               nodalDistance[:,1] **2 + 
                                               nodalDistance[:,2] **2 ) )
            
            print( 'Iteration %i done... accuracy achieved' % i )
            print( '   mean: %.6f mm' % (meanNodalDistance * 1000.) )
            print( '   max:  %.6f mm' % (maxNodalDistance  * 1000.) )
            
            self.meanNodalDistance.append( meanNodalDistance )
            self.maxNodalDistance.append( maxNodalDistance )
            
            
            #
            # Reached optimisation accuracy?
            #
            if self.maxNodalDistance[-1] * 1000 < self.maxErrorAim:
                print( 'Requested nodal position accuracy reached in %i iterations:' % (i+1) )
                break 
            
            
            #
            # Accept the optimisation step?
            #
            if i > 0:
                # did the measure improve?
                if self.maxNodalDistance[-1] < self.maxNodalDistance[ lastImprovedGuess ]:
                    # improved...
                    lastImprovedGuess = i

                    # increase update factor
                    self.updateFactor = np.min( (self.updateFactor * 1.05, 1.0) )
                    self.updateFactors.append( self.updateFactor ) 
                    print('Step successful, current updateFactor to %f' % self.updateFactor )
                    
                    # Update reference state estimate
                    self.referenceSateNodes.append( self.referenceSateNodes[-1] - self.updateFactor * nodalDistance )
                    
                else:
                    # did not improve
                    self.updateFactor = self.updateFactor * 0.7
                    self.updateFactors.append( self.updateFactor ) 
                    print('Step not successful, decreased updateFactor to %f' % self.updateFactor )
                    
                    # get the values from the last successful step
                    nodalDistance = self.deformationHandlers[ lastImprovedGuess + 1 ].deformedNodes - self.xmlModelLoaded.nodes
                    self.referenceSateNodes.append( self.referenceSateNodes[lastImprovedGuess] - self.updateFactor * nodalDistance )
 
            
            else :
                # Update the guessed reference state by the distances evaluated
                self.updateFactors.append(self.updateFactor) 
                self.referenceSateNodes.append( self.referenceSateNodes[-1] - self.updateFactor * nodalDistance )
                lastImprovedGuess = i
            
        
        self.maxNodalDistance = np.array( self.maxNodalDistance )
        bestIteration =  np.nonzero(self.maxNodalDistance == np.min( self.maxNodalDistance ) )[0]    
        
        self.optimalReferenceSateNodes = self.deformationHandlers[bestIteration+1].mdlNodes
        
    
    
    
    def _setStandardModelParameters( self, xmlGenerator ):
        ''' Includes
             - fix constraints
             - material sets
             - output
             - system parameters
        '''
        
        #
        # Set fix constraints
        #
        for i in range( len( self.xmlModelLoaded.fixConstraints ) ):
            xmlGenerator.setFixConstraint( self.xmlModelLoaded.fixConstraints[i]['Nodes'], 
                                            self.xmlModelLoaded.fixConstraints[i]['DOF'] )
            
        #
        # Set material sets
        #
        for i in range( len( self.xmlModelLoaded.materials ) ):
            
            materialType = self.xmlModelLoaded.materials[i]['Type']
            
            try:
                materialName = self.xmlModelLoaded.modelObject.ElementSet[i]['Material']['Name']
            except:
                materialName = 'None'
            
            elasticParams    = self.xmlModelLoaded.materials[ i ][ 'ElasticParams' ]
            materialElements = self.xmlModelLoaded.materials[ i ][ 'Elements'      ]
            
            try:
                numIsoTerms = self.xmlModelLoaded.materials[ i ][ 'NumIsoTerms' ]
                numVolTerms = self.xmlModelLoaded.materials[ i ][ 'NumVolTerms' ]
                viscoParams = self.xmlModelLoaded.materials[ i ][ 'ViscoParams' ]
            
            except:
                numIsoTerms = 0
                numVolTerms = 0
                viscoParams = []
            
            xmlGenerator.setMaterialElementSet( materialType,
                                                materialName, 
                                                elasticParams, 
                                                materialElements, 
                                                numIsoTerms, 
                                                numVolTerms, 
                                                viscoParams )
            
        #
        # Set system parameters
        #
        xmlGenerator.setSystemParameters( self.xmlModelLoaded.systemParams[ 'TimeStep'     ], 
                                          self.xmlModelLoaded.systemParams[ 'TotalTime'    ], 
                                          self.xmlModelLoaded.systemParams[ 'DampingCoeff' ], 
                                          self.xmlModelLoaded.systemParams[ 'HGKappa'      ], 
                                          self.xmlModelLoaded.systemParams[ 'Density'      ] )
        

        #
        # Set output
        #
        xmlGenerator.setOutput( self.xmlModelLoaded.output['Freq'], self.xmlModelLoaded.output['Variables'] )
        
        
    
        #
        # Set shell elements
        #
        if ( len ( self.xmlModelLoaded.shellElements ) != 0 ) and ( len( self.xmlModelLoaded.shellElementSet ) != 0 ):
            
            xmlGenerator.setShellElements( self.xmlModelLoaded.shellElements['Type'], 
                                           self.xmlModelLoaded.shellElements['Elements'] )
            
            for i in range( len( self.xmlModelLoaded.shellElementSet ) ):
                
                self.xmlModelLoaded.shellElementSet[i]
                
                if len( self.xmlModelLoaded.shellElementSet) == 1: # only one membrane material specified
                    xmlGenerator.setShellElementSet( 0, 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialType'      ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialParams'    ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialDensity'   ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialThickness' ])
                    
                else:
                    xmlGenerator.setShellElementSet( self.xmlModelLoaded.shellElementSet[i][ 'Elements'          ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialType'      ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialParams'    ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialDensity'   ], 
                                                     self.xmlModelLoaded.shellElementSet[i][ 'MaterialThickness' ])
                    

    
    
        
    
    
    
if __name__ == '__main__' :
    
    
    # loadedConfiguration the gravitational direction in the original model was [ 0 0 1 ]
    xmlModelFileName = 'W:/philipsBreastProneSupine/referenceState/recoverReal/T4ANP_membrane/model_T4ANPm.xml'
    gravityDirection = np.array( ( 0.0, 1.0, 0. ) ) 
    gravityMagnitude = 10.0
    
    estimator = referenceStateEstimation( xmlModelFileName, gravityDirection, gravityMagnitude )
    estimator.estimateRefenreceState()
    
    
    
    
    
    