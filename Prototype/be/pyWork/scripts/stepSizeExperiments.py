#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import numericalBreastPhantom as numPhantom
import numpy as np
import modelDeformationHandler as mdh
import commandExecution as cmdEx
import matplotlib.pyplot as plt
import matplotlib as mpl
from conversions import numpyArrayToStr
import sys
from xmlModelReader import xmlModelReader
from modelDeformationHandler import modelDeformationHandler
         



class stepSizeExperiments (  ):
    
    def __init__(self, gpu=True, totalTimeIn = 1.0, loadShape='RAMP', configID = '00_step', 
                 maxIterations=None, startIterations=None, iterationIncrement=None, 
                 meshlabSript    = 'Q:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' ):
        
        tetgenVol       = 75
        
        # directory
        baseExperimentDir = 'Q:/philipsBreastProneSupine/referenceState/'
        self.experimentDir     = baseExperimentDir + configID + '/'
        self.plotDir           = self.experimentDir + 'plots/'
        
        self.Eiter   = [] # use this to track iteration numbers for plotting quantities below
        self.Ekin    = []
        self.Estrain = []
        
        # phantom properties
        imageEdgeLength = 400
        
        # general volume mesh properties
        tetgenQ         = 1.5
        
        # system parameters
        timeStep        = 2e-5
        totalTime       = totalTimeIn
        damping         = 50
        
        
        
        ########
        # skin
        simSkin           = False
    
    
        ############
        # material 
        matModelFat       = 'NH'
        matModelSkin      = 'NH'
        matParamsFat      = [  100, 50000 ]
        matParamsSkin     = [ 1000, 50000 ]
        
        ################
        # Visco elastic 
        # for fat only, change material to NHV!
        viscoParams       = []
        viscoNumIsoParams = 0
        viscoNumVolParams = 0
        
        
        #################################
        # Cylindrical base experiments
        cylindricalBase   = False
        
        
        
            #######################################
        # print a summary of the configuration
        print( 'Summary of the simulation:' )
        print( ' Directories...' )
        print( '  -> experimentDir: ' + self.experimentDir ) 
        print( '  -> plotDir: '       + self.plotDir       )
        
        print( ' Phantom properties...' )
        print( '  -> imageEdgeLength: ' + str( imageEdgeLength ) )
        print( '  -> meshlabSript: '    + meshlabSript           )
        print( '  -> tetgenQ: '         + str( tetgenQ )         )
        print( '  -> tetgenVol: '       + str( tetgenVol )       )
        
        print( ' Material parameters...' )
        print( '  -> simSkin: '           + str( simSkin )           )
        print( '  -> matModelFat: '       + str( matModelFat )       )
        print( '  -> matModelSkin: '      + str( matModelSkin )      )
        print( '  -> matParamsFat: '      + str( matParamsFat )      )
        print( '  -> matParamsSkin: '     + str( matParamsSkin )     )
        print( '  -> viscoParams: '       + str( viscoParams )       )
        print( '  -> viscoNumIsoParams: ' + str( viscoNumIsoParams ) )
        print( '  -> viscoNumVolParams: ' + str( viscoNumVolParams ) )
        
        print( ' Simulation system properties...' )
        print( '  -> timeStep: '      + str( timeStep )  )
        print( '  -> totalTime: '     + str( totalTime ) )
        print( '  -> damping: '       + str( damping )   )
        
        
        ######################################
        # 
        # Start with the work
        #    
        if not os.path.exists(self.experimentDir):
            os.mkdir( self.experimentDir ) 
            print( 'directory created...' )
    
        
        if not os.path.exists( self.plotDir ):
            os.mkdir( self.plotDir )
            print( 'directory created...' )
        
        
        #
        # To estimate the influence of the reference state:
        # 1) run the prone simulation
        # 2) Build the model from the loaded state
        # 3) inverse gravity and double it
        #
        print( 'Generating phantom' )
        phantom = numPhantom.numericalBreastPhantom( self.experimentDir, imageEdgeLength, 
                                                     meshlabSript, tetgenVol, tetgenQ, 
                                                     timeStep, totalTime, damping, 
                                                     fatMaterialType  = matModelFat, fatMaterialParams  = matParamsFat, 
                                                     fatViscoNumIsoTerms = viscoNumIsoParams, fatViscoNumVolTerms=viscoNumVolParams,fatViscoParams=viscoParams,  
                                                     skinMaterialType = matModelSkin, skinMaterialParams = matParamsSkin, 
                                                     cylindricalBase = cylindricalBase )
        
        # track over iterations
        aXmlGenP1G   = []
        aDeformP1G   = []
        
        timeSteps = []
        
        # below 50000 the incement is 500
        if (startIterations != None) and (maxIterations != None) and (iterationIncrement != None):
            print('Setting iteration numbers explicitly')
            numIts = range( startIterations, maxIterations+1, iterationIncrement )
            
        else:
            print('Using standard iteration numbers')
            numIts = range( startIterations, 50001, 500 )
            numIts.extend( range( 51000, 100001, 1000 ) )
        
        performedIts = []
        
        for numIt in numIts:
            
            timeStep = totalTime / numIt
            print( 'Starting simulation timeStep: %.2e'     % timeStep )
            print( 'Starting simulation num iterations: %i' % numIt    )
            
            
            p1G = '_prone1G' + str( '_it%06i' % numIt )
            
            deformFileName    = self.experimentDir + 'U.txt'
            deformFileNameP1G = self.experimentDir + 'U' + p1G + '.txt'
            
            gravProne = [0., 0., 1.]
        
            #
            # 1) prone simulation
            #
            phantom.timeStep = timeStep
            xmlGenP1G = phantom.generateXMLmodel( gravProne, gravityMagnitude=10, gravityLoadShape=loadShape, 
                                                  fileIdentifier=p1G, skin=simSkin, outputFrequency=numIt )
            aXmlGenP1G.append( xmlGenP1G )
            

            
        

        
    
    
    def _parseFileForKineticAndStrainEnergy(self, fileName, iterationNum):
        
        f = file(fileName)
        lines = f.readlines()
        
        self.Eiter.append(iterationNum)
        
        for l in lines: 
            if l.count('E kinetic:')==1:
                self.Eiter.append( float( l.split()[-1] ) ) 
        
        if l.count('E kinetic:')==1:
                self.Estrain.append( float( l.split()[-1] ) ) 
        


    def plotAndSaveResults( self ):
        # Use latex plotting, because it looks so great
        plt.rc( 'text', usetex=True )
        plt.rcParams['font.size']=16
        
        xLabel        = '$N \mathrm{[\cdot 10^{3}]}$'
        yLabel        = '$\overline{\| u \|} \; \mathrm{[mm]}$'
        p1meanLabel   = '$\overline{\|u_{p}\|}$'
    
        # plot 
        plt.hold( True )
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(color = 'gray', linestyle='-')
        
        plt.plot( self.iterationNumbers/1000., self.meanDisplacements*1000.,   'b-+', label = p1meanLabel )
        
        #plt.ylim(ymin=0)
        #plt.legend(loc='upper left')
        
        plt.hold( False )
        plt.show()
        plt.savefig( self.plotDir + 'meanDeformOverNumIterations.pdf' )
        plt.savefig( self.plotDir + 'meanDeformOverNumIterations.png', dpi = 300 )
        
        print('Done')



    
if __name__ == '__main__':
    
    s=stepSizeExperiments()
    
    pass
    
    
    