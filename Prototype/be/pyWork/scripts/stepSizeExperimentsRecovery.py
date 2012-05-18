#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import numericalBreastPhantom as numPhantom
import numpy as np
import modelDeformationHandler as mdh
import commandExecution as cmdEx
import matplotlib.pyplot as plt
from conversions import numpyArrayToStr
import sys
from xmlModelReader import xmlModelReader
from modelDeformationHandler import modelDeformationHandler
from glob import glob
import xmlModelReader as xRead




class stepSizeExperimentsRecovery ():
    
    def __init__(self, simDir ):
        
        self.experimentDir     = simDir
        self.plotDir           = self.experimentDir + 'plots/'
        
        
        self.Ekin    = []
        self.Estrain = []
        
        
        #######################################
        # print a summary of the configuration
        print( 'Summary of the simulation:' )
        print( ' Directories...' )
        print( '  -> experimentDir: ' + self.experimentDir ) 
        print( '  -> plotDir: '       + self.plotDir       )
        
        
        ######################################
        # 
        # Start with the work
        #        
        if not os.path.exists( self.plotDir ):
            os.mkdir( self.plotDir )
            print( 'directory created...' )
        
        
        #
        # To estimate the influence of the reference state:
        # 1) run the prone simulation
        # 2) Build the model from the loaded state
        # 3) inverse gravity and double it
        #
        print( 'Recovering phantom data' )
        
        # track over iterations
        aXmlGenP1G   = []
        aDeformP1G   = []
        
        timeSteps = []
        
        #
        # glob from the experiment dir. The vtk files indicate which simulations completed.
        #
        vtkFiles = glob( self.experimentDir + '*_prone1G*.vtk' )
        numIts = []
        
        for vtkF in vtkFiles :
            numIts.append( int( vtkF.split('it')[1].split('.vtk')[0] ) )

        numIts.sort()
        
        for numIt in numIts:
            timeStep = 1.0 / numIt
            timeSteps.append( timeStep )

            
            # reconstruct name....
            p1G = '_prone1G' + str( '_it%06i' % numIt )
            
            deformFileNameP1G = self.experimentDir + 'U' + p1G + '.txt'
            
            print('Model file: \t' + simDir + 'modelFat' + p1G + '.xml')
            xmlGenP1G = xRead.xmlModelReader( simDir + 'modelFat' + p1G + '.xml' ) 
            
            #
            # 1) prone simulation
            #
            aXmlGenP1G.append( xmlGenP1G )
            
            deformP1G = mdh.modelDeformationHandler( aXmlGenP1G[-1], deformFileNameP1G.split('/')[-1] )
            aDeformP1G.append( deformP1G )
            
        
        #
        # Now plot the results
        #
        timeSteps = np.array( timeSteps )
            
            
        #phiMax = 40
        #modDir = 'W:/philipsBreastProneSupine/referenceState/02/'
        
        p1U          = []
        p1meanDisp   = []
    
        
        
        for dh in aDeformP1G :
            #
            # Read the model and the deformations accordingly
            #
            p1U.append( dh )
            
            #
            # Mean deformations
            #
            p1meanDisp.append( np.mean( np.sqrt( p1U[-1].deformVectors[:,0]**2 + 
                                                 p1U[-1].deformVectors[:,1]**2 + 
                                                 p1U[-1].deformVectors[:,2]**2 ) ) )
            
            
        p1meanDisp   = np.array( p1meanDisp )

        # Remember for later use...
        self.xmlGens = aXmlGenP1G
        self.deformHandlers = p1U
        self.meanDisplacements = p1meanDisp
        self.iterationNumbers = np.array( numIts )
        
    
    
    
    def parseLogFilesForKineticAndStrainEnergy( self, logFileBaseName='log', useNumIts = True ):
        
        logFiles = []
        
        if useNumIts :
            for it in self.iterationNumbers :
                logFiles.extend( glob( self.experimentDir + logFileBaseName + str('%06i' %it)+ '.*' ) )
        else:
            logFiles = glob( self.experimentDir + logFileBaseName + '*' )
        
        for logFile in logFiles :
        
            f = file( logFile )
            lines = f.readlines()
            
            for l in lines: 
                if l.count('E kinetic:')==1:
                    self.Ekin.append( float( l.split()[-1] ) ) 
            
                if l.count('E strain:')==1:
                    self.Estrain.append( float( l.split()[-1] ) ) 

        # Convert to numpy array...
        self.Ekin    = np.array( self.Ekin    )
        self.Estrain = np.array( self.Estrain )



    
    def plotAndSaveResults( self ):
        # Use latex plotting, because it looks so great
        plt.rc( 'text', usetex=True )
        plt.rcParams['font.size']=16
        
        xLabel        = '$N \mathrm{[\cdot 10^{3}]}$'
        yLabel        = '$\overline{\| u \|} \; \mathrm{[mm]}$'
        y2Label       = '$E_\mathrm{kin} / E_\mathrm{strain}$'
        meanLabel   = '$\overline{\|u_{p}\|}$'
    
        # plot 
        fig = plt.figure()
        ax  = fig.gca()
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.grid(color = 'gray', linestyle='-')
        
        ax.plot( self.iterationNumbers/1000., self.meanDisplacements*1000.,   'b-+', label = meanLabel )
        ax.set_ylim(bottom=0)
        
        plt.show()
        plt.savefig( self.plotDir + 'meanDeformOverNumIterations.pdf' )
        plt.savefig( self.plotDir + 'meanDeformOverNumIterations.png', dpi = 300 )
        
        #
        # plot combination of mean displacement and energy fraction (kinetic over strain)
        #
        
        if isinstance(self.Ekin, np.ndarray) :
            eLabel= '$E_\mathrm{kin} / E_\mathrm{strain}$'
            plt.hold( True )

            fig = plt.figure()
            ax1 = fig.gca()
            ax1.plot( self.iterationNumbers/1000., self.meanDisplacements*1000., 'b-+', label = meanLabel )
            ax2 = ax1.twinx()
            ax2.plot( self.iterationNumbers/1000., self.Ekin / self.Estrain, 'r-+', label = eLabel )
            ax1.set_xlabel( xLabel  )
            ax1.set_ylabel( yLabel  )
            ax2.set_ylabel( y2Label )
            
            ax1.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            
            #ax1.legend(loc = 'upper left')
            #ax2.legend(loc = 'upper right')
            ax1.grid(color = 'gray', linestyle='-')
            plt.legend( (ax1.get_lines(), ax2.get_lines ()), (meanLabel, eLabel), loc = 'upper left' )
            plt.hold( False )
            plt.show
            plt.savefig( self.plotDir + 'combinedMeanDispEnergy.pdf' )
            plt.savefig( self.plotDir + 'combinedMeanDispEnergy.png', dpi = 300 )
            
        
        print('Done')



    
if __name__ == '__main__':
    
    simDir = 'W:/philipsBreastProneSupine/referenceState/00_step_gpu2/'
    simDir = 'W:/philipsBreastProneSupine/referenceState/00_step_gpu1/'
    simDir = 'W:/philipsBreastProneSupine/referenceState/00_step_totalTime02/'
    s=stepSizeExperimentsRecovery(simDir) 
    s.parseLogFilesForKineticAndStrainEnergy()   
    s.plotAndSaveResults()
    pass
    
    
    