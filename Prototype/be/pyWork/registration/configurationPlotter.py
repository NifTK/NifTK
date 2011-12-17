#! /usr/bin/env python 
# -*- coding: utf-8 -*-

#from registrationTask import RegistrationTask
from matplotlib import rc
from pylab import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show, xlim, ylim
#from itkppRegistrationTask import itkppRegistrationTask
from findExecutable import findExecutable
from matplotlib import *
import os, platform, evaluationListAnalyser, glob
from glob import glob0
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab

class configurationPlotter :
    def __init__( self, baseDir, directories, parameterColNum = 0, summaryFileName = 'params.txt', afterInitialisationVals = None ) :
        ''' afterInitialisationVals is expected to be an array with 4 elements
            afterInitialisationVals=[ meanBreast, percentlileBreast, meanLesion, percentileLesion ]
        '''
        # 1) Read breast and lesion files
        
        self.baseDir     = baseDir
        self.directories = directories
        self.fullDirs    = []
        self.paramColNum = parameterColNum
        
        self.afterInitialsiationVals = afterInitialisationVals
        
        self.summaryFileName = summaryFileName
        
        for dir in self.directories :
            self.fullDirs.append( os.path.realpath( os.path.join(self.baseDir, dir) ) )
            
            if platform.system() == 'Windows' :
                self.fullDirs[-1] = self.fullDirs[-1].replace( '\\', os.altsep )
            
        self.breastFiles = []
        self.lesionFiles = []
        
        # Quantities extracted from files:
        self.xVals                   = []
        self.breastMeanVals          = []
        self.lesionMeanVals          = []
        self.breastPercentileVals    = []
        self.lesionPercentileVals    = []
        self.breastInitialMean       = []
        self.lesionInitialMean       = []
        self.breastInitialPercentile = []
        self.lesionInitialPercentile = []
        
        self._getBreastAndLesionFiles()
        self._interpretParameterFile()


        

    def plotTRE( self, xLabel = '', yMax = None, plotLegend = True, log = False, 
                 plotBreast = True, plotLesion = True, plotBreastInit = True, plotLesionInit = True, plotLatex = True, title = '' ) :
        ''' xLabel is expected to be a string something like '$\frac{s}{\mu}$'
        '''
        print('plotting...')
        
        if plotLatex == True:
            # Use latex plotting, because it looks so great
            rc( 'text', usetex=True )
        
            labelMeanTREBreast    = '$\overline{\mathrm{TRE}}_{B}$'
            labelPercTREBreast    = '$\mathrm{TRE}_{B}^{\%}$'
            labelMeanTRELesion    = '$\overline{\mathrm{TRE}}_{L}$'
            labelPercTRELesion    = '$\mathrm{TRE}_{L}^{\%}$'
            labelIniMeanTREBreast = '$\overline{\mathrm{TRE}}_{B, \mathrm{ini.}}$'
            labelIniPercTREBreast = '$\mathrm{TRE}_{B, \mathrm{ini.}}^{\%}$'
            labelIniMeanTRELesion = '$\overline{\mathrm{TRE}}_{L, \mathrm{ini.}}$'
            labelIniPercTRELesion = '$\mathrm{TRE}_{L, \mathrm{ini.}}^{\%}$'
            labelTRE              = '$\mathrm{TRE} \;\;[ \mathrm{mm} ]$'
    
        else:
            labelMeanTREBreast    = 'mean TRE Breast'
            labelPercTREBreast    = 'perc. TRE Breast'
            labelMeanTRELesion    = 'mean TRE Lesion'
            labelPercTRELesion    = 'perc TRE Lesion'
            labelIniMeanTREBreast = 'ini. mean TRE Breast'
            labelIniPercTREBreast = 'ini. perc TRE Breast'
            labelIniMeanTRELesion = 'ini. mean TRE Lesion'
            labelIniPercTRELesion = 'ini. perc TRE Lesion'
            labelTRE              = 'TRE [mm]'
        
        
        plt.hold( True )
        if plotBreast :
            plt.plot( self.xVals, self.breastMeanVals,       'r-+', label = labelMeanTREBreast )
            plt.plot( self.xVals, self.breastPercentileVals, 'r-x', label = labelPercTREBreast ) 
        
        if plotLesion :
            plt.plot( self.xVals, self.lesionMeanVals,       'b-+', label = labelMeanTRELesion )
            plt.plot( self.xVals, self.lesionPercentileVals, 'b-x', label = labelPercTRELesion )
        
        if len( xLabel ) > 0 :
            xlabel( xLabel )    
        
        if self.afterInitialsiationVals != None :
            if plotBreastInit :
                plt.plot( self.xVals, [ self.afterInitialsiationVals[0] ] * len(self.xVals), 'r--', label = labelIniMeanTREBreast )
                plt.plot( self.xVals, [ self.afterInitialsiationVals[1] ] * len(self.xVals), 'r:',  label = labelIniPercTREBreast )
            
            if plotLesionInit :
                plt.plot( self.xVals, [ self.afterInitialsiationVals[2] ] * len(self.xVals), 'b--', label = labelIniMeanTREBreast )
                plt.plot( self.xVals, [ self.afterInitialsiationVals[3] ] * len(self.xVals), 'b:',  label = labelIniPercTRELesion )
        
        grid(color = 'gray', linestyle='-')

        # Configuring y-axis
        ylabel( labelTRE )
        ylim( ymin = 0 )
        
        if not yMax == None:
            ylim( ymax = yMax )
        
        if plotLegend :
            plt.legend()
        
        if log :
            plt.xscale( 'log' )
        
        if not title == '' :
            plt.title( title )
        
        plt.hold( False )
        plt.show()
        
        print( 'plotting done...' )
        return plt.gcf()
    


        
    def _interpretParameterFile( self ) :
        
        cfgSummaryFile = glob.glob(os.path.join( self.baseDir, self.summaryFileName ) )[0]
        
        f = file( cfgSummaryFile, 'r' )
            
        lines = f.readlines()
        
        for i in range( 0,len(self.directories) ):
            
            cfgID = self.directories[ i ]
            
            # Parse the parameter file for x values (plotting)
            for line in lines :
                if line.startswith( cfgID ) :
                    print('Found' + cfgID)
                
                cols = line.split()
                # remove white spaces
                cols[0] = cols[0].replace(' ','')
                
                # Found the configuration ID_
                if cols[0] == cfgID:
                    self.xVals.append( float( cols[ self.paramColNum ] ) )
                    print('- Found: ' + cfgID + ' param: ' + str( self.xVals[-1] ) )
            
            # Read the evaluation file
            evalFileBreast  = file( self.breastFiles[ i ], 'r' )
            evalFileLesion  = file( self.lesionFiles[ i ], 'r' )
            evalBreastLines = evalFileBreast.readlines()
            evalLesionLines = evalFileLesion.readlines()

            # Check if 60 evaluations are listed in this file...
            if float( evalBreastLines[ 66 ]) != 60:
                print( "WARNING!!! DID NOT FIND 60 EVALUATIONS!!!" )
            

            self.breastMeanVals.append(       float( evalBreastLines[ 80 ] ) )
            self.lesionMeanVals.append(       float( evalLesionLines[ 80 ] ) )
            self.breastPercentileVals.append( float( evalBreastLines[ 82 ] ) )
            self.lesionPercentileVals.append( float( evalLesionLines[ 82 ] ) )
            
            self.breastInitialMean.append(       float( evalBreastLines[ 72 ] ) )
            self.breastInitialPercentile.append( float( evalBreastLines[ 74 ] ) )
            self.lesionInitialMean.append(       float( evalLesionLines[ 72 ] ) )
            self.lesionInitialPercentile.append( float( evalLesionLines[ 74 ] ) )
            
            evalFileBreast.close()
            evalFileLesion.close()
            
        f.close()





    def _getBreastAndLesionFiles( self ) :
        patB = 'eval*breast*'
        patL = 'eval*lesion*'
        
        for dir in self.fullDirs :
            self.breastFiles.append( glob.glob( os.path.join( dir, patB ) )[0] )
            self.lesionFiles.append( glob.glob( os.path.join( dir, patL ) )[0] )
            
            if platform.system() == 'Windows' :
                self.breastFiles[-1] = self.breastFiles[-1].replace( '\\', os.altsep )
                self.lesionFiles[-1] = self.lesionFiles[-1].replace( '\\', os.altsep )
            
            


if __name__ == '__main__' :


    # For further examples on usage see ../scripts/aladinAndF3dPlots.py
    
    baseDir        = 'C:/data/regValidationWithTannerData/outFEIR/'    
    configFileList = ['cfg049','cfg050','cfg051','cfg052','cfg053','cfg054','cfg055','cfg056','cfg057','cfg058','cfg059','cfg060','cfg061','cfg062','cfg063','cfg064' ]
    

    confgPlotter = configurationPlotter( baseDir, configFileList, 2 )
    fig = confgPlotter.plotBreastAndLesion('\mu')
    plt.show()
    print( 'Done' )
