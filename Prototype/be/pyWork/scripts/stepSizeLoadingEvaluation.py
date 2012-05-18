#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import xmlModelReader as xReader
import modelDeformationVisualiser as defVis
import numpy as np
from os.path import exists
import sys
import matplotlib.pyplot as plt




def plotDoubleYAxis( xVals, y1Vals, y2Vals, xLabel, xLabelUint, y1Label, y1LabelUnit, y2Label, y2LabelUnit, plotDirAndBaseName, printLegend=True, y1Max = None ):
    
    plt.rc( 'text', usetex=True )
    plt.rcParams['font.size']=16
    
    fig = plt.figure()
    plt.hold( True )
    ax1 = fig.gca()
    ax1.plot( xVals, y1Vals, 'b-', label = y1Label )
    ax2 = ax1.twinx()
    ax2.plot( xVals, y2Vals, 'r-', label = y2Label )
    ax1.set_xlabel( xLabel        )
    ax1.set_ylabel( y1LabelUnit )
    ax2.set_ylabel( y2LabelUnit )
    ax1.set_ylim( bottom=0 )
    ax2.set_ylim( bottom=0, top=1.1 )
    ax1.grid( color = 'gray', linestyle='-' )
    
    if y1Max != None:
        ax1.set_ylim( top=y1Max )
    
    if printLegend:
        plt.legend( (ax1.get_lines(), ax2.get_lines()), (y1Label, y2Label), loc = 'lower right' )
    
    plt.hold( False )
    plt.show()
    plt.savefig( plotDirAndBaseName + '.pdf' )
    plt.savefig( plotDirAndBaseName + '.png', dpi = 300 )



def evaluate( simDirs, modelPFileBaseNameIn = 'modelFat_prone1G_it050000_totalTime05_rampflat4' ):
    
    for simDir in simDirs:

        modelPFileBaseName = modelPFileBaseNameIn
        #modelSFileBaseName = 'modelFat_supine1G_it050000_totalTime05_rampflat4'
        
        modelFilePRampFlat4 = simDir + modelPFileBaseName + '.xml'
        #modelFileSRampFlat4 = simDir + modelSFileBaseName + '.xml'
        
        deformFilePRampFlat4 = simDir + 'U_' + modelPFileBaseName + '.txt'
        #deformFileSRampFlat4 = simDir + 'U_' + modelSFileBaseName + '.txt'
        
        eKinFilePRampFlat4    = simDir + 'EKinTotal_' + modelPFileBaseName + '.txt'
        #eKinFileSRampFlat4    = simDir + 'EKinTotal_' + modelSFileBaseName + '.txt'
        
        eStrainFilePRampFlat4 = simDir + 'EStrainTotal_' + modelPFileBaseName + '.txt'
        #eStrainFileSRampFlat4 = simDir + 'EStrainTotal_' + modelSFileBaseName + '.txt'
        
        
        
        
        #
        # check file existences 
        #
        if ( (not exists( modelFilePRampFlat4   ) ) or
#             (not exists( modelFileSRampFlat4   ) ) or
             (not exists( deformFilePRampFlat4  ) ) or
#             (not exists( deformFileSRampFlat4  ) ) or
             (not exists( eKinFilePRampFlat4    ) ) or
#             (not exists( eKinFileSRampFlat4    ) ) or
             (not exists( eStrainFilePRampFlat4 ) ) #or 
#             (not exists( eStrainFileSRampFlat4 ) ) 
            ):
            print('At least one file specified here is missing')
            sys.exit() 
        
        
        readerPRampFlat4 = xReader.xmlModelReader( modelFilePRampFlat4 )
        #readerSRampFlat4 = xReader.xmlModelReader( modelFileSRampFlat4 )
        
        
        # the visualiser reads all outputs rather than only the last one...
        visPRampFlat4 = defVis.modelDeformationVisualiser( readerPRampFlat4, deformFilePRampFlat4 )
        #visSRampFlat4 = defVis.modelDeformationVisualiser( readerSRampFlat4, deformFileSRampFlat4 )
        
        
        #
        # look at the mean displacement over the iterations
        #
        itPRampFlat4 = len( visPRampFlat4.deformedNodes )
        #itSRampFlat4 = len( visSRampFlat4.deformedNodes )
        
        meanDispPRampFlat4 = np.zeros( itPRampFlat4 )
        #meanDispSRampFlat4 = np.zeros( itSRampFlat4 )
        
        
        for i in range( itPRampFlat4 ):
            meanDispPRampFlat4[i] = np.mean( np.sqrt( ( visPRampFlat4.displacements[i][:,0] * visPRampFlat4.displacements[i][:,0] ) + 
                                                      ( visPRampFlat4.displacements[i][:,1] * visPRampFlat4.displacements[i][:,1] ) +
                                                      ( visPRampFlat4.displacements[i][:,2] * visPRampFlat4.displacements[i][:,2] )   ) )   
        
        #for i in range( itSRampFlat4 ):
        #    meanDispSRampFlat4[i] = np.mean( np.sqrt( ( visSRampFlat4.displacements[i][:,0] * visSRampFlat4.displacements[i][:,0] ) + 
        #                                              ( visSRampFlat4.displacements[i][:,1] * visSRampFlat4.displacements[i][:,1] ) +
        #                                              ( visSRampFlat4.displacements[i][:,2] * visSRampFlat4.displacements[i][:,2] )   ) )   
        
        #
        # plot the results
        #
        plotDir = simDir + 'plot/'
        
        # generate time and loading function
        tMax = 5.0
        time = np.arange( 0, tMax, tMax/itPRampFlat4 )
        load = time.copy()
        load[load>1.0] = 1.0
        
        # define labels
        eLabel= '$E_\mathrm{kin} / E_\mathrm{strain}$'
        meanLabelUnit = '$\overline{ \| \mathbf{u} \| } \;\mathrm{[mm]}$'
        meanLabel     = '$\overline{ \| \mathbf{u} \| }$'
        loadLabel = '$1/F_g$'
        xLabel = '$t\;\mathrm{[s]}$'
        
        plotDoubleYAxis( time, 1000*meanDispPRampFlat4, load, xLabel, xLabel, 
                         meanLabel, meanLabelUnit, loadLabel, loadLabel, plotDir + 'dispAndLoad')#,
                         #y1Max=35 )
        
        
        
        #
        # What happened to the kinetic and strain energy?
        #
        fKinP       = open( eKinFilePRampFlat4    )
        fStrainP    = open( eStrainFilePRampFlat4 )
        dataKinP    = fKinP.read()
        dataStrainP = fStrainP.read()
        
        fKinP.close()
        fStrainP.close()
        
        dataKinP    = dataKinP.split()
        dataStrainP = dataStrainP.split()
        
        eKinPRampFlat4    = []
        eStrainPRampFlat4 = []
        
        for dKin in dataKinP:
            try:
                eKinPRampFlat4.append(float(dKin))
            except:
                print('Could not convert %s into float' %dKin)
                continue
        
        for dStrain in dataStrainP:
            try:
                eStrainPRampFlat4.append(float(dStrain))
            except:
                print('Could not convert %s into float' % dStrain)
                continue
        
        
        eKinPRampFlat4    = np.array( eKinPRampFlat4    ) 
        eStrainPRampFlat4 = np.array( eStrainPRampFlat4 )
        
        
        #
        # plot kinetic and strain energy
        #
        eKinLabel    = '$E_\mathrm{kin}\;\mathrm{[10^{-3}]}$'
        eStrainLabel = '$E_\mathrm{strain}\;\mathrm{[10^{-3}]}$'
        
        plotDoubleYAxis(time, 1000*eKinPRampFlat4, load, xLabel, xLabel, eKinLabel, eKinLabel, loadLabel, loadLabel, plotDir + 'eKinAndLoad', printLegend=False)#, y1Max=0.9)
        plotDoubleYAxis(time, 1000*eStrainPRampFlat4, load, xLabel, xLabel, eStrainLabel, eStrainLabel, loadLabel, loadLabel, plotDir + 'eStrainAndLoad', printLegend=False)#, y1Max=25)


if __name__ == '__main__':

    simDirs = []

    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D050/' )
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D060/' )
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D070/' )
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D080/' )
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D090/' )
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D100/' ) 
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D150/' )
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D200/' )
    
    #simDirs.append( 'W:/philipsBreastProneSupine/referenceState/01_load/D020/' )
    simDirs.append( 'W:/philipsBreastProneSupine/Meshes/meshMaterials6/' )
    
    evaluate( simDirs, 'model'  )
