#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from glob import glob
import commandExecution as cmdEX
import os



def runNiftySim( xmlModelFile, workDir, exportVTK=True, verbose=True, gpu=True,  log=True ):
    ''' Execute a single simulation and rename the output files if these exist.
    '''
    
    origDir = os.getcwd()
    os.chdir( workDir )
    
    fBase = xmlModelFile.replace( '.xml', '' )
        
    #
    # construct command line
    #
    cmd = 'niftysim'
    params = ' -x ' + xmlModelFile
    
    if verbose:
        params = params + ' -v '
    
    if gpu:
        params = params + ' -sport '
        
    if exportVTK:
        params = params + ' -export ' + fBase + '.vtk ' 
    
    if log:
        logFile = 'log_' + fBase + '.txt' 
    else:
        logFile = None
    
    cmdEX.runCommand( cmd, params, logFile ) 
    
    #
    # Now rename the outputs...
    #
    if os.path.exists( 'E.txt' ) :
        os.rename( 'E.txt', 'E_' + fBase + '.txt' )
    
    if os.path.exists( 'S.txt' ) :
        os.rename( 'S.txt', 'S_' + fBase + '.txt' )

    if os.path.exists( 'F.txt' ) :
        os.rename( 'F.txt', 'F_' + fBase + '.txt' )

    if os.path.exists( 'U.txt' ) :
        os.rename( 'U.txt', 'U_' + fBase + '.txt' )
    
    if os.path.exists( 'EKinTotal.txt' ) :
        os.rename( 'EKinTotal.txt', 'EKinTotal_' + fBase + '.txt' )
    
    if os.path.exists( 'EStrainTotal.txt' ) :
        os.rename( 'EStrainTotal.txt', 'EStrainTotal_' + fBase + '.txt' )
    
    os.chdir( origDir ) 




def runSimulationsInFolder( dirIn, exportVTK=True, verbose=True, gpu=True,  log=True ):
    ''' Parses the given directory and tries to run all xml files as 
        niftySim simulations
    '''

    origDir = os.getcwd()
    os.chdir( dirIn )
    print( 'Working dir: ' + os.getcwd() )
    
    #
    # Get all xml files in the given folder
    #
    xmlFiles = glob( '*.xml' )
    
    xmlFiles.sort()
    
    print('Found the following xml-files:')
    for f in xmlFiles:
        print( f )
    
    
    for f in xmlFiles:
        runNiftySim(f, dirIn, exportVTK, verbose, gpu, log)
    
    os.chdir( origDir )
    
    
    
if __name__ == '__main__':
    
    #d050 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D050/'
    #d060 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D060/'
    #d070 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D070/'
    #d080 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D080/'
    #d090 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D090/'
    #d100 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D100/'
    #d150 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D150/'
    #d200 = 'W:/philipsBreastProneSupine/referenceState/00_load_totalTime01/D200/'
    
    #runSimulationsInFolder( d060 )
    #runSimulationsInFolder( d070 )
    #runSimulationsInFolder( d080 )
    #runSimulationsInFolder( d090 )
    
    #d050 = 'W:/philipsBreastProneSupine/referenceState/01_load/D050/'
    #d100 = 'W:/philipsBreastProneSupine/referenceState/01_load/D100/'
    
    #m = 'W:/philipsBreastProneSupine/Meshes/meshMaterials6/'
    dPoly = 'W:/philipsBreastProneSupine/referenceState/00_float_double/float_gpu_loadPoly/'
    runSimulationsInFolder( dPoly )
    