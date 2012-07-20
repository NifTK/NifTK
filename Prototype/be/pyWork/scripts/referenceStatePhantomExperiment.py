#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numericalBreastPhantom as numPhantom
import commandExecution as cmdEx
import os,sys
import modelDeformationHandler as mdh
import pylab as pl
from conversions import numpyArrayToStr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mayaviPlottingWrap import plotArrayAs3DPoints, plotVectorsAtPoints
import runSimulation as rS
import convergenceAnalyser as cA
from envisage.safeweakref import ref


def referenceStatePhantomExperiment( configID ) : 

    if ( (configID != '00'   ) and (configID != '01'   ) and (configID != '02'   ) and (configID != '03'   ) and
         (configID != '00s'  ) and (configID != '01s'  ) and (configID != '02s'  ) and (configID != '03s'  ) and
         (configID != '00sAB') and (configID != '01sAB') and (configID != '02sAB') and (configID != '03sAB') and
         (configID != '00VE' ) and (configID != '01VE' ) and (configID != '02VE' ) and (configID != '03VE' ) and
         (configID != '00cyl') and (configID != '01cyl') and (configID != '02cyl') and (configID != '03cyl')  ) :
        print('WARNING!!! Configuration might be unknown...') 
        
    
    #
    # Default material parameters
    #
    
    # directory
    baseExperimentDir = 'W:/philipsBreastProneSupine/referenceState/'
    experimentDir     = baseExperimentDir + configID + '/'
    plotDir           = experimentDir + 'plots/'
    
    
    # phantom properties
    imageEdgeLength = 400
    
    # general volume mesh properties
    tetgenQ         = 1.5
    
    # system parameters
    timeStep        = 1e-4 # the critical time step for the 00 mesh resolution is 5e-4
    totalTime       = 5.0 
    damping         = 25
    
    loadShape = 'POLY345FLAT4'
    
    numOutputs = 200
    outputFreq = int( np.ceil( totalTime / timeStep / numOutputs ) )


    ##############################
    # Model resolutions 00 ... 03
    #
    if configID.count('00') == 1 :
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
        tetgenVol       = 75
        
        
    elif configID.count('01') == 1 : 
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
        tetgenVol       = 30
        
    elif configID.count('02') == 1 :
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
        tetgenVol       = 20
    
    elif configID.count('03') == 1 :
        #
        # This setting did not work for ANY step size
        #
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid7.mlx' 
        tetgenVol       = 10
        timeStep        = 1e-6
    else :
        print( 'Unknown configuration!' )
        sys.exit()
  
    ########
    # skin
    simSkin           = False

    if configID.count('s') == 1 :
        simSkin = True

    ############
    # material 
    matModelFat       = 'NH'
    matModelSkin      = 'NH'
    matParamsFat      = [  100, 50000 ]
    matParamsSkin     = [ 1000, 50000 ]
    
    
    if configID.count( 'AB' ) == 1:
        matModelFat     = 'AB'
        matModelSkin    = 'AB'
        matParamsFat    = [  100, 1.25, 50000 ]
        matParamsSkin   = [ 1000, 1.25, 50000 ]


    if configID.count( 'LE' ) == 1:
        matModelFat   = 'LE'
        matModelSkin  = 'LE'
        matParamsFat  = [  299.80, 0.499001 ]
        matParamsSkin = [ 2980.13, 0.490066 ]
    
    

    ################
    # Visco elastic 
    # for fat only, change material to NHV!
    viscoParams       = []
    viscoNumIsoParams = 0
    viscoNumVolParams = 0
    
    if configID.count('VE') == 1 :
        matModelFat       = 'NHV'
        matParamsFat      = [ 100, 50000 ]
        viscoParams       = [ 1.0, 0.2   ]
        viscoNumIsoParams = 1
        viscoNumVolParams = 0

    
    #################################
    # Cylindrical base experiments
    cylindricalBase   = False
    
    if configID.count('cyl') == 1 :
        cylindricalBase = True
    

    #######################################
    # print a summary of the configuration
    print( 'Summary of the simulation:' )
    print( ' Directories...' )
    print( '  -> experimentDir: ' + experimentDir ) 
    print( '  -> plotDir: '       + plotDir       )
    
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
    if not os.path.exists(experimentDir):
        os.mkdir( experimentDir ) 
        print( 'directory created...' )

    
    if not os.path.exists(plotDir):
        os.mkdir( plotDir )
        print( 'directory created...' )
    
    
    
    #
    # To estimate the influence of the reference state:
    # 1) run the prone simulation
    # 2) Build the model from the loaded state
    # 3) inverse gravity and double it
    #
    print( 'Generating phantom' )
    phantom = numPhantom.numericalBreastPhantom( experimentDir, imageEdgeLength, 
                                                 meshlabSript, tetgenVol, tetgenQ, 
                                                 timeStep, totalTime, damping, 
                                                 fatMaterialType  = matModelFat, fatMaterialParams  = matParamsFat, 
                                                 fatViscoNumIsoTerms = viscoNumIsoParams, fatViscoNumVolTerms=viscoNumVolParams,fatViscoParams=viscoParams,  
                                                 skinMaterialType = matModelSkin, skinMaterialParams = matParamsSkin, 
                                                 cylindricalBase = cylindricalBase )
    
    # track over iterations
    errorMeasures      = []
    errorMeasuresNoFix = [] # error measure ignoring the fixed nodes from the calculations
    
    aXmlGenP1G   = []
    aXmlGenP1S2G = []
    aXmlGenS1G   = []
    
    aDeformP1G   = []
    aDeformP1S2G = []
    aDeformS1G   = []
    
    aErrorVects  = []
    aErrorDists  = [] 
    
    angles = range(0, 46, 5)
    
    if configID.count('cyl') == 1:
        angles = range(30, -31, -5) 
    
    for phi in angles :
        
        print( 'Starting simulations angle: %i' % phi )
    
        p1G   = '_prone1G'         + str( '_phi%02i' % phi )
        s1G   = '_supine1G'        + str( '_phi%02i' % phi )
        p1s2G = '_prone1Gsupine2G' + str( '_phi%02i' % phi )
        
        #deformFileName      = experimentDir + 'U.txt'
        #deformFileNameP1G   = experimentDir + 'U' + p1G    + '.txt'
        #deformFileNameP1S2G = experimentDir + 'U' + p1s2G  + '.txt'
        #deformFileNameS1G   = experimentDir + 'U' + s1G    + '.txt'
        
        dZ = np.cos( phi * np.pi / 180. )
        dY = np.sin( phi * np.pi / 180. )
        
        gravProne  = [0., dY, dZ]
        gravSupine = [0.,-dY,-dZ]
    
        #
        # 1) prone simulation
        #
        xmlGenP1G = phantom.generateXMLmodel( gravityVector=gravProne, 
                                              gravityMagnitude=10, 
                                              fileIdentifier=p1G, 
                                              skin=simSkin, 
                                              gravityLoadShape=loadShape, 
                                              outputFrequency=outputFreq )
        aXmlGenP1G.append(xmlGenP1G)
        
        simulationReturn = rS.runNiftySim( os.path.split( phantom.outXmlModelFat )[1] , os.path.split( phantom.outXmlModelFat )[0] )
        
        if simulationReturn != 0 :
            print('Simulation diverged.')
            break

        cA.convergenceAnalyser( phantom.outXmlModelFat )
        plt.close('all')

        deformP1G = mdh.modelDeformationHandler( xmlGenP1G )
        aDeformP1G.append( deformP1G )
        
        #
        # 2) build model from loaded states with
        # 3) inversed and doubled gravity
        #
        xmlGenP1S2G = phantom.generateXMLmodel( gravityVector=gravSupine, 
                                                gravityMagnitude=20, 
                                                fileIdentifier=p1s2G, 
                                                extMeshNodes=deformP1G.deformedNodes * 1000., 
                                                skin=simSkin, 
                                                gravityLoadShape=loadShape,
                                                outputFrequency=outputFreq )
        
        aXmlGenP1S2G.append( xmlGenP1S2G )
         
        #niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport -export ' + phantom.outXmlModelFat.split('.xml')[0] + '.vtk'
    
        simulationReturn = rS.runNiftySim(  os.path.split( phantom.outXmlModelFat )[1] , os.path.split( phantom.outXmlModelFat )[0] )
    
        if simulationReturn != 0 :
            print('Simulation diverged.')
            break

        cA.convergenceAnalyser( phantom.outXmlModelFat )
        plt.close('all')
        
        # rename...
        deformP1S2G = mdh.modelDeformationHandler( xmlGenP1S2G )
        aDeformP1S2G.append( deformP1S2G )
        
        #
        # Now run the supine simulation from the reference state
        #
        xmlGenS1G = phantom.generateXMLmodel( gravityVector=gravSupine, 
                                              gravityMagnitude=10, 
                                              fileIdentifier=s1G, 
                                              skin=simSkin, 
                                              gravityLoadShape=loadShape, 
                                              outputFrequency=outputFreq )
        aXmlGenS1G.append( xmlGenS1G )
        
        simulationReturn = rS.runNiftySim( os.path.split( phantom.outXmlModelFat )[1] , os.path.split( phantom.outXmlModelFat )[0] )
        
        if simulationReturn != 0 :
            print('Simulation diverged.')
            break

        cA.convergenceAnalyser( phantom.outXmlModelFat )
        plt.close('all')
        
        deformS1G = mdh.modelDeformationHandler( xmlGenS1G )
        aDeformS1G.append( deformS1G )
        
        #
        # Analyse and plot the error between the simulations
        #
        errorVects = 1000. * ( deformS1G.deformedNodes - deformP1S2G.deformedNodes ) # errors in mm
        errorDists = np.sqrt( errorVects[:,0] ** 2 + errorVects[:,1] ** 2 + errorVects[:,2] ** 2 ) 
        
        aErrorVects.append( errorVects )
        aErrorDists.append( errorDists )
        
        # Plot as a nice latex-syle pdf/png
        mpl.rc( 'text', usetex=True )
        f = plt.figure()
        plt.hist( errorDists, 150) #, range=(0.0, 30.0) )
        
        pl.ylabel( '$N( \| \mathbf{e} \| )$' )
        pl.xlabel( '$\| \mathbf{e} \| \; \mathrm{[mm]}$' )
        pl.title( '$\phi=%i ^\circ$' %phi )
        pl.grid()
        
        f.savefig( plotDir + str( 'errorHist_phi%02i.pdf' % phi ) )
        f.savefig( plotDir + str( 'errorHist_phi%02i.png' % phi ), dpi=150 )
        f.clf()
        
        errorMeasures.append( np.array( (phi, np.mean(errorDists), np.std(errorDists), np.min(errorDists), np.max(errorDists) ) ) )
        
        
        #
        # Calculate the error measures without the fixed nodes - which do not produce any error by definition
        #  any generator will do!
        #
        fixedNodeNums = aXmlGenP1G[-1].fixConstraintNodes[0]
        allNodeNums   = aXmlGenP1G[-1].allNodesArray
        looseNodeNums = allNodeNums[-np.in1d(allNodeNums, fixedNodeNums)]
        
        plt.hist( errorDists[looseNodeNums], 200)#, range=(0.0, 30.0) )
        
        pl.ylabel( '$N( \| \mathbf{e} \| )$' )
        pl.xlabel( '$\| \mathbf{e} \| \; \mathrm{[mm]}$' )
        pl.title( '$\phi=%i ^\circ$' %phi )
        #pl.ylim( (0,500) )
        pl.grid()
        
        f = plt.gcf()
        f.savefig( plotDir + str( 'errorHist_phi%02i_noFix.pdf' % phi ) )
        f.savefig( plotDir + str( 'errorHist_phi%02i_noFix.png' % phi ), dpi=150 )
        f.clf()
        
        errorMeasuresNoFix.append( np.array( ( phi, 
                                               np.mean( errorDists[looseNodeNums] ), 
                                               np.std ( errorDists[looseNodeNums] ), 
                                               np.min ( errorDists[looseNodeNums] ), 
                                               np.max ( errorDists[looseNodeNums] ) ) ) )
        
        print( 'Angle %i done.' %phi )
    
    
    
    errorMeasures       = np.array( errorMeasures      )
    errorMeasuresNoFix  = np.array( errorMeasuresNoFix )
    resultFileName      = plotDir + 'results.txt'
    resultFileNameNoFix = plotDir + 'resultsNoFix.txt'
    
    resFile = file(resultFileName, 'w')
    resFile.write( numpyArrayToStr(errorMeasures, floatingPoint=True, indent='') )
    resFile.close()
    
    resFileNoFix = file(resultFileNameNoFix, 'w')
    resFileNoFix.write( numpyArrayToStr(errorMeasuresNoFix, floatingPoint=True, indent='') )
    resFileNoFix.close()
    
    #
    # plot the final result
    #
    plt.errorbar( errorMeasures[:,0], errorMeasures[:,1], errorMeasures[:,2], fmt='b-x' )
    
    pl.ylabel( '$ \| e \| $' )
    pl.xlabel( '$ \phi $' )
    #pl.title( 'with fixed' )
    pl.grid()
    
    f = plt.gcf()
    f.savefig( plotDir + str( 'errorOverAngle.pdf' ) )
    f.savefig( plotDir + str( 'errorOverAngle.png' ), dpi=150 )
    f.clf()
    
    #
    # plot the final result 2: ignore fixed
    #
    plt.errorbar( errorMeasuresNoFix[:,0], errorMeasuresNoFix[:,1], errorMeasuresNoFix[:,2], fmt='b-x' )
    
    pl.ylabel( '$ \| e \| $' )
    pl.xlabel( '$ \phi $' )
    #pl.title( 'without fixed' )
    pl.grid()
    
    f = plt.gcf()
    f.savefig( plotDir + str( 'errorOverAngleNoFix.pdf' ) )
    f.savefig( plotDir + str( 'errorOverAngleNoFix.png' ), dpi=150 )
    f.clf()

    print('Done.')



#if __name__ == '__main__' :
#    
#    if len( sys.argv ) != 2:
#        print( 'Usage: Please give one configuration to be executed. Available options are:' )
#        print(' - 00     -> coarse mesh        (approx.  22k elements), fat only, NH [100, 50000] ' )
#        print(' - 01     -> medium coarse mesh (approx.  63k elements), fat only, NH [100, 50000]' )
#        print(' - 02     -> fine mesh          (approx.  86k elements), fat only, NH [100, 50000]' )
#        print(' - 03     -> very fine mesh     (approx. 166k elements), fat only, NH [100, 50000]' )
#        print(' - 00s    -> coarse mesh        (approx.  22k elements), fat+skin, NH [100/1000, 50000]' )
#        print(' - 01s    -> medium coarse mesh (approx.  63k elements), fat+skin, NH [100/1000, 50000]' )
#        print(' - 02s    -> fine mesh          (approx.  86k elements), fat+skin, NH [100/1000, 50000]' )
#        print(' - 03s    -> very fine mesh     (approx. 166k elements), fat+skin, NH [100/1000, 50000]' )
#        print(' - 00sAB  -> coarse mesh        (approx.  22k elements), fat+skin, AB [100/1000, 1.25 50000]' )
#        print(' - 01sAB  -> medium coarse mesh (approx.  63k elements), fat+skin, AB [100/1000, 1.25 50000]' )
#        print(' - 02sAB  -> fine mesh          (approx.  86k elements), fat+skin, AB [100/1000, 1.25 50000]' )
#        print(' - 00VE   -> coarse mesh        (approx.  22k elements), fat only, NHV [100, 50000] [1.0 0.2 1.0 1e10]' )
#        print(' - 00cyl  -> coarse mesh        (approx.  22k elements), fat only, NH [100, 50000], cylindrical base' )
#        print(' - 01cyl  -> medium coarse mesh (approx.  63k elements), fat only, NH [100, 50000], cylindrical base' )
#        print(' - 02cyl  -> fine mesh          (approx.  86k elements), fat only, NH [100, 50000], cylindrical base' )
#        print(' - 03cyl  -> very fine mesh     (approx. 166k elements), fat only, NH [100, 50000], cylindrical base' )
#        sys.exit()
#    
#    configID = sys.argv[1] 
#    referenceStatePhantomExperiment( configID )
#    
    
    

if __name__ == '__main__' :
    referenceStatePhantomExperiment( '00' )
