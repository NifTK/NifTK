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




def referenceStatePhantomExperiment( configID ) :

    if ( (configID != '00'   ) and (configID != '01'   ) and (configID != '02'   ) and (configID != '03'   ) and
         (configID != '00s'  ) and (configID != '01s'  ) and (configID != '02s'  ) and (configID != '03s'  ) and
         (configID != '00sAB') and (configID != '01sAB') and (configID != '02sAB') and (configID != '03sAB') and
         (configID != '00VE' ) and (configID != '01VE' ) and (configID != '02VE' ) and (configID != '03VE' ) and
         (configID != '00cyl') and (configID != '01cyl') and (configID != '02cyl') and (configID != '03cyl')  ) :
        print('Unknown configuration ID!') 
        return
    
    #
    # Default material parameters
    #
    matModel          = 'NH'
    matParamsFat      = [  100, 50000 ]
    matParamsSkin     = [ 1000, 50000 ]
    
    viscoParams       = []
    viscoNumIsoParams = 0
    viscoNumVolParams = 0
    cylindricalBase   = False


    if configID == '00' :
        # Parameter set tested first
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/00'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 75
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = False
        
        
    if configID == '01' : 
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/01/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 30
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = False
        
    if configID == '02' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/02/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 20
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         =False
    
    
    if configID == '03' :
        #
        # This setting did not work for ANY step size
        #
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/03/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid7.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 10
        tetgenQ         = 1.5
        timeStep        = 1e-6
        totalTime       = 1.0
        damping         = 50
        simSkin         = False
        
    ####################
    # Skin experiments
    # NH, skin 100, fat 1000
    if configID == '00s' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/00s/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 75
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
        
    
    if configID == '01s' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/01s/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx'  
        imageEdgeLength = 400
        tetgenVol       = 30
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
    
    
    if configID == '02s' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/02s/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx'  
        imageEdgeLength = 400
        tetgenVol       = 20
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
        
    
    if configID == '03s' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/03s/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid7.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 10
        tetgenQ         = 1.5
        timeStep        = 1e-6
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
    
    ###############################
    # Skin experiments
    # AB, 1.25, skin 100, fat 1000
    if configID.count( 'AB' ) == 1:
        matModel        = 'AB'
        matParamsFat    = [  100, 1.25, 50000 ]
        matParamsSkin   = [ 1000, 1.25, 50000 ]
    
    if configID == '00sAB' :
        
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/00sAB/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 75
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
    
    
    if configID == '01sAB' :    
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/01sAB/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx'  
        imageEdgeLength = 400
        tetgenVol       = 30
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
    
    if configID == '02sAB' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/02sAB/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx'  
        imageEdgeLength = 400
        tetgenVol       = 20
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
    
    if configID == '03sAB' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/03sAB/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid7.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 10
        tetgenQ         = 1.5
        timeStep        = 1e-6
        totalTime       = 1.0
        damping         = 50
        simSkin         = True
        
    ##############################
    # Visco elastic experiments
    #
    if configID.count('VE') == 1 :
        viscoParams       = [ 1.0, 0.2, 1.0, 1e10 ]
        viscoNumIsoParams = 1
        viscoNumVolParams = 1

    
    if configID == '00VE' :
        # Parameter set tested first
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/00/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 75
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = False
    
    #################################
    # Cylindrical base experiments
    #
    if configID.count('cyl') == 1 :
        cylindricalBase = True
    
    if configID == '00cyl' :
        # Parameter set tested first
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/00cyl/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 75
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = False    
        
    if configID == '01cyl' : 
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/01cyl/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 30
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         = False
        
    if configID == '02cyl' :
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/02cyl/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 20
        tetgenQ         = 1.5
        timeStep        = 2e-5
        totalTime       = 1.0
        damping         = 50
        simSkin         =False
    
    
    if configID == '03cyl' :
        #
        # This setting did not work for ANY step size
        #
        experimentDir   = 'W:/philipsBreastProneSupine/referenceState/03cyl/'
        meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid7.mlx' 
        imageEdgeLength = 400
        tetgenVol       = 10
        tetgenQ         = 1.5
        timeStep        = 1e-6
        totalTime       = 1.0
        damping         = 50
        simSkin         = False

    plotDir = experimentDir + 'plots/'

    ######################################
    # 
    # Start with the work
    #    
    if not os.path.exists(plotDir):
        print( 'Error: Cannot find specified plotting directory' )
        sys.exit()
    
    if not os.path.exists(experimentDir):
        print( 'Error: Cannot find specified experiment directory' )
        sys.exit()
    
    
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
                                                 fatMaterialType  = matModel, fatMaterialParams  = matParamsFat, 
                                                 skinMaterialType = matModel, skinMaterialParams = matParamsSkin, 
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
    
    
    
    for phi in range(0,91,5) :
        
        print( 'Starting simulations angle: %i' % phi )
    
        p1G   = '_prone1G'         + str( '_phi%02i' % phi )
        s1G   = '_supine1G'        + str( '_phi%02i' % phi )
        p1s2G = '_prone1Gsupine2G' + str( '_phi%02i' % phi )
        
        deformFileName      = experimentDir + 'U.txt'
        deformFileNameP1G   = experimentDir + 'U' + p1G    + '.txt'
        deformFileNameP1S2G = experimentDir + 'U' + p1s2G  + '.txt'
        deformFileNameS1G   = experimentDir + 'U' + s1G    + '.txt'
        
        dZ = np.cos( phi * np.pi / 180. )
        dY = np.sin( phi * np.pi / 180. )
        
        gravProne  = [0., dY, dZ]
        gravSupine = [0.,-dY,-dZ]
    
        #
        # 1) prone simulation
        #
        xmlGenP1G = phantom.generateXMLmodel( gravProne, 10, p1G, skin=simSkin)
        aXmlGenP1G.append(xmlGenP1G)
        
        niftySimCmd    = 'niftySim'
        niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport '
        
        if cmdEx.runCommand( niftySimCmd, niftySimParams ) != 0 :
            print('Simulation diverged.')
            break
        
        
        
        # rename the deformation file 
        if os.path.exists( deformFileNameP1G ) :
            os.remove(deformFileNameP1G)
        
        os.rename( deformFileName, deformFileNameP1G )
        deformP1G = mdh.modelDeformationHandler( xmlGenP1G, deformFileNameP1G.split('/')[-1] )
        aDeformP1G.append( deformP1G )
        #
        # 2) build model from loaded states with
        # 3) inversed and doubled gravity
        #
        xmlGenP1S2G = phantom.generateXMLmodel( gravSupine, 20, p1s2G, deformP1G.deformedNodes * 1000., skin=simSkin )
        aXmlGenP1S2G.append( xmlGenP1S2G )
         
        niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport '
    
        if cmdEx.runCommand( niftySimCmd, niftySimParams ) != 0 :
            print('Simulation diverged.')
            break
        
        # rename...
        if os.path.exists( deformFileNameP1S2G ) : 
            os.remove( deformFileNameP1S2G )
        os.rename( deformFileName, deformFileNameP1S2G )
        deformP1S2G = mdh.modelDeformationHandler( xmlGenP1S2G, deformFileNameP1S2G.split('/')[-1] )
        aDeformP1S2G.append( deformP1S2G )
        
        #
        # Now run the supine simulation from the reference state
        #
        xmlGenS1G = phantom.generateXMLmodel( gravSupine, 10, s1G, skin=simSkin )
        aXmlGenS1G.append( xmlGenS1G )
        
        niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport '
        
        if cmdEx.runCommand( niftySimCmd, niftySimParams ) != 0 :
            print('Simulation diverged.')
            break
        
        # rename...
        if os.path.exists( deformFileNameS1G ):
            os.remove( deformFileNameS1G ) 
            
        os.rename( deformFileName, deformFileNameS1G )
        deformS1G = mdh.modelDeformationHandler( xmlGenS1G, deformFileNameS1G.split('/')[-1] )
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
        plt.hist( errorDists, 150) #, range=(0.0, 30.0) )
        
        pl.ylabel( '$N( \| e \| )$' )
        pl.xlabel( '$\| e \|$' )
        pl.title( '$\phi=%i$' %phi )
        pl.grid()
        
        f = plt.gcf()
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
        
        pl.ylabel( '$N( \| e \| )$' )
        pl.xlabel( '$\| e \|$' )
        pl.title( '$\phi=%i$' %phi )
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



if __name__ == '__main__' :
    
    if len( sys.argv ) != 2:
        print( 'Usage: Please give one configuration to be executed. Available options are:' )
        print(' - 00     -> coarse mesh        (approx.  22k elements), fat only, NH [100, 50000] ' )
        print(' - 01     -> medium coarse mesh (approx.  63k elements), fat only, NH [100, 50000]' )
        print(' - 02     -> fine mesh          (approx.  86k elements), fat only, NH [100, 50000]' )
        print(' - 03     -> very fine mesh     (approx. 166k elements), fat only, NH [100, 50000]' )
        print(' - 00s    -> coarse mesh        (approx.  22k elements), fat+skin, NH [100/1000, 50000]' )
        print(' - 01s    -> medium coarse mesh (approx.  63k elements), fat+skin, NH [100/1000, 50000]' )
        print(' - 02s    -> fine mesh          (approx.  86k elements), fat+skin, NH [100/1000, 50000]' )
        print(' - 03s    -> very fine mesh     (approx. 166k elements), fat+skin, NH [100/1000, 50000]' )
        print(' - 00sAB  -> coarse mesh        (approx.  22k elements), fat+skin, AB [100/1000, 1.25 50000]' )
        print(' - 01sAB  -> medium coarse mesh (approx.  63k elements), fat+skin, AB [100/1000, 1.25 50000]' )
        print(' - 02sAB  -> fine mesh          (approx.  86k elements), fat+skin, AB [100/1000, 1.25 50000]' )
        print(' - 00VE   -> coarse mesh        (approx.  22k elements), fat only, NHV [100, 50000] [1.0 0.2 1.0 1e10]' )
        print(' - 00cyl  -> coarse mesh        (approx.  22k elements), fat only, NH [100, 50000], cylindrical base' )
        print(' - 01cyl  -> medium coarse mesh (approx.  63k elements), fat only, NH [100, 50000], cylindrical base' )
        print(' - 02cyl  -> fine mesh          (approx.  86k elements), fat only, NH [100, 50000], cylindrical base' )
        print(' - 03cyl  -> very fine mesh     (approx. 166k elements), fat only, NH [100, 50000], cylindrical base' )
        sys.exit()
    
    configID = sys.argv[1] 
    referenceStatePhantomExperiment( configID )
    
    
    
    
    
