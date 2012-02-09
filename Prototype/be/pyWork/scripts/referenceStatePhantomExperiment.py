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

## Parameter set tested first
#experimentDir   = 'W:/philipsBreastProneSupine/referenceState/'
#meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
#plotDir         = experimentDir + 'plots/fatGravityRot/' 
#imageEdgeLength = 400
#tetgenVol       = 75

#experimentDir   = 'W:/philipsBreastProneSupine/referenceState/01/'
#meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
#plotDir         = experimentDir + 'plots/fatGravityRot/' 
#imageEdgeLength = 400
#tetgenVol       = 30
#tetgenQ         = 1.5
#timeStep        = 2e-5
#totalTime       = 1.0
#damping         = 50

#experimentDir   = 'W:/philipsBreastProneSupine/referenceState/02/'
#meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid.mlx' 
#plotDir         = experimentDir + 'plots/fatGravityRot/' 
#imageEdgeLength = 400
#tetgenVol       = 20
#tetgenQ         = 1.5
#timeStep        = 2e-5
#totalTime       = 1.0
#damping         = 50

experimentDir   = 'W:/philipsBreastProneSupine/referenceState/03/'
meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_mid7.mlx' 
plotDir         = experimentDir + 'plots/fatGravityRot/' 
imageEdgeLength = 400
tetgenVol       = 10
tetgenQ         = 1.5
timeStep        = 1e-6
totalTime       = 1.0
damping         = 50


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
phantom = numPhantom.numericalBreastPhantom( experimentDir, imageEdgeLength, meshlabSript, tetgenVol, tetgenQ, timeStep, totalTime, damping )

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
    xmlGenP1G = phantom.generateXMLmodelFatOnly( gravProne, 10, p1G )
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
    xmlGenP1S2G = phantom.generateXMLmodelFatOnly( gravSupine, 20, p1s2G, deformP1G.deformedNodes * 1000. )
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
    xmlGenS1G = phantom.generateXMLmodelFatOnly( gravSupine, 10, s1G )
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
