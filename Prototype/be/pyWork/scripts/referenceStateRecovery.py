#! /usr/bin/env python 
# -*- coding: utf-8 -*-

# Purpose: Test to see if Tim's approach works for more complex shaped phantoms
# 
# The overall plan
# 1) generate the numerical breast phantom
# 2) apply gravity +g (in the direction of prone)
#    this will produce the state where the model is usually built from 
#  +->  3) apply gravity -g
#  |       This is the (initial) estimate of reference state.
#          Needs to be refined.
#  |    4) measure the difference between simulation original starting position. 
#  +--  5) update the estimated reference state
# 
# 5) compare the reference state with the original model 
#
#

import numericalBreastPhantom as numPhantom
import numpy as np
import sys, os
import commandExecution as cmdEx
import modelDeformationHandler as mdh


#
# Some parameters
#
experimentDir   = 'W:/philipsBreastProneSupine/referenceState/recovery01/'
meshlabSript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx' 
imageEdgeLength = 400
tetgenVol       = 75

gravityVectP = [0.,0.,-1.]
gravityVectS = [0.,0., 1.]
gravityMag   = 10
niftySimCmd  = 'niftySim'


p1G                 = 'p1G'
p1s1G               = 'p1s1G'  # estimation of the reference state
deformFileName      = experimentDir   + 'U.txt'
deformFileNameP1G   = experimentDir   + 'U' + p1G   + '.txt'
deformFileNameP1S1G   = experimentDir + 'U' + p1s1G + '.txt'



    
if not os.path.exists(experimentDir):
    print( 'Error: Cannot find specified experiment directory' )
    sys.exit()

#
# 1) Generate initial breast phantom
#
print( 'Generating phantom' )
phantom   = numPhantom.numericalBreastPhantom( experimentDir, imageEdgeLength, meshlabSript, tetgenVol )
xmlGenP1G = phantom.generateXMLmodelFatOnly( gravityVectP, gravityMag, 'simProne' )

#
# 2) simulate gravity -> new prone reference
#
niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport '

if cmdEx.runCommand( niftySimCmd, niftySimParams ) != 0 :
    print('Simulation diverged.')
    break

# rename the deformation file 
os.rename( deformFileName, deformFileNameP1G )
deformP1G = mdh.modelDeformationHandler( xmlGenP1G, deformFileNameP1G.split('/')[-1] )




#
# Try to get the unknown reference state from the prone image 
#
proneNodes  = deformP1G.deformedNodes
xmlGenP1S1G = phantom.generateXMLmodelFatOnly( gravityVectS, gravityMag, 'simProneSupine', proneNodes*1000 )


#
# 3) simulate gravity -> reference state estimate
#
niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport '

if cmdEx.runCommand( niftySimCmd, niftySimParams ) != 0 :
    print('Simulation diverged.')
    break

# rename the deformation file 
os.rename( deformFileName, deformFileNameP1G )
deformP1S1G = mdh.modelDeformationHandler( xmlGenP1S1G, deformFileNameP1G.split('/')[-1] )

refStateEstNodes = deformP1S1G.deformedNodes
 

#
# From this initial estimate, now 
# a) simulate prone
# b) measure difference between prone and simulated prone 
# c) update estimated reference state
#

###
#
#
phantom.generateXMLmodelFatOnly( gravityVectP, gravityMag, 'simProneFromEst', refStateEstNodes )

niftySimParams = ' -x ' + phantom.outXmlModelFat + ' -v -sport '

if cmdEx.runCommand( niftySimCmd, niftySimParams ) != 0 :
    print('Simulation diverged.')
    break

# rename the deformation file 
os.rename( deformFileName, deformFileNameP1G )
deformP1G = mdh.modelDeformationHandler( xmlGenP1G, deformFileNameP1G.split('/')[-1] )





