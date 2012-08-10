#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' This script builds the xml model for prone-to-supine simulation based on 
    meshes which were constructed with the script build Mesh.py 
'''

#import numpy as np

#from numpy.core.fromnumeric import nonzero
#import scipy.linalg as linalg
#import xmlModelGenerator
from mayaviPlottingWrap import plotArrayAs3DPoints, plotArraysAsMesh, plotVectorsAtPoints
import numpy as np
import xmlModelGenerator as xGen
import imageJmeasurementReader as ijResReader
import nibabel as nib
import vtk
from vtk.util import numpy_support as VN
import getNodesCloseToMask as ndProx
import materialSetGenerator
import commandExecution as cmdEx
import f3dRegistrationTask as f3dTask
from lowAndHighModelCoordinates import lowAndHighModelCoordinates
from runSimulation import runNiftySim
import convergenceAnalyser as ca
import os

# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be corrected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

meshDir            = 'W:/philipsBreastProneSupine/Meshes/meshMaterials6Supine/'
breastVolMeshName  = meshDir + 'breastSurf_impro.1.vtk'    # volume mesh    
xmlFileOut         = meshDir + 'modelSupine.xml'
chestWallMaskImage = 'W:/philipsBreastProneSupine/SegmentationSupine/segmOutChestPectMuscFatGland_voi_dilateCW.nii'
labelImage         = 'W:/philipsBreastProneSupine/SegmentationSupine/segmOutChestPectMuscFatGland.nii'
skinMaskImage      = 'W:/philipsBreastProneSupine/SegmentationSupine/segmOutChestPectMuscFatGland_voi_dilateAir.nii'


ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( breastVolMeshName )
ugr.Update()

boxVolMesh = ugr.GetOutput()
boxVolMeshPoints = VN.vtk_to_numpy( boxVolMesh.GetPoints().GetData() )
boxVolMeshCells = VN.vtk_to_numpy( boxVolMesh.GetCells().GetData() )
boxVolMeshCells = boxVolMeshCells.reshape( boxVolMesh.GetNumberOfCells(), 
                                           boxVolMeshCells.shape[0] / boxVolMesh.GetNumberOfCells() )


surfaceExtractor = vtk.vtkDataSetSurfaceFilter()
surfaceExtractor.SetInput( boxVolMesh )
surfaceExtractor.Update()
breastSurfMeshPoints = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPoints().GetData() )

if not 'matGen' in locals():
    matGen = materialSetGenerator.materialSetGenerator( boxVolMeshPoints, 
                                                        boxVolMeshCells, 
                                                        labelImage, 
                                                        skinMaskImage, 
                                                        boxVolMesh, 
                                                        95, 105, 180, 3 ) # fat, gland, muscle, number tet-nodes to be surface element

#
# Find model boundaries
#
deltaX = 5
deltaY = 3
deltaZ = 3

lowXPoints, lowXIdx, highXPoints, highXIdx, lowYPoints, lowYIdx, highYPoints, highYIdx, lowZPoints, lowZIdx, highZPoints, highZIdx = lowAndHighModelCoordinates( boxVolMeshPoints, deltaX, deltaY, deltaZ )

print( 'Found %i low x points'  % len( lowXIdx  ) )
print( 'Found %i high x points' % len( highXIdx ) )
print( 'Found %i low y points'  % len( lowYIdx  ) )
print( 'Found %i high y points' % len( highYIdx ) )
print( 'Found %i low z points'  % len( lowZIdx  ) )
print( 'Found %i high z points' % len( highZIdx ) )

plotArrayAs3DPoints( lowXPoints,  ( 0, 0, 1.0 ) ) # sternum
plotArrayAs3DPoints( lowZPoints,  ( 0, 0, 1.0 ) ) # inferior 
plotArrayAs3DPoints( highZPoints, ( 0, 0, 1.0 ) ) # superior

#
# Find the points close to the chest surface
#
( ptsCloseToChest, idxCloseToChest ) = ndProx.getNodesWithtinMask( chestWallMaskImage, 200, boxVolMeshPoints, breastSurfMeshPoints )
plotArrayAs3DPoints( ptsCloseToChest, (1.0,1.0,1.0) )


# This little helper array is used for gravity load and material definition
allNodesArray    = np.array( range( boxVolMeshPoints.shape[0] ) )
allElemenstArray = np.array( range( boxVolMeshCells.shape[0]  ) )


genFix = xGen.xmlModelGenrator(  boxVolMeshPoints / 1000., boxVolMeshCells[ : , 1:5], 'T4ANP' )

# Fix constraints
genFix.setFixConstraint( lowXIdx,  0 )         # sternum

#genFix.setFixConstraint( lowZIdx,  0 )         # inferior breast boundary
#genFix.setFixConstraint( lowZIdx,  1 )         # inferior breast boundary
genFix.setFixConstraint( lowZIdx,  2 )         # inferior breast boundary

#genFix.setFixConstraint( highZIdx, 0 )         # superior breast boundary
#genFix.setFixConstraint( highZIdx, 1 )
genFix.setFixConstraint( highZIdx, 2 )
#genFix.setFixConstraint( highYIdx, 0 )         # mid-axillary line
genFix.setFixConstraint( highYIdx, 1 )         # mid-axillary line
#genFix.setFixConstraint( highYIdx, 2 )         # mid-axillary line
genFix.setFixConstraint( idxCloseToChest, 0 )  # chest wall
genFix.setFixConstraint( idxCloseToChest, 1 )  # chest wall
genFix.setFixConstraint( idxCloseToChest, 2 )  # chest wall


#
# visco elastic parameters
#
#genFix.setMaterialElementSet( 'NHV', 'FAT',     [  200, 50000], np.union1d( matGen.fatElements, matGen.skinElements), 1, 0, [1.0, 0.4] )
#genFix.setMaterialElementSet( 'NHV', 'GLAND',   [  400, 50000], matGen.glandElements,  1, 0, [1.0, 0.4] )
#genFix.setMaterialElementSet( 'NHV', 'MUSCLE',  [  800, 50000], matGen.muscleElements, 1, 0, [1.0, 0.4] )

#
# elastic parameters
#
genFix.setMaterialElementSet( 'NH', 'FAT',     [  150, 50000], np.union1d( matGen.fatElements, matGen.skinElements) )
genFix.setMaterialElementSet( 'NH', 'GLAND',   [  300, 50000], matGen.glandElements )
genFix.setMaterialElementSet( 'NH', 'MUSCLE',  [  600, 50000], matGen.muscleElements )

genFix.setShellElements('T3', matGen.shellElements )
genFix.setShellElementSet(0, 'NeoHookean', [1000], 1000, 0.005)

genFix.setGravityConstraint( [0., 1., 0. ], 20., allNodesArray, 'POLY345FLAT' )
genFix.setOutput( 750, ['U', 'EKinTotal', 'EStrainTotal'] )
genFix.setSystemParameters( timeStep=4.e-5, totalTime=2.0, dampingCoefficient=25, hgKappa=0.00, density=1000 )    
genFix.writeXML( xmlFileOut )



#
# run the simulation and analyse some quantities
#
runNiftySim( os.path.basename( xmlFileOut ), meshDir )
analyser = ca.convergenceAnalyser( xmlFileOut )

import sys
sys.exit()

#
# run the simulation
#


# directories
#meshDir            = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/'
regDirFEIR         = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/regFEIR/'
regDirAladin       = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/regAladin/'
regDirF3D          = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/regF3D/'

# xml model and generated output image
strSimulatedSupine = meshDir + 'out.nii'

# original images
strProneImg        = 'W:/philipsBreastProneSupine/proneCrop2Pad-zeroOrig.nii'

strSupineImg       = 'W:/philipsBreastProneSupine/rigidAlignment/supine1kTransformCrop2Pad_zeroOrig.nii'

# run the simulation and resampling at the same time
simCommand = 'ucltkDeformImageFromNiftySimulation'
simParams   = ' -i '    + strProneImg
simParams  += ' -x '    + xmlFileOut 
simParams  += ' -o '    + strSimulatedSupine
simParams  += ' -mval 0 ' 
simParams  += ' -interpolate bspl '

# run the simulation
print('Starting niftySim-Ulation')
cmdEx.runCommand( simCommand, simParams )


f3dReg = f3dTask.f3dRegistrationTask( strSimulatedSupine, strSupineImg, strSimulatedSupine, regDirF3D, 'NA', 
                                      bendingEnergy=0.0025, logOfJacobian=0.025, finalGridSpacing=5, numberOfLevels=5, maxIterations=300, gpu=True)

f3dReg.run()
f3dReg.constructNiiDeformationFile()


dispImg = nib.load( f3dReg.dispFieldITK )
dispData = dispImg.get_data()
dispAffine = dispImg.get_affine()
dispAffine[0,0] = - dispAffine[0,0]  # quick and dirty
dispAffine[1,1] = - dispAffine[1,1]
 

#
# get the displacement condition for the chest nodes
#
dVect = []

for i in range( ptsCloseToChest.shape[0] ) :
    curIDX = np.array( np.round( np.dot( dispAffine, np.hstack( ( ptsCloseToChest[i,:], 1 ) ) ) ), dtype = np.int )
    dVect.append( np.array( ( dispData[curIDX[0], curIDX[1], curIDX[2], 0, 0], 
                              dispData[curIDX[0], curIDX[1], curIDX[2], 0, 1], 
                              dispData[curIDX[0], curIDX[1], curIDX[2], 0, 2] ) ) )
    
dVect = np.array( dVect )







#visualiser = vis.modelDeformationVisualiser( genFix )


#
# compare the obtained deformation with some manually picked correspondences.   
#
# There needs to be an offset correction, as the images were cropped
#
strManuallyPickedPointsFileName = 'W:/philipsBreastProneSupine/visProneSupineDeformVectorField/Results.txt'

# cropping and padding region properties
offsetPix = np.array( [259,91,0] )

# get the image spacing
# as medSurfer only considers the pixel spacing we can ignore the origin for now 
chestWallImg   = nib.load( chestWallMaskImage )
affineTrafoMat = chestWallImg.get_affine()

# scale the offset vector
scaleX = np.abs( affineTrafoMat[0,0] )
scaleY = np.abs( affineTrafoMat[1,1] )
offsetMM = np.dot( np.abs( affineTrafoMat[0:3, 0:3] ), offsetPix )
(table, header) = ijResReader.readFileAsArray( strManuallyPickedPointsFileName )

pointsProne     = table[0:table.shape[0]:2,:]
pointsSupine    = table[1:table.shape[0]:2,:]

pointsProne     = pointsProne[:,5:8]
pointsSupine    = pointsSupine[:,5:8]

pointsPronePrime  = pointsProne  - np.tile( offsetMM, ( pointsProne.shape[0], 1 )  )
pointsSupinePrime = pointsSupine - np.tile( offsetMM, ( pointsSupine.shape[0],1 ) )
plotArrayAs3DPoints( pointsPronePrime,  (1,0,1) )
plotArrayAs3DPoints( pointsSupinePrime, (1,1,1) )

plotVectorsAtPoints( pointsSupinePrime - pointsPronePrime, pointsPronePrime )

#os.chdir( origWorkDir )


