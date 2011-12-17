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
import os, sys
import fileCorrespondence as fc 
import numpy as np
import xmlModelGenerator as xGen
import modelDeformationVisualiser as vis
import commandExecution as cmdEx
import imageJmeasurementReader as ijResReader
import nibabel as nib
import vtk
from vtk.util import numpy_support as VN
import getNodesCloseToMask as ndProx

import materialSetGenerator as matGen


# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be coorected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

meshDir            = 'W:/philipsBreastProneSupine/Meshes/meshMaterials/'
breastVolMeshName  = meshDir + 'breastSurf_improLR.1.vtk'    # volume mesh
breastSurfMeshName = meshDir + 'breastSurf_improLR.stl'    
xmlFileOut         = meshDir + 'model.xml'
chestWallMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/proneMaskChestWall2-pad.nii'

origWorkDir = os.getcwd()
os.chdir( meshDir )

ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( breastVolMeshName )
ugr.Update()

breastVolMesh = ugr.GetOutput()
breastVolMeshPoints = VN.vtk_to_numpy( breastVolMesh.GetPoints().GetData() )
breastVolMeshCells = VN.vtk_to_numpy( breastVolMesh.GetCells().GetData() )
breastVolMeshCells = breastVolMeshCells.reshape( breastVolMesh.GetNumberOfCells(),breastVolMeshCells.shape[0]/breastVolMesh.GetNumberOfCells() )

#plotArrayAs3DPoints( breastVolMeshPoints , (1.,0.,0.) )

minXCoordinate = np.min( breastVolMeshPoints[:,0] )

stlR = vtk.vtkSTLReader()
stlR.SetFileName( breastSurfMeshName )
stlR.Update()

#breastSurfMesh  = stlR.GetOutput()
#breastSurfMeshPoints = VN.vtk_to_numpy( breastSurfMesh.GetPoints().GetData() )

#
# 
#
surfaceExtractor = vtk.vtkDataSetSurfaceFilter()
surfaceExtractor.SetInput( breastVolMesh )
surfaceExtractor.Update()
breastSurfMeshPoints = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPoints().GetData() )
        
mg        = matGen.materialSetGenerator(breastVolMeshPoints, breastVolMeshCells, None, None, breastVolMesh, 0, 0, 0)
skinNodes = mg.skinElements
fatNodes  = mg.fatElemetns


# find those nodes of the model, which are close to the sternum... i.e. low x-values
# 9mm should be ok...
deltaX = 5

lowXPoints = []
lowXIdx    = []

for i in range( breastVolMeshPoints.shape[0] ):
    if breastVolMeshPoints[i,0] < ( minXCoordinate + deltaX ) :
        lowXIdx.append( i )
        lowXPoints.append( [breastVolMeshPoints[i,0], breastVolMeshPoints[i,1], breastVolMeshPoints[i,2] ] )
    
lowXPoints = np.array( lowXPoints )
lowXIdx    = np.array( lowXIdx    )

print( 'Found %i points within an x range between [ -inf ; %f ]' % (len( lowXIdx ), minXCoordinate + deltaX ) )
plotArrayAs3DPoints(lowXPoints, ( 0, 0, 1.0 ) )


#
# Find the points close to the chest surface
#
( ptsCloseToChest, idxCloseToChest ) = ndProx.getNodesCloseToMask( chestWallMaskImage, 200, breastVolMeshPoints, 9, breastSurfMeshPoints )
plotArrayAs3DPoints( ptsCloseToChest, (1.0,1.0,1.0) )


#
# prepare the cylinder parameters
#
cylOrigin = np.array((8.25,190.0,-10)) / 1000
cylAxis = np.array((0,0,1))
cylRadius =  89.625 /1000. 
cylLength = 220.0 / 1000.
cylDispOrigin = np.zeros((1,3))
cylRadChange= 0


###############################
# Which bits go into the model
#
# 1) [X] nodes and elements of the deformable model. Remember that all coordinates need to be in mm!!!
# 2) [X] boundary surface
# 3) [X] fix nodes close to sternum only in x-direction
# 4) [X] material properties
# 5) [X] system parameters
#

# This little helper array is used for gravity load and material definition
allNodesArray    = np.array( range( breastVolMeshPoints.shape[0] ) )
allElemenstArray = np.array( range( breastVolMeshCells.shape[0]  ) )


genFix = xGen.xmlModelGenrator(  breastVolMeshPoints / 1000., breastVolMeshCells[ : , 1:5], 'T4' )

#genFix.setFixConstraint( lowXIdx, 0 )
#genFix.setFixConstraint( lowXIdx, 1 )
#genFix.setFixConstraint( lowXIdx, 2 )
genFix.setFixConstraint( idxCloseToChest, 0 )
genFix.setFixConstraint( idxCloseToChest, 1 )
genFix.setFixConstraint( idxCloseToChest, 2 )
genFix.setMaterialElementSet( 'NH', 'FAT', [500, 50000], allElemenstArray )
#genFix.setMaterialElementSet( 'NH', 'FAT', [500, 50000], fatElemetns  )
#genFix.setMaterialElementSet( 'NH', 'SKIN', [5000, 50000], skinElements )
genFix.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray, 'STEP' )
genFix.setOutput( 500, 'U' )
genFix.setSystemParameters( timeStep=1e-4, totalTime=1, dampingCoefficient=35, hgKappa=0.05, density=1000 )    
genFix.writeXML( xmlFileOut )
    
#
# run the simulations
#
simCommand = 'niftySim'
simParams = '-sport -v -x ' + xmlFileOut

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

os.chdir( origWorkDir )
