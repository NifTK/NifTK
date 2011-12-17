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
#import findExecutable
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


# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be coorected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

meshDir = 'W:/philipsBreastProneSupine/Meshes/meshImprovement/'
breastVolMeshName = meshDir + 'breastSurf_impro.1.vtk'    # volume mesh
chestSurfMeshName = meshDir + 'chestWallSurf_impro.vtk'   # surface mesh

chestWallMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/proneMaskChestWall2-pad.nii'

origWorkDir = os.getcwd()
os.chdir( meshDir )

xmlFileOut     = meshDir + 'model.xml'

ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( breastVolMeshName )
ugr.Update()

breastVolMesh = ugr.GetOutput()
breastVolMeshPoints = VN.vtk_to_numpy( breastVolMesh.GetPoints().GetData() )
breastVolMeshCells = VN.vtk_to_numpy( breastVolMesh.GetCells().GetData() )
breastVolMeshCells = breastVolMeshCells.reshape( breastVolMesh.GetNumberOfCells(),breastVolMeshCells.shape[0]/breastVolMesh.GetNumberOfCells() )


pdr = vtk.vtkPolyDataReader()
pdr.SetFileName( chestSurfMeshName )
pdr.Update()

chestSurfMesh = pdr.GetOutput()
chestSurfMeshPoints = VN.vtk_to_numpy( chestSurfMesh.GetPoints().GetData() )
chestSurfPolys = VN.vtk_to_numpy( chestSurfMesh.GetPolys().GetData() )
chestSurfPolys = chestSurfPolys.reshape( chestSurfMesh.GetPolys().GetNumberOfCells(),chestSurfPolys.shape[0]/chestSurfMesh.GetPolys().GetNumberOfCells() )

# gain access to the meshes
#mBreast = vmr.vtkMeshFileReader( breastVolMeshName )
#mChest  = vmr.vtkMeshFileReader( chestSurfMeshName )

plotArrayAs3DPoints( breastVolMeshPoints , (1.,0.,0.) )
plotArrayAs3DPoints( chestSurfMeshPoints,  (0.,1.,0.) )


minXCoordinate = np.min( breastVolMeshPoints[:,0] )

# find those nodes of the model, which are close to the sternum... i.e. low x-values
# 9mm should be ok...
deltaX = 9

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
(ptsCloseToChest, idxCloseToChest)=ndProx.getNodesCloseToMask(chestWallMaskImage, 200, breastVolMeshPoints, 10)
plotArrayAs3DPoints(ptsCloseToChest, (1.0,1.0,1.0))

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



gen = xGen.xmlModelGenrator(  breastVolMeshPoints / 1000., breastVolMeshCells[ : , 1:5] )

gen.setFixConstraint( lowXIdx, 0 )
gen.setFixConstraint( lowXIdx, 1 )
gen.setFixConstraint( lowXIdx, 2 )

gen.setFixConstraint( idxCloseToChest, 0 )
gen.setFixConstraint( idxCloseToChest, 1 )
gen.setFixConstraint( idxCloseToChest, 2 )

gen.setMaterialElementSet( 'NH', 'FAT', [500, 50000], allElemenstArray )
gen.setGravityConstraint( [0.3, 1, 0 ], 10., allNodesArray, 'RAMP' )
gen.setOutput( 5000, 'U' )
gen.setSystemParameters( timeStep=1e-6, totalTime=1, dampingCoefficient=75, hgKappa=0.05, density=1000 )
#gen.setContactSurface( chestSurfMeshPoints[:,0:3] / 1000, chestSurfPolys[ : , 1:4 ], idxCloseToChest, 'T3' )

gen.writeXML( xmlFileOut )

print( 'Done...' )



# Now it's nice to gain access to the result. 
#visualiser = vis.modelDeformationVisualiser( gen )
#visualiser.animateDeformation()
#visualiser.deformationAsVectors()



# compare the obtained deformation with some manually picked correspondences. 
  

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

