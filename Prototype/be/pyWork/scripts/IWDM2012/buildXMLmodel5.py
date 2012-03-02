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

# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be coorected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

meshDir            = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/'
breastVolMeshName  = meshDir + 'breastSurf_impro.1.vtk'    # volume mesh
breastSurfMeshName = meshDir + 'breastSurf_impro.stl'    
xmlFileOut         = meshDir + 'model.xml'
chestWallMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh-dilateR2I4.nii'
labelImage         = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad.nii'
skinMaskImage      = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-AirThresh-dilateR2I4.nii'


ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( breastVolMeshName )
ugr.Update()

breastVolMesh = ugr.GetOutput()
breastVolMeshPoints = VN.vtk_to_numpy( breastVolMesh.GetPoints().GetData() )
breastVolMeshCells = VN.vtk_to_numpy( breastVolMesh.GetCells().GetData() )
breastVolMeshCells = breastVolMeshCells.reshape( breastVolMesh.GetNumberOfCells(),breastVolMeshCells.shape[0]/breastVolMesh.GetNumberOfCells() )

stlR = vtk.vtkSTLReader()
stlR.SetFileName( breastSurfMeshName )
stlR.Update()

surfaceExtractor = vtk.vtkDataSetSurfaceFilter()
surfaceExtractor.SetInput( breastVolMesh )
surfaceExtractor.Update()
breastSurfMeshPoints = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPoints().GetData() )

if not 'matGen' in locals():
    matGen = materialSetGenerator.materialSetGenerator( breastVolMeshPoints, 
                                                        breastVolMeshCells, 
                                                        labelImage, 
                                                        skinMaskImage, 
                                                        breastVolMesh, 
                                                        95, 105, 180, 3 ) # fat, gland, muscle, number tet-nodes to be surface element


# find those nodes of the model, which are close to the sternum... i.e. low x-values
minXCoordinate = np.min( breastVolMeshPoints[:,0] )
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
( ptsCloseToChest, idxCloseToChest ) = ndProx.getNodesWithtinMask( chestWallMaskImage, 200, breastVolMeshPoints, breastSurfMeshPoints )
plotArrayAs3DPoints( ptsCloseToChest, (1.0,1.0,1.0) )


# This little helper array is used for gravity load and material definition
allNodesArray    = np.array( range( breastVolMeshPoints.shape[0] ) )
allElemenstArray = np.array( range( breastVolMeshCells.shape[0]  ) )


genFix = xGen.xmlModelGenrator(  breastVolMeshPoints / 1000., breastVolMeshCells[ : , 1:5], 'T4ANP' )

genFix.setFixConstraint( lowXIdx, 0 )
#genFix.setFixConstraint( lowXIdx, 1 )
#genFix.setFixConstraint( lowXIdx, 2 )
genFix.setFixConstraint( idxCloseToChest, 0 )
genFix.setFixConstraint( idxCloseToChest, 1 )
genFix.setFixConstraint( idxCloseToChest, 2 )

#genFix.setMaterialElementSet( 'NH', 'FAT', [500, 50000], allElemenstArray )

genFix.setMaterialElementSet( 'NHV', 'FAT',    [  250, 50000], matGen.fatElemetns, 1, 0, [1.0, 0.2] )
genFix.setMaterialElementSet( 'NH', 'SKIN',    [ 2000, 50000], matGen.skinElements   )
genFix.setMaterialElementSet( 'NH', 'GLAND',   [  500, 50000], matGen.glandElements  )
genFix.setMaterialElementSet( 'NH', 'MUSCLE',  [ 1000, 50000], matGen.muscleElements )

genFix.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray, 'RAMP' )
genFix.setOutput( 5000, 'U' )
genFix.setSystemParameters( timeStep=1e-4, totalTime=1, dampingCoefficient=50, hgKappa=0.05, density=1000 )    
genFix.writeXML( xmlFileOut )



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


