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
import feirRegistrationTask as feirTask

# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be corrected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

useFEIR                   = True
updateFactor              = 0.75
numIterations             = 5

meshDir                   = 'W:/philipsBreastProneSupine/Meshes/meshMaterials4/'
regDirF3D                 = meshDir + 'regF3D/'
regDirFEIR                = meshDir + 'regFEIR/'
breastVolMeshName         = meshDir + 'breastSurf_impro.1.vtk'    # volume mesh    
xmlFileOut                = meshDir + 'model.xml'

chestWallMaskImage        = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh.nii'
chestWallMaskImageDilated = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh-dilateR2I4.nii'
labelImage                = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM2_Crp2-pad.nii'
skinMaskImage             = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-AirThresh-dilateR2I4.nii'

# original images
strProneImg               = 'W:/philipsBreastProneSupine/proneCrop2Pad-zeroOrig.nii'
strSupineImg              = 'W:/philipsBreastProneSupine/rigidAlignment/supine1kTransformCrop2Pad_zeroOrig.nii'




#
# compare the obtained deformation with some manually picked correspondences: TRE
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





ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( breastVolMeshName )
ugr.Update()

# Get the volume mesh
breastVolMesh = ugr.GetOutput()
breastVolMeshPoints = VN.vtk_to_numpy( breastVolMesh.GetPoints().GetData() )
breastVolMeshCells = VN.vtk_to_numpy( breastVolMesh.GetCells().GetData() )
breastVolMeshCells = breastVolMeshCells.reshape( breastVolMesh.GetNumberOfCells(),breastVolMeshCells.shape[0]/breastVolMesh.GetNumberOfCells() )

# Get the surface from the volume mesh (as tetgen added some nodes)
surfaceExtractor = vtk.vtkDataSetSurfaceFilter()
surfaceExtractor.SetInput( breastVolMesh )
surfaceExtractor.Update()
breastSurfMeshPoints = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPoints().GetData() )

# calculate the material parameters (still to be improved)
# this only needs to run once...
if not 'matGen' in locals():
    matGen = materialSetGenerator.materialSetGenerator( breastVolMeshPoints, 
                                                        breastVolMeshCells, 
                                                        labelImage, 
                                                        skinMaskImage, 
                                                        breastVolMesh, 
                                                        95, 105, 180, 3 ) # fat, gland, muscle, number tet-nodes to be surface element

plotArrayAs3DPoints( matGen.skinElementMidPoints, (0., 1., 0.) )


# find those nodes of the model, which are close to the sternum... i.e. low x-values
# and those nodes of the model, which are close to mid-axillary line. i.e. high y-values
minXCoordinate = np.min( breastVolMeshPoints[:,0] )
deltaX = 5

maxYCoordinate = np.max( breastVolMeshPoints[:,1] )
deltaY = deltaX

lowXPoints  = []
lowXIdx     = []
highYPoints = []
highYIdx    = []

for i in range( breastVolMeshPoints.shape[0] ):
    if breastVolMeshPoints[i,0] < ( minXCoordinate + deltaX ) :
        lowXIdx.append( i )
        lowXPoints.append( [breastVolMeshPoints[i,0], breastVolMeshPoints[i,1], breastVolMeshPoints[i,2] ] )
    
    if breastVolMeshPoints[i,1] > ( maxYCoordinate - deltaY ) :
        highYIdx.append( i )
        highYPoints.append( [breastVolMeshPoints[i,0], breastVolMeshPoints[i,1], breastVolMeshPoints[i,2] ] )
    
lowXPoints  = np.array( lowXPoints )
lowXIdx     = np.array( lowXIdx    )
highYPoints = np.array( highYPoints )
highYIdx    = np.array( highYIdx    )

print( 'Found %i points within a x range between [ -inf ; %f ]' % (len( lowXIdx  ), minXCoordinate + deltaX ) )
print( 'Found %i points within a y range between [ %f ; +inf ]' % (len( highYIdx ), maxYCoordinate - deltaY ) )
plotArrayAs3DPoints( lowXPoints,  ( 0, 0, 1.0 ) )
plotArrayAs3DPoints( highYPoints, ( 0, 1.0, 1.0 ) )


#
# Find the points on the chest surface
#
( ptsCloseToChest, idxCloseToChest ) = ndProx.getNodesWithtinMask( chestWallMaskImageDilated, 200, breastVolMeshPoints, breastSurfMeshPoints)
plotArrayAs3DPoints( ptsCloseToChest, (1.0,1.0,1.0) )


# This little helper array is used for gravity load and material definition
allNodesArray    = np.array( range( breastVolMeshPoints.shape[0] ) )
allElemenstArray = np.array( range( breastVolMeshCells.shape[0]  ) )

#
# Generate the xml-file
#
genFix = xGen.xmlModelGenrator(  breastVolMeshPoints / 1000., breastVolMeshCells[ : , 1:5], 'T4' )

genFix.setFixConstraint( lowXIdx,  0 )
genFix.setFixConstraint( lowXIdx,  1 )
genFix.setFixConstraint( lowXIdx,  2 )

genFix.setFixConstraint( highYIdx, 0 )
genFix.setFixConstraint( highYIdx, 1 )
genFix.setFixConstraint( highYIdx, 2 )

genFix.setFixConstraint( idxCloseToChest, 0 )
genFix.setFixConstraint( idxCloseToChest, 1 )
genFix.setFixConstraint( idxCloseToChest, 2 )


#genFix.setMaterialElementSet( 'NH', 'FAT',    [  400, 50000], allElemenstArray    )
genFix.setMaterialElementSet( 'NH', 'FAT',    [  150, 50000], matGen.fatElemetns    )
genFix.setMaterialElementSet( 'NH', 'SKIN',   [ 1600, 50000], matGen.skinElements   )
genFix.setMaterialElementSet( 'NH', 'GLAND',  [  300, 50000], matGen.glandElements  )
genFix.setMaterialElementSet( 'NH', 'MUSCLE', [  800, 50000], matGen.muscleElements )

genFix.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray, 'RAMP' )
genFix.setOutput( 5000, 'U' )
genFix.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=50, hgKappa=0.05, density=1000 )    
genFix.writeXML( xmlFileOut )


#
# remember which nodes (numbers and coordinates) were fixed, so these can be used later on.
# and initialise these with zero (=fixed) displacement
#
prevFixedNodes = np.unique( np.hstack( ( genFix.fixConstraintNodes[0], 
                                         genFix.fixConstraintNodes[1], 
                                         genFix.fixConstraintNodes[2] ) ) )
dispVects      = np.zeros( (prevFixedNodes.shape[0], 3) )

for i in range( numIterations ) :
    
    #
    # run the simulation and the resampling
    #
    
    # xml model and generated output image
    strSimulatedSupine         = meshDir + 'out'      + str('%03i' %i) + '.nii'
    strSimulatedSupineLabelImg = meshDir + 'outLabel' + str('%03i' %i) + '.nii'

    # run the simulation and resampling at the same time
    simCommand = 'niftkDeformImageFromNiftySimulation'
    simParams   = ' -x '    + xmlFileOut 
    simParams  += ' -i '    + strProneImg
    simParams  += ' -o '    + strSimulatedSupine

    # also deform the label image!
    simParams  += ' -iL '    + labelImage
    simParams  += ' -oL '    + strSimulatedSupineLabelImg
    
    
    
    # niftyReg and FEIR use different indicators for "mask"
    if useFEIR :
        simParams  += ' -mval -1 ' 
    else :
        simParams  += ' -mval 0 '
         
    simParams  += ' -interpolate bspl '
    
    # run the simulation
    print('Starting niftySimulation')
    cmdEx.runCommand( simCommand, simParams )

    
    #
    # Check if the resampled image was created 
    #
    simSup = nib.load( strSimulatedSupine )
    
    if np.min( simSup.get_data() ) == np.max( simSup.get_data() ):
        print('No more simulation results after %i iterations' % i)
        break  
    
    if useFEIR :
        feirReg = feirTask.feirRegistrationTask( strSimulatedSupine, strSupineImg, regDirFEIR, 'NA', 
                                                 mu=0.0025*(2**-8), lm=0.0, mode='fast', mask=True, displacementConvergence=0.01 )#, planStr='n')
    
        feirReg.run()
        feirReg.constructNiiDeformationFile()
        feirReg.resampleSourceImage()
        dispImg = nib.load( feirReg.dispFieldITK )
        
    else :
        f3dReg = f3dTask.f3dRegistrationTask( strSimulatedSupine, strSupineImg, strSimulatedSupine, regDirF3D, 'NA', 
                                              bendingEnergy=0.007, logOfJacobian=0.0, finalGridSpacing=5, numberOfLevels=5, maxIterations=300, gpu=True)
        
        f3dReg.run()
        f3dReg.constructNiiDeformationFile()
        dispImg = nib.load( f3dReg.dispFieldITK )
    
    
    # read the deformation field
    dispData = dispImg.get_data()
    dispAffine = dispImg.get_affine()
    dispAffine[0,0] = - dispAffine[0,0]  # quick and dirty
    dispAffine[1,1] = - dispAffine[1,1]
     
    #
    # generate the new model with the updated boundary conditions...
    #
    # idea: all nodes that were fixed in the previous model are now replaced by the displacements from the 
    #       registration
    #
    
    fixedPoints     = []
    dispVectsUpdate = []
    
    for n in prevFixedNodes:
        fixedPoints.append( breastVolMeshPoints[n,:] )
        curIDX = np.array( np.round( np.dot( dispAffine, np.hstack( ( breastVolMeshPoints[n,:], 1 ) ) ) ), dtype = np.int )
        dispVectsUpdate.append( np.array( ( dispData[curIDX[0], curIDX[1], curIDX[2], 0, 0], 
                                            dispData[curIDX[0], curIDX[1], curIDX[2], 0, 1], 
                                            dispData[curIDX[0], curIDX[1], curIDX[2], 0, 2] ) ) )
    
    # compose the displacement as a simple addition
    dispVects = updateFactor * np.array( dispVectsUpdate ) + dispVects
    
    gen2 = xGen.xmlModelGenrator( breastVolMeshPoints / 1000., breastVolMeshCells[ : , 1:5], 'T4' )
    
    gen2.setDifformDispConstraint( 'RAMP', prevFixedNodes, dispVects / 1000. )
    #gen2.setMaterialElementSet( 'NH', 'FAT',    [  400, 50000], allElemenstArray    ) # homogeneous material for debugging only
    gen2.setMaterialElementSet( 'NH', 'FAT',    [  150, 50000], matGen.fatElemetns    )
    gen2.setMaterialElementSet( 'NH', 'SKIN',   [ 1600, 50000], matGen.skinElements   )
    gen2.setMaterialElementSet( 'NH', 'GLAND',  [  300, 50000], matGen.glandElements  )
    gen2.setMaterialElementSet( 'NH', 'MUSCLE', [  800, 50000], matGen.muscleElements )
    
    gen2.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray, 'RAMP' )
    gen2.setOutput( 5000, 'U' )
    gen2.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=50, hgKappa=0.05, density=1000 )    
    xmlFileOut = meshDir + 'modelD' + str( '%03i' %i ) + '.xml'
    gen2.writeXML( xmlFileOut )
    
    
    
    
#
# Plan:
#  1) Extract the contact surface between chest wall and breast tissue from label image
#     - Original chest wall and deformed PM need to be combined
#  2) Generate a mesh from this 
#  3) Let the breast tissue slide on this (this model generation can be taken from previous tests) 
#  4) 
#
    
    
    
    
    
    
    
    
    
        