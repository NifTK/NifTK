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
import vtk2stl
import stlBinary2stlASCII
from vtk.util import numpy_support as VN
import getNodesCloseToMask as ndProx
import materialSetGenerator
import commandExecution as cmdEx
import f3dRegistrationTask as f3dTask
import feirRegistrationTask as feirTask
import os, sys
import vtkVolMeshHandler as vmh
import modelDeformationHandler as mDefH




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
FEIRmode                  = 'fast'
FEIRmu                    = 0.0025 * 2 ** -8
FEIRlambda                = 0.0
FEIRconvergence           = 0.01
FEIRplanStr               = 'n'

updateFactor              = 1.0
numIterations             = 1

meshDir                   = 'W:/philipsBreastProneSupine/Meshes/meshMaterials4/'
mlxDir                    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/'
regDirF3D                 = meshDir + 'regF3D/'
regDirFEIR                = meshDir + 'regFEIR/'
breastVolMeshName         = meshDir + 'breastSurf_impro.1.vtk'    # volume mesh    
breastVolMeshName2        = meshDir + 'breastSurf2_impro.1.vtk'    # volume mesh for sliding par    
xmlFileOut                = meshDir + 'model.xml'

chestWallMaskImage        = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh.nii'
chestWallMaskImageDilated = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh-dilateR2I4.nii'
labelImage                = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM2_Crp2-pad.nii'
skinMaskImage             = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-AirThresh-dilateR2I4.nii'

# original images
strProneImg               = 'W:/philipsBreastProneSupine/proneCrop2Pad-zeroOrig.nii'
strSupineImg              = 'W:/philipsBreastProneSupine/rigidAlignment/supine1kTransformCrop2Pad_zeroOrig.nii'

#
# Make sure the registration dirs exists 
#
if not os.path.exists( regDirF3D ) :
    os.mkdir( regDirF3D )
if not os.path.exists( regDirFEIR ) :
    os.mkdir( regDirFEIR )




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


#
# Handle the breast volume mesh
#
breastMesh  = vmh.vtkVolMeshHandler( breastVolMeshName )
breastMesh2 = vmh.vtkVolMeshHandler( breastVolMeshName2 )


# calculate the material parameters (still to be improved)
# this only needs to run once...
if not 'matGen2' in locals():
    matGen2 = materialSetGenerator.materialSetGenerator( breastMesh2.volMeshPoints, 
                                                         breastMesh2.volMeshCells, 
                                                         labelImage, 
                                                         skinMaskImage, 
                                                         breastMesh2.volMesh, 
                                                         95, 105, 180, 3 ) # fat, gland, muscle, number tet-nodes to be surface element
if not 'matGen' in locals():
    matGen = materialSetGenerator.materialSetGenerator( breastMesh.volMeshPoints, 
                                                        breastMesh.volMeshCells, 
                                                        labelImage, 
                                                        skinMaskImage, 
                                                        breastMesh.volMesh, 
                                                        95, 105, 180, 3 ) # fat, gland, muscle, number tet-nodes to be surface element


plotArrayAs3DPoints( matGen.skinElementMidPoints, (0., 1., 0.) )


# find those nodes of the model, which are close to the sternum... i.e. low x-values
# and those nodes of the model, which are close to mid-axillary line. i.e. high y-values
minXCoordinate = np.min( breastMesh.volMeshPoints[:,0] )
deltaX = 5

maxYCoordinate = np.max( breastMesh.volMeshPoints[:,1] )
deltaY = deltaX

lowXPoints  = []
lowXIdx     = []
highYPoints = []
highYIdx    = []

for i in range( breastMesh.volMeshPoints.shape[0] ):
    if breastMesh.volMeshPoints[i,0] < ( minXCoordinate + deltaX ) :
        lowXIdx.append( i )
        lowXPoints.append( [breastMesh.volMeshPoints[i,0], breastMesh.volMeshPoints[i,1], breastMesh.volMeshPoints[i,2] ] )
    
    if breastMesh.volMeshPoints[i,1] > ( maxYCoordinate - deltaY ) :
        highYIdx.append( i )
        highYPoints.append( [breastMesh.volMeshPoints[i,0], breastMesh.volMeshPoints[i,1], breastMesh.volMeshPoints[i,2] ] )
    
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
( ptsCloseToChest, idxCloseToChest ) = ndProx.getNodesWithtinMask( chestWallMaskImageDilated, 200, 
                                                                   breastMesh.volMeshPoints, 
                                                                   breastMesh.surfMeshPoints )
plotArrayAs3DPoints( ptsCloseToChest, (1.0,1.0,1.0) )


# This little helper array is used for gravity load and material definition
allNodesArray    = np.array( range( breastMesh.volMeshPoints.shape[0] ) )
allElemenstArray = np.array( range( breastMesh.volMeshCells.shape[0]  ) )

#
# Generate the xml-file
#
genFix = xGen.xmlModelGenrator(  breastMesh.volMeshPoints / 1000., breastMesh.volMeshCells[ : , 1:5], 'T4' )

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
genFix.setMaterialElementSet( 'NH', 'FAT',    [  200, 50000], matGen.fatElemetns    )
genFix.setMaterialElementSet( 'NH', 'SKIN',   [ 2400, 50000], matGen.skinElements   )
genFix.setMaterialElementSet( 'NH', 'GLAND',  [  400, 50000], matGen.glandElements  )
genFix.setMaterialElementSet( 'NH', 'MUSCLE', [  600, 50000], matGen.muscleElements )

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

#
# remember the output images
#
strSimulatedSupine         = []
strSimulatedSupineLabelImg = []
strOutDVFImg               = []

i=0

while True :
#for i in range( numIterations ) :
    
    #
    # run the simulation and the resampling
    #
    
    # xml model and generated output image
    strSimulatedSupine.append(         meshDir + 'out'      + str('%03i' %i) + '.nii' )
    strSimulatedSupineLabelImg.append( meshDir + 'outLabel' + str('%03i' %i) + '.nii' )
    strOutDVFImg.append(               meshDir + 'outDVF'   + str('%03i' %i) + '.nii' )

    # run the simulation and resampling at the same time
    simCommand = 'niftkDeformImageFromNiftySimulation'
    simParams   = ' -x '    + xmlFileOut 
    simParams  += ' -i '    + strProneImg
    simParams  += ' -o '    + strSimulatedSupine[-1]

    # also deform the label image!
    simParams  += ' -iL '    + labelImage
    simParams  += ' -oL '    + strSimulatedSupineLabelImg[-1]
    simParams  += ' -oD '    + strOutDVFImg[-1]
    
    
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
    # Check if simulation diverged 
    #
    simSup = nib.load( strSimulatedSupine[-1] )
    
    if np.min( simSup.get_data() ) == np.max( simSup.get_data() ):
        
        # remove those files which do not describe a valid 
        print('No more simulation results after %i iterations' % i)
        
        simSup = nib.load( strSimulatedSupine[-2] )
        strSimulatedSupine.pop( -1 )
        strSimulatedSupineLabelImg.pop( -1 )
        strOutDVFImg.pop(-1)
        break  
    
    
    if i >= numIterations :
        break
    
    #
    # Do the registration step
    #
    if useFEIR :
        feirReg = feirTask.feirRegistrationTask( strSimulatedSupine[-1], strSupineImg, regDirFEIR, 'NA', 
                                                 mu=FEIRmu, lm=FEIRlambda, mode=FEIRmode, mask=True, 
                                                 displacementConvergence=FEIRconvergence, planStr=FEIRplanStr)
    
        feirReg.run()
        feirReg.constructNiiDeformationFile()
        feirReg.resampleSourceImage()
        dispImg = nib.load( feirReg.dispFieldITK )
        
    else :
        f3dReg = f3dTask.f3dRegistrationTask( strSimulatedSupine[-1], strSupineImg, strSimulatedSupine[-1], regDirF3D, 'NA', 
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
        fixedPoints.append( breastMesh.volMeshPoints[n,:] )
        curIDX = np.array( np.round( np.dot( dispAffine, np.hstack( ( breastMesh.volMeshPoints[n,:], 1 ) ) ) ), dtype = np.int )
        dispVectsUpdate.append( np.array( ( dispData[curIDX[0], curIDX[1], curIDX[2], 0, 0], 
                                            dispData[curIDX[0], curIDX[1], curIDX[2], 0, 1], 
                                            dispData[curIDX[0], curIDX[1], curIDX[2], 0, 2] ) ) )
    
    # compose the displacement as a simple addition
    dispVects = updateFactor * np.array( dispVectsUpdate ) + dispVects
    
    gen2 = xGen.xmlModelGenrator( breastMesh.volMeshPoints / 1000., breastMesh.volMeshCells[ : , 1:5], 'T4' )
    
    gen2.setDifformDispConstraint( 'RAMP', prevFixedNodes, dispVects / 1000. )
    #gen2.setMaterialElementSet( 'NH', 'FAT',    [  400, 50000], allElemenstArray    ) # homogeneous material for debugging only
    gen2.setMaterialElementSet( 'NH', 'FAT',    [  200, 50000], matGen.fatElemetns    )
    gen2.setMaterialElementSet( 'NH', 'SKIN',   [ 2400, 50000], matGen.skinElements   )
    gen2.setMaterialElementSet( 'NH', 'GLAND',  [  400, 50000], matGen.glandElements  )
    gen2.setMaterialElementSet( 'NH', 'MUSCLE', [  600, 50000], matGen.muscleElements )
    
    gen2.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray, 'RAMP' )
    gen2.setOutput( 5000, 'U' )
    gen2.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=50, hgKappa=0.05, density=1000 )    
    xmlFileOut = meshDir + 'modelD' + str( '%03i' %i ) + '.xml'
    gen2.writeXML( xmlFileOut )
    
    i = i + 1

fixedPoints = np.array( fixedPoints )
#
# Plan:
#  1) Extract the contact surface between pectoral muscle and breast tissue from label image
#     - Original chest wall and deformed PM need to be combined
#     - 
#  2) Generate a mesh from this 
#  3) Let the breast tissue slide on this (this model generation can be taken from previous tests) 
#  4) 

# Image that will be deformed and thresholded to extract the sliding surface...
modLabelImage             = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM3_Crp2-pad.nii'
defChestWallImg           = meshDir + 'deformedChest.nii' 
defChestWallSurfMeshVTK   = meshDir + 'defChestWallSurf.vtk'
defChestWallSurfMeshSmesh = meshDir + 'defChestWallSurf.smesh'

deformParams  = ' -i '   + modLabelImage
deformParams += ' -o '   + defChestWallImg
deformParams += ' -interpolate lin '
deformParams += ' -def ' + strOutDVFImg[-1] # use the second to last image

cmdEx.runCommand( 'niftkValidateDeformationVectorField', deformParams )

#
# threshold the image
#
threshParams  = ' -i ' + defChestWallImg
threshParams += ' -o ' + defChestWallImg
threshParams += ' -u 255 ' 
threshParams += ' -l 170 ' 
threshParams += ' -in 255 ' 
threshParams += ' -out 0 ' 

cmdEx.runCommand( 'niftkThreshold', threshParams )

#
# Crop the outside to avoid filling holes in the mesh...
#
cropParams = defChestWallImg + ' ' + defChestWallImg + ' 220 270 190 0 '
padParams  = defChestWallImg + ' ' + defChestWallImg + ' 230 280 200 0 '
cmdEx.runCommand( 'niftkPadImage', cropParams )
cmdEx.runCommand( 'niftkPadImage', padParams  )


#
# give the deformed image to medsurfer
# - parameters same as those used in buildMeshMaterial4
#



# Parameters used for directory meshMaterials4
medSurferParms  = ' -iso 80 '      
medSurferParms += ' -df 0.85 '     #was 0.8  
medSurferParms += ' -shrink 4 4 4 ' # was 2 2 2 
medSurferParms += ' -presmooth '
medSurferParms += ' -niter 80 ' 
#medSurferParms += ' -postsmooth ' # added to avoid meshlab


# Build the chest wall mesh (contact constraint):
medSurfCWParams  = ' -img '   + defChestWallImg 
medSurfCWParams += ' -vtk '   + defChestWallSurfMeshVTK
medSurfCWParams += ' -surf '  + defChestWallSurfMeshSmesh
medSurfCWParams += medSurferParms

cmdEx.runCommand( 'medSurfer', medSurfCWParams )

# get access to the vtk file
pdr = vtk.vtkPolyDataReader()
pdr.SetFileName( defChestWallSurfMeshVTK )
pdr.Update()
chestSurfMesh = pdr.GetOutput()
chestSurfMeshPoints = VN.vtk_to_numpy( chestSurfMesh.GetPoints().GetData() )
chestSurfPolys = VN.vtk_to_numpy( chestSurfMesh.GetPolys().GetData() )
chestSurfPolys = chestSurfPolys.reshape( chestSurfMesh.GetPolys().GetNumberOfCells(),chestSurfPolys.shape[0]/chestSurfMesh.GetPolys().GetNumberOfCells() )



pdr = vtk.vtkPolyDataReader()
pdr.SetFileName( defChestWallSurfMeshVTK )
pdr.Update()

chestSurfMesh = pdr.GetOutput()
chestSurfMeshPoints = VN.vtk_to_numpy( chestSurfMesh.GetPoints().GetData() )
chestSurfPolys = VN.vtk_to_numpy( chestSurfMesh.GetPolys().GetData() )
chestSurfPolys = chestSurfPolys.reshape( chestSurfMesh.GetPolys().GetNumberOfCells(),chestSurfPolys.shape[0]/chestSurfMesh.GetPolys().GetNumberOfCells() )


############################################
# Now build the breast tissue model: 
# - Fat and gland only 
#

# fix points with low x-coordinate
# find those nodes of the model, which are close to the sternum... i.e. low x-values
# and those nodes of the model, which are close to mid-axillary line. i.e. high y-values
minXCoordinate = np.min( breastMesh2.volMeshPoints[:,0] )
deltaX = 5

maxYCoordinate = np.max( breastMesh2.volMeshPoints[:,1] )
deltaY = deltaX

lowXPoints2  = []
lowXIdx2     = []

for i in range( breastMesh2.volMeshPoints.shape[0] ):
    if breastMesh2.volMeshPoints[i,0] < ( minXCoordinate + deltaX ) :
        lowXIdx2.append( i )
        lowXPoints2.append( [breastMesh2.volMeshPoints[i,0], breastMesh2.volMeshPoints[i,1], breastMesh2.volMeshPoints[i,2] ] )
        
lowXPoints2 = np.array( lowXPoints2 )
lowXIdx2    = np.array( lowXIdx2    )

# This little helper array is used for gravity load and material definition
allNodesArray2    = np.array( range( breastMesh2.volMeshPoints.shape[0] ) )
allElemenstArray2 = np.array( range( breastMesh2.volMeshCells.shape[0]  ) )

#
# Slight offset into the negative y-direction is required to prevent surfaces to overlap
#
offsetVal        = -5 # in mm
offsetArray      = np.zeros_like( breastMesh2.volMeshPoints )
offsetArray[:,1] = offsetVal


# Sliding xmlFile generator
genS = xGen.xmlModelGenrator( (breastMesh2.volMeshPoints + offsetArray )/ 1000., breastMesh2.volMeshCells[ : , 1:5], 'T4' )

genS.setMaterialElementSet( 'NH', 'FAT',    [  200, 50000], matGen2.fatElemetns    )
genS.setMaterialElementSet( 'NH', 'SKIN',   [ 2400, 50000], matGen2.skinElements   )
genS.setMaterialElementSet( 'NH', 'GLAND',  [  400, 50000], matGen2.glandElements  )

if matGen2.muscleElements.shape != 0 :
    genS.setMaterialElementSet( 'NH', 'MUSCLE', [  600, 50000], matGen2.muscleElements )

#genS.setContactSurfaceVTKFile(defChestWallSurfMeshVTK, 'T3', allNodesArray2.shape[0] )
genS.setContactSurface( chestSurfMeshPoints[:,0:3] / 1000., chestSurfPolys[ : , 1:4 ], allNodesArray2, 'T3' )

genS.setFixConstraint( lowXIdx2, 0 )
genS.setFixConstraint( lowXIdx2, 2 )

genS.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray2, 'RAMP' )
genS.setOutput( 5000, 'U' )
genS.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=50, hgKappa=0.05, density=1000 )    

xmlFileOut = meshDir + 'modelS.xml'
genS.writeXML( xmlFileOut )





strSimulatedSupine.append(         meshDir + 'outS.nii'      )
strSimulatedSupineLabelImg.append( meshDir + 'outLabelS.nii' )
strOutDVFImg.append(               meshDir + 'outDVFS.nii'   )

# run the simulation and resampling at the same time
simCommand = 'niftkDeformImageFromNiftySimulation'
simParams   = ' -x '    + xmlFileOut 
simParams  += ' -i '    + strProneImg
simParams  += ' -o '    + strSimulatedSupine[-1]
simParams  += ' -offset ' + str('%.3f,%.3f,%.3f' % (0, -offsetVal, 0 ))


# also deform the label image!
simParams  += ' -iL '    + labelImage
simParams  += ' -oL '    + strSimulatedSupineLabelImg[-1]
simParams  += ' -oD '    + strOutDVFImg[-1]


# niftyReg and FEIR use different indicators for "mask"
if useFEIR :
    simParams  += ' -mval -1 ' 
else :
    simParams  += ' -mval 0 '
     
simParams  += ' -interpolate bspl '

# run the simulation
print('Starting niftySimulation')
cmdEx.runCommand( simCommand, simParams )



####################################################
####################################################
####################################################


#
# Check if simulation diverged 
#
simSup = nib.load( strSimulatedSupine[-1] )

if np.min( simSup.get_data() ) == np.max( simSup.get_data() ):

    # remove those files which do not describe a valid 
    print('No more simulation results after %i iterations' % i)
    

    #simSup = nib.load( strSimulatedSupine[-2] )
    strSimulatedSupine.pop( -1 )
    strSimulatedSupineLabelImg.pop( -1 )
    strOutDVFImg.pop(-1)
    sys.exit()  



#
# Do the registration step
#
if useFEIR :
    feirReg = feirTask.feirRegistrationTask( strSimulatedSupine[-1], strSupineImg, regDirFEIR, 'NA', 
                                             mu=FEIRmu, lm=FEIRlambda, mode=FEIRmode, mask=True, 
                                             displacementConvergence=FEIRconvergence, planStr=FEIRplanStr)
    
    feirReg.run()
    feirReg.constructNiiDeformationFile()
    feirReg.resampleSourceImage()
    dispImg = nib.load( feirReg.dispFieldITK )

else :
    f3dReg = f3dTask.f3dRegistrationTask( strSimulatedSupine[-1], strSupineImg, strSimulatedSupine[-1], regDirF3D, 'NA', 
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
#
# remember which nodes (numbers and coordinates) were fixed, so these can be used later on.
# and initialise these with zero (=fixed) displacement
#
# TODO: This is not the zero vector but a combination 

prevDeform = mDefH.modelDeformationHandler( genS )
prevDeform.deformationVectors()

dispVects = prevDeform.deformationVectors( matGen2.skinNodes ) * 1000. # given in m thus multiply by 1000
# but need to be limited to the nodes given only

skinPoints      = []
dispVectsUpdate = []

for n in matGen2.skinNodes:
    skinPoints.append( breastMesh2.volMeshPoints[n,:] )
    curIDX = np.array( np.round( np.dot( dispAffine, np.hstack( ( breastMesh2.volMeshPoints[n,:], 1 ) ) ) ), dtype = np.int )
    dispVectsUpdate.append( np.array( ( dispData[ curIDX[0], curIDX[1], curIDX[2], 0, 0 ], 
                                        dispData[ curIDX[0], curIDX[1], curIDX[2], 0, 1 ], 
                                        dispData[ curIDX[0], curIDX[1], curIDX[2], 0, 2 ] ) ) )

## compose the displacement as a simple addition
dispVects = updateFactor * np.array( dispVectsUpdate ) + dispVects


#
# Same generator as previous one (sliding and gravity) but now with skin displacement
#

# Sliding xmlFile generator
genSD = xGen.xmlModelGenrator( (breastMesh2.volMeshPoints + offsetArray )/ 1000., breastMesh2.volMeshCells[ : , 1:5], 'T4' )

genSD.setMaterialElementSet( 'NH', 'FAT',    [  200, 50000], matGen2.fatElemetns    )
genSD.setMaterialElementSet( 'NH', 'SKIN',   [ 2400, 50000], matGen2.skinElements   )
genSD.setMaterialElementSet( 'NH', 'GLAND',  [  400, 50000], matGen2.glandElements  )

if matGen2.muscleElementMidPoints.shape[0] != 0 :
    genSD.setMaterialElementSet( 'NH', 'MUSCLE', [  600, 50000], matGen2.muscleElements )

genSD.setContactSurface( chestSurfMeshPoints[:,0:3] / 1000., chestSurfPolys[ : , 1:4 ], allNodesArray2, 'T3' )

#genSD.setFixConstraint( lowXIdx2, 0 )
#genSD.setFixConstraint( lowXIdx2, 2 )
genSD.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray2, 'RAMP' )
genSD.setDifformDispConstraint('RAMP', matGen2.skinNodes, dispVects / 1000. )
genSD.setOutput( 5000, 'U' )
genSD.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=50, hgKappa=0.05, density=1000 )    

xmlFileOut = meshDir + 'modelSD.xml'
genSD.writeXML( xmlFileOut )



strSimulatedSupine.append(         meshDir + 'outSD.nii'      )
strSimulatedSupineLabelImg.append( meshDir + 'outLabelSD.nii' )
strOutDVFImg.append(               meshDir + 'outDVFSD.nii'   )

# run the simulation and resampling at the same time
simParams   = ' -x '    + xmlFileOut 
simParams  += ' -i '    + strProneImg
simParams  += ' -o '    + strSimulatedSupine[-1]
simParams  += ' -offset ' + str('%.3f,%.3f,%.3f' % (0, -offsetVal, 0 ))


# also deform the label image!
simParams  += ' -iL '    + labelImage
simParams  += ' -oL '    + strSimulatedSupineLabelImg[-1]
simParams  += ' -oD '    + strOutDVFImg[-1]


# niftyReg and FEIR use different indicators for "mask"
if useFEIR :
    simParams  += ' -mval -1 ' 
else :
    simParams  += ' -mval 0 '
     
simParams  += ' -interpolate bspl '

# run the simulation
print('Starting niftySimulation')
cmdEx.runCommand( simCommand, simParams )
    
    
    
    

    
    
    
    
    