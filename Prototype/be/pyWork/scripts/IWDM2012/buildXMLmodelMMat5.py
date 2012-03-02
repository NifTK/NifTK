#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' This script builds the xml model for prone-to-supine simulation based on 
    meshes which were constructed with the script build Mesh.py 
'''

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
import pointWithinTetrahedron as pInTet
from glob import glob
from maskFromSurface import maskFromSurface



fastDebug                 = False
useFEIR                   = False
FEIRmode                  = 'standard'
FEIRmu                    = 0.0025 * 2 ** -7.5
FEIRlambda                = 0.0
FEIRconvergence           = 0.005
FEIRplanStr               = 'n'

F3DbendingEnergy1          = 0.02
F3DlogOfJacobian1          = 0.0
F3DfinalGridSpacing1       = 3
F3DnumberOfLevels1         = 4
F3DmaxIterations1          = 300
F3Dgpu1                    = False

F3DbendingEnergy2          = 0.01
F3DlogOfJacobian2          = 0.01
F3DfinalGridSpacing2       = 10
F3DnumberOfLevels2         = 3
F3DmaxIterations2          = 300
F3Dgpu2                    = False

#matModel                  = 'NH'
#matFatParams              = [  500, 50000 ]
#matGlandParams            = [  750, 50000 ]
#matMuscleParams           = [ 1000, 50000 ]
#matSkinParams             = [ 2500, 50000 ]

# Specify the material
matSkinModel              = 'NH'
matSkinParams             = [ 1000, 50000 ]
matSkinViscoNumIsoTerms   = 0
matSkinViscoNumVolTerms   = 0
matSkinViscoParams        = []

matFatModel               = 'NHV'
matFatParams              = [ 150, 50000 ]
matFatViscoNumIsoTerms    = 1
matFatViscoNumVolTerms    = 0
matFatViscoParams         = [1.0, 0.25]

matGlandModel             = 'NHV'
matGlandParams            = [ 300, 50000 ]
matGlandViscoNumIsoTerms  = 1
matGlandViscoNumVolTerms  = 0
matGlandViscoParams       = [1.0, 0.25]

matMuscleModel            = 'NH'
matMuscleParams           = [  800, 50000 ]
matMuscleViscoNumIsoTerms = 0
matMuscleViscoNumVolTerms = 0
matMuscleViscoParams      = []



updateFactor              = 0.5
numIterations             = 5

meshDir                   = 'W:/philipsBreastProneSupine/Meshes/meshMaterials5/'
mlxDir                    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/'
regDirF3D                 = meshDir + 'regF3D/'
regDirFEIR                = meshDir + 'regFEIR/'
breastVolMeshName         = meshDir + 'breastSurf_impro.1.vtk'     # volume mesh    
breastVolMeshName2        = meshDir + 'breastSurf2_impro.1.vtk'    # volume mesh for sliding par    
pectSurfMeshName          = meshDir + 'pectWallSurf_impro.stl'  
defPectSurfMeshName       = pectSurfMeshName.split('.')[0] + '_def.stl' 
defPectSurfMaskImgName    = pectSurfMeshName.split('.')[0] + '_def.nii' 
xmlFileOut                = meshDir + 'model.xml'
logFileOut                = meshDir + 'log.txt'

TREtrack                  = []

#chestWallMaskImage        = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh.nii'
#chestWallMaskImageDilated = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh-dilateR2I4.nii'
labelImage                = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM2_Crp2-pad.nii'
modLabelImage             = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM3_Crp2-pad.nii'
pronePectMuscleMaskImage  = ''
skinMaskImage             = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-AirThresh-dilateR2I4.nii'
pectoralMuscleMaskImage   = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM3_Crp2-pad-PMThresh'


# original images
strProneImg               = 'W:/philipsBreastProneSupine/proneCrop2Pad-zeroOrig.nii'
strSupineImg              = 'W:/philipsBreastProneSupine/rigidAlignment/supine1kTransformCrop2Pad_zeroOrig.nii'

# msaked images
strMaskedSupineImg       = meshDir + 'supineMasked.nii' 


# Make sure the registration directories exists 
if not os.path.exists( regDirF3D ) :
    os.mkdir( regDirF3D )
if not os.path.exists( regDirFEIR ) :
    os.mkdir( regDirFEIR )

print( 'Parameters selected: ' )
print( ' - Fat parameters:    ' + matFatModel + ', ' + str( matFatParams    ) )
print( ' - Gland parameters:  ' + str( matGlandParams  ) )
print( ' - Muscle parameters: ' + str( matMuscleParams ) )
print( ' - Skin parameters:   ' + str( matSkinParams   ) )

if useFEIR == False : 
    print( 'Warning: All files in the reigstration directory will be deleted by running this script!\n -> ' + regDirF3D )

####################################
#
# Evaluate the TRE (preparations)
#
breastMesh2 = vmh.vtkVolMeshHandler( breastVolMeshName2 )
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
labelImg       = nib.load( labelImage )
affineTrafoMat = labelImg.get_affine()

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

plotArrayAs3DPoints( breastMesh2.volMeshPoints, (0.5,0.5,0.5) )
plotVectorsAtPoints( pointsSupinePrime - pointsPronePrime, pointsPronePrime )

# the startTRE is 
startTRE = []

for i in range( pointsPronePrime.shape[0] ):
    startTRE.append( np.linalg.norm( pointsSupinePrime[i] - pointsPronePrime[i] ) )

TREtrack.append( startTRE )
print( 'TRE: '         )
print( str( TREtrack ) )


#
# Find the elements in which the selected points lie, needs to be done only once!
#
elementNumbers  = []
baryCoords      = []
euklideanCoords = [] 
pointNums       = []

for p in pointsPronePrime :
    # elements of the model -> get coordinates
    for elNum in range( breastMesh2.volMeshCells.shape[0] ):
        (n,aNum,bNum,cNum,dNum) = breastMesh2.volMeshCells[elNum,:]
        aCds =  breastMesh2.volMeshPoints[aNum,:]
        bCds =  breastMesh2.volMeshPoints[bNum,:]
        cCds =  breastMesh2.volMeshPoints[cNum,:]
        dCds =  breastMesh2.volMeshPoints[dNum,:]
        
        (isInside, aBarCds, bBarCds, cBarCds, dBarCds) = pInTet.pointWithinTetrahedron(aCds, bCds, cCds, dCds, p)
        
        if isInside:
            print('Found point ' + str(p) +' in element number ' + str(elNum) )#
            elementNumbers.append( elNum )
            baryCoords.append( np.array( (aBarCds, bBarCds, cBarCds, dBarCds) ) )
            euklideanCoords.append( np.array( (aCds, bCds, cCds, dCds) ) )
            pointNums.append( np.array( (aNum, bNum, cNum, dNum) ) )
            break


#
# register ... to get the sliding surface
#
if useFEIR :
    print('Starting FEIR registration')
    strProneChestWallMuscleImage = 'W:\philipsBreastProneSupine\ManualSegmentation\prone_CwM_-1.nii'
    feirReg = feirTask.feirRegistrationTask( strProneChestWallMuscleImage, strSupineImg, regDirFEIR, 'NA', 
                                             mu=FEIRmu, lm=FEIRlambda, mode=FEIRmode, mask=True, 
                                             displacementConvergence=FEIRconvergence, planStr=FEIRplanStr)

    feirReg.run()
    feirReg.constructNiiDeformationFile()
    feirReg.resampleSourceImage()
    dispImg = nib.load( feirReg.dispFieldITK )
    regDeformField = feirReg.dispFieldITK
    
else :
    print('Cleaning registration directory')
    
    fileList = glob( regDirF3D + '/*' )
    
    for f in fileList : 
        os.remove( f )

    print('Starting f3d registration')
    strProneChestWallMuscleImage = 'W:\philipsBreastProneSupine\ManualSegmentation\prone_CwM_0.nii'
    f3dReg = f3dTask.f3dRegistrationTask( strProneChestWallMuscleImage, strSupineImg, strProneChestWallMuscleImage, regDirF3D, 'NA', 
                                          bendingEnergy=F3DbendingEnergy1, logOfJacobian=F3DlogOfJacobian1, 
                                          finalGridSpacing=F3DfinalGridSpacing1, numberOfLevels=F3DnumberOfLevels1, 
                                          maxIterations=F3DmaxIterations1, gpu=F3Dgpu1 )
    
    f3dReg.run()
    f3dReg.constructNiiDeformationFile()
    dispImg = nib.load( f3dReg.dispFieldITK )
    regDeformField = f3dReg.dispFieldITK
    

dispData   = dispImg.get_data()
dispAffine = dispImg.get_affine()

dispAffine[0,0] = -dispAffine[0,0] # Correct for itk peculiarity
dispAffine[1,1] = -dispAffine[1,1]
 
dispAffine = np.linalg.inv( dispAffine )


#
# now read the pectoral muscle surface mesh and adapt points according to the registration.
#
print( 'Deforming pectoral muscle surface' )
pectSurfReader = vtk.vtkSTLReader()
pectSurfReader.SetFileName( pectSurfMeshName )
pectSurfReader.Update()

pectSurfMesh = pectSurfReader.GetOutput()
pectSurfMeshPoints = VN.vtk_to_numpy( pectSurfMesh.GetPoints().GetData() )
pectSurfMeshPolys  = VN.vtk_to_numpy( pectSurfMesh.GetPolys().GetData() )
pectSurfMeshPolys  = pectSurfMeshPolys.reshape( pectSurfMesh.GetNumberOfCells(), pectSurfMeshPolys.shape[0] / pectSurfMesh.GetNumberOfCells() )

dispVects = [] # these will be in mm as sampled from registration deformation

for i in range( pectSurfMeshPoints.shape[0] ) :
    curP = pectSurfMeshPoints[ i, : ]
    curIDX = np.array( np.round( np.dot( dispAffine, np.hstack( ( pectSurfMeshPoints[ i, : ], 1 ) ) ) ), dtype = np.int )
    
    if useFEIR:
        dispVects.append( np.array( ( dispData[curIDX[0], curIDX[1], curIDX[2], 0, 0], 
                                      dispData[curIDX[0], curIDX[1], curIDX[2], 0, 1], 
                                      dispData[curIDX[0], curIDX[1], curIDX[2], 0, 2] ) ) )
    else:
        dispVects.append( np.array( ( dispData[curIDX[0], curIDX[1], curIDX[2], 0, 0], 
                                      dispData[curIDX[0], curIDX[1], curIDX[2], 0, 1], 
                                      dispData[curIDX[0], curIDX[1], curIDX[2], 0, 2] ) ) )   
 

dispVects = np.array( dispVects ) 
pectSurfMeshPointsDef = pectSurfMeshPoints + dispVects

# write the deformed surface to stl-file for further use
defPts = vtk.vtkPoints()
defPts.SetData( VN.numpy_to_vtk(pectSurfMeshPointsDef) )

pectSurfMesh.SetPoints( defPts )

stlWriter = vtk.vtkSTLWriter()
stlWriter.SetInput( pectSurfMesh )
stlWriter.SetFileName( defPectSurfMeshName )
stlWriter.Update()

print('Generating Mask image from surface')
surf2mask = maskFromSurface( pectSurfMesh, strSupineImg, 0, 1 )
surf2mask.saveMaskToNii( defPectSurfMaskImgName )


#
# Mask the supine image --> Will be the source/template image for the next registrations...
#
maskingCmd     = 'niftkMultiply'
maskingParams  = ' -i ' + strSupineImg  
maskingParams += ' -j ' + defPectSurfMaskImgName  
maskingParams += ' -o ' + strMaskedSupineImg

cmdEx.runCommand(maskingCmd, maskingParams )




########################################################
#
# Handle the breast volume mesh
#
# calculate the material parameters (still to be improved)
# this only needs to run once...
if not 'matGen2' in locals():
    matGen2 = materialSetGenerator.materialSetGenerator( breastMesh2.volMeshPoints, 
                                                         breastMesh2.volMeshCells, 
                                                         labelImage, 
                                                         skinMaskImage, 
                                                         breastMesh2.volMesh, 
                                                         95, 105, 180, 3 )     # fat, gland, muscle, number tet-nodes to be surface element

strSimulatedSupine         = []
strSimulatedSupineLabelImg = []
strOutDVFImg               = []
xmlGens                    = []

############################################
# Now build the breast tissue model: 
# - Fat and gland only 
# - sliding on pectoral muscle

# fix points with low x-coordinate
# find those nodes of the model, which are close to the sternum... i.e. low x-values
minXCoordinate = np.min( breastMesh2.volMeshPoints[:,0] )
deltaX = 5

maxYCoordinate = np.max( breastMesh2.volMeshPoints[:,1] )

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
offsetVal        = -7 # in mm
offsetArray      = np.zeros_like( breastMesh2.volMeshPoints )
offsetArray[:,1] = offsetVal


# Sliding xmlFile generator
print('Generate FEM model: sliding.')
genS = xGen.xmlModelGenrator( (breastMesh2.volMeshPoints + offsetArray )/ 1000., breastMesh2.volMeshCells[ : , 1:5], 'T4ANP' )

genS.setMaterialElementSet( matFatModel,   'FAT',   matFatParams, matGen2.fatElemetns, 
                            matFatViscoNumIsoTerms, matFatViscoNumVolTerms, matFatViscoParams )
genS.setMaterialElementSet( matSkinModel,  'SKIN',  matSkinParams, matGen2.skinElements, 
                            matSkinViscoNumIsoTerms, matSkinViscoNumVolTerms, matSkinViscoParams )
genS.setMaterialElementSet( matGlandModel, 'GLAND', matGlandParams, matGen2.glandElements,
                            matGlandViscoNumIsoTerms, matGlandViscoNumVolTerms, matGlandViscoParams  )

if matGen2.muscleElements.shape != 0 :
    genS.setMaterialElementSet( matMuscleModel, 'MUSCLE', matMuscleParams, matGen2.muscleElements,
                                matMuscleViscoNumIsoTerms, matMuscleViscoNumVolTerms, matMuscleViscoParams )

genS.setContactSurface( pectSurfMeshPointsDef[:,0:3] / 1000., pectSurfMeshPolys[ : , 1:4 ], allNodesArray2, 'T3' )

genS.setFixConstraint( lowXIdx2, 0 )
genS.setFixConstraint( lowXIdx2, 2 )

genS.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray2, 'RAMP' )
genS.setOutput( 5000, 'U' )
genS.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=75, hgKappa=0.05, density=1000 )    

xmlFileOut = meshDir + 'modelS.xml'
genS.writeXML( xmlFileOut )

xmlGens.append( genS )
strSimulatedSupine.append        ( meshDir + 'outS.nii'      )
strSimulatedSupineLabelImg.append( meshDir + 'outLabelS.nii' )
strOutDVFImg.append              ( meshDir + 'outDVFS.nii'   )


# run the simulation 
simCommand  = 'niftkDeformImageFromNiftySimulation'
simParams   = ' -x '      + xmlFileOut 
simParams  += ' -i '      + strProneImg
simParams  += ' -o '      + strSimulatedSupine[-1]
simParams  += ' -offset ' + str('%.3f,%.3f,%.3f' % (0, offsetVal, 0 ))


# also deform the label image!
simParams  += ' -iL '    + labelImage
simParams  += ' -oL '    + strSimulatedSupineLabelImg[-1]
simParams  += ' -oD '    + strOutDVFImg[-1]
simParams  += ' -interpolate bspl '


# niftyReg and FEIR use different indicators for "mask"
if useFEIR :
    simParams  += ' -mval -1 ' 
else :
    simParams  += ' -mval 0 '
     

# run the simulation
print('Starting niftySimulation')
cmdEx.runCommand( simCommand, simParams, onlyPrintCommand=fastDebug, logFileName=logFileOut )




################################
#
# Check if simulation diverged 
#
simSup = nib.load( strSimulatedSupine[-1] )

if np.min( simSup.get_data() ) == np.max( simSup.get_data() ):

    # remove those files which do not describe a valid 
    print('Sorry, simulation diverged!')
    strSimulatedSupine.pop( -1 )
    strSimulatedSupineLabelImg.pop( -1 )
    strOutDVFImg.pop(-1)
    xmlGens.pop(-1)
    sys.exit()  



#####################
#
# TRE calculations
#
#
# In the deformed mesh (simulation result), the barycentric coordinates are the same, the elements are the same 
# only the points were deformed 
#   - The difference between the pointSupinePrime and the simulated point (can be calculated from baryCds) 
#     is the TRE!
#
deformS = mDefH.modelDeformationHandler( genS )
dNdsS   = deformS.deformedNodes # in m
treS    = []

for i in range( pointsSupinePrime.shape[0] ) :
    pRef = pointsSupinePrime[ i ]
    pSim = ( dNdsS[ pointNums[i][0] ] * baryCoords[i][0] + 
             dNdsS[ pointNums[i][1] ] * baryCoords[i][1] + 
             dNdsS[ pointNums[i][2] ] * baryCoords[i][2] + 
             dNdsS[ pointNums[i][3] ] * baryCoords[i][3] ) * 1000.
    
    treS.append( np.linalg.norm( pRef - pSim ) )

TREtrack.append( treS )
print( 'TRE: '         )
print( str( TREtrack ) )



#######
#
# loop
#
for it in range( numIterations ) :
    #
    # Do the registration step
    #
    if useFEIR :
        print( 'Starting FEIR registration' )
        feirReg = feirTask.feirRegistrationTask( strSimulatedSupine[-1], strMaskedSupineImg, regDirFEIR, 'NA', 
                                                 mu=FEIRmu, lm=FEIRlambda, mode=FEIRmode, mask=True, 
                                                 displacementConvergence=FEIRconvergence, planStr=FEIRplanStr )
    
        feirReg.run()
        feirReg.constructNiiDeformationFile()
        feirReg.resampleSourceImage()
        dispImg = nib.load( feirReg.dispFieldITK )
    
    else :
        print( 'Starting f3d registration' )
        f3dReg = f3dTask.f3dRegistrationTask( strSimulatedSupine[-1], strMaskedSupineImg, '', regDirF3D, 'NA', 
                                              bendingEnergy=F3DbendingEnergy2, logOfJacobian=F3DlogOfJacobian2, 
                                              finalGridSpacing=F3DfinalGridSpacing2, numberOfLevels=F3DnumberOfLevels2, 
                                              maxIterations=F3DmaxIterations2, gpu=F3Dgpu2 )
        
        f3dReg.run()
        f3dReg.constructNiiDeformationFile()
        dispImg = nib.load( f3dReg.dispFieldITK )
    
    
    # read the deformation field
    dispData = dispImg.get_data()
    dispAffine = np.linalg.inv( dispImg.get_affine() )
    dispAffine[0,0] = - dispAffine[0,0]  # quick and dirty
    dispAffine[1,1] = - dispAffine[1,1]
    
    #
    # generate the new model with the updated boundary conditions...
    #
    prevDeform  = mDefH.modelDeformationHandler( xmlGens[-1] )
    dispVects   = prevDeform.deformationVectors( matGen2.skinNodes ) * 1000.       # given in m thus multiply by 1000
    deformedNds = prevDeform.deformedModelNodes() * 1000                           # given in mm
    
    skinPoints      = []
    defSkinPoints   = []
    dispVectsUpdate = []
    dispImgRange    = dispData.shape
    
    for n in matGen2.skinNodes:
        skinPoints.append( breastMesh2.volMeshPoints[n,:] )
        defSkinPoints.append( prevDeform.deformedNodes[n,:] )
        curIDX = np.array( np.round( np.dot( dispAffine, np.hstack( ( deformedNds[n,:], 1 ) ) ) ), dtype = np.int )
        
        # Clip! 
        curIDX[0] = np.max( (curIDX[0], 0                 ) )
        curIDX[0] = np.min( (curIDX[0], dispImgRange[0]-1 ) )
        curIDX[1] = np.max( (curIDX[1], 0                 ) )
        curIDX[1] = np.min( (curIDX[1], dispImgRange[1]-1 ) )
        curIDX[2] = np.max( (curIDX[2], 0                 ) )
        curIDX[2] = np.min( (curIDX[2], dispImgRange[2]-1 ) )
        
        dispVectsUpdate.append( np.array( ( dispData[ curIDX[0], curIDX[1], curIDX[2], 0, 0 ], 
                                            dispData[ curIDX[0], curIDX[1], curIDX[2], 0, 1 ], 
                                            dispData[ curIDX[0], curIDX[1], curIDX[2], 0, 2 ] ) ) )
    
    ## compose the displacement as a simple addition
    dispVects = updateFactor * np.array( dispVectsUpdate ) + dispVects
    
    
    #
    # Same generator as previous one (sliding and gravity) but now with skin displacement
    # TODO: Take care not to fix displacement nodes. Skipped for now!
    #
    
    # Sliding + displacement xmlFile generator
    print('Generate FEM model: sliding and displacement.')
    genSD = xGen.xmlModelGenrator( (breastMesh2.volMeshPoints + offsetArray )/ 1000., breastMesh2.volMeshCells[ : , 1:5], 'T4ANP' )
    
    genSD.setMaterialElementSet( matFatModel,   'FAT',   matFatParams, matGen2.fatElemetns, 
                                 matFatViscoNumIsoTerms, matFatViscoNumVolTerms, matFatViscoParams )
    genSD.setMaterialElementSet( matSkinModel,  'SKIN',  matSkinParams, matGen2.skinElements, 
                                 matSkinViscoNumIsoTerms, matSkinViscoNumVolTerms, matSkinViscoParams )
    genSD.setMaterialElementSet( matGlandModel, 'GLAND', matGlandParams, matGen2.glandElements,
                                 matGlandViscoNumIsoTerms, matGlandViscoNumVolTerms, matGlandViscoParams  )
    
    if matGen2.muscleElements.shape != 0 :
        genSD.setMaterialElementSet( matMuscleModel, 'MUSCLE', matMuscleParams, matGen2.muscleElements,
                                     matMuscleViscoNumIsoTerms, matMuscleViscoNumVolTerms, matMuscleViscoParams )
    
    genSD.setContactSurface( pectSurfMeshPointsDef[:,0:3] / 1000., pectSurfMeshPolys[ : , 1:4 ], allNodesArray2, 'T3' )
    
    #genSD.setFixConstraint( lowXIdx2, 0 )
    #genSD.setFixConstraint( lowXIdx2, 2 )
    genSD.setGravityConstraint( [0., 1, 0 ], 20, allNodesArray2, 'RAMP' )
    genSD.setDifformDispConstraint('RAMP', matGen2.skinNodes, dispVects / 1000. )
    genSD.setOutput( 5000, 'U' )
    genSD.setSystemParameters( timeStep=0.5e-4, totalTime=1, dampingCoefficient=75, hgKappa=0.05, density=1000 )    
    
    xmlFileOut = meshDir + 'modelSD' + str('%02i' % it) + '.xml'
    genSD.writeXML( xmlFileOut )
    
    xmlGens.append( genSD )
    
    strSimulatedSupine.append        ( meshDir + 'outSD'      + str('%02i' % it) + '.nii' )
    strSimulatedSupineLabelImg.append( meshDir + 'outLabelSD' + str('%02i' % it) + '.nii' )
    strOutDVFImg.append              ( meshDir + 'outDVFSD'   + str('%02i' % it) + '.nii' )
    
    # run the simulation and resampling at the same time
    simParams   = ' -x '      + xmlFileOut 
    simParams  += ' -i '      + strProneImg
    simParams  += ' -o '      + strSimulatedSupine[-1]
    simParams  += ' -offset ' + str('%.3f,%.3f,%.3f' % (0, offsetVal, 0 ))
    
    
    # also deform the label image!
    simParams  += ' -iL ' + labelImage
    simParams  += ' -oL ' + strSimulatedSupineLabelImg[-1]
    simParams  += ' -oD ' + strOutDVFImg[-1]
    simParams  += ' -interpolate bspl '
    
    
    # niftyReg and FEIR use different indicators for "mask"
    if useFEIR :
        simParams  += ' -mval -1 ' 
    else :
        simParams  += ' -mval 0 '
         
    
    # run the simulation
    print('Starting niftySimulation')
    cmdEx.runCommand( simCommand, simParams, logFileName=logFileOut )
    
    
    
    
    
    
    #########################
    #
    # TRE calculations
    #
    #
    # In the deformed mesh (simulation result), the barycentric coordinates are the same, the elements are the same 
    # only the points were deformed 
    #   - The difference between the pointSupinePrime and the simulated point (can be calculated from baryCds) 
    #     is the TRE!
    #
    deformSD = mDefH.modelDeformationHandler( genSD )
    dNdsSD       = deformSD.deformedNodes # in m
    
    treSD = []
    
    for i in range( pointsSupinePrime.shape[0] ) :
        pRef = pointsSupinePrime[ i ]
        pSim = ( dNdsSD[ pointNums[i][0] ] * baryCoords[i][0] + 
                 dNdsSD[ pointNums[i][1] ] * baryCoords[i][1] + 
                 dNdsSD[ pointNums[i][2] ] * baryCoords[i][2] + 
                 dNdsSD[ pointNums[i][3] ] * baryCoords[i][3] ) * 1000.
        #
        # WatchOut! offset
        #
        treSD.append( np.linalg.norm( pRef - pSim ) )
    
    TREtrack.append( treSD )
    print( 'TRE: '         )
    print( str( TREtrack ) )

    print( 'Done iteration %i.' %i )



    
    
    
    