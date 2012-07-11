#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import vtk
import numpy as np
from vtk.util import numpy_support as VN
import xmlModelGenerator as xGen
from lowAndHighModelCoordinates import lowAndHighModelCoordinates
import runSimulation as rS
import convergenceAnalyser as cA
import matplotlib.pyplot as plt



######################################
# Model parameters used later on...
#
simDirH8               = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/'
simDirT4               = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/'
simDirT4ANP            = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/'

defaultVTKMeshH8       = 'W:/philipsBreastProneSupine/referenceState/boxModel/meshes/boxH8_15.vtk'
defaultVTKMeshT4       = 'W:/philipsBreastProneSupine/referenceState/boxModel/meshes/boxT4_15.vtk'

xmlModelNameGravH8     = 'box_H8_grav.xml'
xmlModelNameDispH8     = 'box_H8_disp.xml'
xmlModelNameForceH8    = 'box_H8_force.xml'

xmlModelNameGravT4     = 'box_T4_grav.xml'
xmlModelNameDispT4     = 'box_T4_disp.xml'
xmlModelNameForceT4    = 'box_T4_force.xml'

xmlModelNameGravT4ANP  = 'box_T4ANP_grav.xml'
xmlModelNameDispT4ANP  = 'box_T4ANP_disp.xml'
xmlModelNameForceT4ANP = 'box_T4ANP_force.xml'

density        = 1000.0
dampingCoeff   = 25.
totalTime      = 5.
timeStep       = 1.e-5 
materialType   = 'NH'
materialParams = [100.067, 50000]

# AB experiments
# materialType = 'AB'
# materialParams = [100.067, 1.5, 50000]

#
# The FEBio equivalent should be  [initial mod. 100.067, links 2.25, bulk mod. 50000]
#

gravDir = np.array( ( 0, 0, 1.0 ) )
gravMag = 10.0

displacementMag = 0.005
hgKappa = 0.075

forceMagX = 0.
forceMagY = 2e-4
forceMagZ = 2e-4

numOutput = 200.

useGPU = True
#
######################################
outputFreq = int( np.ceil( totalTime / timeStep / numOutput ) )


######################################
# Load the meshes from the vtk file
######################################

#
# Load T4 mesh
#
ugrT4 = vtk.vtkUnstructuredGridReader()
ugrT4.SetFileName( defaultVTKMeshT4 )
ugrT4.Update()

boxVolMeshT4       = ugrT4.GetOutput()
boxVolMeshPointsT4 = VN.vtk_to_numpy( boxVolMeshT4.GetPoints().GetData() ) 
boxVolMeshCellsT4  = VN.vtk_to_numpy( boxVolMeshT4.GetCells().GetData() )
boxVolMeshCellsT4  = boxVolMeshCellsT4.reshape( boxVolMeshT4.GetNumberOfCells(),boxVolMeshCellsT4.shape[0] / 
                                                boxVolMeshT4.GetNumberOfCells() )
boxVolMeshCellsT4 = boxVolMeshCellsT4[:,1:9]


#
# Load H8 mesh
#
ugrH8 = vtk.vtkUnstructuredGridReader()
ugrH8.SetFileName( defaultVTKMeshH8 )
ugrH8.Update()

boxVolMeshH8       = ugrH8.GetOutput()
boxVolMeshPointsH8 = VN.vtk_to_numpy( boxVolMeshH8.GetPoints().GetData() ) 
boxVolMeshCellsH8  = VN.vtk_to_numpy( boxVolMeshH8.GetCells().GetData() )
boxVolMeshCellsH8  = boxVolMeshCellsH8.reshape( boxVolMeshH8.GetNumberOfCells(),boxVolMeshCellsH8.shape[0] / 
                                                boxVolMeshH8.GetNumberOfCells() )
boxVolMeshCellsH8 = boxVolMeshCellsH8[:,1:9]




################################
# Prepare  boundary conditions
################################

#
# Find model boundaries
#
deltaX = 1.e-5
deltaY = 1.e-5
deltaZ = 1.e-5
lowXPointsT4, lowXIdxT4, highXPointsT4, highXIdxT4, lowYPointsT4, lowYIdxT4, highYPointsT4, highYIdxT4, lowZPointsT4, lowZIdxT4, highZPointsT4, highZIdxT4 = lowAndHighModelCoordinates( boxVolMeshPointsT4, deltaX, deltaY, deltaZ )
lowXPointsH8, lowXIdxH8, highXPointsH8, highXIdxH8, lowYPointsH8, lowYIdxH8, highYPointsH8, highYIdxH8, lowZPointsH8, lowZIdxH8, highZPointsH8, highZIdxH8 = lowAndHighModelCoordinates( boxVolMeshPointsH8, deltaX, deltaY, deltaZ )



###########################
# gravity: T4 T4ANP and H8 
###########################

genGravT4    = xGen.xmlModelGenrator( boxVolMeshPointsT4, boxVolMeshCellsT4, 'T4'    )
genGravT4ANP = xGen.xmlModelGenrator( boxVolMeshPointsT4, boxVolMeshCellsT4, 'T4ANP' )
genGravH8    = xGen.xmlModelGenrator( boxVolMeshPointsH8, boxVolMeshCellsH8, 'H8'    )

# set the fix constraint
genGravH8.setFixConstraint( lowZIdxH8, 0 )
genGravH8.setFixConstraint( lowZIdxH8, 1 )
genGravH8.setFixConstraint( lowZIdxH8, 2 )

genGravT4.setFixConstraint( lowZIdxT4, 0 )
genGravT4.setFixConstraint( lowZIdxT4, 1 )
genGravT4.setFixConstraint( lowZIdxT4, 2 )

genGravT4ANP.setFixConstraint( lowZIdxT4, 0 )
genGravT4ANP.setFixConstraint( lowZIdxT4, 1 )
genGravT4ANP.setFixConstraint( lowZIdxT4, 2 )

# define gravity loading
genGravH8.setGravityConstraint   ( gravDir, gravMag, genGravH8.allNodesArray,    'POLY345FLAT4' )
genGravT4.setGravityConstraint   ( gravDir, gravMag, genGravT4.allNodesArray,    'POLY345FLAT4' )
genGravT4ANP.setGravityConstraint( gravDir, gravMag, genGravT4ANP.allNodesArray, 'POLY345FLAT4' )

# define material
genGravH8.setMaterialElementSet   ( materialType, 'Fat', materialParams, genGravH8.allElemenstArray    )
genGravT4.setMaterialElementSet   ( materialType, 'Fat', materialParams, genGravT4.allElemenstArray    )
genGravT4ANP.setMaterialElementSet( materialType, 'Fat', materialParams, genGravT4ANP.allElemenstArray )


# define system parameters
genGravH8.setSystemParameters   ( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)
genGravT4.setSystemParameters   ( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)
genGravT4ANP.setSystemParameters( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)

genGravH8.setOutput   ( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genGravT4.setOutput   ( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genGravT4ANP.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )

# write the model file
genGravH8.writeXML   ( simDirH8    + xmlModelNameGravH8    )
genGravT4.writeXML   ( simDirT4    + xmlModelNameGravT4    )
genGravT4ANP.writeXML( simDirT4ANP + xmlModelNameGravT4ANP )



###############
# displacement
###############
genDispH8    = xGen.xmlModelGenrator( boxVolMeshPointsH8, boxVolMeshCellsH8, 'H8'    )
genDispT4    = xGen.xmlModelGenrator( boxVolMeshPointsT4, boxVolMeshCellsT4, 'T4'    )
genDispT4ANP = xGen.xmlModelGenrator( boxVolMeshPointsT4, boxVolMeshCellsT4, 'T4ANP' )

# set the fix constraint
genDispH8.setFixConstraint( lowZIdxH8, 0 )
genDispH8.setFixConstraint( lowZIdxH8, 1 )
genDispH8.setFixConstraint( lowZIdxH8, 2 )

genDispT4.setFixConstraint( lowZIdxT4, 0 )
genDispT4.setFixConstraint( lowZIdxT4, 1 )
genDispT4.setFixConstraint( lowZIdxT4, 2 )

genDispT4ANP.setFixConstraint( lowZIdxT4, 0 )
genDispT4ANP.setFixConstraint( lowZIdxT4, 1 )
genDispT4ANP.setFixConstraint( lowZIdxT4, 2 )


# set displacement constraint
genDispH8.setUniformDispConstraint( 0, 'POLY345FLAT4', highZIdxH8, 0.0)
genDispH8.setUniformDispConstraint( 1, 'POLY345FLAT4', highZIdxH8, 0.0)
genDispH8.setUniformDispConstraint( 2, 'POLY345FLAT4', highZIdxH8, displacementMag )

genDispT4.setUniformDispConstraint( 0, 'POLY345FLAT4', highZIdxT4, 0.0)
genDispT4.setUniformDispConstraint( 1, 'POLY345FLAT4', highZIdxT4, 0.0)
genDispT4.setUniformDispConstraint( 2, 'POLY345FLAT4', highZIdxT4, displacementMag )

genDispT4ANP.setUniformDispConstraint( 0, 'POLY345FLAT4', highZIdxT4, 0.0)
genDispT4ANP.setUniformDispConstraint( 1, 'POLY345FLAT4', highZIdxT4, 0.0)
genDispT4ANP.setUniformDispConstraint( 2, 'POLY345FLAT4', highZIdxT4, displacementMag )

genDispH8.setMaterialElementSet   ( materialType, 'Fat', materialParams, genDispH8.allElemenstArray    )
genDispT4.setMaterialElementSet   ( materialType, 'Fat', materialParams, genDispT4.allElemenstArray    )
genDispT4ANP.setMaterialElementSet( materialType, 'Fat', materialParams, genDispT4ANP.allElemenstArray )

genDispH8.setSystemParameters   ( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)
genDispT4.setSystemParameters   ( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)
genDispT4ANP.setSystemParameters( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)


genDispH8.setOutput   ( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genDispT4.setOutput   ( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genDispT4ANP.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )


genDispH8.writeXML   ( simDirH8    + xmlModelNameDispH8 )
genDispT4.writeXML   ( simDirT4    + xmlModelNameDispT4 )
genDispT4ANP.writeXML( simDirT4ANP + xmlModelNameDispT4ANP )



########
# force
########
genForceH8    = xGen.xmlModelGenrator( boxVolMeshPointsH8, boxVolMeshCellsH8, 'H8'    )
genForceT4    = xGen.xmlModelGenrator( boxVolMeshPointsT4, boxVolMeshCellsT4, 'T4'    )
genForceT4ANP = xGen.xmlModelGenrator( boxVolMeshPointsT4, boxVolMeshCellsT4, 'T4ANP' )

# set the fix constraint
genForceH8.setFixConstraint( lowZIdxH8, 0 )
genForceH8.setFixConstraint( lowZIdxH8, 1 )
genForceH8.setFixConstraint( lowZIdxH8, 2 )

genForceT4.setFixConstraint( lowZIdxT4, 0 )
genForceT4.setFixConstraint( lowZIdxT4, 1 )
genForceT4.setFixConstraint( lowZIdxT4, 2 )

genForceT4ANP.setFixConstraint( lowZIdxT4, 0 )
genForceT4ANP.setFixConstraint( lowZIdxT4, 1 )
genForceT4ANP.setFixConstraint( lowZIdxT4, 2 )

# set force constraint
if forceMagX != 0. :
    genForceH8.setUniformForceConstraint( 0, 'POLY345FLAT4', highZIdxH8, forceMagX )
if forceMagY != 0. :
    genForceH8.setUniformForceConstraint( 1, 'POLY345FLAT4', highZIdxH8, forceMagY )
if forceMagZ != 0. :
    genForceH8.setUniformForceConstraint( 2, 'POLY345FLAT4', highZIdxH8, forceMagZ )

if forceMagX != 0. :
    genForceT4.setUniformForceConstraint( 0, 'POLY345FLAT4', highZIdxT4, forceMagX )
if forceMagY != 0. :
    genForceT4.setUniformForceConstraint( 1, 'POLY345FLAT4', highZIdxT4, forceMagY )
if forceMagZ != 0. :
    genForceT4.setUniformForceConstraint( 2, 'POLY345FLAT4', highZIdxT4, forceMagZ )

if forceMagX != 0. :
    genForceT4ANP.setUniformForceConstraint( 0, 'POLY345FLAT4', highZIdxT4, forceMagX )
if forceMagY != 0. :
    genForceT4ANP.setUniformForceConstraint( 1, 'POLY345FLAT4', highZIdxT4, forceMagY )
if forceMagZ != 0. :
    genForceT4ANP.setUniformForceConstraint( 2, 'POLY345FLAT4', highZIdxT4, forceMagZ )

genForceH8.setMaterialElementSet   ( materialType, 'Fat', materialParams, genForceH8.allElemenstArray    )
genForceT4.setMaterialElementSet   ( materialType, 'Fat', materialParams, genForceT4.allElemenstArray    )
genForceT4ANP.setMaterialElementSet( materialType, 'Fat', materialParams, genForceT4ANP.allElemenstArray )

genForceH8.setSystemParameters   ( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)
genForceT4.setSystemParameters   ( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)
genForceT4ANP.setSystemParameters( timeStep, totalTime, dampingCoeff, hgKappa, 1000.)


genForceH8.setOutput   ( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genForceT4.setOutput   ( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genForceT4ANP.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )


genForceH8.writeXML   ( simDirH8    + xmlModelNameForceH8 )
genForceT4.writeXML   ( simDirT4    + xmlModelNameForceT4 )
genForceT4ANP.writeXML( simDirT4ANP + xmlModelNameForceT4ANP )




#
# run the simulation and look at the convergence
#
rS.runSimulationsInFolder( simDirH8,    gpu=useGPU )
rS.runSimulationsInFolder( simDirT4,    gpu=useGPU )
rS.runSimulationsInFolder( simDirT4ANP, gpu=useGPU )

cA.convergenceAnalyser( simDirH8 + xmlModelNameGravH8  )
cA.convergenceAnalyser( simDirH8 + xmlModelNameDispH8  )
cA.convergenceAnalyser( simDirH8 + xmlModelNameForceH8 )
plt.close( 'all' )

cA.convergenceAnalyser( simDirT4 + xmlModelNameGravT4  )
cA.convergenceAnalyser( simDirT4 + xmlModelNameDispT4  )
cA.convergenceAnalyser( simDirT4 + xmlModelNameForceT4 )
plt.close( 'all' )

cA.convergenceAnalyser( simDirT4ANP + xmlModelNameGravT4ANP  )
cA.convergenceAnalyser( simDirT4ANP + xmlModelNameDispT4ANP  )
cA.convergenceAnalyser( simDirT4ANP + xmlModelNameForceT4ANP ) 
plt.close( 'all' )

