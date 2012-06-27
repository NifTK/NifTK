#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import xmlModelReader as xRead
import vtk
import numpy as np
from vtk.util import numpy_support as VN
import xmlModelGenerator as xGen
import mayaviPlottingWrap as mpw
import meshStatistics as meshStats
from lowAndHighModelCoordinates import lowAndHighModelCoordinates
import runSimulation as rS
import criticalTimeStep as cTS
import convergenceAnalyser as cA
import matplotlib.pyplot as plt


######################################
# Model parameters used later on...
#
meshDir          = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/'
defaultVTKMesh   = 'W:/philipsBreastProneSupine/referenceState/boxModel/meshes/boxH8_150.vtk'
xmlModelNameGrav = 'box_H8_grav.xml'
xmlModelNameDisp = 'box_H8_disp.xml'

density        = 1000.0
dampingCoeff   = 5.
totalTime      = 5.
timeStep       = 1.e-5 
materialType   = 'NH'
materialParams = [100.067, 50000]

gravDir = np.array( ( 0, 0, 1.0 ) )
gravMag = 4.0

numOutput = 100
#
######################################

outputFreq = int( totalTime / timeStep / numOutput )


ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( defaultVTKMesh )
ugr.Update()

boxVolMesh = ugr.GetOutput()
boxVolMeshPoints = VN.vtk_to_numpy( boxVolMesh.GetPoints().GetData() )
boxVolMeshCells = VN.vtk_to_numpy( boxVolMesh.GetCells().GetData() )
boxVolMeshCells = boxVolMeshCells.reshape( boxVolMesh.GetNumberOfCells(),boxVolMeshCells.shape[0]/boxVolMesh.GetNumberOfCells() )

#
# Find model boundaries
#
deltaX = 1.e-5
deltaY = 1.e-5
deltaZ = 1.e-5
lowXPoints, lowXIdx, highXPoints, highXIdx, lowYPoints, lowYIdx, highYPoints, highYIdx, lowZPoints, lowZIdx, highZPoints, highZIdx = lowAndHighModelCoordinates( boxVolMeshPoints, deltaX, deltaY, deltaZ )


genGrav = xGen.xmlModelGenrator( boxVolMeshPoints, boxVolMeshCells[:,1:9], 'H8' )

# set the fix constraint
genGrav.setFixConstraint( lowZIdx, 0 )
genGrav.setFixConstraint( lowZIdx, 1 )
genGrav.setFixConstraint( lowZIdx, 2 )

# define material
genGrav.setMaterialElementSet( materialType, 'Fat', materialParams, genGrav.allElemenstArray )

# define gravity loading
genGrav.setGravityConstraint( gravDir, gravMag, genGrav.allNodesArray, 'POLY345FLAT4' )

# define system parameters
genGrav.setSystemParameters( timeStep, totalTime, dampingCoeff, 0, 1000.)
genGrav.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )

# write the model file
genGrav.writeXML( meshDir + xmlModelNameGrav )






genDisp = xGen.xmlModelGenrator( boxVolMeshPoints, boxVolMeshCells[:,1:9], 'H8' )

# set the fix constraint
genDisp.setFixConstraint( lowZIdx, 0 )
genDisp.setFixConstraint( lowZIdx, 1 )
genDisp.setFixConstraint( lowZIdx, 2 )

genDisp.setUniformDispConstraint( 0, 'POLY345FLAT4', highZIdx, 0.0)
genDisp.setUniformDispConstraint( 1, 'POLY345FLAT4', highZIdx, 0.0)
genDisp.setUniformDispConstraint( 2, 'POLY345FLAT4', highZIdx, 0.005)

genDisp.setMaterialElementSet( materialType, 'Fat', materialParams, genDisp.allElemenstArray )

genDisp.setSystemParameters( timeStep, totalTime, dampingCoeff, 0, 1000.)
genDisp.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genDisp.writeXML( meshDir + xmlModelNameDisp )



#
# run the simulation and look at the convergence
#
rS.runSimulationsInFolder( meshDir )

cA.convergenceAnalyser( meshDir + xmlModelNameGrav )
cA.convergenceAnalyser( meshDir + xmlModelNameDisp )

plt.close( 'all' )




