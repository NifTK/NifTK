#! /usr/bin/env python 
# -*- coding: utf-8 -*-

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


meshDir           = 'W:/philipsBreastProneSupine/referenceState/boxModel/meshes/'
simDir            = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/'
boxMeshName       = 'boxT4.vtk'
xmlModelNameT4    = 'boxGravModel_T4.xml'
xmlModelNameT4ANP = 'boxGravModel_T4ANP.xml'

density      = 1000.0
dampingCoeff = 50.
totalTime    = 5.
timeStep     = 1.e-5 
materialType = 'NH'
materialParams = [100.067, 50000]

gravDir = np.array( (0,0,1.0 ) )
gravMag = 10.

numOutput  = 100
outputFreq = int( totalTime / timeStep / numOutput )


ugr = vtk.vtkUnstructuredGridReader()
ugr.SetFileName( meshDir + boxMeshName )
ugr.Update()

boxVolMesh = ugr.GetOutput()
boxVolMeshPoints = VN.vtk_to_numpy( boxVolMesh.GetPoints().GetData() )
boxVolMeshCells = VN.vtk_to_numpy( boxVolMesh.GetCells().GetData() )
boxVolMeshCells = boxVolMeshCells.reshape( boxVolMesh.GetNumberOfCells(),boxVolMeshCells.shape[0]/boxVolMesh.GetNumberOfCells() )
boxVolMeshCells = boxVolMeshCells[:,1:5]

#mpw.plotArrayAs3DPoints(boxVolMeshPoints * 10., (1.0, 1.0, 1.0) )

mS = meshStats.meshStatistics( boxVolMeshPoints, boxVolMeshCells )
                
#
# lower and upper bounds
#

boxLowerBounds = np.min( boxVolMeshPoints, axis = 0 )
boxUpperBounds = np.max( boxVolMeshPoints, axis = 0 )

edgeLength = 0.015

boxVolMeshPoints = edgeLength * boxVolMeshPoints + edgeLength/2


deltaX = 1.e-5
deltaY = 1.e-5
deltaZ = 1.e-5
lowXPoints, lowXIdx, highXPoints, highXIdx, lowYPoints, lowYIdx, highYPoints, highYIdx, lowZPoints, lowZIdx, highZPoints, highZIdx = lowAndHighModelCoordinates( boxVolMeshPoints, deltaX, deltaY, deltaZ )

#
# Generate T4 model
#
genT4 = xGen.xmlModelGenrator( boxVolMeshPoints, boxVolMeshCells, 'T4' )

# set the fix constraint
genT4.setFixConstraint( lowZIdx, 0 )
genT4.setFixConstraint( lowZIdx, 1 )
genT4.setFixConstraint( lowZIdx, 2 )

genT4.setMaterialElementSet( materialType, 'Fat', materialParams, genT4.allElemenstArray )

genT4.setGravityConstraint( gravDir, gravMag, genT4.allNodesArray, 'POLY345FLAT4' )
genT4.setSystemParameters( timeStep, totalTime, dampingCoeff, 0, 1000.)
genT4.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genT4.writeXML( simDir + xmlModelNameT4 )

#
# Generate T4ANP model
#
genT4ANP = xGen.xmlModelGenrator( boxVolMeshPoints, boxVolMeshCells, 'T4ANP' )

# set the fix constraint
genT4ANP.setFixConstraint( lowZIdx, 0 )
genT4ANP.setFixConstraint( lowZIdx, 1 )
genT4ANP.setFixConstraint( lowZIdx, 2 )

genT4ANP.setMaterialElementSet( materialType, 'Fat', materialParams, genT4ANP.allElemenstArray )

genT4ANP.setGravityConstraint( gravDir, gravMag, genT4.allNodesArray, 'POLY345FLAT4' )
genT4ANP.setSystemParameters( timeStep, totalTime, dampingCoeff, 0, 1000.)
genT4ANP.setOutput( outputFreq, ['U', 'EKinTotal', 'EStrainTotal' ] )
genT4ANP.writeXML( simDir + xmlModelNameT4ANP )


#
# Critical time step calculation
#
c = cTS.criticalTimeStep( xmlFileName  = simDir + xmlModelNameT4 )

#
# run the simulation and look at the convergence
#
rS.runSimulationsInFolder( simDir )

cA.convergenceAnalyser( simDir + xmlModelNameT4 )
plt.close( 'all' )
cA.convergenceAnalyser( simDir + xmlModelNameT4ANP )
plt.close( 'all' )


