#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' This script builds the xml model for prone-to-supine simulation based on 
'''

#import numpy as np

#from numpy.core.fromnumeric import nonzero
#import scipy.linalg as linalg
#import xmlModelGenerator
from mayaviPlottingWrap import plotArrayAs3DPoints, plotArraysAsMesh, plotVectorsAtPoints
import os, sys
#import findExecutable
import smeshFileReader as smr
import vtkMeshFileReader as vmr
import fileCorrespondence as fc 
import numpy as np
import xmlModelGenerator as xGen
import modelDeformationVisualiser as vis
import commandExecution as cmdEx
import imageJmeasurementReader as ijResReader
import nibabel as nib


# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be coorected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

chestWallMaskImage    = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii'
breastTissueMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii'

meshDir     =  'W:/philipsBreastProneSupine/Meshes/004/'
origWorkDir = os.getcwd()
os.chdir( meshDir )

xmlFileOut     = meshDir + 'model1.xml'
xmlDispFileOut = meshDir + 'modelDisp.xml'

chestWallSurfMeshSmesh = meshDir + 'chestWallSurf.smesh'
chestWallSurfMeshVTK   = meshDir + 'chestWallSurf.vtk'
breastSurfMeshSmesh    = meshDir + 'breastSurf.smesh'
breastSurfMeshVTK      = meshDir + 'breastSurf.vtk'


# Smeshing parameters which should work for both meshes (chest contact and soft tissue)...
medSurferParms  = ' -presmooth '
medSurferParms += ' -iso 250 '      
medSurferParms += ' -df 0.95 '       
medSurferParms += ' -shrink 4 4 4 ' 
medSurferParms += ' -niter 50 '
#medSurferParms += ' -bandw 2 '

# Build the soft tissue mesh:
medSurfBreastParams  = ' -img '  + breastTissueMaskImage 
medSurfBreastParams += ' -surf ' + breastSurfMeshSmesh
medSurfBreastParams += ' -vtk '  + breastSurfMeshVTK
medSurfBreastParams += medSurferParms

cmdEx.runCommand( 'medSurfer', medSurfBreastParams )

# Build the chest wall mesh (contact constraint):
medSurfCWParams  = ' -img '  + chestWallMaskImage 
medSurfCWParams += ' -surf ' + chestWallSurfMeshSmesh
medSurfCWParams += ' -vtk '  + chestWallSurfMeshVTK
medSurfCWParams += medSurferParms

cmdEx.runCommand( 'medSurfer', medSurfCWParams )

# gain access to the created meshes
smrBreast = smr.smeshFileReader( breastSurfMeshSmesh    )
smrChest  = smr.smeshFileReader( chestWallSurfMeshSmesh )

plotArrayAs3DPoints( smrBreast.nodes[:,1:4], (1.,0.,0.) )
plotArrayAs3DPoints( smrChest.nodes[:,1:4],  (0.,1.,0.) )

# mesh plotting...
plotArraysAsMesh( smrChest.nodes[:,1:4], smrChest.facets[:,1:4] )


# build the volume mesh for the breast tissue 
# these parameters are pretty much standard, but should work for now
tetVolParams = ' -pq1.41OK ' + breastSurfMeshSmesh
cmdEx.runCommand( 'tetgen', tetVolParams )

# gain access to the output vtk file.
vtkFileList = fc.getFiles( meshDir, '*1.vtk' )

if len(vtkFileList) != 1 :
    print( 'Warning: Did not find exactly one file ending wiht 1.vtk !' )

vmrBreast      = vmr.vtkMeshFileReader( vtkFileList[0] )
minXCoordinate = np.min( vmrBreast.points[:,0] )

# find those nodes of the model, which are close to the sternum... i.e. low x-values
# 9mm should be ok...
deltaX = 9

lowXPoints = []
lowXIdx    = []

for i in range( vmrBreast.points.shape[0] ):
    if vmrBreast.points[i,0] < ( minXCoordinate + deltaX ) :
        lowXIdx.append( i )
        lowXPoints.append( [vmrBreast.points[i,0], vmrBreast.points[i,1], vmrBreast.points[i,2] ] )
    
lowXPoints = np.array( lowXPoints )
lowXIdx    = np.array( lowXIdx    )

print( 'Found %i points within an x range between [ -inf ; %f ]' % (len( lowXIdx ), minXCoordinate + deltaX ) )
#plotArrayAs3DPoints(lowXPoints, ( 0, 0, 1.0 ) )




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
allNodesArray    = np.array( range( vmrBreast.points.shape[0] ) )
allElemenstArray = np.array( range( vmrBreast.cells.shape[0]  ) )

gen = xGen.xmlModelGenrator(  vmrBreast.points / 1000, vmrBreast.cells[ : , 1:5] )

gen.setFixConstraint( lowXIdx, 0 )
gen.setFixConstraint( lowXIdx, 2 )
gen.setMaterialElementSet( 'NH', 'Fat', [500, 50000], allElemenstArray )
gen.setGravityConstraint( [0.3, 1, 0 ], 40., allNodesArray, 'RAMP' )
gen.setOutput( 5000, 'U' )
gen.setSystemParameters( timeStep=1e-6, totalTime=1, dampingCoefficient=100, hgKappa=0.05, density=1000 )
gen.setContactSurface( smrChest.nodes[:,1:4] / 1000, smrChest.facets[ : , 1:4 ], allNodesArray, 'T3' )

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
offsetPix = np.array( [250,80,0] )

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
# Now try the displacement constraint
#

# Precondition: find the node, which is closest to the aleola... (second point pair)
# Find the node number of the volume mesh, which is closest to the point: pointsProne[1]
pointsProne[1]

minDist = 1e10
minIdx  = -1
d = np.tile( pointsPronePrime[1,:], ( vmrBreast.points.shape[0], 1 ) ) - vmrBreast.points

for i in range( vmrBreast.points.shape[0] ) :
    curDist = np.dot( d[i,:], d[i,:] )
    if curDist < minDist:
        minDist = curDist
        minIdx  = i

print('Found node %i (%f.3, %f.3, %f.3) to be close to the point (%f.3, %f.3, %f.3)' % (minIdx, vmrBreast.points[ minIdx, 0 ],
                                                                                                vmrBreast.points[ minIdx, 1 ],
                                                                                                vmrBreast.points[ minIdx, 2 ],
                                                                                                pointsPronePrime[1,0],
                                                                                                pointsPronePrime[1,1],
                                                                                                pointsPronePrime[1,2] ))

# Get the direction of the displacement
dispVec = (pointsSupinePrime - pointsPronePrime)[1,:]


# now set the displacement constraint for this node...
dispNode= np.array( [minIdx] )

dispGen = xGen.xmlModelGenrator(  vmrBreast.points / 1000., vmrBreast.cells[ : , 1:5] )
#dispGen.setDispConstraint(0, 'RAMP', dispNode, dispVec[0]/1000. )
#dispGen.setDispConstraint(1, 'RAMP', dispNode, dispVec[1]/1000. )
#dispGen.setDispConstraint(2, 'RAMP', dispNode, dispVec[2]/1000. )

dispGen.setFixConstraint( lowXIdx, 0 )
dispGen.setFixConstraint( lowXIdx, 2 )
dispGen.setMaterialElementSet( 'NH', 'Fat', [500, 1e7], allElemenstArray )
dispGen.setGravityConstraint( [0.3, 1, 0 ], 100., allNodesArray, 'RAMP' )
dispGen.setOutput( 5000, 'U' )
dispGen.setSystemParameters( timeStep=0.5e-6, totalTime=1, dampingCoefficient=60, hgKappa=0.05, density=1000 )
dispGen.setContactSurface( smrChest.nodes[:,1:4] / 1000, smrChest.facets[ : , 1:4 ], allNodesArray, 'T3' )

dispGen.writeXML( xmlDispFileOut )

print( 'Done...' )



# go back to where you belong...
os.chdir( origWorkDir )


