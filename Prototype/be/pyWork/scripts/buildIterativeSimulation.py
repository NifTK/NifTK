#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' This script builds the xml model for prone-to-supine simulation based on 
'''

from mayaviPlottingWrap import plotArrayAs3DPoints, plotArraysAsMesh, plotVectorsAtPoints
import os
import smeshFileReader as smr
import vtkMeshFileReader as vmr
import fileCorrespondence as fc 
import numpy as np
import xmlModelGenerator as xGen
import modelDeformationVisualiser as vis
import commandExecution as cmdEx

regenerateObjects = True

# Prepare iterations
chestWallMaskImage    = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii'
breastTissueMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii'

meshDir     =  'W:/philipsBreastProneSupine/Meshes/005/'
logFileName = meshDir + 'log.txt'
origWorkDir = os.getcwd()
os.chdir( meshDir )

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

if not regenerateObjects :
    cmdEx.runCommand( 'medSurfer', medSurfBreastParams, logFileName )

# Build the chest wall mesh (contact constraint):
medSurfCWParams  = ' -img '  + chestWallMaskImage 
medSurfCWParams += ' -surf ' + chestWallSurfMeshSmesh
medSurfCWParams += ' -vtk '  + chestWallSurfMeshVTK
medSurfCWParams += medSurferParms

if not regenerateObjects :
    cmdEx.runCommand( 'medSurfer', medSurfCWParams, logFileName )

# gain access to the created meshes
smrBreast = smr.smeshFileReader( breastSurfMeshSmesh    )
smrChest  = smr.smeshFileReader( chestWallSurfMeshSmesh )

#plt.hold( True  )
plotArrayAs3DPoints( smrBreast.nodes[:,1:4], (1.,0.,0.) )
plotArrayAs3DPoints( smrChest.nodes[:,1:4],  (0.,1.,0.) )
#plt.hold( False )

# mesh plotting...
plotArraysAsMesh( smrChest.nodes[:,1:4], smrChest.facets[:,1:4] )


# build the volume mesh for the breast tissue 
# these parameters are pretty much standard, but should work for now
tetVolParams = ' -pq1.41OK ' + breastSurfMeshSmesh
if not regenerateObjects :
    cmdEx.runCommand( 'tetgen', tetVolParams, logFileName )

# gain access to the output vtk file.
vtkFileList = fc.getFiles( meshDir, '*1.vtk' )

if len(vtkFileList) != 1 :
    print( 'Warning: Did not find exactly one file ending with 1.vtk !' )

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
plotArrayAs3DPoints(lowXPoints, ( 0, 0, 1.0 ) )




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

gens = [] # xmlGenerators
viss = [] # visualisers

for i in range( 1, 14 ) :
    
    strIDX = str('%03i' % i)
    print('-->Running iteration ' + strIDX )
    
    if i == 1:
        gen = xGen.xmlModelGenrator(  vmrBreast.points / 1000, vmrBreast.cells[ : , 1:5] )
    else: 
        gen = xGen.xmlModelGenrator(  viss[-1].deformedNodes[-1], vmrBreast.cells[ : , 1:5] ) 

    gen.setFixConstraint( lowXIdx, 0 )
    gen.setFixConstraint( lowXIdx, 2 )
    gen.setMaterialElementSet( 'NH', 'Fat', [500, 50000], allElemenstArray )
    gen.setGravityConstraint( [0.3, 1, 0 ], 10., allNodesArray, 'RAMP' )
    gen.setOutput( 5000, 'U' )
    gen.setSystemParameters( timeStep=0.5e-5, totalTime=1, dampingCoefficient=100, hgKappa=0.05, density=1000 )
    gen.setContactSurface( smrChest.nodes[:,1:4] / 1000, smrChest.facets[ : , 1:4 ], allNodesArray, 'T3' )
    xmlFileOut = meshDir + 'model' + strIDX + '.txt'
    
    gen.writeXML( xmlFileOut )
    
    # remember what you did...
    gens.append( gen )
    
    # run the simulation
    niftySimParams = ' -x ' + xmlFileOut + ' -v -sport '
    if not regenerateObjects :
        cmdEx.runCommand( 'niftySimPR', niftySimParams, logFileName  )
    
        # the quirky bit about niftysim: all the deformations are called U.txt
        if not os.path.exists( meshDir + 'U.txt'):
            print('Error: Deformation file was not created...')
    
            
        os.rename( meshDir + 'U.txt', meshDir + 'U' + strIDX + '.txt' )
    
    visualiser = vis.modelDeformationVisualiser( gen, 'U' + strIDX + '.txt' )
    viss.append( visualiser )
    
print( 'Done...' )


os.chdir( origWorkDir )
