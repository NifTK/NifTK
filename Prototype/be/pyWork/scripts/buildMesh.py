#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' This script builds meshes from segmented volume data
    MeshLab is used to refine and improve the results 
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
import vtk2stl

# starting from the images
# 1) soft tissue (currently seen as homogeneous material... to be coorected later on)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland-clippedShoulder-pad.nii
#    -> intensity value 255
#
# 2) chest wall (fixed undeformable structure)
#    -> W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii
#    -> intensity value 255
#

#meshDir               =  'W:/philipsBreastProneSupine/Meshes/meshImprovement/'
breastTissueMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/proneMaskMuscleFatGland2-cleaned-pad.nii'

chestWallMaskImage    = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh.nii'
breastTissueMaskImage = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad.nii'
meshDir               = 'W:/philipsBreastProneSupine/Meshes/meshMaterials2/'
origWorkDir           = os.getcwd()
os.chdir( meshDir )

#xmlFileOut     = meshDir + 'model1.xml'
#xmlDispFileOut = meshDir + 'modelDisp.xml'

chestWallSurfMeshSmesh = meshDir + 'chestWallSurf.smesh'
chestWallSurfMeshVTK   = meshDir + 'chestWallSurf.vtk'
breastSurfMeshSmesh    = meshDir + 'breastSurf.smesh'
breastSurfMeshVTK      = meshDir + 'breastSurf.vtk'

#
# parameters used for meshImprovement
#
# Smeshing parameters which should work for both meshes (chest contact and soft tissue)...
#medSurferParms  = ' -presmooth '
medSurferParms  = ' -iso 160 '      
medSurferParms += ' -df 0.8 '       
medSurferParms += ' -shrink 3 3 3 ' 
#medSurferParms += ' -niter 50 '
#medSurferParms += ' -bandw 2 '

#
# Parameters used for meshMaterial
#
medSurferParms  = ' -iso 80 '      
medSurferParms += ' -df 0.8 '       
medSurferParms += ' -shrink 2 2 2 '
medSurferParms += ' -presmooth'
medSurferParms += ' -niter 40'




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
#tetVolParams = ' -pq1.41OK ' + breastSurfMeshSmesh
#cmdEx.runCommand( 'tetgen', tetVolParams )

# gain access to the output vtk file.
#vtkFileList = fc.getFiles( meshDir, '*1.vtk' )

#if len(vtkFileList) != 1 :
#    print( 'Warning: Did not find exactly one file ending with 1.vtk !' )

#vmrBreast      = vmr.vtkMeshFileReader( vtkFileList[0] )
#minXCoordinate = np.min( vmrBreast.points[:,0] )



# convert the breast and chest wall mesh to stl files for modification
vtk2stl.vtk2stl([chestWallSurfMeshVTK, breastSurfMeshVTK])

####################################################################
# folder: meshImprovement
# The resulting files were improved using MeshLab 
# - Filters -> Remeshing -> Uniform Mesh Resampling -> 2% Precision
# - Filters -> Smoothing -> Laplacian Smoothing -> 2 Steps
#

####################################################################
# folder: meshMaterials
# The resulting files were improved using MeshLab 
# - Filters -> Remeshing -> Uniform Mesh Resampling -> 2mm Precision, offset 0.5mm
# - Filters -> Smoothing -> Laplacian Smoothing -> 3 Steps
#


# go back to where you belong...
os.chdir( origWorkDir )
