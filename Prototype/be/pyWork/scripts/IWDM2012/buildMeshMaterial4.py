#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' Explore a promising way to build an even more reliable mesh 
'''

from mayaviPlottingWrap import plotArrayAs3DPoints, plotArraysAsMesh, plotVectorsAtPoints
import os
import smeshFileReader as smr
import commandExecution as cmdEx
import vtk2stl
import stlBinary2stlASCII

chestWallMaskImage     = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh.nii'
breastTissueMaskImage  = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM2_Crp2-pad.nii'
breastTissueMaskImage2 = 'W:/philipsBreastProneSupine/ManualSegmentation/FatGlandTissueMask2_Crp2_pad.nii'  # fat gland skin only

mlxDir                 = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/'
meshDir                = 'W:/philipsBreastProneSupine/Meshes/meshMaterials4/'

origWorkDir            = os.getcwd()
os.chdir( meshDir )


chestWallSurfMeshSmesh = meshDir + 'chestWallSurf.smesh'
chestWallSurfMeshVTK   = meshDir + 'chestWallSurf.vtk'
breastSurfMeshSmesh    = meshDir + 'breastSurf.smesh'
breastSurfMeshVTK      = meshDir + 'breastSurf.vtk'
breastSurfMeshSmesh2   = meshDir + 'breastSurf2.smesh'
breastSurfMeshVTK2     = meshDir + 'breastSurf2.vtk'


# Parameters used for directory meshMaterials4
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


# Build the soft tissue mesh 2:
medSurfBreastParams2  = ' -img '  + breastTissueMaskImage2 
medSurfBreastParams2 += ' -surf ' + breastSurfMeshSmesh2
medSurfBreastParams2 += ' -vtk '  + breastSurfMeshVTK2
medSurfBreastParams2 += medSurferParms

cmdEx.runCommand( 'medSurfer', medSurfBreastParams2 )

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


# convert the breast and chest wall mesh to stl files for modification
vtk2stl.vtk2stl([chestWallSurfMeshVTK, breastSurfMeshVTK, breastSurfMeshVTK2])

####################################################################
# folder: meshImprovement
# The resulting files were improved using MeshLab 
# - Filters -> Remeshing -> Uniform Mesh Resampling -> 2mm Precision, offset 0.5mm
# - Filters -> Smoothing -> Laplacian Smoothing -> 3 Steps
# - Filters -> Iso Parametrisation -> 140 / 180 / Best Heuristic / 10 / true
# - Filters -> Iso Parametrisation Remeshing -> 8  
#
# These actions are summarised in the script 
# -> surfProcessing.mlx
#
# The above can now be done in an automated way by using meshlabserver and an appropriate script file 
# Only the output of meshlabserer needs to be converted from binary format to ascii format 
#

#
# Improve the mesh quality for the breast 
#

# some file names
breastSurfBaseName     = breastSurfMeshVTK.split('.')[0] 
breastSurfMeshSTL      = breastSurfBaseName + '.stl' 
improBreastSurfMeshSTL = breastSurfBaseName + '_impro.stl'

if not os.path.exists( breastSurfMeshSTL ) :
    print('ERRROR: Breast surface stl file does not exist.')
    exit()
    
# run meshlab improvements 

meshLabCommand         = 'meshlabserver'
meshlabScript          = mlxDir + 'surfProcessing.mlx'
meshLabParamrs         = ' -i ' + breastSurfMeshSTL
meshLabParamrs        += ' -o ' + improBreastSurfMeshSTL
meshLabParamrs        += ' -s ' + meshlabScript

cmdEx.runCommand( meshLabCommand, meshLabParamrs )
stlBinary2stlASCII.stlBinary2stlASCII( improBreastSurfMeshSTL )

# run tetgen and build the volume mesh
tetVolParams = ' -pq1.42a10K ' + improBreastSurfMeshSTL
cmdEx.runCommand( 'tetgen', tetVolParams )

#
# Build the 2nd mesh needs to be coarser as niftySim crashes otherwise
# some file names
#
breastSurfBaseName2     = breastSurfMeshVTK2.split('.')[0] 
breastSurfMeshSTL2      = breastSurfBaseName2 + '.stl' 
improBreastSurfMeshSTL2 = breastSurfBaseName2 + '_impro.stl'

if not os.path.exists( breastSurfMeshSTL2 ) :
    print('ERRROR: Breast surface stl file does not exist.')
    exit()
    
# run meshlab improvements 
meshlabScriptCoarse    = mlxDir + 'surfProcessing_coarse.mlx'
meshLabParamrs         = ' -i ' + breastSurfMeshSTL2
meshLabParamrs        += ' -o ' + improBreastSurfMeshSTL2
#meshLabParamrs        += ' -s ' + meshlabScript
meshLabParamrs        += ' -s ' + meshlabScriptCoarse

cmdEx.runCommand( meshLabCommand, meshLabParamrs )

# convert the output file to ASCII format
stlBinary2stlASCII.stlBinary2stlASCII( improBreastSurfMeshSTL2 )

# build the volume mesh
#tetVolParams = ' -pq1.42a50K ' + improBreastSurfMeshSTL2 # changed this to coarser setting 50 instead of 10
tetVolParams = ' -pq1.42a100K ' + improBreastSurfMeshSTL2
cmdEx.runCommand( 'tetgen', tetVolParams )

# go back to where you belong...
os.chdir( origWorkDir )










