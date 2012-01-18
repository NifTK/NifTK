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

#chestWallMaskImage     = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM_Crp2-pad-CWThresh.nii'
pectoralMuscMaskImage  = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwM_Crp2-pad.nii' 
#breastTissueMaskImage  = 'W:/philipsBreastProneSupine/ManualSegmentation/CombinedMasks_CwAGFM2_Crp2-pad.nii'
breastTissueMaskImage2 = 'W:/philipsBreastProneSupine/ManualSegmentation/FatGlandTissueMask2_Crp2_pad.nii'  # fat gland skin only

mlxDir                 = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/'
meshDir                = 'W:/philipsBreastProneSupine/Meshes/meshMaterials5/'

origWorkDir            = os.getcwd()
os.chdir( meshDir )


pectWallSurfMeshSmesh  = meshDir + 'pectWallSurf.smesh'
pectWallSurfMeshVTK    = meshDir + 'pectWallSurf.vtk'
breastSurfMeshSmesh2   = meshDir + 'breastSurf2.smesh'
breastSurfMeshVTK2     = meshDir + 'breastSurf2.vtk'


# Parameters used for directory meshMaterials5
medSurferParms  = ' -iso 80 '      
medSurferParms += ' -df 0.8 '       
medSurferParms += ' -shrink 2 2 2 '
medSurferParms += ' -presmooth'
medSurferParms += ' -niter 40'



# Build the soft tissue mesh 2:
medSurfBreastParams2  = ' -img '  + breastTissueMaskImage2 
medSurfBreastParams2 += ' -surf ' + breastSurfMeshSmesh2
medSurfBreastParams2 += ' -vtk '  + breastSurfMeshVTK2
medSurfBreastParams2 += medSurferParms

cmdEx.runCommand( 'medSurfer', medSurfBreastParams2 )

# Build the pectoral muscle interface mesh (contact constraint):
medSurfPWParams  = ' -img '  + pectoralMuscMaskImage 
medSurfPWParams += ' -surf ' + pectWallSurfMeshSmesh
medSurfPWParams += ' -vtk '  + pectWallSurfMeshVTK
medSurfPWParams += medSurferParms

cmdEx.runCommand( 'medSurfer', medSurfPWParams )

# gain access to the created meshes
smrBreast2 = smr.smeshFileReader( breastSurfMeshSmesh2 )
smrPect    = smr.smeshFileReader( pectWallSurfMeshSmesh )

plotArrayAs3DPoints( smrBreast2.nodes[:,1:4], (1.,0.,0.) )
plotArrayAs3DPoints( smrPect.nodes[:,1:4],    (0.,1.,0.) )

# mesh plotting...
plotArraysAsMesh( smrPect.nodes[:,1:4], smrPect.facets[:,1:4] )


# convert the breast and chest wall mesh to stl files for modification
vtk2stl.vtk2stl( [pectWallSurfMeshVTK, breastSurfMeshVTK2] )


#
# Improve the mesh quality for the breast 
#
meshLabCommand         = 'meshlabserver'
meshlabScript          = mlxDir + 'surfProcessing.mlx'
meshlabScriptCoarse    = mlxDir + 'surfProcessing_coarse.mlx'


breastSurfBaseName2     = breastSurfMeshVTK2.split('.')[0] 
breastSurfMeshSTL2      = breastSurfBaseName2 + '.stl' 
improBreastSurfMeshSTL2 = breastSurfBaseName2 + '_impro.stl'

if not os.path.exists( breastSurfMeshSTL2 ) :
    print('ERRROR: Breast surface stl file does not exist.')
    exit()
    
# run meshlab improvements 
meshLabParamrs         = ' -i ' + breastSurfMeshSTL2
meshLabParamrs        += ' -o ' + improBreastSurfMeshSTL2
#meshLabParamrs        += ' -s ' + meshlabScript
meshLabParamrs        += ' -s ' + meshlabScriptCoarse

cmdEx.runCommand( meshLabCommand, meshLabParamrs )

# convert the output file to ASCII format
stlBinary2stlASCII.stlBinary2stlASCII( improBreastSurfMeshSTL2 )

# build the volume mesh
tetVolParams = ' -pq1.42a75K ' + improBreastSurfMeshSTL2 # changed this to coarser setting 50 instead of 10
cmdEx.runCommand( 'tetgen', tetVolParams )

import sys
sys.exit()

#
# Improve the mesh quality for the pectoral muscle interface 
#
pectWallSurfBaseName     = pectWallSurfMeshVTK.split('.')[0] 
pectWallSurfMeshSTL      = pectWallSurfBaseName + '.stl' 
impropectWallSurfMeshSTL = pectWallSurfBaseName + '_impro.stl'

if not os.path.exists( pectWallSurfMeshSTL ) :
    print('ERRROR: Breast surface stl file does not exist.')
    exit()
    
# run meshlab improvements 
meshLabParamrs         = ' -i ' + pectWallSurfMeshSTL
meshLabParamrs        += ' -o ' + impropectWallSurfMeshSTL
#meshLabParamrs        += ' -s ' + meshlabScript
meshLabParamrs        += ' -s ' + meshlabScriptCoarse

cmdEx.runCommand( meshLabCommand, meshLabParamrs )

# convert the output file to ASCII format
stlBinary2stlASCII.stlBinary2stlASCII( impropectWallSurfMeshSTL )



# go back to where you belong...
os.chdir( origWorkDir )














