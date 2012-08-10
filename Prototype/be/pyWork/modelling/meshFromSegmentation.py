#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' Objectives:
    - Skin membrane elements
    - Large model fow
'''

from mayaviPlottingWrap import plotArrayAs3DPoints, plotArraysAsMesh, plotVectorsAtPoints
import os
import smeshFileReader as smr
import commandExecution as cmdEx
import vtk2stl
import stlBinary2stlASCII



class meshFromSegmentation:
    def __init__(self, breastTissueMaskIn, meshDir ):
        
        self.breastTissueMask = breastTissueMaskIn
        self.meshDir          = meshDir 
        #
        # Parameters used for meshMaterial
        #
        self.medSurferParms  = ' -iso 80 '      
        self.medSurferParms += ' -df 0.8 '       
        self.medSurferParms += ' -shrink 2 2 2 '
        self.medSurferParms += ' -presmooth'
        self.medSurferParms += ' -niter 40'
        
        self.chestWallSurfMeshSmesh = self.meshDir + 'chestWallSurf.smesh'
        self.chestWallSurfMeshVTK   = self.meshDir + 'chestWallSurf.vtk'
        self.breastSurfMeshSmesh    = self.meshDir + 'breastSurf.smesh'
        self.breastSurfMeshVTK      = self.meshDir + 'breastSurf.vtk'
        
        self.meshLabCommand         = 'meshlabserver'
    
    
    
    
    def generateBreastMesh( self, mlxFile ):

        # Build the soft tissue mesh:
        origWorkDir           = os.getcwd()
        os.chdir( self.meshDir )
        
        # Run medsurfer for the surface extraction
        self.medSurfBreastParams  = ' -img '  + self.breastTissueMask 
        self.medSurfBreastParams += ' -surf ' + self.breastSurfMeshSmesh
        self.medSurfBreastParams += ' -vtk '  + self.breastSurfMeshVTK
        self.medSurfBreastParams += self.medSurferParms
        
        cmdEx.runCommand( 'medSurfer', self.medSurfBreastParams )
        
        # convert the breast and chest wall mesh to stl files for modification
        vtk2stl.vtk2stl( [self.breastSurfMeshVTK] )
        
        if os.path.exists( mlxFile ):
            #
            # Improve the mesh quality 
            #
            
            breastSurfBaseName          = self.breastSurfMeshVTK.split('.')[0] 
            self.breastSurfMeshSTL      = breastSurfBaseName + '.stl' 
            self.improBreastSurfMeshSTL = breastSurfBaseName + '_impro.stl'
            
            if not os.path.exists( self.breastSurfMeshSTL ) :
                print('ERRROR: Breast surface stl file does not exist.')
                return
            
            
            # run meshlab improvements 
            self.meshLabParamrs         = ' -i ' + self.breastSurfMeshSTL
            self.meshLabParamrs        += ' -o ' + self.improBreastSurfMeshSTL
            self.meshLabParamrs        += ' -s ' + mlxFile

            cmdEx.runCommand( self.meshLabCommand, self.meshLabParamrs )
            
            # convert the output file to ASCII format
            stlBinary2stlASCII.stlBinary2stlASCII( self.improBreastSurfMeshSTL )
            
            
            # build the volume mesh
            tetVolParams = ' -pq1.41a50K ' + self.improBreastSurfMeshSTL
            cmdEx.runCommand( 'tetgen', tetVolParams )


        
        os.chdir( origWorkDir )





if __name__ == '__main__':
    
    meshDir          = 'W:/philipsBreastProneSupine/Meshes/meshMaterials6Supine/'
    meshlabScript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_smooth6.mlx'
    breastTissueMask = 'W:/philipsBreastProneSupine/SegmentationSupine/segmOutChestPectMuscFatGland_voi.nii'
    
    meshGenerator = meshFromSegmentation( breastTissueMask, meshDir )
    meshGenerator.generateBreastMesh( meshlabScript )


