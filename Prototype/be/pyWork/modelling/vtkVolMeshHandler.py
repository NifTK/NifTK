#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import vtk
from vtk.util import numpy_support as VN

class vtkVolMeshHandler :
    ''' On initialisation the given mesh will be read, the points and mesh cells
        are stored as member variables and the surface is extracted.
    '''
    
    def __init__( self, vtkVolumeMeshName ):
        
        #
        # Handle the breast volume mesh
        #
        ugr = vtk.vtkUnstructuredGridReader()
        ugr.SetFileName( vtkVolumeMeshName )
        ugr.Update()
        
        # Get the volume mesh
        self.volMesh = ugr.GetOutput()
        self.volMeshPoints = VN.vtk_to_numpy( self.volMesh.GetPoints().GetData() )
        self.volMeshCells = VN.vtk_to_numpy( self.volMesh.GetCells().GetData() )
        self.volMeshCells = self.volMeshCells.reshape( self.volMesh.GetNumberOfCells(), self.volMeshCells.shape[0] / self.volMesh.GetNumberOfCells() )
        
        # Get the surface from the volume mesh
        surfaceExtractor = vtk.vtkDataSetSurfaceFilter()
        surfaceExtractor.SetInput( self.volMesh )
        surfaceExtractor.Update()
        self.surfMeshPoints = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPoints().GetData() )
