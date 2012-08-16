#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import vtk
from vtk.util import numpy_support as VN



class nodesAndElementsFromVTKFile:
    
    def __init__(self, vtkFileName):
        
        self.vtkFileName = vtkFileName
        
    
        self.ugr = vtk.vtkUnstructuredGridReader()
        self.ugr.SetFileName( self.vtkFileName )
        self.ugr.Update()
        
        self.vtkMesh = self.ugr.GetOutput()
        self.meshPoints = VN.vtk_to_numpy( self.vtkMesh.GetPoints().GetData() ) 
        self.meshCells = VN.vtk_to_numpy( self.vtkMesh.GetCells().GetData() )
        self.meshCells = self.meshCells.reshape( self.vtkMesh.GetNumberOfCells(), 
                                                 self.meshCells.shape[0] / self.vtkMesh.GetNumberOfCells() )
        self.meshCells = self.meshCells[:,1:]    
        
