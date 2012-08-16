#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import meshStatistics 
import nodesAndElementsFromVTKFile as ndsAndEls
import vtk

def scaleVTKMesh( vtkFileIn, vtkFileOut, scaleFactor ):
    ''' Meshes need to be unstructured grids
    '''
    
    nAe   = ndsAndEls.nodesAndElementsFromVTKFile( vtkFileIn )
    stats = meshStatistics.meshStatistics( nAe.meshPoints * scaleFactor, nAe.meshCells )
    
    w = vtk.vtkUnstructuredGridWriter()
    w.SetInput( stats.unstructuredGrid )
    w.SetFileName( vtkFileOut )
    w.Update()
        
