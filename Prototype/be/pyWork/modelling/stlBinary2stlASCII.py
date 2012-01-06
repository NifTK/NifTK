#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import vtk

def stlBinary2stlASCII( fileNameIn, fileNameOut=None ):
    ''' @summary: Converts a binary stl mesh file into its ascii verision. 
        @param fileNameIn: Binary stl input file
        @param fileNameOut: ASCII stl output file (optinal). If not given, then given input file will be replaced. 
    '''    

    
    if fileNameOut == None :
        fileNameOut=fileNameIn
    
    # read the input file
    reader = vtk.vtkSTLReader()
    reader.SetFileName( fileNameIn )
    reader.Update()
    
    # write the output 
    writer = vtk.vtkSTLWriter()
    writer.SetInput( reader.GetOutput() )
    writer.SetFileName( fileNameOut )
    writer.SetFileTypeToASCII()
    writer.Update()
    
