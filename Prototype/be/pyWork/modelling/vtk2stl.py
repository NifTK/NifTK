import sys
import vtk
#import os
#import tkFileDialog

def vtk2stl(files):
    pdr = vtk.vtkPolyDataReader()

    for f in files:
        print 'Reading VTK data from', f, '...'
        pdr.SetFileName(f)

        clean = vtk.vtkCleanPolyData()
        clean.SetInput(pdr.GetOutput())

        triangles = vtk.vtkTriangleFilter()
        triangles.SetInput(clean.GetOutput())
        
        # convert to STL and save:
        if f.endswith('.vtk'):
            outfile = f[:-3]
        else:
            outfile = f
        outfile = outfile + 'stl'

        print 'Writing STL data to', outfile, '...'

        sw = vtk.vtkSTLWriter()
        sw.SetFileName(outfile)
        sw.SetInput(triangles.GetOutput())
        #sw.SetFileTypeToBinary()
        sw.Write()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        print( 'Give input vtk file' )
        #files = tkFileDialog.askopenfilename(multiple=1)
    vtk2stl(files)