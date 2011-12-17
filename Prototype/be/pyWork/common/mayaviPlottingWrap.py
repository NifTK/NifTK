#! /usr/bin/env python 
# -*- coding: utf-8 -*-



from enthought.mayavi.mlab import points3d, triangular_mesh, quiver3d
import numpy as np

def plotArrayAs3DPoints( arrayIn, colorIn=(.5, .5, .5) ):
    '''
        3D point plotting. numpy arrays expected
    '''
    if arrayIn.shape == (3,) or arrayIn.shape == (4,):
        arrayIn = np.tile(arrayIn, (2,1))
    
    x = arrayIn[:,0]
    y = arrayIn[:,1]
    z = arrayIn[:,2]

    return points3d( x, y, z, color = colorIn, scale_factor=1. )



def plotArraysAsMesh( arrayPointsIn, arrayTrianglesIn ) :
    '''
        3D mesh plotting. numpy arrays expected
    '''
    tri = []
    for i in range( arrayTrianglesIn.shape[0] ) :
        tri.append( ( arrayTrianglesIn[i,0], arrayTrianglesIn[i,1], arrayTrianglesIn[i,2] ) )
    
    
    return triangular_mesh( arrayPointsIn[:,0], arrayPointsIn[:,1], arrayPointsIn[:,2], tri )


def plotVectorsAtPoints( arrayVectors, arrayPoints ):
    '''
        deformation visualisation
    '''
    
    quiver3d( arrayPoints[:,0], arrayPoints[:,1], arrayPoints[:,2], arrayVectors[:,0], arrayVectors[:,1], arrayVectors[:,2],scale_factor=1., mode='arrow', resolution=20, scale_mode = 'vector'  )
    