#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author:  Bjoern Eiben
@summary: Read a file generated with tetgen and modifiy it, so that it does no longer show a wrong behaviour 
          in paraview (I suspect the elments are numbered in a wrong way. ) 
'''


import vtkMeshFileReader as vmr
import numpy as np

from numpy.core.fromnumeric import nonzero
import scipy.linalg as linalg
import xmlModelGenerator
from mayaviPlottingWrap import plotArrayAs3DPoints



#strFileName = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/proneMaskMuscleFatGland.1.vtk'
strFileName = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/002/smesh-shoulderClipped.1.vtk'
vtkFileOut  = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/002/smesh-shoulderClipped-adapted.vtk'
xmlFileOut  = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/002/model.xml'


strFileName = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/003/smesh-shoulderClipped.1.vtk'
vtkFileOut  = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/003/smesh-shoulderClipped-adapted.vtk'
xmlFileOut  = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/003/model.xml'


#strFileName = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/004/breast.1.vtk'
#vtkFileOut  = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/004/breast.vtk'
#xmlFileOut  = 'Z:/documents/Project/philipsBreastProneSupine/Meshes/004/model.xml'



reader = vmr.vtkMeshFileReader( strFileName )

# presumably the numbering is wrong...
reader.cells[:,1:5] = reader.cells[:,1:5] - np.min( reader.cells[:,1:5] )

# Write the file:
reader.writeToFile( vtkFileOut )


# Well it's nice to look at the data...
plotArrayAs3DPoints( reader.points )




#
# Now read the image data and determine which nodes need to be fixed (first boundary condition to test...)
#

import nibabel as nib
maskImg = nib.load('Z:/documents/Project/philipsBreastProneSupine/ManualSegmentation/CombinedMasksCropped-pad.nii')

# the spacing matrix
hdr  = maskImg.get_header()
qMat = hdr.get_qform()

# Well there is always sth. wrong... medSurfer only uses spacing and not the origin...
qMat = np.abs( qMat )
qMat[0:3,3]=np.zeros(3)

img  = maskImg.get_data()
qMatInv = linalg.inv(qMat)


# Threshold data
img[ nonzero( img < 200 ) ] = 0

# create an image which holds the y-border values (min y coordinate)
yMap = np.zeros( ( img.shape[0], img.shape[2] ) )

for x in range( img.shape[0] ) :
    for z in range( img.shape[2] ):
        #(a,b) = nonzero( img[x,:,z] > 128 )
        a = nonzero( img[x,:,z] > 128 )
        
        if a[0].size != 0 :
            # get the minimal entry which is different from zero
            yMap[x,z] = np.min( a[0] ) * qMat[1,1]
        else :
            # else set to maximum value
            yMap[x,z] = ( img.shape[1] + 5 ) * qMat[1,1]

# 
chestPoints    = []
chestPointsIdx = []

for i in range( reader.points.shape[0] ) :
    pRealWorld = reader.points[i,:]
    pDiscrete = np.dot( qMatInv[0:3,0:3], pRealWorld ).round()

    x = max( min( int( pDiscrete[0] ), img.shape[0]-1 ), 0 )
    z = max( min( int( pDiscrete[2] ), img.shape[2]-1 ), 0 )
    xz = ( x, z )
    
    dist = yMap[xz] - pRealWorld[1] 
    
    if np.abs( dist ) < 5 :
        chestPoints.append(pRealWorld)
        chestPointsIdx.append(i)

chestPoints    = np.array( chestPoints    )
chestPointsIdx = np.array( chestPointsIdx )

#plotArrayAs3DPoints( chestPoints )


# This little helper array is used for gravity load and material definition
allNodesArray    = np.array( range( reader.points.shape[0] ) )
allElemenstArray = np.array( range( reader.cells.shape[0]  ) )

gen = xmlModelGenerator.xmlModelGenrator( reader.points / 1000, reader.cells[:,1:5], 'T4ANP' )
# start with a homogeneous material
gen.setFixConstraint( chestPointsIdx, 0 )
gen.setFixConstraint( chestPointsIdx, 1 )
gen.setFixConstraint( chestPointsIdx, 2 )
gen.setMaterialElementSet( 'NH', 'Fat', [100, 50000], allElemenstArray )
gen.setGravityConstraint( [0.3, 1, 0 ], 10., allNodesArray, 'RAMP' )
gen.setOutput( 100000, 'U' )
gen.setSystemParameters( 1e-5, 5, 100, 0.05, 1000 )
gen.setContactSurface( chestPoints, chestPoints, chestPoints, 'TX' )
gen.writeXML( xmlFileOut )


print( 'Done...' )


