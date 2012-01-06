#! /usr/bin/env python 
# -*- coding: utf-8 -*-




import pointWithinTetrahedron as pInT
import nibabel as nib
import numpy as np
import numpy.linalg as la



class modelToImageTransform :
    
    def __init__ ( self, points, elements, deformationVectors, referenceImageName ):
        
        self.points   = points
        self.elements = elements
        
        self.referenceImage = nib.load( referenceImageName )
        self.refImgData     = self.referenceImage.get_data()
        self.refImgI2X      = self.referenceImage.get_affine()        # 4x4 matrix holding the index-to-point transformation
        
        # rotate the affine transformation
        rotZ90=  np.eye(4)
        rotZ90[0,0] = -1
        rotZ90[1,1] = -1
        self.refImgI2X = np.dot(rotZ90, self.refImgI2X )
        
        self.refImgX2I      = la.inv( self.refImgI2X )
        self.generateMaskImage()
    
    
    
    
    def generateMaskImage( self ) :
        self.maskImgData = np.zeros_like( self.refImgData, np.uint8 )
        
        
        # iterate through the elements and check for every physical point if this 
        # is inside the tetrahedron. If so, set the mask pixel to 255
        
        numElements = self.elements.shape[0]
        
        for i in range( numElements ) :
            
            if ( (i+1) % ( numElements / 100 ) ) == 0 :
                print('... %3i percent done' % ( 100*i/numElements + 1) )
            
            # Get the bounding coordinates of the element            
            p1 = self.points[ self.elements[ i, 0 ], : ]
            p2 = self.points[ self.elements[ i, 1 ], : ]
            p3 = self.points[ self.elements[ i, 2 ], : ]
            p4 = self.points[ self.elements[ i, 3 ], : ]
            
            pts = np.vstack( (p1, p2, p3, p4) )
            mPt = np.min( pts, 0 )  # minimal xyz-coordinate
            MPt = np.max( pts, 0 )  # maximal xyz-coordinate
            
            # convert mPt and MPt into image index
            mIdx = np.array( np.floor( np.dot( self.refImgX2I,np.hstack( (mPt,1) ) ) ), dtype=np.int )   
            MIdx = np.array( np.ceil(  np.dot( self.refImgX2I,np.hstack( (MPt,1) ) ) ), dtype=np.int )
            
            for iZ in range( mIdx[2], MIdx[2] ) : #TODO: check if +1 is necessary
                for iY in range( mIdx[1], MIdx[1] ) :
                    for iX in range( mIdx[0], MIdx[0] ) :
                        curP = np.dot( self.refImgI2X, np.array( (iX, iY, iZ, 1. ) ) )
                        
                        
                        r = pInT.pointWithinTetrahedron( p1, p2, p3, p4, curP[0:3] )
                        
                        if r[0] :
                            self.maskImgData[iX, iY, iZ] = 255
            
        #     
        self.maskImg = nib.nifti1.Nifti1Image( self.maskImgData, self.referenceImage.get_affine() )    
        
