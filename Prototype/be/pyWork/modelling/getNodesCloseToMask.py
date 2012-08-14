#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from numpy.core.fromnumeric import nonzero
import scipy.linalg as linalg



def getNodesCloseToMask( maskImgName, imgThreshold, nodePoints, distance, allowedPoints=None ):
    
    ''' @param maskImgName: Path to nii file name which holds the mask
        @param imgThreshold: Threshold for the mask image
        @param nodePoints: Point coordinates from the mesh model
        @param distance: Tolerable distance between mask and mesh points
        @param allowedPoints: Give a list with allowed points, if the fixed nodes should be restricted
    '''
    
    print('WARNING: If the image data has an offset, then the coordinate transformations are not correct!!! See ToDo!!!')
    
    maskImg = nib.load( maskImgName )
    
    # the spacing matrix
    hdr  = maskImg.get_header()
    qMat = hdr.get_qform()
    
    # Well there is always sth. wrong... medSurfer only uses spacing and not the origin...
    qMat        = np.abs( qMat )
    qMat[0:3,3] = np.zeros(3)
    
    # TODO: Change
    matImgAffine = maskImg.get_affine()
    rot90Z       = np.array(([-1,0,0,0],[0,-1,0,0], [0,0,1,0], [0,0,0,1]))           # for itk written images        
    matXToI      = np.dot( np.linalg.inv( maskImg.get_affine()),    rot90Z )  # matrix which converts the real world coordinate X to the image index I 
    
    
    img         = maskImg.get_data()
    qMatInv     = linalg.inv(qMat)
    
    
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
    foundPoints    = []
    foundPointsIdx = []
    
    for i in range( nodePoints.shape[0] ) :
        pRealWorld = nodePoints[i,:]
        pDiscrete = np.dot( qMatInv[0:3,0:3], pRealWorld ).round()
    
        x = max( min( int( pDiscrete[0] ), img.shape[0]-1 ), 0 )
        z = max( min( int( pDiscrete[2] ), img.shape[2]-1 ), 0 )
        xz = ( x, z )
        
        dist = yMap[xz] - pRealWorld[1] 
        
        if np.abs( dist ) < distance :
            
            if allowedPoints == None :
            
                foundPoints.append(pRealWorld)
                foundPointsIdx.append(i)
            else:
                # before appending, check if they are in the allowed list!
                d=np.tile(pRealWorld, (allowedPoints.shape[0],1) ) - allowedPoints
                minDst = np.min( np.sqrt( (d[:,0]*d[:,0]) + (d[:,1]*d[:,1]) + (d[:,2]*d[:,2]) ) )
                
                if minDst < 1e-5 :
                    foundPoints.append(pRealWorld)
                    foundPointsIdx.append(i)
    

    foundPoints    = np.array( foundPoints    )
    foundPointsIdx = np.array( foundPointsIdx )

    return ( foundPoints, foundPointsIdx )





def getNodesWithtinMask( maskImgName, imgThreshold, nodePoints, allowedPoints=None ):
    
    ''' @param maskImgName: Path to nii file name which holds the mask
        @param imgThreshold: Threshold for the mask image
        @param nodePoints: Point coordinates from the mesh model
        @param allowedPoints: Give a list with allowed points, if the fixed nodes should be restricted
    '''
    
    maskImg = nib.load( maskImgName )
    
    # the spacing matrix
    #hdr  = maskImg.get_header()
    #qMat = hdr.get_qform()
    
    # Well there is always sth. wrong... medSurfer only uses spacing and not the origin...
    # TODO: Find a better way to this solution...
    #qMat        = np.abs( qMat )
    #qMat[0:3,3] = np.zeros(3)
    
    #idxCentre    = np.array( (np.around( np.dot( self.labelXToIMat, np.hstack( ( cdsTetCentre, 1 ) ) ) ) ), dtype=np.int )
    matImgAffine = maskImg.get_affine()
    rot90Z       = np.array(([-1,0,0,0],[0,-1,0,0], [0,0,1,0], [0,0,0,1]))           # for itk written images        
    matXToI      = np.dot( np.linalg.inv( maskImg.get_affine()),    rot90Z )  # matrix which converts the real world coordinate X to the image index I 
    
    img          = maskImg.get_data()
    #qMatInv     = linalg.inv(qMat)
    
    
    # Threshold data
    img[ nonzero( img < 200 ) ] = 0
    
    
    # 
    foundPoints    = []
    foundPointsIdx = []
    
    for i in range( nodePoints.shape[0] ) :
        pRealWorld = nodePoints[i,:]
        
        pDiscrete  = np.array( (np.around( np.dot( matXToI, np.hstack( ( pRealWorld, 1 ) ) ) ) ), dtype=np.int ) 
        #pDiscrete = np.dot( qMatInv[0:3,0:3], pRealWorld ).round()
        
        # sample the mask image at the given discrete point
        curLabel = img[pDiscrete[0], pDiscrete[1], pDiscrete[2]]
        
        if curLabel !=0 :
            
            if allowedPoints == None :
            
                foundPoints.append(pRealWorld)
                foundPointsIdx.append(i)
                
            else:
                # before appending, check if they are in the allowed list!
                d      = np.tile(pRealWorld, (allowedPoints.shape[0],1) ) - allowedPoints
                minDst = np.min( np.sqrt( (d[:,0]*d[:,0]) + (d[:,1]*d[:,1]) + (d[:,2]*d[:,2]) ) )
                
                if minDst < 1e-5 :
                    foundPoints.append( pRealWorld )
                    foundPointsIdx.append(i)
    
    
    foundPoints    = np.array( foundPoints    )
    foundPointsIdx = np.array( foundPointsIdx )

    return ( foundPoints, foundPointsIdx )



