#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np


def pointWithinTetrahedron(t1, t2, t3, t4, p) :
    ''' Implements a test if a given point is within the tetrahedron 
        specified by the four corner points p1...p4.
        @param p1: numpy array holding 3D coordinates of the corner point of the tetrahedron 
        @return: (bollInside, b1, b2, b3, b4) boolen holding information if point is inside and
                 barycentric coordintes
    '''
    
    #
    # Calculate the determinants of the matrices
    #
    
    # convert to homogeneous coordinates
    t1h = np.hstack((t1,1.))
    t2h = np.hstack((t2,1.))
    t3h = np.hstack((t3,1.))
    t4h = np.hstack((t4,1.))
    ph  = np.hstack((p, 1.))
    
    M0 = np.vstack( (t1h, t2h, t3h, t4h) ) 
    M1 = np.vstack( (ph,  t2h, t3h, t4h) ) 
    M2 = np.vstack( (t1h, ph,  t3h, t4h) ) 
    M3 = np.vstack( (t1h, t2h, ph,  t4h) ) 
    M4 = np.vstack( (t1h, t2h, t3h, ph ) ) 
    
    D0 = np.linalg.det(M0)
    D1 = np.linalg.det(M1)
    D2 = np.linalg.det(M2)
    D3 = np.linalg.det(M3)
    D4 = np.linalg.det(M4)
    
    if D0 == 0:
        print('Error: points are coplanar!')
        return
    
    b1 = D1/D0
    b2 = D2/D0
    b3 = D3/D0
    b4 = D4/D0

    isInside = False
    
    if (D0<0) and (D1<0) and (D2<0)and (D3<0) and (D4<0) :
        isInside = True
    
    if (D0>0) and (D1>0) and (D2>0)and (D3>0) and (D4>0) :
        isInside = True
    

    return (isInside, b1, b2, b3, b4)    
