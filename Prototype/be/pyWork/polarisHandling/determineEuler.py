#! /usr/bin/env python 
# -*- coding: utf-8 -*-


#
# This method implements the conversion from a rotation matrix R (3x3 array) into the 
# Euler angles which are used by the Polaris Vicra system. 
# 
# Rotation around the x,y and z axis are also named in the vicra software as
# rX: yaw
# rY: pitch
# rZ: roll
#

import math


def determineEuler( rotationMatrixIn ):
    
    R = rotationMatrixIn
    
    rX = 0. # yaw
    rY = 0. # pitch
    rZ = 0. # roll
    
    rZ    = math.atan2( R[1,0], R[0,0] )
    cosRz = math.cos( rZ )
    sinRz = math.sin( rZ )
    
    rY    = math.atan2( -R[2,0], ( cosRz * R[0,0] ) + ( sinRz * R[1,0] ) ) 
    rX    = math.atan2( (sinRz * R[0,2]) - (cosRz * R[1,2]), (-sinRz * R[0,1]) + (cosRz * R[1,1]) )
    
    return rX, rY, rZ