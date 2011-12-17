#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from rigidHelpers import *
import cv
import numpy as np

pDest = getViosionRTCalibrationCoordinates()

#
# This is now the manual part:
#  Getting image coordinates according to the real world coordinates 
#
pImg = np.array( [ [ 439, 441, 442, 443, 445, 446, 437, 436, 434, 432, 430, 487, 511, 559, 583, 630, 654, 702, 392, 368, 320, 295, 248, 223, 175 ],
                   [ 358, 313, 290, 247, 225, 184, 403, 428, 475, 501, 551, 359, 360, 362, 362, 364, 364, 366, 357, 356, 354, 354, 352, 351, 350 ],
                   np.zeros(25),np.ones(25)] )

pointCounts = np.array([[25]])

# The image these points were read from is:
imgPath = 'Z:\documents\Project\Seed\calibration\photo.JPG'
img     = cv.LoadImage( imgPath , cv.CV_LOAD_IMAGE_GRAYSCALE )

camMat           = cv.CreateMat( 3, 3, cv.CV_32FC1 )
distortionCoeffs = cv.CreateMat( 4, 1, cv.CV_32FC1 )
rotVecs          = cv.CreateMat( 1, 3, cv.CV_32FC1 )
rotMat           = cv.CreateMat( 3, 3, cv.CV_32FC1 )
transVecs        = cv.CreateMat( 1, 3, cv.CV_32FC1 )
# CalibrateCamera2( objectPoints, imagePoints, pointCounts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs [, flags]) -> None
cv.CalibrateCamera2( cv.fromarray( pDest[0:3,:] ),
                     cv.fromarray( pImg [0:3,:] ), 
                     cv.fromarray( pointCounts ),
                     cv.GetSize(img),
                     camMat,
                     distortionCoeffs,
                     rotVecs,
                     transVecs, cv.CV_CALIB_ZERO_DISPARITY + cv.CV_CALIB_ZERO_TANGENT_DIST + cv.CV_CALIB_FIX_PRINCIPAL_POINT )

cv.Rodrigues2( rotVecs, rotMat )  

# now create the undistorted image to see the effects
imgUnDist = cv.CreateImage( cv.GetSize( img ), img.depth, img.channels )
cv.Undistort2(img, imgUnDist, camMat, distortionCoeffs )
cv.ShowImage( 'original',    img       )
cv.ShowImage( 'undistorted', imgUnDist )
cv.SaveImage( 'Z:\documents\Project\Seed\calibration\photo_undist.JPG', imgUnDist )
print 'Done'
