# rotation functions 

import numpy as np
import math

# generate the rotation matrix around the x-axis
rotMatX = lambda thetaX : np.array( [ [ 1.,               0.,                0., 0. ],
                                      [ 0., math.cos(thetaX), -math.sin(thetaX), 0. ],
                                      [ 0., math.sin(thetaX),  math.cos(thetaX), 0. ],
                                      [ 0.,               0.,                0., 1. ] ] )

# generate the rotation matrix around the y-axis
rotMatY = lambda thetaY : np.array( [ [  math.cos(thetaY), 0., math.sin(thetaY), 0. ], 
                                      [                0., 1.,               0., 0. ],
                                      [ -math.sin(thetaY), 0., math.cos(thetaY), 0. ],
                                      [                0., 0.,               0., 1. ] ] )

# generate the rotation matrix around the z-axis
rotMatZ = lambda thetaZ : np.array( [ [ math.cos(thetaZ), -math.sin(thetaZ), 0., 0. ],
                                      [ math.sin(thetaZ),  math.cos(thetaZ), 0., 0. ],
                                      [               0.,                0., 1., 0. ],
                                      [               0.,                0., 0., 1. ] ] )


def testByReference( x ):
    x = x+1
    

# rotate a point x around the x-axis about thetaX
rotX = lambda thetaX, x : np.dot( rotMatX( thetaX ), x )
# rotate a point x around the y-axis about thetaY
rotY = lambda thetaY, x : np.dot( rotMatY( thetaY ), x )
# rotate a point x around the z-axis about thetaZ
rotZ = lambda thetaZ, x : np.dot( rotMatZ( thetaZ ), x )

# combined rotation 
rotXYZ = lambda thetaX, thetaY, thetaZ, x : rotZ( thetaZ, rotY( thetaY, rotX( thetaX, x ) ) )

rotMatZYX = lambda X,Y,Z : np.dot( rotMatZ(Z), np.dot( rotMatY( Y ) , rotMatX( X ) ) )

# translation 
translate = lambda tx, ty, tz, x : np.dot( np.array( [ [1., 0., 0., tx],
                                                       [0., 1., 0., ty],
                                                       [0., 0., 1., tz],
                                                       [0., 0., 0., 1.] ] ), x )
rigid = lambda rX, rY, rZ, tx, ty, tz, x: translate( tx, ty, tz, rotZ( rZ, rotY( rY, rotX( rX, x)  ) ) )



def getViosionRTCalibrationCoordinates() :
    return np.array( [ [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0, 5.0, 7.5, 12.5, 15.0, 20.0, 22.5, 27.5, -5.0, -7.5, -12.5, -15.0, -20.0, -22.5, -27.5 ],
                       [0.0, 5.0, 7.5, 12.5, 15.0, 20.0, -5.0, -7.5, -12.5, -15.0, -20.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       np.zeros(25), np.ones(25)] )

## optimisation remnants which were never used
# for the optimisation (finding rotation and translation parameters) the errorfunction holds the 2-norm 
#fitfunc = lambda p, x : rigid( p[0], p[1], p[2], p[3], p[4], p[5], x )
#errfunc = lambda p, x, xDest : fitfunc( p, x ) - xDest 

#iniParams = [0.,0.,0.,0.,0.,0.]

#endParams, success = optimize.leastsq( errfunc, iniParams[:], args=( p, pDest ) )


