#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import polarisRecordingReader as polRec
import orthogonalProcurstesPointRegistration as oppr
from rigidHelpers import *
from determineEuler import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy import *
from scipy import optimize


# generate the file names
cvsBase  = 'Z:/documents/Project/Seed/VRT_data/vicraRecordings/visioRT_calibratioPlate/pos03_'
csvFiles = []

for i in range( 1, 26 ) :
    csvFiles.append( cvsBase + str('%03d' % i ) + '.csv' )

# extract the recordings
recordings = []

for fileName in csvFiles:
    recordings.append( polRec.polarisRecordingReader( fileName ) )
    
    
# now reformat the points for plotting (later on!)
pointsX = []
pointsY = []
pointsZ = []

for rec in recordings :
    pointsX.append( rec.tools[2].meanTx )
    pointsY.append( rec.tools[2].meanTy )
    pointsZ.append( rec.tools[2].meanTz )



# rearrange into another format
p = array([pointsX, pointsY, pointsZ, ones(25)]).reshape( 4, 25 )

# the desired points are given in the following array
pDest = array([[0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0, 5.0, 7.5, 12.5, 15.0, 20.0, 22.5, 27.5, -5.0, -7.5, -12.5, -15.0, -20.0, -22.5, -27.5 ],
               [0.0, 5.0, 7.5, 12.5, 15.0, 20.0, -5.0, -7.5, -12.5, -15.0, -20.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
               zeros(25), ones(25)])
# convert to mm
pDest[0:3] = 10 * pDest[0:3]


# solve the orthogonal Procurstes problem
procSolver = oppr.orthogonalProcrustesPointRegistration( p, pDest )
procSolver.register()
pReg = dot( procSolver.homRigTransfromMat, p )

# now some errors are interesting
#  - how far away are the transformed points from the desired points?
 
errors = sqrt( np.sum( (pReg - pDest)[0:3,:]**2 ,0) )
((pReg - pDest)[0,:])**2 + ((pReg - pDest)[1,:])**2 + ((pReg - pDest)[2,:])**2
print 'Mean Error: ' + str( np.mean(errors) )


# Now plot some more results...
plt.hold( True )
figReg = plt.figure()
axReg  = figReg.gca( projection = '3d' )

axReg.scatter( pReg[0,:],  pReg[1,:],  pReg[2,:],  marker='o' )
axReg.scatter( pDest[0,:], pDest[1,:], pDest[2,:], marker='^' )

# Some more plotting into the old Graph...

plt.hold( False )
plt.show()


plt.hold( True )
fig = plt.figure()
ax = fig.gca( projection = '3d' )

#mpl.rc( 'text', usetex=True )
ax.scatter( pointsX, pointsY, pointsZ )
ax.scatter( pReg[0,:],  pReg[1,:],  pReg[2,:],  marker='o' )
ax.scatter( pDest[0,:], pDest[1,:], pDest[2,:], marker='^' )
plt.hold( False )
plt.show()




#
# Test section to see whether the angle recovery works all right...
#

rX =  1.2
rY = -0.3
rZ =  0.15

M = rotMatZYX( rX, rY, rZ )


















