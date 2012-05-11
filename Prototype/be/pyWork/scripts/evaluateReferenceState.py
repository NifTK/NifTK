#! /usr/bin/env python 
# -*- coding: utf-8 -*- 

import os
import sys
from xmlModelReader import xmlModelReader
from modelDeformationHandler import modelDeformationHandler
import numpy as np
import matplotlib.pyplot as plt
 


phiMax = 40
modDir = 'W:/philipsBreastProneSupine/referenceState/02/'

p1mod        = []
s1mod        = []
p1s2mod      = []
p1def        = []
s1def        = []
p1s2def      = []
p1Reader     = []
s1Reader     = []
p1s2Reader   = []
p1U          = []
s1U          = []
p1s2U        = []
p1meanDisp   = []
s1meanDisp   = []
p1s2meanDisp = []

phis = [] 

for p in range( 0, phiMax+1, 5 ) :
    phis.append(p)
    phi     = str( '%02i' %p )
    
    #
    # Set the correct file names
    #
    p1mod.append(   modDir + 'modelFat_prone1G_phi'         + phi + '.xml' )
    s1mod.append(   modDir + 'modelFat_supine1G_phi'        + phi + '.xml' ) 
    p1s2mod.append( modDir + 'modelFat_prone1Gsupine2G_phi' + phi + '.xml' )
    
    p1def.append(   modDir + 'U_prone1G_phi'         + phi + '.txt' ) 
    s1def.append(   modDir + 'U_supine1G_phi'        + phi + '.txt' ) 
    p1s2def.append( modDir + 'U_prone1Gsupine2G_phi' + phi + '.txt' ) 
    
    #
    # Check file existence
    #
    if not( os.path.exists( p1mod[-1]   ) and 
            os.path.exists( s1mod[-1]   ) and
            os.path.exists( p1s2mod[-1] ) and 
            os.path.exists( p1def[-1]   ) and 
            os.path.exists( s1def[-1]   ) and
            os.path.exists( p1s2def[-1] ) ):
        print "Error: At least one file is missing! Exiting..."
        sys.exit()
    
    #
    # Read the model and the deformations accordingly
    #
    p1Reader.append(   xmlModelReader( p1mod[-1]   ) ) 
    s1Reader.append(   xmlModelReader( s1mod[-1]   ) )
    p1s2Reader.append( xmlModelReader( p1s2mod[-1] ) ) 
    
    p1U.append(   modelDeformationHandler( p1Reader[-1],   p1def[-1]   ) )
    s1U.append(   modelDeformationHandler( s1Reader[-1],   s1def[-1]   ) )
    p1s2U.append( modelDeformationHandler( p1s2Reader[-1], p1s2def[-1] ) )
    
    #
    # Mean deformations
    #
    p1meanDisp.append( np.mean( np.sqrt( p1U[-1].deformVectors[:,0]**2 + 
                                         p1U[-1].deformVectors[:,1]**2 + 
                                         p1U[-1].deformVectors[:,2]**2 ) ) )
    
    s1meanDisp.append( np.mean( np.sqrt( s1U[-1].deformVectors[:,0]**2 + 
                                         s1U[-1].deformVectors[:,1]**2 + 
                                         s1U[-1].deformVectors[:,2]**2 ) ) )
    
    p1s2meanDisp.append( np.mean( np.sqrt( p1s2U[-1].deformVectors[:,0]**2 + 
                                           p1s2U[-1].deformVectors[:,1]**2 + 
                                           p1s2U[-1].deformVectors[:,2]**2 ) ) )
    
    
    
    
p1meanDisp   = np.array( p1meanDisp   )
s1meanDisp   = np.array( s1meanDisp   )
p1s2meanDisp = np.array( p1s2meanDisp )
phis         = np.array(phis)

# Use latex plotting, because it looks so great
plt.rc( 'text', usetex=True )
plt.rcParams['font.size']=16

xLabel = '$\phi \; \mathrm{[^\circ]}$'
yLabel = '$\overline{\|u\|} \; \mathrm{[mm]}$'
p1meanLabel   = '$\overline{\|u_{p}\|}$'
s1meanLabel   = '$\overline{\|u_{s}\|}$'
p1s2meanLabel = '$\overline{\|u_{p2s}\|}$'

# plot 
plt.hold( True )
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.grid(color = 'gray', linestyle='-')

plt.plot( phis, p1meanDisp,   'b-+', label = p1meanLabel )
plt.plot( phis, s1meanDisp,   'r-+', label = s1meanLabel )
plt.plot( phis, p1s2meanDisp, 'k-+', label = p1s2meanLabel )

plt.ylim(ymin=0)
plt.legend(loc='upper left')

plt.hold( False )
plt.show()
plt.savefig( modDir + 'test.pdf' )
plt.savefig( modDir + 'test.png', dpi = 300)


