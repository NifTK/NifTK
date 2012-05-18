#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from stepSizeExperimentsRecovery import stepSizeExperimentsRecovery
import numpy as np
import matplotlib.pyplot as plt

# Define the directories
simDir01 = 'W:/philipsBreastProneSupine/referenceState/00_step_totalTime01/'
simDir02 = 'W:/philipsBreastProneSupine/referenceState/00_step_totalTime02/'
simDir05 = 'W:/philipsBreastProneSupine/referenceState/00_step_totalTime05/'
simDir10 = 'W:/philipsBreastProneSupine/referenceState/00_step_totalTime10/'
plotDir  = 'W:/philipsBreastProneSupine/referenceState/00_step_totalTime01_to_10_plots/'

# Analyse the results
SR01 = stepSizeExperimentsRecovery( simDir01 )
SR02 = stepSizeExperimentsRecovery( simDir02 )
SR05 = stepSizeExperimentsRecovery( simDir05 )
SR10 = stepSizeExperimentsRecovery( simDir10 )

# Get the kinetic and strain energies at the end of the simulation
SR01.parseLogFilesForKineticAndStrainEnergy( 'tmp', True )
SR02.parseLogFilesForKineticAndStrainEnergy( 'log', True )
SR05.parseLogFilesForKineticAndStrainEnergy( 'log', True )
SR10.parseLogFilesForKineticAndStrainEnergy( 'log', True )

# calculate the means within the stable step size region
useFirstNResults = 10
ignoreFirstNResults = 2

eKinMean01 = np.mean( SR01.Ekin[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
eKinMean02 = np.mean( SR02.Ekin[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
eKinMean05 = np.mean( SR05.Ekin[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
eKinMean10 = np.mean( SR10.Ekin[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )

eStrainMean01 = np.mean( SR01.Estrain[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
eStrainMean02 = np.mean( SR02.Estrain[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
eStrainMean05 = np.mean( SR05.Estrain[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
eStrainMean10 = np.mean( SR10.Estrain[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )

meanDisp01 = np.mean( SR01.meanDisplacements[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
meanDisp02 = np.mean( SR02.meanDisplacements[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
meanDisp05 = np.mean( SR05.meanDisplacements[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )
meanDisp10 = np.mean( SR10.meanDisplacements[ ignoreFirstNResults : useFirstNResults + ignoreFirstNResults ] )


tTotal      = np.array( (1.0,           2.0,           5.0,           10.0         ) )
eKinMean    = np.array( (eKinMean01,    eKinMean02,    eKinMean05,    eKinMean10   ) )
eStrainMean = np.array( (eStrainMean01, eStrainMean02, eStrainMean05, eStrainMean10) )
meanDisp    = np.array( (meanDisp01,    meanDisp02,    meanDisp05,    meanDisp10   ) )


# do some plotting


plt.rc( 'text', usetex=True )
plt.rcParams[ 'font.size' ] = 16

xLabel        = '$T_\mathrm{total} \; \mathrm{[s]}$'
yDispLabel    = '$\overline{\| u \|} \; \mathrm{[mm]}$'
yEkinLabel    = '$E_\mathrm{kin} $'
yEstrainLabel = '$E_\mathrm{strain}$'
yEFracLabel   = '$E_\mathrm{kin} / E_\mathrm{strain}$'
#meanLabel   = '$\overline{\|u_{p}\|}$'

# plot mean displacements
plt.hold(True)
figDisp = plt.figure()
ax      = figDisp.gca()

ax.set_xlabel( xLabel     )
ax.set_ylabel( yDispLabel )

ax.grid(color = 'gray', linestyle='-')

ax.plot( tTotal, meanDisp, 'b-+', label = yDispLabel )
ax.set_ylim( bottom = 0 )
ax.set_xlim( left   = 0 )
ax.set_xlim( right  =11 )

plt.hold( False )
plt.show()
plt.savefig( plotDir + 'meanDispOverTotalTime.pdf' )
plt.savefig( plotDir + 'meanDispOverTotalTime.png', dpi = 300 )



#######################
# plot kinetic energy 
plt.hold(True)
figEKin = plt.figure()
ax      = figEKin.gca()

ax.set_xlabel( xLabel     )
ax.set_ylabel( yEkinLabel )

ax.grid(color = 'gray', linestyle='-')

ax.plot( tTotal, eKinMean,   'b-+', label = yEkinLabel )
ax.set_ylim( bottom = 0 )
ax.set_xlim( left   = 0 )
ax.set_xlim( right  =11 )


plt.hold( False )
plt.show()
plt.savefig( plotDir + 'meanEKinOverTotalTime.pdf' )
plt.savefig( plotDir + 'meanEKinOverTotalTime.png', dpi = 300 )



#######################
# plot strain energy 
plt.hold(True)
figEstrain = plt.figure()
ax         = figEstrain.gca()

ax.set_xlabel( xLabel     )
ax.set_ylabel( yEstrainLabel )

ax.grid(color = 'gray', linestyle='-')

ax.plot( tTotal, eStrainMean,   'b-+', label = yEstrainLabel )
ax.set_ylim( bottom = 0 )
ax.set_xlim( left   = 0 )
ax.set_xlim( right  =11 )

plt.hold( False )
plt.show()
plt.savefig( plotDir + 'meanEStrainOverTotalTime.pdf' )
plt.savefig( plotDir + 'meanEStrainOverTotalTime.png', dpi = 300 )



#######################
# plot kinetic strain energy fraction 
plt.hold(True)
figEstrain = plt.figure()
ax         = figEstrain.gca()

ax.set_xlabel( xLabel     )
ax.set_ylabel( yEFracLabel )

ax.grid(color = 'gray', linestyle='-')

ax.plot( tTotal, eKinMean/eStrainMean,   'b-+', label = yEFracLabel )
ax.set_ylim( bottom = 0 )
ax.set_xlim( left   = 0 )
ax.set_xlim( right  =11 )

plt.hold( False )
plt.show()
plt.savefig( plotDir + 'meanEKinStrainFractionOverTotalTime.pdf' )
plt.savefig( plotDir + 'meanEKinStrainFractionOverTotalTime.png', dpi = 300 )
