#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import feirRegistrationTask         as feirTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
from os import path, makedirs

####
# Parameters for the registration 
####

# directory with input images
imgDir   = 'X:/NiftyRegValidationWithTannerData/nii'
imgExt   = 'nii'

####
# Parameters for the evaluation 
####

referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'



# Get the image files
imgFiles = fc.getFiles( imgDir, imgExt )

lTargets  = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources  = fc.getPrecontrast( imgFiles )


####
# FEIR specific parameters
####

# Series of experiments
lm               = 0
mode             = 'h'
mask             = True
displacementConv = 0.01
paramArray = []

# Starting number, consider generating it automatically...
n = 1
# Adjusting to local disk as network is running out of space...
baseRegDir        = 'C:/data/regValidationWithTannerData/outFEIR/'
paraMeterFileName = 'params.txt'


# Did the file existed already?
paramFileExisted = path.exists( path.join( baseRegDir, paraMeterFileName ) )

# track the parameters in a text file in the registration base folder
parameterFile = file( path.join( baseRegDir, paraMeterFileName ), 'a+' )

# write a small header if the file did not exists already
if not paramFileExisted :
    parameterFile.write( '%10s  %60s  %14s  %14s  %8s  %5s  %6s\n'  % ('regID', 'regDir', 'mu', 'lm', 'mode', 'mask', 'conv' ) )
    parameterFile.write( '%10s  %60s  %10s  %10s  %8s  %5s  %6s\n'  % ('----------', '------------------------------------------------------------', '--------------', '--------------', '--------', '-----', '------' ) )
else :
    lines = parameterFile.readlines()
    lastCols = lines[-1].split()
    n = int(lastCols[0].split('cfg')[1]) + 1 


# Here the varying bits are filled with life. 
for i in range(-10,6) :
    # Construct regID
    regID = 'cfg' + str( '%03d' % n )
    mu = 0.0025 * (2**i)

    regDir = baseRegDir + regID + '/'
    
    print( 'regID: ' + regID                          )
    print( ' - mu       = ' + str( mu               ) )  
    print( ' - lm       = ' + str( lm               ) )    
    print( ' - mask     = ' + str( mask             ) )
    print( ' - dsplConv = ' + str( displacementConv ) )
    
    paramArray.append( [regID, regDir, mu, lm, mode, mask, displacementConv] )
    n += 1
    
    parameterFile.write( '%10s  %60s  %14.12f  %14.12f  %8s  %5s  %6.4f\n'  % (regID, regDir, mu, lm, mode, mask, displacementConv ) )
    parameterFile.flush()


parameterFile.close()

# Now build the registrations
for params in paramArray :
    # Generate registration task list

    regID            = params[0]
    regDir           = params[1]
    mu               = params[2]
    lm               = params[3]
    mode             = params[4]
    mask             = params[5]
    displacementConv = params[6]
    
    # Create the registration directory if it does not exist
    if not path.exists( regDir ):
        makedirs( regDir )
    
    regTaskList  = []
    
    for target in lTargets :
        source = fc.matchTargetAndSource( target, lSources, imgExt )
        
        # create the registration task...
        task = feirTask.feirRegistrationTask( path.join( imgDir, target), 
                                              path.join( imgDir, source), 
                                              regDir, regID,
                                              mu, lm, mode, mask, displacementConv )
        regTaskList.append( task )
        
    lExecuter = executer.registrationTaskListExecuter( regTaskList, 4 )
    lExecuter.execute()
    
    
    # now generate the evaluation tasks:
    evalFileName = regDir + '/eval.txt'
    evalTaskList = []
    
    for regTask in regTaskList :
        # referenceDeformDirIn, maskDirIn, registrationTaskIn, evalFileIn = 'eval.txt'
        evalTaskList.append( evTask.evaluationTask( referenceDeformDir, maskDir, regTask, evalFileName ) )
    
    evalExecuter = executer.registrationTaskListExecuter( evalTaskList, 1 )
    evalExecuter.execute()
    
    
    analyser = evalListAnalyser.evaluationListAnalyser( evalTaskList )
    analyser.printInfo()
    analyser.writeSummaryIntoFile()

    
print( 'Done...\n' )






    