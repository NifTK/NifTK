#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author:  Bjoern Eiben

@summary: This script differs from the previous, as the parameter "planstr"
          is now available. This enables the choice of different pre-registrations
          (affine (p), translation (t), rotation (r), scale (s), shear(h), 
          non-rigid (n)). 
'''

import feirRegistrationTask         as feirTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
import numpy                        as np
from os import path, makedirs

####
# Parameters for the registration 
####

# directory with input images
imgDir   = 'X:/NiftyRegValidationWithTannerData/nii'
#imgDir   = 'C:/data/RegValidation/nii'
imgExt   = 'nii'

####
# Parameters for the evaluation 
####

#referenceDeformDir = imgDir + '/deformationFields'
referenceDeformDir = 'C:/data/regValidationWithTannerData/deformationFields/'
maskDir            = imgDir 



# Get the image files
imgFiles = fc.getFiles( imgDir, imgExt )

lTargets  = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources  = fc.getPrecontrast( imgFiles )


####
# FEIR specific parameters
####

# Series of experiments
mus              = 0.0025 * 2. ** np.array( range( -10, 6 ) ) 
mode             = 'fast'
mask             = True
displacementConv = 0.01
planstr          = 'n'

paramArray       = []


# Starting number, consider generating it automatically...
n = 200

baseRegDir        = 'C:/data/regValidationWithTannerData/outFEIR/'
paraMeterFileName = 'params.txt'



# Did the file existed already?
paramFileExisted = path.exists( path.join( baseRegDir, paraMeterFileName ) )

# track the parameters in a text file in the registration base folder
parameterFile = file( path.join( baseRegDir, paraMeterFileName ), 'a+' )

# write a small header if the file did not exists already
if not paramFileExisted :
    parameterFile.write( '%10s  %60s  %14s  %14s  %8s  %5s  %6s  %8s\n'  % ('regID', 'regDir', 'mu', 'lm', 'mode', 'mask', 'conv', 'planstr' ) )
    parameterFile.write( '%10s  %60s  %14s  %14s  %8s  %5s  %6s  %8s\n'  % ('----------', '------------------------------------------------------------', '--------------', '--------------', '--------', '-----', '------', '--------' ) )
else :
    lines = parameterFile.readlines()
    lastCols = lines[-1].split()
    n = int(lastCols[0].split('cfg')[1]) + 1 



# Here the varying bits are filled with life.
for mu in mus :
        
    for i in [0] :
        
        # Construct regID
        regID = 'cfg' + str( '%03d' % n )
        lm    = mu * i
    
        regDir = baseRegDir + regID + '/'
        
        print( 'regID: ' + regID                          )
        print( ' - mu       = ' + str( mu               ) )  
        print( ' - lm       = ' + str( lm               ) )    
        print( ' - mask     = ' + str( mask             ) )
        print( ' - dsplConv = ' + str( displacementConv ) )
        print( ' - planstr  = ' + str( planstr          ) )
        
        paramArray.append( [regID, regDir, mu, lm, mode, mask, displacementConv, planstr] )
        n += 1
        
        parameterFile.write( '%10s  %60s  %14.7e  %14.7e  %8s  %5s  %6.4f  %8s\n'  % (regID, regDir, mu, lm, mode, mask, displacementConv, str( planstr ) ) )
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
    palnstr          = params[7]
    
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
                                              mu, lm, mode, mask, displacementConv, palnstr )
        regTaskList.append( task )
        
    lExecuter = executer.registrationTaskListExecuter( regTaskList, 3 )
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

    