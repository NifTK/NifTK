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
from os import path

####
# Parameters for the registration 
####

# directory with input images
imgDir   = 'X:/NiftyRegValidationWithTannerData/nii'
imgExt   = 'nii'

# directory for registration results
regDir   = 'X:/NiftyRegValidationWithTannerData/outFEIR/cfg03'

# registration ID
regID = 'cfg03'

# feir specific parameters
mu               = 0.0025
lm               = 0.0
mode             = 'fast'
mask             = True
displacementConv = 0.001
    
####
# Parameters for the evaluation 
####

referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
#
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'
evalFileName       = regDir + '/eval.txt'



imgFiles = fc.getFiles( imgDir, imgExt )

lTargets  = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources  = fc.getPrecontrast( imgFiles )


# Generate registration task list

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
evalTaskList = []

for regTask in regTaskList :
    # referenceDeformDirIn, maskDirIn, registrationTaskIn, evalFileIn = 'eval.txt'
    evalTaskList.append( evTask.evaluationTask( referenceDeformDir, maskDir, regTask, evalFileName ) )

evalExecuter = executer.registrationTaskListExecuter( evalTaskList, 1 )
evalExecuter.execute()


analyser = evalListAnalyser.evaluationListAnalyser( evalTaskList )
analyser.printInfo()
analyser.writeSummaryIntoFile()




    


