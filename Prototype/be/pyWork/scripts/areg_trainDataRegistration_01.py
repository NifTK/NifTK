#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import aregRegistrationTask         as aregTask
import evaluationTask               as evTask
import evaluationListAnalyser       as evalListAnalyser
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
from os import path

####
# Parameters for the registration 
####

# directory with input images
imgDir   = 'X:/NiftyRegValidationWithTannerData/giplZ/'
imgExt   = "gipl.Z"

# directory for registration results
regDir   = 'C:/data/regValidationWithTannerData/outRREG/def/'

# registration ID
regID = 'def'

# areg specific parameters
parameterFile   = 'C:/data/regValidationWithTannerData/outRREG/AffineRegn.params'
targetThreshold = 0



####
# Parameters for the evaluation 
####

referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'
evalFileName       = regDir + '/eval.txt'



imgFiles = fc.getFiles( imgDir, imgExt )

lTargets  = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources  = fc.getPrecontrast( imgFiles )

regTaskList  = []

for target in lTargets :
    source = fc.matchTargetAndSource( target, lSources, imgExt )
    
    # create the registration task...
    task = aregTask.aregRegistrationTask( path.join( imgDir, target), path.join( imgDir, source), regDir, regID,
                                          parameterFile, targetThreshold )
    regTaskList.append( task )
    
lExecuter = executer.registrationTaskListExecuter( regTaskList, 8 )
lExecuter.execute()


# now generate the evaluation tasks:
evalTaskList = []

for regTask in regTaskList :
    # referenceDeformDirIn, maskDirIn, registrationTaskIn, evalFileIn = 'eval.txt'
    evalTaskList.append( evTask.evaluationTask( referenceDeformDir, maskDir, regTask, evalFileName ) )

evalExecuter = executer.registrationTaskListExecuter( evalTaskList, 8 )
evalExecuter.execute()




analyser = evalListAnalyser.evaluationListAnalyser( evalTaskList )
analyser.writeSummaryIntoFile()
analyser.printInfo()














