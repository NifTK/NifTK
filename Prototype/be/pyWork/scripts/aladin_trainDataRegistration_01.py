#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import aladinRegistrationTask       as aladinTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
from evaluationListAnalyser import *
from os import path

####
# Parameters for the registration 
####

# directory with input images
imgDir   = 'X:/NiftyRegValidationWithTannerData/nii/'
imgExt   = "nii"

# directory for registration results
regDir   = 'X:/NiftyRegValidationWithTannerData/outAladin/def'
tMaskDir = 'X:/NiftyRegValidationWithTannerData/nii/masks'

# registration ID
regID = 'def'

###
# aladin parameters
###
rigOnly       = True
maxItPerLevel = 5
levels        = 3 
percentBlock  = 50 
percentInlier = 50 


####
# Parameters for the evaluation 
####

referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'
targetMaskDir      = 'X:/NiftyRegValidationWithTannerData/nii/masks'
evalFileName       = regDir + '/eval.txt'



imgFiles = fc.getFiles( imgDir, imgExt )

lTargets     = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources     = fc.getPrecontrast( imgFiles )
lTargetMasks = fc.getAnyBreastMasks(fc.getFiles( tMaskDir ))
regTaskList  = []

for target in lTargets :
    source = fc.matchTargetAndSource( target, lSources, imgExt )
    
    
    tMask = fc.matchTargetAndTargetMask( target, lTargetMasks )
    # create the registration task...         targetIn,                   sourceIn,                   maskIn,                            outputPath, registrationID, 
    task = aladinTask.aladinRegistrationTask( path.join( imgDir, target), path.join( imgDir, source), path.join( targetMaskDir, tMask ), regDir, regID, 
                                              rigOnly, maxItPerLevel, levels, percentBlock, percentInlier )
                                          
    regTaskList.append( task )
    
lExecuter = executer.registrationTaskListExecuter( regTaskList, 8 )
#lExecuter.execute()


# now generate the evaluation tasks:
evalTaskList = []

for regTask in regTaskList :
    # referenceDeformDirIn, maskDirIn, registrationTaskIn, evalFileIn = 'eval.txt'
    evalTaskList.append( evTask.evaluationTask( referenceDeformDir, maskDir, regTask, evalFileName ) )

evalExecuter = executer.registrationTaskListExecuter( evalTaskList, 8 )
#evalExecuter.execute()


analyser = evaluationListAnalyser( evalTaskList )
analyser.printInfo()
















