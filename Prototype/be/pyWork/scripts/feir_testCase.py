#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author:  Bjoern Eiben
@summary: Reproducing the error which occurred during experiments. 
'''

import feirRegistrationTask         as feirTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
from os import path, makedirs

# directory with input images
# imgDir   = 'X:/NiftyRegValidationWithTannerData/nii'
imgDir   = 'G:/data/RegValidation/nii'
imgExt   = 'nii'

# Parameters for the evaluation 
#referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
referenceDeformDir = imgDir + '/deformationFields'
maskDir            = imgDir

regDir = 'G:/data/RegValidation/outFEIR/testCase'


# Get the image files
imgFiles = fc.getFiles( imgDir, imgExt )

lTargets  = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources  = fc.getPrecontrast( imgFiles )

# Series of experiments
mu               = 0.0025 * (2 ** (-6))  # optimal mu for previous experiments
lm               = -mu 
mode             = 'fast'
mask             = True
displacementConv = 0.01

regID = 'errorRep'

# generate target and source
target = lTargets[0]
source = fc.matchTargetAndSource( target, lSources )

regTaskList = []

regTask = feirTask.feirRegistrationTask( path.join( imgDir, target ), 
                                      path.join( imgDir, source ), 
                                      regDir , regID,
                                      mu, lm, mode, mask, displacementConv ) 

regTask.run()





evalFileName = regDir + '/eval.txt'


evalTask = evTask.evaluationTask( referenceDeformDir, maskDir, regTask, evalFileName )
evalTask.run()
    




