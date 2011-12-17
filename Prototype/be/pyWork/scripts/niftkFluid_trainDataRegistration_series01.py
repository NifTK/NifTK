#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import niftkFluidRegistrationTask   as fluidTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
import registrationInitialiser      as regInitialiser
from os import path, makedirs
# from time import sleep



####
# Parameters for the registration 
####
# directory with input images
imgDir     = 'X:/NiftyRegValidationWithTannerData/nii'
regMaskDir = 'X:/NiftyRegValidationWithTannerData/nii/masks'
imgExt     = 'nii'



####
# Parameters for the evaluation 
####

#referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
referenceDeformDir = 'C:/data/regValidationWithTannerData/deformationFields/'
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'



# Get the image files
imgFiles     = fc.getFiles( imgDir, imgExt )

lTargets     = fc.limitToTrainData( fc.getFullDeformed ( imgFiles ) ) 
lSources     = fc.getPrecontrast( imgFiles )
lTargetMasks = fc.getAnyBreastMasks(fc.getFiles( regMaskDir ))

####
# fluid specific parameters
####
    
#numLevels          = 1
#regInterpolator    = 2
#finalInterpolator  = 3
#similarityNum      = 4
#forceStr           = 'cc'
#stepSize           = 0.125
#minDeformMagnitude = 0.01
#checkSimilarity    = 1
#maxItPerLevel      = 300
#regirdStepSize     = 1.0
#minCostFuncChange  = 1e-15
#lameMu             = 0.01
#lameLambda         = 0
#startLevel         = 0
#stopLevel          = 0
#iteratingStepSize  = 0.7
#numDilations       = 5
    
# new setting, trying sth. new...    
numLevels          = 3
regInterpolator    = 2
finalInterpolator  = 3
similarityNum      = 4
forceStr           = 'cc'
stepSize           = 0.125
minDeformMagnitude = 0.01
checkSimilarity    = 1
maxItPerLevel      = 300
regirdStepSize     = 1.0
minCostFuncChange  = 1e-15
lameMu             = 0.01
lameLambda         = 0
startLevel         = 2
stopLevel          = 2
iteratingStepSize  = 0.7
numDilations       = 5

affineInit         = 'C:/data/regValidationWithTannerData/outRREG/def/'   # directory with the rreg initialisations


# Series of experiments
paramArray = []

# Starting number
n = 1

baseRegDir        = 'C:/data/regValidationWithTannerData/outNIFTKFluid/'
paraMeterFileName = 'params.txt'


# Did the file existed already?
paramFileExisted = path.exists( path.join( baseRegDir, paraMeterFileName ) )

# track the parameters in a text file in the registration base folder
parameterFile = file( path.join( baseRegDir, paraMeterFileName ), 'a+' )

# write a small header if the file did not exists already
if not paramFileExisted :
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %60s\n'
                                                                                   % ('regID', 'regDir', # defautl params...
                                                                                      'numLevels', 
                                                                                      'regInterpType', 
                                                                                      'finInterpType', 
                                                                                      'simNum', 
                                                                                      'forceStr', 
                                                                                      'stepSize', 
                                                                                      'minDeformMag',
                                                                                      'checkSim',
                                                                                      'maxIt',
                                                                                      'regridStep',
                                                                                      'minCostChange',
                                                                                      'mu',
                                                                                      'lambda',
                                                                                      'startLevel',  
                                                                                      'stopLevel',
                                                                                      'iteratingStep',
                                                                                      'numDilations',
                                                                                      'affineInit' ) )
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %60s\n'  
                                                                       % ('----------', 
                                                                          '------------------------------------------------------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '------------------------------------------------------------' ) )
else :
    lines = parameterFile.readlines()
    lastCols = lines[-1].split()
    n = int(lastCols[0].split( 'cfg' )[1]) + 1 



# Here the varying bits are filled with life. 
for i in range(1,2) :
    # Construct regID
    regID = 'cfg' + str( '%03d' % n )
    
    regDir = baseRegDir + regID + '/'
    print( 'Adding new task: '                               ) 
    print( ' - regID         = ' + str( regID              ) ) 
    print( ' - regDir        = ' + str( regDir             ) )
    print( ' - numLevels     = ' + str( numLevels          ) )
    print( ' - regInterpType = ' + str( regInterpolator    ) )
    print( ' - finInterpType = ' + str( finalInterpolator  ) )
    print( ' - forceStr      = ' + str( forceStr           ) )
    print( ' - stepSize      = ' + str( stepSize           ) )
    print( ' - minDeformMag  = ' + str( minDeformMagnitude ) )
    print( ' - checkSim      = ' + str( checkSimilarity    ) )
    print( ' - maxIt         = ' + str( maxItPerLevel      ) )
    print( ' - regridStep    = ' + str( regirdStepSize     ) )
    print( ' - minCostChange = ' + str( minCostFuncChange  ) )
    print( ' - mu            = ' + str( lameMu             ) )
    print( ' - lambda        = ' + str( lameLambda         ) )
    print( ' - startLevel    = ' + str( startLevel         ) )
    print( ' - stopLevel     = ' + str( stopLevel          ) )
    print( ' - iteratingStep = ' + str( iteratingStepSize  ) )
    print( ' - numDilations  = ' + str( numDilations       ) )
    print( ' - affineInit    = ' + str( affineInit         ) )

    paramArray.append( [ regID, regDir, numLevels, regInterpolator, finalInterpolator, similarityNum, forceStr, 
                         stepSize, minDeformMagnitude, checkSimilarity, maxItPerLevel, regirdStepSize, minCostFuncChange, 
                         lameMu, lameLambda, startLevel, stopLevel, iteratingStepSize, numDilations ] )
    n += 1
    
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %60s\n'
                         % ( regID, regDir, numLevels, regInterpolator, finalInterpolator, similarityNum, forceStr, 
                             stepSize, minDeformMagnitude, checkSimilarity, maxItPerLevel, regirdStepSize, minCostFuncChange, 
                             lameMu, lameLambda, startLevel, stopLevel, iteratingStepSize, numDilations, affineInit ) )

    parameterFile.flush()

parameterFile.close()



# Now build the registrations
for params in paramArray :
    # Generate registration task list
    regID              = params[ 0]
    regDir             = params[ 1] 
    numLevels          = params[ 2]
    regInterpolator    = params[ 3]
    finalInterpolator  = params[ 4]
    similarityNum      = params[ 5]
    forceStr           = params[ 6]
    stepSize           = params[ 7]
    minDeformMagnitude = params[ 8]
    checkSimilarity    = params[ 9]
    maxItPerLevel      = params[10]
    regirdStepSize     = params[11]
    minCostFuncChange  = params[12]
    lameMu             = params[13]
    lameLambda         = params[14]
    startLevel         = params[15]
    stopLevel          = params[16]
    iteratingStepSize  = params[17]
    numDilations       = params[18]
    
    # Create the registration directory if it does not exist
    if not path.exists( regDir ):
        makedirs( regDir )
    
    regTaskList  = []
    
    for target in lTargets :
        source = fc.matchTargetAndSource( target, lSources, imgExt )
        tMask = fc.matchTargetAndTargetMask( target, lTargetMasks )
        
        # find the initialisation file
        init           = regInitialiser.registrationInitialiser( 'rreg', 'niftkFluid', path.join( imgDir, source), path.join( imgDir, target), affineInit )
        affineInitFile = init.getInitialisationFile()
        
        # create the registration task...
        task = fluidTask.niftkFluidRegistrationTask( path.join( imgDir, target), 
                                                     path.join( imgDir, source), 
                                                     path.join( regMaskDir, tMask ),
                                                     regDir, regID, numLevels, regInterpolator, finalInterpolator, similarityNum, forceStr, 
                                                     stepSize, minDeformMagnitude, checkSimilarity, maxItPerLevel, regirdStepSize, minCostFuncChange, 
                                                     lameMu, lameLambda, startLevel, stopLevel, iteratingStepSize, numDilations, affineInitFile )
        regTaskList.append( task )
        
    lExecuter = executer.registrationTaskListExecuter( regTaskList, 2 )
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






    