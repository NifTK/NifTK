#! /usr/bin/env python 
# -*- coding: utf-8 -*-

# This scipt compares the execution time of typical configurations of 
# - niftyReg reg_aladin and reg_f3d
# - feir (fast, standard, h)

# The following configurations were tested
# reg_f3d (6 configurations)
# - single multi {2,3} resolution,          3x 
# - dx = 20mm control point spacing,        1x
# - pv = 0.995 volume constraint weight     1x
# - CPU and GPU version                     2x

# reg_aladin (12 configurations)
# - percentI = 50%                          1x
# - CPU and GPU version                     2x
# - rigid and affine                        2x
# - levels                                  3x


# FEIR (6 configurations)
# - mu_E = -6 : mu = 0.0025 * 2 ** (-6)     1x
# - modes fast, standard, h                 3x
# - lamba = {0 , 2mu}                       2x

import fileCorrespondence           as fc
import registrationInitialiser      as regInitialiser
import f3dRegistrationTask          as f3dTask
import aladinRegistrationTask       as aladinTask
import feirRegistrationTask         as feirTask
import evaluationTask               as evTask
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
from os import path, makedirs
import sys

#################################################
#
# reg_f3d speed test section
#

# directory with input images
imgDir     = 'X:/NiftyRegValidationWithTannerData/nii'
regMaskDir = 'X:/NiftyRegValidationWithTannerData/nii/masks'
imgExt     = 'nii'

# directory for the evaluation
referenceDeformDir = 'C:/data/regValidationWithTannerData/deformationFields/'
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'


# Get the image files
imgFiles     = fc.getFiles( imgDir, imgExt )

lTargets     = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources     = fc.getPrecontrast( imgFiles )
lTargetMasks = fc.getAnyBreastMasks(fc.getFiles( regMaskDir ))

bendingEnergy        = 0.0
logOfJacobian        = 0.995
finalGridSpacing     = 20.0
numbersOfLevels      = [1,2,3]
maxIterations        = 300
#affineInit           = 'C:/data/regValidationWithTannerData/outRREG/def/'
affineInit           = None
gpus                 = [False, True]

paramArray = []

# Starting number
n = 1

baseRegDir        = 'C:/data/regValidationWithTannerData/outF3d/'
paraMeterFileName = 'params.txt'


# Did the file existed already?
paramFileExisted = path.exists( path.join( baseRegDir, paraMeterFileName ) )

# track the parameters in a text file in the registration base folder
parameterFile = file( path.join( baseRegDir, paraMeterFileName ), 'a+' )

# write a small header if the file did not exists already
if not paramFileExisted :
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %60s  %14s\n'  % ('regID', 'regDir', 'bendingEnergy', 'logOfJacobian', 'finalSpacing', 'numberOfLevels', 'maxIterations', 'affineInit', 'gpu' ) )
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %60s  %14s\n'  %  ('----------', 
                                                                          '------------------------------------------------------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '------------------------------------------------------------', 
                                                                          '--------------' ) )
else :
    lines = parameterFile.readlines()
    lastCols = lines[-1].split()
    n = int(lastCols[0].split( 'cfg' )[1]) + 1 

for gpu in gpus:
    for numberOfLevels in numbersOfLevels :
        # Construct regID
        regID = 'cfg' + str( '%03d' % n )
        
        regDir = baseRegDir + regID + '/'
        
        print( 'regID: ' + regID                                  )
        print( ' - regDir           = ' + str( regDir           ) )
        print( ' - bendingEnergy    = ' + str( bendingEnergy    ) )  
        print( ' - logOfJacobian    = ' + str( logOfJacobian    ) )    
        print( ' - finalGridSpacing = ' + str( finalGridSpacing ) )
        print( ' - numberOfLevels   = ' + str( numberOfLevels   ) )
        print( ' - maxIterations    = ' + str( maxIterations    ) )
        print( ' - affineInit       = ' + str( affineInit       ) )
        print( ' - gpu              = ' + str( gpu              ) )
    
        paramArray.append( [ regID, regDir, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInit, gpu ] )
        n += 1
        
        parameterFile.write( '%10s  %60s  %14.12f  %14.12f  %14.12f  %14.12f  %14.12f  %60s  %14s\n'  % (regID, regDir, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInit, gpu ) )
        parameterFile.flush()


parameterFile.close()


# Now build the registrations
for params in paramArray :
    # Generate registration task list

    regID            = params[0]
    regDir           = params[1]
    bendingEnergy    = params[2]
    logOfJacobian    = params[3]
    finalGridSpacing = params[4]
    numberOfLevels   = params[5]
    maxIterations    = params[6]
    affineInit       = params[7]
    gpu              = params[8]
    
    
    # Create the registration directory if it does not exist
    if not path.exists( regDir ):
        makedirs( regDir )
    
    regTaskList  = []
    
    for target in lTargets :
        source = fc.matchTargetAndSource( target, lSources, imgExt )
        tMask  = fc.matchTargetAndTargetMask( target, lTargetMasks )
        
        if affineInit == None :
            affineInitFile = None
        else :
            # find the initialisation file
            init           = regInitialiser.registrationInitialiser('rreg','reg_f3d', path.join( imgDir, source), path.join( imgDir, target), affineInit )
            affineInitFile = init.getInitialisationFile()
        
        # create the registration task...
        task = f3dTask.f3dRegistrationTask( path.join( imgDir, target), 
                                            path.join( imgDir, source), 
                                            path.join( regMaskDir, tMask ),
                                            regDir, regID,
                                            bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInitFile, gpu )
        regTaskList.append( task )
        
    lExecuter = executer.registrationTaskListExecuter( regTaskList, 1 )
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

    
print( 'f3d tests done...\n' )



sys.exit(0)














####################################################
#
# aladin speed test section
#


# Series of experiments
rigOnlys      = [ True, False ]
maxItPerLevel = 5
numsLevels    = [1,2,3] 
percentBlock  = 50 
percentInlier = 50 
gpus          = [True, False] 

paramArray = []


# Starting number
n = 1

baseRegDir        = 'C:/data/regValidationWithTannerData/outAladin/'
paraMeterFileName = 'params.txt'


# Did the file existed already?
paramFileExisted = path.exists( path.join( baseRegDir, paraMeterFileName ) )

# track the parameters in a text file in the registration base folder
parameterFile = file( path.join( baseRegDir, paraMeterFileName ), 'a+' )

# write a small header if the file did not exists already
if not paramFileExisted :
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %14s\n'  % ('regID', 'regDir', 'rigOnly', 'maxItPerLevel', 'levels', 'percentBlock', 'percentInlier', 'gpu' ) )
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s  %14s\n'  %  ('----------', 
                                                                          '------------------------------------------------------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------' ) )
else :
    lines = parameterFile.readlines()
    lastCols = lines[-1].split()
    n = int(lastCols[0].split('cfg')[1]) + 1 


for gpu in gpus :
    for rigOnly in rigOnlys :
        for levels in numsLevels :
        
            # Construct regID
            regID = 'cfg' + str( '%03d' % n )
            
            #percentInlier = i
        
            regDir = baseRegDir + regID + '/'
            
            print( 'regID: ' + regID                               )
            print( ' - rigOnly       = ' + str( rigOnly          ) )  
            print( ' - maxItPerLevel = ' + str( maxItPerLevel    ) )    
            print( ' - levels        = ' + str( levels           ) )
            print( ' - percentBlock  = ' + str( percentBlock     ) )
            print( ' - percentInlier = ' + str( percentInlier    ) )
            print( ' - gpu           = ' + str( gpu              ) )
            
            paramArray.append( [regID, regDir, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier, gpu] )
            n += 1
            
            parameterFile.write( '%10s  %60s  %14s  %14d  %14d  %14d  %14d  %14s\n'  % (regID, regDir, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier, gpu ) )
            parameterFile.flush()


parameterFile.close()

# Now build the registrations
for params in paramArray :
    # Generate registration task list

    regID            = params[0]
    regDir           = params[1]
    rigOnly          = params[2]
    maxItPerLevel    = params[3]
    levels           = params[4]
    percentBlock     = params[5]
    percentInlier    = params[6]
    gpu              = params[7]
    
    # Create the registration directory if it does not exist
    if not path.exists( regDir ):
        makedirs( regDir )
    
    regTaskList  = []
    
    for target in lTargets :
        source = fc.matchTargetAndSource( target, lSources, imgExt )
        tMask = fc.matchTargetAndTargetMask( target, lTargetMasks )
        
        # create the registration task...
        task = aladinTask.aladinRegistrationTask( path.join( imgDir, target), 
                                                  path.join( imgDir, source), 
                                                  path.join( regMaskDir, tMask ),
                                                  regDir, regID,
                                                  rigOnly, maxItPerLevel, levels, percentBlock, percentInlier, gpu )
        regTaskList.append( task )
        
    lExecuter = executer.registrationTaskListExecuter( regTaskList, 1 )
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


sys.exit(0)







###############################################################
#
# FEIR speed test section
#

# Series of experiments
mu               = 0.0025 * 2. ** (-6) 
lms              = [ 0. , 2.* mu ]
modes            = ['fast', 'standard', 'h']
mask             = True
displacementConv = 0.01
planstr          = 'ptrshn'

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
for mode in modes :
    for lm in lms :
        
        # Construct regID
        regID = 'cfg' + str( '%03d' % n )
    
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
        
    lExecuter = executer.registrationTaskListExecuter( regTaskList, 1 )
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

    


























