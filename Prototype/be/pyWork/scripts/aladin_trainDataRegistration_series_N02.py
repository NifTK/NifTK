#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import aladinRegistrationTask       as aladinTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
from os import path, makedirs

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

referenceDeformDir = 'X:/NiftyRegValidationWithTannerData/nii/deformationFields'
maskDir            = 'X:/NiftyRegValidationWithTannerData/nii'



# Get the image files
imgFiles     = fc.getFiles( imgDir, imgExt )

lTargets     = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources     = fc.getPrecontrast( imgFiles )
lTargetMasks = fc.getAnyBreastMasks(fc.getFiles( regMaskDir ))

####
# aladin specific parameters
####

# Series of experiments
rigOnly       = True
maxItPerLevel = 5
levels        = 3 
percentBlock  = 50 
percentInlier = 50 

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
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %14s\n'  % ('regID', 'regDir', 'rigOnly', 'maxItPerLevel', 'levels', 'percentBlock', 'percentInlier' ) )
    parameterFile.write( '%10s  %60s  %14s  %14s  %14s  %14s  %6s\n'  %  ('----------', 
                                                                          '------------------------------------------------------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------', 
                                                                          '--------------' ) )
else :
    lines = parameterFile.readlines()
    lastCols = lines[-1].split()
    n = int(lastCols[0].split('cfg')[1]) + 1 


# Here the varying bits are filled with life. 
# RIGID
rigOnly = True

for i in range( 5, 101, 5 ) :
    # Construct regID
    regID         = 'cfg' + str( '%03d' % n )
    percentBlock  = i

    regDir = baseRegDir + regID + '/'
    
    print( 'regID: ' + regID                          )
    print( ' - rigOnly       = ' + str( rigOnly          ) )  
    print( ' - maxItPerLevel = ' + str( maxItPerLevel    ) )    
    print( ' - levels        = ' + str( levels           ) )
    print( ' - percentBlock  = ' + str( percentBlock     ) )
    print( ' - percentInlier = ' + str( percentInlier    ) )
    
    paramArray.append( [regID, regDir, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier] )
    n += 1
    
    parameterFile.write( '%10s  %60s  %14s  %14d  %14d  %14d  %14d\n'  % (regID, regDir, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier ) )
    parameterFile.flush()
    
# Here the varying bits are filled with life. 
# AFFINE
rigOnly = False

for i in range( 5, 101, 5 ) :
    # Construct regID
    regID         = 'cfg' + str( '%03d' % n )
    percentBlock  = i

    regDir = baseRegDir + regID + '/'
    
    print( 'regID: ' + regID                          )
    print( ' - rigOnly       = ' + str( rigOnly          ) )  
    print( ' - maxItPerLevel = ' + str( maxItPerLevel    ) )    
    print( ' - levels        = ' + str( levels           ) )
    print( ' - percentBlock  = ' + str( percentBlock     ) )
    print( ' - percentInlier = ' + str( percentInlier    ) )
    
    paramArray.append( [regID, regDir, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier] )
    n += 1
    
    parameterFile.write( '%10s  %60s  %14s  %14d  %14d  %14d  %14d\n'  % (regID, regDir, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier ) )
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
                                                  rigOnly, maxItPerLevel, levels, percentBlock, percentInlier )
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






    