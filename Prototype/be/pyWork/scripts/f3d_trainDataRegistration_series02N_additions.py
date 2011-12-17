#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import f3dRegistrationTask          as f3dTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
import registrationInitialiser      as regInitialiser
from os import path, makedirs
from time import sleep



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

lTargets     = fc.limitToTrainData( fc.getDeformed ( imgFiles ) ) 
lSources     = fc.getPrecontrast( imgFiles )
lTargetMasks = fc.getAnyBreastMasks(fc.getFiles( regMaskDir ))

####
# f3d specific parameters
####
bendingEnergy        = 0.0
#logOfJacobians       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] # to be varied over the iterations...
#logOfJacobians       = [ 0.92, 0.94, 0.96, 0.98 ] # to be varied over the iterations...
logOfJacobians       = [ 0.91, 0.93, 0.97, 0.99, 0.995 ] # to be varied over the iterations...
finalGridSpacing     = 60.0
numberOfLevels       = 1
maxIterations        = 300
affineInit           = 'C:/data/regValidationWithTannerData/outAladin/cfg168/'   # directory with the rreg initialisations
#affineInit           = 'C:/data/regValidationWithTannerData/outRREG/def/'   # !!! remember to change initialisation method as well!!!directory with the rreg initialisations # 
gpu                  = False

# Series of experiments


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

#spacings = [40, 20, 10]
#
#for finalGridSpacing in spacings :
#    # Here the varying bits are filled with life. 
#    for logOfJacobian in logOfJacobians :
#        # Construct regID
#        regID = 'cfg' + str( '%03d' % n )
#        
#        regDir = baseRegDir + regID + '/'
#        
#        print( 'regID: ' + regID                                  )
#        print( ' - regDir           = ' + str( regDir           ) )
#        print( ' - bendingEnergy    = ' + str( bendingEnergy    ) )  
#        print( ' - logOfJacobian    = ' + str( logOfJacobian    ) )    
#        print( ' - finalGridSpacing = ' + str( finalGridSpacing ) )
#        print( ' - numberOfLevels   = ' + str( numberOfLevels   ) )
#        print( ' - maxIterations    = ' + str( maxIterations    ) )
#        print( ' - affineInit       = ' + str( affineInit       ) )
#        print( ' - gpu              = ' + str( gpu              ) )
#        
#        
#        paramArray.append( [ regID, regDir, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInit, gpu ] )
#        n += 1
#        
#        parameterFile.write( '%10s  %60s  %14.12f  %14.12f  %14.12f  %14.12f  %14.12f  %60s  %14s\n'  % (regID, regDir, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInit, gpu ) )
#        parameterFile.flush()
#
#
## m10
#finalGridSpacing = 10.0
#numberOfLevels   = 2
#
## Here the varying bits are filled with life. 
#for i in logOfJacobians :
#    # Construct regID
#    regID = 'cfg' + str( '%03d' % n )
#    logOfJacobian = i
#
#    regDir = baseRegDir + regID + '/'
#    
#    print( 'regID: ' + regID                                  )
#    print( ' - regDir           = ' + str( regDir           ) )
#    print( ' - bendingEnergy    = ' + str( bendingEnergy    ) )  
#    print( ' - logOfJacobian    = ' + str( logOfJacobian    ) )    
#    print( ' - finalGridSpacing = ' + str( finalGridSpacing ) )
#    print( ' - numberOfLevels   = ' + str( numberOfLevels   ) )
#    print( ' - maxIterations    = ' + str( maxIterations    ) )
#    print( ' - affineInit       = ' + str( affineInit       ) )
#    print( ' - gpu              = ' + str( gpu              ) )
#    
#    
#    paramArray.append( [ regID, regDir, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInit, gpu ] )
#    n += 1
#    
#    parameterFile.write( '%10s  %60s  %14.12f  %14.12f  %14.12f  %14.12f  %14.12f  %60s  %14s\n'  % (regID, regDir, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInit, gpu ) )
#    parameterFile.flush()



# m5
finalGridSpacing = 5.0
numberOfLevels   = 3

# Here the varying bits are filled with life. 
for i in logOfJacobians :
    # Construct regID
    regID = 'cfg' + str( '%03d' % n )
    logOfJacobian = i

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
        
        # find the initialisation file
        init           = regInitialiser.registrationInitialiser('reg_aladin','reg_f3d', path.join( imgDir, source), path.join( imgDir, target), affineInit )
        affineInitFile = init.getInitialisationFile()
        
        # create the registration task...
        task = f3dTask.f3dRegistrationTask( path.join( imgDir, target), 
                                            path.join( imgDir, source), 
                                            path.join( regMaskDir, tMask ),
                                            regDir, regID,
                                            bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInitFile, gpu )
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






    