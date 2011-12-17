#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
from fileCorrespondence import getFiles
import os, platform


class externalRegistrationTask( RegistrationTask ) :
    ''' Implements a dummy registration
        
        Use when the registration of Christine Tanner's data was performed externally and the actual task needs to be 
        assumed from the file name.
        
        The deformation filed should follow this naming convention:
        S1i1_to_S1i3AP10_regMethod_Transform.nii
    '''
    
    def __init__( self, deformationFieldPath ) :
        ''' 
            Check the default parameters            
        '''
        
        # Assume the registration
        
        # The output path is the base directory of the deformationFieldPath
        outputPath = os.path.split( deformationFieldPath )[0]
        
        # Do not care about the path...
        (deformationFieldName, fileExt) = os.path.splitext( os.path.split( deformationFieldPath )[1] )
        deformationFieldNameComps = deformationFieldName.split('_')
        
        # get rid of empty strings... if double under scores were used 
        while deformationFieldNameComps.count( '' ) != 0 :
            deformationFieldNameComps.remove('')
        
        targetIn = deformationFieldNameComps[2] + '.nii'
        sourceIn = deformationFieldNameComps[0] + '.nii'
        maskIn   = '' 
        registrationID = "id"

        RegistrationTask.__init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID )
        
        self.dofID      = 'dof'
        self.regAppID   = '__extern__'
        self.regCommand = 'extern'
        
        self.constructOutputFileNames( deformationFieldPath )
        #self.constructRegistationCommand( mu, lm, mode, mask, displacementConvergence )
        
        
        
    def run( self ) :
        print( "The registration was already performed externally. Nothing to do. " )
        #self.runRegistrationTask()
        #self.cleanUpAfterRegistration()
        
        
        
        
    def constructRegistationCommand( self, mu, lm, mode, mask, displacementConvergence ) :
        ''' Put together the parameters in the fashion it is expected by feir 
        '''
        pass

    
    
    
    def cleanUpAfterRegistration( self ) :
        ''' The following post registration steps are necessary:
                - delete the unwanted ics files, which were constructed
        '''
        pass
        
        
        
        
    def constructOutputFileNames( self, deformationFiledPath ) :
        ''' Generates the names of the registration outputs...
        '''
        
        RegistrationTask.constructOutputBaseFileName( self )
        
        self.outDOF     = os.path.realpath( deformationFiledPath )
        self.outImage   = 'n.a.'
        self.outLog     = os.path.realpath( os.path.splitext( deformationFiledPath)[0] + '_log.txt'  )
        
        self.target     = os.path.realpath( self.target )
        self.source     = os.path.realpath( self.source )
        
        if len( self.mask ) != 0 :
            self.mask       = os.path.realpath( self.mask )
        
        
        # replace the "\\" by "/" as this causes confusion in the command line...
        if platform.system() == 'Windows' :
            #output
            self.outImage = self.outImage.replace( '\\', os.altsep )
            self.outLog   = self.outLog.replace  ( '\\', os.altsep )
            self.outDOF   = self.outDOF.replace  ( '\\', os.altsep )
            self.outPath  = self.outPath.replace ( '\\', os.altsep )
            #input
            self.target   = self.target.replace  ( '\\', os.altsep )
            self.source   = self.source.replace  ( '\\', os.altsep )
            self.mask     = self.mask.replace    ( '\\', os.altsep )
    
    
    
    
    
    
if __name__ == '__main__' :
    
    from evaluationTask import *
    
    print( 'Starting test of python handling of assumed task' )
    print( 'Step 1: Build fake registration object' )
    
    deformationFieldToEvaluate = 'X:/NiftyRegValidationWithTannerData/testAssume/S1i1__regTo__S1i3AP5__feir__cfg045_dof.nii'
    
    regTask = externalRegistrationTask( 'X:/NiftyRegValidationWithTannerData/testAssume/S1i1__regTo__S1i3AP5__feir__cfg045_dof.nii' )
    regTask.printInfo()
    regTask.run()
    
    
    print( 'Step 2: Evaluation' )
    
    # Some standard file locations
    dirRefDeforms   = 'X:\\NiftyRegValidationWithTannerData\\nii\\deformationFields'
    dirMasks        = 'X:\\NiftyRegValidationWithTannerData\\nii'
    evalFile        = os.path.join( os.path.split(deformationFieldToEvaluate)[0], 'eval.txt' )
    
    evalTask        = evaluationTask( dirRefDeforms,
                                      dirMasks, 
                                      regTask,
                                      evalFile )
    
    evalTask.run()
    
    print( 'Done...' )
    
    
    
    
    
    
    