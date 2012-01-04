#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
import os, platform


class f3dRegistrationTask( RegistrationTask ) :
    ''' Implementation of the registration task which is done by reg_f3d.
        
    '''
    
    def __init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID, 
                  bendingEnergy = 0.01, logOfJacobian = 0, 
                  finalGridSpacing = -5, numberOfLevels = 3, maxIterations = 100, affineInitialisation = None, gpu = True ) :
        ''' 
            The default values represent the default values of reg_f3d. Not all parameters of reg_f3d
            are available via this interface. Please extend this if you need to.
            
            Parameters for base class:
            @param targetIn:        Target image, give complete path 
            @param sourceIn:        Source image, give complete path 
            @param maskIn:          Optional mask, give complete path
            @param registrationID:  Identifier of the specific task
            
            reg_f3d specific parameters
            @param bendingEnergy:     Weight of the bending energy term
            @param logOfJacobian:     Weight of the log of the Jacobian 
            @param finalGridSpacing:  Final grid scacing in mm (if > 0) or in voxels (if < 0)
            @param numberOfLevels:    Number of pyramid level to use
            @param maxIterations:     Number of maximum iterations to perform
            
        '''
        
        # prepare the base class
        RegistrationTask.__init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID )
        
        self.dofID    = 'dof'
        self.regAppID = '__f3d__'
        
        self.constructOutputFileNames()
        self.constructRegistationCommand( bendingEnergy, logOfJacobian, 
                                          finalGridSpacing, numberOfLevels, maxIterations,
                                          affineInitialisation, gpu )
        
        

        
    def run( self ) :
        self.runRegistrationTask()
        


        
    def constructRegistationCommand( self, bendingEnergy, logOfJacobian, finalGridSpacing, numberOfLevels, maxIterations, affineInitialisation, gpu ) :
        ''' Put together the parameters in the fashion it is expected by reg_f3d 
        '''
        self.regCommand = 'reg_f3d'
        
        # Outputs: source, target, output image and output transformation
        self.regParams  = ' -source ' + self.source
        self.regParams += ' -target ' + self.target
        self.regParams += ' -result ' + self.outImage
        self.regParams += ' -cpp '    + self.outDOF
        
        # further parameters
        if len( self.mask ) != 0 :
            self.regParams += ' -tmask ' + self.mask
        
           
        self.regParams += ' -be '    + str( bendingEnergy    )
        self.regParams += ' -jl '    + str( logOfJacobian    )
        self.regParams += ' -sx '    + str( finalGridSpacing )
        self.regParams += ' -ln '    + str( numberOfLevels   )
        self.regParams += ' -maxit ' + str( maxIterations    ) 
        
        if affineInitialisation != None : 
            self.regParams += ' -aff ' + affineInitialisation 
            
        if gpu :
            self.regParams += ' -gpu '
        
        
        
        
    def constructOutputFileNames( self ) :
        ''' Generates the names of the registation outputs...

        '''
        RegistrationTask.constructOutputBaseFileName( self )
        
        
        self.outDOF     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_cpp.nii' ) )
        self.outImage   = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_out.nii' ) )
        self.outLog     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_log.txt' ) )
        
        self.target     = os.path.realpath( self.target )
        self.source     = os.path.realpath( self.source )
        
        if len( self.mask ) != 0 :
            self.mask       = os.path.realpath( self.mask   )
        
        
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
    
    
    def constructNiiDeformationFile( self ) :
        
        self.temporaryFiles =[]
        
        ###
        # Generate the position field
        ###
        
        # the position field name
        posField = os.path.splitext( self.outDOF )[0].split( 'cpp' )[0] + 'pos.nii' 
        
        cppConversionCommand = 'reg_transform'
        cppConversionParams  =  ' -target '  + self.target
        cppConversionParams  += ' -cpp2def ' + self.outDOF + ' ' + posField
        
        self.temporaryFiles.append( posField )
        self.runTask( cppConversionCommand, cppConversionParams )
        
        ###
        # Generate the displacement field
        ###
        
        # displacement field name
        dispFieldNII = os.path.splitext( self.outDOF )[0].split( 'cpp' )[0] + 'dispN.nii'
        
        dispConversionParams =  ' -target '    + self.target
        dispConversionParams += ' -def2disp '  + posField + ' ' + dispFieldNII
        
        self.temporaryFiles.append( dispFieldNII )
        self.runTask( cppConversionCommand, dispConversionParams )
        
        
        ###
        # Convert the vectors of the field to simulate the registration in a 180degress rotated coo-system
        ###
        self.dispFieldITK = os.path.splitext( self.outDOF )[0].split( 'cpp' )[0] + 'dispI.nii'
        
        vectConvCommand = 'ucltkConvertNiftiVectorImage'
        vectConvParams  = ' -i ' + dispFieldNII
        vectConvParams += ' -o ' + self.dispFieldITK
        
        #self.temporaryFiles.append( dispFieldITK )
        self.runTask( vectConvCommand, vectConvParams )
        
        ###
        # Clean up after work is done...
        ###
        for item in self.temporaryFiles :
            os.remove( item )
    

    
if __name__ == '__main__' :
    
    from evaluationTask import *
    from registrationInitialiser import *
    
    print( 'Starting test of reg_f3d' )
    
    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10.nii'
    maskIn     = 'X:\\NiftyRegValidationWithTannerData\\nii\\masks\\S1i3AP10mask_breast.nii'
    outputPath = 'X:\\NiftyRegValidationWithTannerData\\outF3d\\'
    
    initialisationDir = 'C:/data/regValidationWithTannerData/outRREG/def/'
    
    
    initialiser = registrationInitialiser( 'rreg', 'reg_f3d', sourceIn, targetIn, initialisationDir )
    iniFile     = initialiser.getInitialisationFile();
    
    
    regID      = 'ini'
    
    regTask = f3dRegistrationTask( targetIn, sourceIn, maskIn, outputPath, regID, affineInitialisation = iniFile )
    regTask.printInfo()
    regTask.run()
    
    
    print( 'Step 2: Evaluation' )
    f3dEvalOutFile = outputPath + 'evalOut.txt'
    
    # Some standard file locations
    dirRefDeforms   = 'X:\\NiftyRegValidationWithTannerData\\nii\\deformationFields'
    dirMasks        = 'X:\\NiftyRegValidationWithTannerData\\nii'
        
    evalTask        = evaluationTask( dirRefDeforms,
                                      dirMasks, 
                                      regTask,
                                      f3dEvalOutFile )
    
    evalTask.run()

    print( 'Done...' )
    