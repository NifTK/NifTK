#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
from fileCorrespondence import getFiles
import os, platform


class feirRegistrationTask( RegistrationTask ) :
    ''' Implementation of the registration task which is done by FEIR (PHILIPS).
        
        outDOF takes the dummy file name for the deformation vector field. This will be 
        replaced with the combined vector valued image, which is constructed for the
        evaluation. 
    '''
    
    def __init__( self, targetIn, sourceIn, outputPath, registrationID, 
                  mu = 0.0025, lm = 0.0, mode = 'fast', mask = False, displacementConvergence = 0.01, planStr = None ) :
        ''' 
            Check the default parameters            
        '''
        
        # prepare the base class
        # the mask is handled differently in FEIR, thus give an empty string to the super-class
        maskIn = '' 
        RegistrationTask.__init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID )
        
        self.dofID    = 'dof'
        self.regAppID = '__feir__'
        
        self.constructOutputFileNames()
        self.constructRegistationCommand( mu, lm, mode, mask, displacementConvergence, planStr )
        
        
        
    def run( self ) :
        self.runRegistrationTask()
        self.cleanUpAfterRegistration()
        
        
        
        
    def constructRegistationCommand( self, mu, lm, mode, mask, displacementConvergence, planStr ) :
        ''' Put together the parameters in the fashion it is expected by feir 
        '''
        self.regCommand = 'feir'
        
        # Outputs: source, target, output image and output transformation
        # feir reference (target) and template (source)
        self.regParams += ' r ' + self.target   
        self.regParams += ' t ' + self.source   
        self.regParams += ' o ' + self.outImage 
        
        # mode options fast, standard and h
        self.regParams += ' mode '         + mode
        self.regParams += ' mu '           + str( mu )
        self.regParams += ' lambda '       + str( lm )
        self.regParams += ' displ_update ' + str( displacementConvergence )
        
        
        if mask :
            self.regParams += ' scheme MRsupine '
        
        
        # save to mhd data (only works in conjuncion with ics which needs to be deleted further on, nii not supported)
        self.regParams += ' save2ics 1 '
        self.regParams += ' save2mhd 1 '        
        
        # For TEST purposes only!!!
        if planStr != None :
            self.regParams += ' planstr ' + planStr + ' '
    
    
    
    
    
    def cleanUpAfterRegistration( self ) :
        ''' The following post registration steps are necessary:
                - delete the unwanted ics files, which were constructed
        '''
        
        # Fill the tmp file list manually to avoid clashes with other calls when running several threads
        self.tmpFileList  = []
        self.tmpFileList.append( self.outImage + '.x.ics' ) 
        self.tmpFileList.append( self.outImage + '.x.ids' )
        self.tmpFileList.append( self.outImage + '.x.inf' )
        
        self.tmpFileList.append( self.outImage + '.y.ics' ) 
        self.tmpFileList.append( self.outImage + '.y.ids' )
        self.tmpFileList.append( self.outImage + '.y.inf' )

        self.tmpFileList.append( self.outImage + '.z.ics' ) 
        self.tmpFileList.append( self.outImage + '.z.ids' )
        self.tmpFileList.append( self.outImage + '.z.inf' )
        
        
        for item in self.tmpFileList :
            os.remove( os.path.join( self.outPath, item ) )
        
        
        
        
    def constructOutputFileNames( self ) :
        ''' Generates the names of the registation outputs...

        '''
        RegistrationTask.constructOutputBaseFileName( self )
        
        # 
        self.outDOF     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_dof.nii'  ) )
        self.outImage   = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_out'      ) )
        self.outLog     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_log.txt'  ) )
        
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
    
    
    
    
    def constructNiiDeformationFile( self ):
        ''' The FEIR result is rotated by 180 degrees. But in contrast to the 
            nifty coordinate system not only the vector components need to be 
            exchanged, but the whole image must be additionally rotated. This 
            is accounted for in the construction process of the deformation 
            field. 
        '''
        
        print( 'Starting FEIR deformation field composition...' )
        deformFiledComposerCmd = 'niftkCompose3DVectorImageFromComponentImages'
        
        # Find out what the name of the deformation field components is...
        # and then generate the deformation vector field
        
        deformationVectorField = self.outDOF
        
        deformationVectorFieldExt = os.path.splitext( deformationVectorField )
        deformationVectorFieldExt = deformationVectorFieldExt[-1]
        
        if deformationVectorFieldExt != '.nii' :
            print('ERROR: DOF file has not the expected file extension.')
            return
        
        paramsFieldComp  = ' -x ' + self.outImage + '.x.mhd'
        paramsFieldComp += ' -y ' + self.outImage + '.y.mhd'
        paramsFieldComp += ' -z ' + self.outImage + '.z.mhd'
        
        paramsFieldComp += ' -invertX -invertY '
        paramsFieldComp += ' -flipXY '

        paramsFieldComp += ' -o ' + deformationVectorField

        self.runTask( deformFiledComposerCmd, paramsFieldComp )
        
        self.dispFieldITK = deformationVectorField
        
        # Check if the deformation field was successfully created
        if os.path.isfile( deformationVectorField ) != True : 
            print( 'ERROR: The deformation vector field was not generated! ' )
            return



        
    def resampleSourceImage( self ):
        
        resampleCommand = 'niftkValidateDeformationVectorField'
        resampleParams  = ' -i ' + self.source
        resampleParams += ' -o ' + self.outImage + '.nii'
        resampleParams += ' -def ' +self.outDOF
        resampleParams += ' -interpolate bspl '
        
        self.runTask( resampleCommand, resampleParams )
        # check if the nii vector image already exists
        
        
        
        pass
        
        
    
if __name__ == '__main__' :
    
    from evaluationTask import *
    
    print( 'Starting test of python handling of feir' )
    print( 'Step 1: Registration' )
    
    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10.nii'
    outputPath = 'X:\\NiftyRegValidationWithTannerData\\testFEIR\\nonlinOnly\\'
    
    regID      = 'def'
    
    # FEIR parameters:
    mu               = 0.0025
    lm               = 0
    mode             = 'fast'
    mask             = True
    displacementConv = 0.01
    
    
    regTask = feirRegistrationTask( targetIn, sourceIn, outputPath, regID, mu, lm, mode, mask, displacementConv )
    regTask.printInfo()
    regTask.run()
    
    
    print( 'Step 2: Evaluation' )
    feirEvalOutFile = outputPath + 'evalOut.txt'
    
    # Some standard file locations
    dirRefDeforms   = 'X:\\NiftyRegValidationWithTannerData\\nii\\deformationFields'
    dirMasks        = 'X:\\NiftyRegValidationWithTannerData\\nii'
        
    evalTask        = evaluationTask( dirRefDeforms,
                                      dirMasks, 
                                      regTask,
                                      feirEvalOutFile )
    
    evalTask.run()
    
    print( 'Done...' )
    
    
    
    
    
    
    