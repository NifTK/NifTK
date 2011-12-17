#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
import os, platform


class aladinRegistrationTask( RegistrationTask ) :
    ''' Implementation of the registration task which is done by reg_aladin.
        
        outDOF takes the matrix which is produced by this registration.
        the extension mat is reserved for the matrix in itk coordinates used for the evaluation
        
    '''
    
    def __init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID, 
                  rigOnly=False, maxItPerLevel=5, levels=3, percentBlock=50, percentInlier=50, gpu=False, affineInit = None ) :
        ''' 
            The default values represent the default values of reg_aladin. Not all parameters of reg_aladin
            are available via this interface. Please extend this if you need to.
            
            Parameters for base class:
            @param targetIn:        Target image, give complete path 
            @param sourceIn:        Source image, give complete path 
            @param maskIn:          Optional mask, give complete path
            @param registrationID:  Identifier of the specific task
            
            reg_aladin specific parameters
            @param rigOnly:         Set to true to perform a rigid registration only
            @param maxItPerLevel:   Number of iteration per level 
            @param levels:          Number of level to perform
            @param percentBlock:    Percentage of block to use
            @param percentInlier:   Percentage of inlier for the LTS
            
        '''
        
        # prepare the base class
        RegistrationTask.__init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID )
        
        self.dofID    = 'dof'
        self.regAppID = '__aladin__'
        
        self.constructOutputFileNames()
        self.constructRegistationCommand( rigOnly, maxItPerLevel, levels, percentBlock, percentInlier, gpu, affineInit )
        
        
        
    def run( self ) :
        self.runRegistrationTask()
        
        
    def constructRegistationCommand( self, rigOnly, maxItPerLevel, levels, percentBlock, percentInlier, gpu, affineInit ) :
        ''' Put together the parameters in the fashion it is expected by reg_aladin 
        '''
        self.regCommand = 'reg_aladin'
        
        # Outputs: source, target, output image and output transformation
        self.regParams += ' -source ' + self.source
        self.regParams += ' -target ' + self.target
        self.regParams += ' -result ' + self.outImage
        self.regParams += ' -aff '    + self.outDOF
        
        # further parameters
        if len( self.mask ) != 0 :
            self.regParams += ' -tmask ' + self.mask
        
        if rigOnly :
            self.regParams += ' -rigOnly'
            
        self.regParams += ' -maxit ' + str( maxItPerLevel )
        self.regParams += ' -ln '    + str( levels        )
        self.regParams += ' -%v '    + str( percentBlock  )
        self.regParams += ' -%i '    + str( percentInlier )
        
        if gpu :
            self.regParams += ' -gpu '
        
        if affineInit != None :
            self.regParams += ' -inaff ' + str( affineInit )
        
         
        
        
    def constructOutputFileNames( self ) :
        ''' Generates the names of the registation outputs...

        '''
        RegistrationTask.constructOutputBaseFileName( self )
        
        
        self.outDOF     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_dof.txt'  ) )
        self.outImage   = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_out.nii' ) )
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
    
    
    
    
    
    
if __name__ == '__main__' :
    
    print( 'Starting test of reg_aladin' )
    
    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10.nii'
    maskIn     = 'X:\\NiftyRegValidationWithTannerData\\nii\\masks\\S1i3AP10mask_breast.nii'
    outputPath = 'X:\\NiftyRegValidationWithTannerData\\outAladin\\test\\'
    
    rigOnly    = True
    regID      = 'rig001'
    
    regTask = aladinRegistrationTask( targetIn, sourceIn, maskIn, outputPath, regID, rigOnly )
    regTask.printInfo()
    regTask.run()
    
    print( 'Done...' )
    