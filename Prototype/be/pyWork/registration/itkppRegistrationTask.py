#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
import os, platform




class itkppRegistrationTask( RegistrationTask ) : 
    ''' Class implementing the interface to the registration of the itk++
        package (rigid - rreg and affine - areg). A mask image is not supported
        
        Usage: - Initialise a class by giving the source and target image as well as the 
                 output path and the registrationID (string defining the registraion)
                 task = rregRegistrationTask( target, source, outputPath, registrationID 
                 parameterFile, targetThreshold )
                 
               - run the registration task by calling runRegistrationTask() (implemented in base class)
                 task.run()
                       
    
        itk++ registrations do not generate the output image by default, hence
        a post-registration command needs to be executed. 
    '''
    

    
    def __init__ ( self, targetIn, sourceIn, outputPath, registrationID, 
                   parameterFile, targetThreshold ) :
        ''' Initialise the regsitration task, and its base class.
        '''
        
        # Note: Mask image is not supported
        RegistrationTask.__init__( self, targetIn, sourceIn, '', outputPath, registrationID )

        # separator between source and target
        self.dofID  = 'dof'
        
        
        
        
    def run( self ):
        ''' This calls the registration command from the base class and any 
            post processing needed by rreg/areg (transformation and matrix generation)
        '''
        RegistrationTask.runRegistrationTask( self )
        RegistrationTask.runTask( self, self.postRegCommand, self.postRegParams )
        
        self.taskComplete = True
        
        
        
    def constructOutputFileNames( self ) :
        ''' Construct the file names which are finally used by the registration 
            to write the results to. The filenames rely on the outFileBaseName 
            constructed in the base-class
        '''
        RegistrationTask.constructOutputBaseFileName( self )

        self.outDOF     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_dof.dof'  ) )
        self.outImage   = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_out.gipl' ) )
        self.outLog     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_log.txt'  ) )
        
        self.target = os.path.realpath( self.target )
        self.soruce = os.path.realpath( self.source )
        
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
        
        
        
    def constructRegistrationCommand( self, parameterFileIn, targetThreshold ) :
        ''' Construct the command line as it is expected by rreg
            target, source, parameter file (special), threshold in target image
        '''
        
        # Make the path to the parameter file windows compatible
        parameterFile = os.path.realpath( parameterFileIn )
        
        if platform.system() == 'Windows' :
            parameterFile = parameterFile.replace( '\\', os.altsep )
        
        #self.regCommand = 'rreg'
        
        # target and source
        self.regParams = self.target + ' ' + self.source 
        
        # parameter file
        self.regParams += ' -parameter ' + parameterFile
        
        # threshold
        self.regParams += ' -Tp ' + str( targetThreshold )
        
        # dofout
        self.regParams += ' -dofout ' + self.outDOF       
        
        
        
    def constructPostRegistrationCommand( self ):
        ''' This method generates 
            1) the output image as it is not constructed by an itk++ registration but by transformation
              (itk++ package)
            2) the homogenous matrix
        '''
         
        # Output image generation 
        # Usage: transformation [source] [output] <options>
        # Options used: -dofin dofFile - target targetImage
        self.postRegCommand = 'transformation'
        
        # source image
        self.postRegParams = self.source 
        
        # output image  
        self.postRegParams += ' ' + self.outImage      
        
        # dof file
        self.postRegParams += ' -dofin ' + self.outDOF
        
        # target image
        self.postRegParams += ' -target ' + self.target 
        