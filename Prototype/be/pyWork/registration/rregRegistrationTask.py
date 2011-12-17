#! /usr/bin/env python 
# -*- coding: utf-8 -*-

#from registrationTask import RegistrationTask
from itkppRegistrationTask import itkppRegistrationTask
from findExecutable import findExecutable
import os, platform

class rregRegistrationTask( itkppRegistrationTask ) : 
    ''' Class implementing the interface to the rigid registration of the itk++
        package. 
        
        Usage: - Initialise a class by giving the source and target image as well as the 
                 output path and the registrationID (string defining the registraion)
                 task = rregRegistrationTask( target, source, outputPath, registrationID 
                 parameterFile, targetThreshold )
                 
               - run the registration task by calling runRegistrationTask() (implemented in base class)
                 task.run()
                       
    
        rreg does not generate the output image by default, hence
        a post-registration command needs to be executed. 
    '''
    

    
    def __init__ ( self, targetIn, sourceIn, outputPath, registrationID, 
                   parameterFile, targetThreshold ) :
        ''' Initialise the regsitration task, and its base class.
        '''
        
        itkppRegistrationTask.__init__( self, targetIn, sourceIn, outputPath, registrationID , 
                                        parameterFile, targetThreshold )
        
        self.regCommand = 'rreg'
        self.regAppID   = '__rreg__' 
        
        # configure the registration specific bits and pieces
        self.constructOutputFileNames()
        self.constructRegistrationCommand( parameterFile, targetThreshold )
        self.constructPostRegistrationCommand()
        
        
   
    
if __name__ == '__main__' : 
    target   = 'X:/NiftyRegValidationWithTannerData/giplZ/S1i3AP10.gipl.Z'
    source   = 'X:/NiftyRegValidationWithTannerData/giplZ/S1i1.gipl.Z'
    outDir   = 'X:/NiftyRegValidationWithTannerData/testRREG/' 
    paraFile = 'X:/NiftyRegValidationWithTannerData/testRREG/AffineRegn.params'
    regID    = 'ID'
    
    task = rregRegistrationTask( target, source, outDir, regID, paraFile, 0 )
    task.printInfo()
    task.run()
    print( 'Done.' )
    
    
    
    
