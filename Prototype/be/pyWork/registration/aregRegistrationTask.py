#! /usr/bin/env python 
# -*- coding: utf-8 -*-

#from registrationTask import RegistrationTask
from itkppRegistrationTask import itkppRegistrationTask
from findExecutable import findExecutable
import os, platform

class aregRegistrationTask( itkppRegistrationTask ) : 
    ''' Class implementing the interface to the rigid registration of the itk++
        package. 
        
        Usage: - Initialise a class by giving the source and target image as well as the 
                 output path and the registrationID (string defining the registraion)
                 task = aregRegistrationTask( target, source, outputPath, registrationID 
                 parameterFile, targetThreshold )
                 
               - run the registration task by calling runRegistrationTask() (implemented in base class)
                 task.run()
    '''
    

    
    def __init__ ( self, targetIn, sourceIn, outputPath, registrationID, 
                   parameterFile, targetThreshold ) :
        ''' Initialise the registration task, and its base class.
        '''
        
        itkppRegistrationTask.__init__( self, targetIn, sourceIn, outputPath, registrationID, 
                                        parameterFile, targetThreshold )
        
        self.regCommand = 'areg'
        self.regAppID   = '__areg__' 
        
        # configure the registration specific bits and pieces
        self.constructOutputFileNames()
        self.constructRegistrationCommand( parameterFile, targetThreshold )
        self.constructPostRegistrationCommand()
        
        
   
    
if __name__ == '__main__' : 
    target   = 'X:/NiftyRegValidationWithTannerData/giplZ/S1i3AP10.gipl.Z'
    source   = 'X:/NiftyRegValidationWithTannerData/giplZ/S1i1.gipl.Z'
    out      = 'X:/NiftyRegValidationWithTannerData/testAREG/' 
    paraFile = 'X:/NiftyRegValidationWithTannerData/testAREG/AffineRegn.params'
    regID    = 'ID'
    
    task = aregRegistrationTask( target, source, out, regID, paraFile, 0 )
    task.printInfo()
    task.run()
    print( 'Done.' )
    
    
    
    
