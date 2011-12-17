#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
from fileCorrespondence import getFiles
import os, platform, glob, subprocess, shlex



class registrationInitialiser :
    ''' Helper class to feed a registration result into another registration 
        
        Currently this is for rreg/areg initialisation for reg_f3d
    '''
    
    def __init__( self, initialMethod, targetMethod, sourceImage, targetImage, initialRegDir ) :
        
        self.initialMethod   = initialMethod
        self.targetMethod    = targetMethod
        self.initialRegDir   = initialRegDir
        
        self.targetImage     = targetImage
        self.sourceImage     = sourceImage
        
        # handle the case, if the "full" target image was used...
        self.targetImage = self.targetImage.split('full')[0]
        
        # Output generated later on
        self.outFileName          = ''
        self.correspondingDOFFile = ''
        self.outFileAlreadyExists = False
        self.outLog               = ''
        
        # Chcek if the method was implemented yet.
        if not self._checkInitialisationImplemented() :
            print( 'Sorry not yet implemented...' );
            return
        
        # adapt for specific Windows needs ;)
        if platform.system() == 'Windows' :
            self.targetImage   = self.targetImage.replace( '\\', os.altsep )  
            self.sourceImage   = self.sourceImage.replace( '\\', os.altsep )
            self.initialRegDir = self.initialRegDir.replace( '\\', os.altsep )  
        
        # find corresponding file
        self._findCorrespondingDOFFile()
        self._constructOutFileName()
        
    
    
    def _checkInitialisationImplemented( self ):
        implem = False
        
        if ( self.targetMethod == 'reg_f3d' and ( self.initialMethod == 'rreg' or self.initialMethod == 'areg' ) ):
            implem = True
            self.conversionRequested = 'rreg__to__reg_f3d'
            
        if ( self.targetMethod == 'niftkFluid' and ( self.initialMethod == 'rreg' or self.initialMethod == 'areg' ) ):
            implem = True
            self.conversionRequested = 'rreg__to__niftkFluid'
            
        if ( self.targetMethod == 'feir' and ( self.initialMethod == 'rreg' or self.initialMethod == 'areg' ) ):
            implem = True
            self.conversionRequested = 'rreg__to__feir'
        
        if ( self.targetMethod == 'reg_f3d' and ( self.initialMethod == 'reg_aladin' or self.initialMethod == 'aladin' ) ):
            implem = True
            self.conversionRequested = 'aladin__to__f3d'
        
        return implem
        
    
        
    def _findCorrespondingDOFFile( self ) :
        
        # Generate the pattern...
        pattern = os.path.split( os.path.splitext( self.sourceImage )[0] )[1] + '*' + os.path.split( os.path.splitext( self.targetImage )[0] )[1] + '*' + 'dof.*'
        
        # Look for it in the file system
        candidates = glob.glob( os.path.join(self.initialRegDir, pattern) )
        
        # Be sure you only found one.
        if len( candidates ) != 1:
            print( 'Could not establish a correspondence. Sorry.' )
            self.correspondingDOFFile = ''
        else:
            self.correspondingDOFFile = candidates[0]
        
        # create the log file
        self.outLog = os.path.join( self.initialRegDir, 'log_dof_conversion.txt' )
        
        # adapt for specific Windows needs ;)
        if platform.system() == 'Windows' :
            self.correspondingDOFFile = self.correspondingDOFFile.replace( '\\', os.altsep )
            self.outLog              = self.outLog.replace( '\\', os.altsep )
    
    
    
    
    def _constructOutFileName( self ) :
        
        # NiftyReg
        if  self.conversionRequested == 'rreg__to__reg_f3d' : # self.targetMethod == 'reg_f3d' and ( self.initialMethod == 'rreg' or self.initialMethod == 'areg' ) :
            self.outFileName = os.path.join( self.initialRegDir, os.path.splitext( os.path.split( self.correspondingDOFFile )[1] )[0] + '_matRot.txt' )
        
        # niftkFluid
        if  self.conversionRequested == 'rreg__to__niftkFluid' : # self.targetMethod == 'reg_f3d' and ( self.initialMethod == 'rreg' or self.initialMethod == 'areg' ) :
            self.outFileName = os.path.join( self.initialRegDir, os.path.splitext( os.path.split( self.correspondingDOFFile )[1] )[0] + '_uclAff.txt' )
        
        if  self.conversionRequested == 'rreg__to__feir' : 
            self.outFileName      = os.path.join( self.initialRegDir, os.path.splitext( os.path.split( self.correspondingDOFFile )[1] )[0] + '_rregOut.nii' )
            self.intermedFileName = os.path.join( self.initialRegDir, os.path.splitext( os.path.split( self.correspondingDOFFile )[1] )[0] + '_uclAff.txt' )
        
        # Thanks, this is the simplest case yet. f3d understands files produced by aladin
        if self.conversionRequested == 'aladin__to__f3d' :
            self.outFileName      = self.correspondingDOFFile
             
            
        # adapt for specific Windows needs ;)
        if platform.system() == 'Windows' :
            self.outFileName = self.outFileName.replace( '\\', os.altsep )    
            if self.conversionRequested == 'rreg__to__feir' :
                self.intermedFileName = self.intermedFileName.replace( '\\', os.altsep )
                
        # Does the file already exist?
        if os.path.exists( self.outFileName ):
            self.outFileAlreadyExists = True 
        
        if  self.conversionRequested == 'rreg__to__feir' : 
            if os.path.exists( self.intermedFileName ):
                self.intermedFileAlreadyExists = True 

    
    def getInitialisationFile( self ):
        ''' This returns the parameter which is expected by the target methd
        '''
        # NiftyReg
        if  self.conversionRequested == 'rreg__to__reg_f3d' : # self.targetMethod == 'reg_f3d' and ( self.initialMethod == 'rreg' or self.initialMethod == 'areg' ) :
            if not self.outFileAlreadyExists :
                self._itkPPtof3dConversion()
            return self.outFileName
        
        # niftkFluid
        elif self.conversionRequested == 'rreg__to__niftkFluid' : 
            if not self.outFileAlreadyExists :
                self._itkPPtoNIFTKFluidConversion()
            return self.outFileName
        
        # feir
        elif self.conversionRequested == 'rreg__to__feir' :
            if not self.outFileAlreadyExists : 
                self._itkPPtoFEIRConversion()
            return self.outFileName
        
        # aladin f3d
        elif self.conversionRequested == 'aladin__to__f3d' :
            if not self.outFileAlreadyExists :
                print( 'Error, it should not have come this far..., sorry.' )
                return ''
            return self.outFileName
        
        # none so far...
        else :
            print( 'Not yet implemented, sorry.' )
            return ''




        
    def _itkPPtoFEIRConversion( self ) :
        if not self.intermedFileAlreadyExists:
            print( 'Implement starting FEIR initialisation...' )
            return
            
        cmd     = 'niftkTransformation'
        params  = ' -ti ' + self.targetImage
        params += ' -si ' + self.sourceImage
        params += ' -g '  + self.intermedFileName
        params += ' -o '  + self.outFileName


        if findExecutable( cmd ) == None :
            print( 'Error: Executable could not be found!' )
            return
        
        logFile = file( self.outLog ,'a+' )
        logFile.write( ' TASK \n' )
        logFile.write( '======\n\n' )
        logFile.write( ' command\n ---> ' + cmd + ' ' + params  + '\n\n' )
        logFile.flush()
        
        cmd       = shlex.split ( cmd + ' ' + params )
        self.proc = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
        
        logFile.write('\n TASK Done \n')
        logFile.write('===========\n\n')
        logFile.flush()
        logFile.close()
        
        return



    def _itkPPtof3dConversion( self ) :
        
        # Construct the name of the output dof
        # Check if it already exists
        
        cmd     = 'niftkITKppRigidDOFConversion'
        params  = ' -dof '    + self.correspondingDOFFile
        params += ' -target ' + self.targetImage
        params += ' -mat '    + self.outFileName
        params += ' -rot '
        
        if findExecutable( cmd ) == None :
            print( 'Error: Executable could not be found!' )
            return
        
        logFile = file( self.outLog ,'a+' )
        logFile.write( ' TASK \n' )
        logFile.write( '======\n\n' )
        logFile.write( ' command\n ---> ' + cmd + ' ' + params  + '\n\n' )
        logFile.flush()
        #self.commandTrack.append( cmdIn + ' ' + paramsIn )
        cmd       = shlex.split ( cmd + ' ' + params )
        self.proc = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
        
        logFile.write('\n TASK Done \n')
        logFile.write('===========\n\n')
        logFile.flush()
        logFile.close()

        return


    def _itkPPtoNIFTKFluidConversion( self ):
        
        # Construct the name of the output dof
        # Check if it already exists
        
        cmd     = 'niftkITKppRigidDOFConversion'
        params  = ' -dof '    + self.correspondingDOFFile
        params += ' -target ' + self.targetImage
        params += ' -ucl '    + self.outFileName
        
        if findExecutable( cmd ) == None :
            print( 'Error: Executable could not be found!' )
            return
        
        logFile = file( self.outLog ,'a+' )
        logFile.write( ' TASK \n' )
        logFile.write( '======\n\n' )
        logFile.write( ' command\n ---> ' + cmd + ' ' + params  + '\n\n' )
        logFile.flush()
        #self.commandTrack.append( cmdIn + ' ' + paramsIn )
        cmd       = shlex.split ( cmd + ' ' + params )
        self.proc = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
        
        logFile.write('\n TASK Done \n')
        logFile.write('===========\n\n')
        logFile.flush()
        logFile.close()

        return


if __name__ == '__main__' :
    
    # Minimal requirements to create an f3d task. 
#    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
#    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10.nii'
#    maskIn     = 'X:\\NiftyRegValidationWithTannerData\\nii\\masks\\S1i3AP10mask_breast.nii'
#    outputPath = 'X:\\NiftyRegValidationWithTannerData\\outF3d\\'
#    
#    
#    # Additionally the regDir of the initialisation needs to be specified....
#    initialDir = 'C:/data/regValidationWithTannerData/outRREG/def'
#    initialMethod = 'rreg'
#    targetMethod = 'reg_f3d'
#    initialiser = registrationInitialiser( initialMethod, targetMethod, sourceIn, targetIn, initialDir )
#    retVal = initialiser.getInitialisationFile()
#    
#    print( 'Dof file for initialisation is: ' + retVal )
#    





    # Minimal requirements to create an f3d task. 
    
#    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
#    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10.nii'
#    maskIn     = 'X:\\NiftyRegValidationWithTannerData\\nii\\masks\\S1i3AP10mask_breast.nii'
#    outputPath = 'X:\\NiftyRegValidationWithTannerData\\outF3d\\'
#    
#    
#    # Additionally the regDir of the initialisation needs to be specified....
#    initialDir = 'C:/data/regValidationWithTannerData/outRREG/def'
#    initialMethod = 'rreg'
#    targetMethod = 'niftkFluid'
#    initialiser = registrationInitialiser( initialMethod, targetMethod, sourceIn, targetIn, initialDir )
#    retVal = initialiser.getInitialisationFile()
#    
#    print( 'Dof file for initialisation is: ' + retVal )
    
    
    
    
    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10.nii'
    maskIn     = 'X:\\NiftyRegValidationWithTannerData\\nii\\masks\\S1i3AP10mask_breast.nii'
    outputPath = 'X:\\NiftyRegValidationWithTannerData\\outF3d\\'
    
    
    # Additionally the regDir of the initialisation needs to be specified....
    initialDir = 'C:/data/regValidationWithTannerData/outRREG/def'
    initialMethod = 'rreg'
    targetMethod = 'feir'
    initialiser = registrationInitialiser( initialMethod, targetMethod, sourceIn, targetIn, initialDir )
    retVal = initialiser.getInitialisationFile()
    
    print( 'Dof file for initialisation is: ' + retVal )

    