#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import subprocess, shlex, os, time
from findExecutable import findExecutable


class RegistrationTask :
    ''' A class which represents a registration task 
        Each registration algorithm will need to implement a subclass for functionality
        Handling of registration initialisations need to be considered in the subclasses. 
    '''

    # outPath
    # outImage
    # outDOF    -> construct name by subclass!
    # outLog
    
    def __init__( self, targetIn, sourceIn, maskIn, outputPath = None, registrationID = 'defID' ) :
        ''' Initialisation of this class
            Parameters:
                 targetIn          target image (complete path)
                 soruceIn          source image (complete path)
                 outputPath        path for the registration results
                 registrationID    string which identifies this registration task
        '''
        
        # set the class members to default values
        self.target       = targetIn        # target image
        self.source       = sourceIn        # source image
        self.mask         = maskIn          # mask (in target space)
        self.regCommand   = ''              # executable registration program
        self.regParams    = ''              # parameters of the registration program
        self.regID        = registrationID  # string with the registration ID e.g. for differentiation in the file name
        self.regDuration  = 0               # duration of the registration in seconds 
        self.taskComplete = False           # Will be set to true, once the registration has completed
        self.commandTrack = []              # Lists the executed commands
        
        
        if outputPath == None :
            self.outPath = os.getcwd()
        else :
            self.outPath = outputPath
        
        # initialise output file names and construct them before execution
        self.outDOF   = ''
        self.outImage = ''
        self.outLog   = ''
        
        # helpers for file name construction 
        # separator between source and target
        self.regTo   = '__regTo__'
        self.regAppID = '__MYREG__' 
        
        
        
    
    
    def runRegistrationTask( self ) :
        ''' Runs the registration task, if the command is found in the path
        '''

        logFile = file( self.outLog ,'a+' )
        logFile.write( ' REGISTRATION TASK \n' )
        logFile.write( '===================\n\n' )
        
        # find the executalbe first...
        if ( findExecutable( self.regCommand ) == None ) :
            print( 'ERROR: Could not find the executable. Return with no job done.' )
            logFile.write('\n ERROR: Could not find the executable. Return with no job done. \n')
            logFile.write('\n REGISTRATION TASK Done \n')
            logFile.write('========================\n\n')
            logFile.flush()
            logFile.close()
    
            return
            
        # create the logging file
        logFile.write( ' registration command\n ---> ' + self.regCommand  + ' ' + self.regParams + '\n\n' )
        logFile.flush()
        
        # Prepare the command and run it
        self.commandTrack.append( self.regCommand + ' ' + self.regParams )
        cmd       = shlex.split ( self.regCommand + ' ' + self.regParams )
        tic = time.clock()
        self.proc = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
        toc = time.clock()
        
        self.regDuration = toc - tic
        
        logFile.write('\nDuration: %.4f s\n' %(toc - tic))
        logFile.write('\n REGISTRATION TASK Done \n')
        logFile.write('========================\n\n')
        logFile.flush()
        logFile.close()
    
    
    
    
    def runTask( self, cmdIn ='' , paramsIn = '' ) :
        ''' Any commands which are needed before or after the registration can be 
            performed with this convenience function
        '''
        if cmdIn == '' :
            print('Nothing to do for this task')
            return
        elif findExecutable( cmdIn ) == None :
            print( 'Could not find the executable: ' + cmdIn + '\nReturn with no job done.' )
            return
        
        logFile = file( self.outLog ,'a+' )
        logFile.write( ' TASK \n' )
        logFile.write( '======\n\n' )
        logFile.write( ' command\n ---> ' + cmdIn + ' ' + paramsIn  + '\n\n' )
        logFile.flush()
        self.commandTrack.append( cmdIn + ' ' + paramsIn )
        cmd       = shlex.split ( cmdIn + ' ' + paramsIn )
        self.proc = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
        
        logFile.write('\n TASK Done \n')
        logFile.write('===========\n\n')
        logFile.flush()
        logFile.close()
        
    
    
    def constructOutputBaseFileName( self ) :
        ''' Only the base name is usually in common between the registration methods
            This requires 'self.regAppID' to be set correctly in the subclass
        '''
        ( sPath, sFileName ) = os.path.split( self.source )
        ( tPath, tFileName ) = os.path.split( self.target )
        sStrings = str.split( sFileName, '.' ) 
        tStrings = str.split( tFileName, '.' )
        
        self.outFileBaseName = sStrings[0] + self.regTo + tStrings[0] + self.regAppID + self.regID 
        

     
        
    def printInfo( self ) :
        ''' Prints information about the current class to the command line.
        '''
        
        print( 'This is the RegistrationTask class' )
        print( ' --> reg. command:    %s' % self.regCommand )
        print( ' --> reg. parameters: %s' % self.regParams  )
        print( ' --> Target image:    %s' % self.target     )
        print( ' --> Mask image:      %s' % self.mask       )
        print( ' --> Source image:    %s' % self.source     )
        print( ' --> Output path:     %s' % self.outPath    )
        print( ' --> Output image:    %s' % self.outImage   )
        print( ' --> Output dofs:     %s' % self.outDOF     )
        print( ' --> Output log:      %s' % self.outLog     )
        
    
   
if __name__ == '__main__' : 
    
    T = 'X:/NiftyRegValidationWithTannerData/nii/S1i3AP10.nii'
    S = 'X:/NiftyRegValidationWithTannerData/nii/S1i1.nii'
    
    task = RegistrationTask( T, S )
    task.printInfo()
    