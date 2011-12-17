#! /usr/bin/env python 
# -*- coding: utf-8 -*-

# Evaluation: Comparing Tanner simulations with registration results

import fileCorrespondence as fc
from registrationTask import RegistrationTask
import subprocess, shlex, os, platform, bz2Wrap
from findExecutable import findExecutable



class evaluationTask :
    ''' This class implements the evaluation of the Tanner data.
    '''
    
    def __init__( self, referenceDeformDirIn, maskDirIn, registrationTaskIn, evalFileIn = 'eval.txt' )  :
        ''' The initialisation gives all the information which is needed to evaluate the
            registration quality
            The log file of the registration task will further be used for logging.
            Two different evaluation files will be produced: one for the breast mask, and one for the lesion mask.
            The default 'eval.txt' will result in eval_breast__regAppID__regID.txt and eval_lesion__regAppID__regID.txt            
            
            @param referenceDeformDirIn: 
                   directory where the reference deformation fields can be found (nii format)
            @param maskDirIn : 
                   directory where the masks for the breast and the lesion can be found (in 
                   the source image space!)
            @param registrationTaskIn: 
                   the class which describes the registration task (must be a subclass of regsitrationTask)
            @param evalFile
                   the text file will be used to write the evaluation results to. Give a complete path
        '''
        
        # check if the task is really a registration task...
        if isinstance( registrationTaskIn, RegistrationTask ) == False :
            print( 'ERROR: Expected a registration task as a second argument!' )
            return
         
        self.registrationTask       = registrationTaskIn

        # set the directory with the reference deformations...
        self.referenceDefromDir    = os.path.realpath( referenceDeformDirIn )
        self.maskDir               = os.path.realpath( maskDirIn            )
        
        # construct the name for the evaluation files.
        
        regAppID = self.registrationTask.regAppID
        regID    = self.registrationTask.regID
        
         
        self.evalFileBreast = os.path.splitext( evalFileIn )[0] + '_breast' + regAppID + regID + os.path.splitext( evalFileIn )[1]
        self.evalFileLesion = os.path.splitext( evalFileIn )[0] + '_lesion' + regAppID + regID + os.path.splitext( evalFileIn )[1]
        self.evalFileBreast = os.path.realpath( self.evalFileBreast )
        self.evalFileLesion = os.path.realpath( self.evalFileLesion)
        
        
                
        if platform.system() == 'Windows' :
            self.referenceDefromDir = self.referenceDefromDir.replace( '\\', os.altsep )
            self.maskDir            = self.maskDir.replace( '\\', os.altsep )
            self.evalFileBreast     = self.evalFileBreast.replace( '\\', os.altsep )
            self.evalFileLesion     = self.evalFileLesion.replace( '\\', os.altsep )
            
        # Track the executed commands and set up some vars
        self.commandTrack           = [] 
        self.temporaryFiles         = []
        self.taskComplete           = False
        
        self.maskBreast             = '' 
        self.maskLesion             = ''
        self.referenceDeform        = ''
        
        # match the three parameters above
        self.matchImages()
        
        


    def matchImages( self ) :
        ''' Tries to determine the breast / lesion mask and the deformation field 
        '''

        # match reference deformation
        deformFields         = fc.getDeformationFields( fc.getFiles( self.referenceDefromDir ) )
        self.referenceDeform = fc.matchTargetAndDeformationField( self.registrationTask.target, deformFields )        
        
        # match lesion mask (in source image space)
        lesionMasks     = fc.getSourceLesionMasks( fc.getFiles( self.maskDir ) )
        self.maskLesion = fc.matchSourceAndSoruceMask( self.registrationTask.source, lesionMasks )
        
        breastMasks     = fc.getSourceBreastMasks( fc.getFiles( self.maskDir ) )
        self.maskBreast = fc.matchSourceAndSoruceMask( self.registrationTask.source, breastMasks )
        
        # build complete paths
        self.referenceDeform = os.path.realpath( os.path.join( self.referenceDefromDir, self.referenceDeform ) )
        self.maskLesion      = os.path.realpath( os.path.join( self.maskDir, self.maskLesion ) )
        self.maskBreast      = os.path.realpath( os.path.join( self.maskDir, self.maskBreast ) )
        
        # checks and path for windows...
        if platform.system() == 'Windows' :
            self.referenceDeform = self.referenceDeform.replace( '\\', os.altsep )
            self.maskLesion      = self.maskLesion.replace     ( '\\', os.altsep )
            self.maskBreast      = self.maskBreast.replace     ( '\\', os.altsep )
       
        
       
        
    def run( self ) :
        ''' Only call this to start the evaluation
        	Note: areg and rreg are the same from the evaluation perspective
        '''
        if self.registrationTask.regCommand == 'rreg' :
            self.runRREGEvaluation()
            
        elif self.registrationTask.regCommand == 'areg' :
            self.runRREGEvaluation()
        
        elif self.registrationTask.regCommand == 'reg_aladin' :
            self.runRegAladinEvaluation()
        
        elif self.registrationTask.regCommand == 'reg_f3d' :
            self.runRegF3dEvaluation()
       
        elif self.registrationTask.regCommand == 'feir' :
            self.runFEIREvaluation()
        
        elif self.registrationTask.regCommand == 'niftkFluid' :
            self.runUcltkFluidEvaluation()
        
        elif self.registrationTask.regCommand == 'extern' :
            self.runAssumedRegEvaluation()    
        
        else :
            print( 'not yet implemented, sorry' )
        
        print( 'Done.' )
        
    
    
    
    def runRegAladinEvaluation( self ) :
        ''' The coordinate systems of reg_aladin and the itk evaluation coordinate system are 
            rotated by 180 degrees around the z-axis. Thus the matrix needs to be adapted
        '''
        
        print ( 'Starting reg_aladin evaluation...' )
        
        evalCmd           = 'niftkDeformationFieldTargetRegistrationErrorWithHistogram'

        # parameters for the breast region
        evalParamsBreast  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsBreast += ' -mat'                                   # Use a matrix rather than a deformation field
        evalParamsBreast += ' -matRot'                                # nifti and itk coordinates are rotated
        evalParamsBreast += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsBreast += ' -mask ' + self.maskBreast               # name of the mask image
        evalParamsBreast += ' -st '   + self.evalFileBreast           # output file name 
        evalParamsBreast += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsBreast += ' '       + self.registrationTask.outDOF  # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsBreast )
        
        # parameters for the lesion region
        evalParamsLesion  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsLesion += ' -mat'                                   # Use a matrix rather than a deformation field
        evalParamsLesion += ' -matRot'                                # nifti and itk coordinates are rotated
        evalParamsLesion += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsLesion += ' -mask ' + self.maskLesion               # name of the mask image
        evalParamsLesion += ' -st '   + self.evalFileLesion           # output file name 
        evalParamsLesion += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsLesion += ' '       + self.registrationTask.outDOF  # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsLesion )
        
        
    
    
    def runRegF3dEvaluation( self ):
        # generate position filed from cpp file (DOF)
        # generate displacement displacement vector field (DVF) from position field
        # convert the generated DVF (imitate registration in 180 degrees rotated system)
        # evaluate as usual...
        # clean up
        
        print( 'Starting reg_f3d evaluation...' )
        
        ###
        # Generate the position field
        ###
        
        # the position field name
        posField = os.path.splitext( self.registrationTask.outDOF )[0].split( 'cpp' )[0] + 'pos.nii' 
        
        cppConversionCommand = 'reg_transform'
        cppConversionParams  =  ' -target '  + self.registrationTask.target
        cppConversionParams  += ' -cpp2def ' + self.registrationTask.outDOF + ' ' + posField
        
        self.temporaryFiles.append( posField )
        self.runCmd( cppConversionCommand, cppConversionParams )
        
        ###
        # Generate the displacement field
        ###
        
        # displacement field name
        dispFieldNII = os.path.splitext( self.registrationTask.outDOF )[0].split( 'cpp' )[0] + 'dispN.nii'
        
        dispConversionParams =  ' -target '    + self.registrationTask.target
        dispConversionParams += ' -def2disp '  + posField + ' ' + dispFieldNII
        
        self.temporaryFiles.append( dispFieldNII )
        self.runCmd( cppConversionCommand, dispConversionParams )
        
        
        ###
        # Convert the vectors of the field to simulate the registration in a 180degress rotated coo-system
        ###
        dispFieldITK = os.path.splitext( self.registrationTask.outDOF )[0].split( 'cpp' )[0] + 'dispI.nii'
        
        vectConvCommand = 'niftkConvertNiftiVectorImage'
        vectConvParams  = ' -i ' +  dispFieldNII
        vectConvParams += ' -o ' + dispFieldITK
        
        self.temporaryFiles.append( dispFieldITK )
        self.runCmd( vectConvCommand, vectConvParams )
                 
        
        ###
        # Finally evaluate
        ###
        
        evalCmd           = 'niftkDeformationFieldTargetRegistrationErrorWithHistogram'

        # parameters for the breast region
        evalParamsBreast  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsBreast += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsBreast += ' -mask ' + self.maskBreast               # name of the mask image
        evalParamsBreast += ' -st '   + self.evalFileBreast           # output file name 
        evalParamsBreast += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsBreast += ' '       + dispFieldITK                  # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsBreast )
        
        # parameters for the lesion region
        evalParamsLesion  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsLesion += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsLesion += ' -mask ' + self.maskLesion               # name of the mask image
        evalParamsLesion += ' -st '   + self.evalFileLesion           # output file name 
        evalParamsLesion += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsLesion += ' '       + dispFieldITK                  # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsLesion )
        
        bz2Wrap.bZipFile( dispFieldITK )
        
        ###
        # Clean up after work is done...
        ###
        for item in self.temporaryFiles :
            os.remove( item )

        
        
        
        
    
    def runRREGEvaluation( self ) :
        # the registration produced a dof-file which needs to be evaluated...
        # 1) construct the homogeneous matrix with niftkITKpluplusRigidDofConversion
        #    (required params: dof target mat)
        # 2) 
        print( 'Starting rreg/areg evaluation...' )
        
        dof2MatrixCmd    = 'niftkITKppRigidDofConversion'
        compressCmd      = 'compress'
        evalCmd          = 'niftkDeformationFieldTargetRegistrationErrorWithHistogram'
        
        if self.checkExecutables( [dof2MatrixCmd, compressCmd, evalCmd] ) != True :
            return

        target        = self.registrationTask.target 
        wasCompressed = False
        
        if target.endswith('.Z') == True :
            print( 'Uncompressing image...' )
            wasCompressed = True
            target = target.replace('.Z','')
            self.runCmd( compressCmd, ' -d ' + self.registrationTask.target )
            
        
        # construct the name of the matrix:
        print( 'Constructing the homogeneous transformation matrix...' )
        outMatrix = self.registrationTask.outDOF.replace( 'dof.dof', 'mat.txt' )
        
        dof2MatrixParams  = ' -dof '    + self.registrationTask.outDOF    
        dof2MatrixParams += ' -target ' + target
        dof2MatrixParams += ' -mat '    + outMatrix
        
        self.temporaryFiles.append( outMatrix )
        self.runCmd( dof2MatrixCmd, dof2MatrixParams )
        
        # put the matrix into the evaluation software
        # breast and lesion...
        print( 'Evaluating the rigid matrix...' )
        evalParamsBreast =  ' -f'                              # Tanner simulations were in Lagragian frame
        evalParamsBreast += ' -mat'                            # Use a matrix rather than a deformation field
        evalParamsBreast += ' -mvalue 255'                     # value of the mask (inside)
        evalParamsBreast += ' -mask ' + self.maskBreast        # name of the mask image
        evalParamsBreast += ' -st '   + self.evalFileBreast    # output file name 
        evalParamsBreast += ' '       + self.referenceDeform   # field1 the reference deformation
        evalParamsBreast += ' '       + outMatrix              # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsBreast )
        
        evalParamsLesion =  ' -f'                              # Tanner simulations were in Lagragian frame
        evalParamsLesion += ' -mat'                            # Use a matrix rather than a deformation field
        evalParamsLesion += ' -mvalue 255'                     # value of the mask (inside)
        evalParamsLesion += ' -mask ' + self.maskLesion        # name of the mask image
        evalParamsLesion += ' -st '   + self.evalFileLesion    # output file name 
        evalParamsLesion += ' '       + self.referenceDeform   # field1 the reference deformation
        evalParamsLesion += ' '       + outMatrix              # field2 the matrix

        self.runCmd( evalCmd, evalParamsLesion )
        
        # compress again
        if wasCompressed == True :
            print( 'Compressing result again...' )
            self.runCmd( compressCmd, target )
            
        self.taskComplete = True




    def runFEIREvaluation( self ):
        ''' The FEIR result is rotated by 180 degrees. But in contrast to the 
            nifty coordinate system not only the vector components need to be 
            exchanged, but the whole image must be additionally rotated. This 
            is accounted for in the construction process of the deformation 
            field. 
        '''
        
        print( 'Starting FEIR evaluation...' )
        deformFiledComposerCmd = 'niftkCompose3DVectorImageFromComponentImages'
        evalCmd                = 'niftkDeformationFieldTargetRegistrationErrorWithHistogram'
        
        # Find out what the name of the deformation field components is...
        # and then generate the deformation vector field
        
        deformationVectorField = self.registrationTask.outDOF
        
        deformationVectorFieldExt = os.path.splitext( deformationVectorField )
        deformationVectorFieldExt = deformationVectorFieldExt[-1]
        if deformationVectorFieldExt != '.nii' :
            print('ERROR: DOF file has not the expected file extension.')
            return
        
        paramsFieldComp  = ' -x ' + self.registrationTask.outImage + '.x.mhd'
        paramsFieldComp += ' -y ' + self.registrationTask.outImage + '.y.mhd'
        paramsFieldComp += ' -z ' + self.registrationTask.outImage + '.z.mhd'
        
        paramsFieldComp += ' -invertX -invertY '
        paramsFieldComp += ' -flipXY '

        paramsFieldComp += ' -o ' + deformationVectorField

        self.runCmd( deformFiledComposerCmd, paramsFieldComp )
        
        # Check if the deformation field was successfully created
        if os.path.isfile( deformationVectorField ) != True : 
            print( 'ERROR: The deformation vector field was not generated! ' )
            return
        
        # the original data is no more needed 
        self.temporaryFiles.append( self.registrationTask.outImage + '.x.mhd' )
        self.temporaryFiles.append( self.registrationTask.outImage + '.y.mhd' )
        self.temporaryFiles.append( self.registrationTask.outImage + '.z.mhd' )
        self.temporaryFiles.append( self.registrationTask.outImage + '.x.raw' )
        self.temporaryFiles.append( self.registrationTask.outImage + '.y.raw' )
        self.temporaryFiles.append( self.registrationTask.outImage + '.z.raw' )
        
        
        # put the newly composed deformation vector field into the evaluation software
        # breast and lesion...
        print( 'Evaluating the deformation field...' )
        evalParamsBreast =  ' -f'                              # Tanner simulations were in Lagragian frame
        evalParamsBreast += ' -mvalue 255'                     # value of the mask (inside)
        evalParamsBreast += ' -mask ' + self.maskBreast        # name of the mask image
        evalParamsBreast += ' -st '   + self.evalFileBreast    # output file name 
        evalParamsBreast += ' '       + self.referenceDeform   # field1 the reference deformation
        evalParamsBreast += ' '       + deformationVectorField # field2 the registration result
        
        self.runCmd( evalCmd, evalParamsBreast )
        
        evalParamsLesion =  ' -f'                              # Tanner simulations were in Lagragian frame
        evalParamsLesion += ' -mvalue 255'                     # value of the mask (inside)
        evalParamsLesion += ' -mask ' + self.maskLesion        # name of the mask image
        evalParamsLesion += ' -st '   + self.evalFileLesion    # output file name 
        evalParamsLesion += ' '       + self.referenceDeform   # field1 the reference deformation
        evalParamsLesion += ' '       + deformationVectorField # field2 the registration result

        self.runCmd( evalCmd, evalParamsLesion )

        # Clean the original component images
        for item in self.temporaryFiles :
            os.remove( item )

        # if the compress command is available, then compress the deformation vector field (otherwise storage might become an issue)
        compressCmd    = '7z'
        compressParams = 'a ' + deformationVectorField + '.bz2 ' + deformationVectorField  
        self.runCmd( compressCmd, compressParams )

        # delete the original deformation vector field
        if os.path.isfile( deformationVectorField + '.bz2 ' ) :
            os.remove( deformationVectorField )
        
        self.registrationTask.outDOF = deformationVectorField + '.bz2 '
        
        self.taskComplete = True
    
    
    
    
    def runUcltkFluidEvaluation( self ) :
        #
        # The current implementation of the niftkFluid registration gives the DVF in 
        # voxel and NOT in real world spacing. Thus the dof must be modified by the
        # evaluation software (-vox2)  
        #
        print( 'Starting niftkFluid evaluation...' )
        
        evalCmd           = 'niftkDeformationFieldTargetRegistrationErrorWithHistogram'
        
        evalParamsBreast  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsBreast += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsBreast += ' -vox2'                                  # The DVF is given in voxels by niftkFluid 
        evalParamsBreast += ' -mask ' + self.maskBreast               # name of the mask image
        evalParamsBreast += ' -st '   + self.evalFileBreast           # output file name 
        evalParamsBreast += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsBreast += ' '       + self.registrationTask.outDOF  # field2 the niftk dof out
        
        self.runCmd( evalCmd, evalParamsBreast )
        
        evalParamsLesion  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsLesion += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsLesion += ' -vox2'                                  # The DVF is given in voxels by niftkFluid
        evalParamsLesion += ' -mask ' + self.maskLesion               # name of the mask image
        evalParamsLesion += ' -st '   + self.evalFileLesion           # output file name 
        evalParamsLesion += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsLesion += ' '       + self.registrationTask.outDOF  # field2 the niftk dof out

        self.runCmd( evalCmd, evalParamsLesion )
        self.taskComplete = True


    
    def runAssumedRegEvaluation( self ) :
        
        evalCmd           = 'niftkDeformationFieldTargetRegistrationErrorWithHistogram'

        # parameters for the breast region
        evalParamsBreast  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsBreast += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsBreast += ' -mask ' + self.maskBreast               # name of the mask image
        evalParamsBreast += ' -st '   + self.evalFileBreast           # output file name 
        evalParamsBreast += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsBreast += ' '       + self.registrationTask.outDOF  # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsBreast )
        
        # parameters for the lesion region
        evalParamsLesion  = ' -f'                                     # Tanner simulations were in Lagragian frame
        evalParamsLesion += ' -mvalue 255'                            # value of the mask (inside)
        evalParamsLesion += ' -mask ' + self.maskLesion               # name of the mask image
        evalParamsLesion += ' -st '   + self.evalFileLesion           # output file name 
        evalParamsLesion += ' '       + self.referenceDeform          # field1 the reference deformation
        evalParamsLesion += ' '       + self.registrationTask.outDOF  # field2 the matrix
        
        self.runCmd( evalCmd, evalParamsLesion ) 
        pass        
    
    
    
    
    def runCmd( self, cmdIn ='' , paramsIn = '' ) :
        ''' Any commands which are needed before or after the registration can be 
            performed with this convenience function
        '''
        if cmdIn == '' :
            print('Nothing to do for this task')
            return
        elif findExecutable( cmdIn ) == None :
            print( 'Could not find the executable. Return with no job done.' )
            return
        
        logFile = file( self.registrationTask.outLog ,'a+' )
        logFile.write( ' EVALUATION TASK \n' )
        logFile.write( '=================\n\n' )
        logFile.write( ' command\n ---> ' + cmdIn + ' ' + paramsIn  + '\n\n' )
        print        ( ' command\n ---> ' + cmdIn + ' ' + paramsIn            )
        logFile.flush()
        self.commandTrack.append( cmdIn + ' ' + paramsIn )
        cmd       = shlex.split ( cmdIn + ' ' + paramsIn )
        self.proc = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
        
        logFile.write('\n EVALUATION TASK Done \n')
        logFile.write('======================\n\n')
        logFile.flush()
        logFile.close()
        
        
        
        


    
    
    
    def checkExecutables( self, executableListIn ) :
        ''' Checks if the executables in the given list are in the path... 
        '''
        success = True
        
        for executable in executableListIn :

            if findExecutable( executable ) == None :
                print( 'ERROR: required executable is not in the path: ' + executable )
                success = False
            
        return success
        
        
        
         
        
        
        
        
if __name__ == '__main__' : 
    from rregRegistrationTask import *
    from aregRegistrationTask import *
    from aladinRegistrationTask import *
    from feirRegistrationTask import *
    
    evalRREG   = False 
    evalAREG   = False
    evalAladin = False
    evalFEIR   = True
    
    dirRefDeforms   = 'X:\\NiftyRegValidationWithTannerData\\nii\\deformationFields'
    dirMasks        = 'X:\\NiftyRegValidationWithTannerData\\nii'
    
    #
    # rreg test section
    #
    if evalRREG :
    
    	# Registration 
        rregTarget = 'Y:\\testData\\S1i3AP10.gipl.Z'
        rregSource = 'Y:\\testData\\S1i1.gipl.Z'

        rregRegTask = rregRegistrationTask( rregTarget, rregSource, 'Y:\\testData\\outRREG', 'myID', 'Y:\\testData\\AffineRegn.params', 0 )
        rregRegTask.printInfo()
        rregRegTask.run()


		# Evaluation        
        rregEvalOutFileBase = 'Y:\\testData\\outRREG\\evalOut.txt'
        
        rregEevalTask = evaluationTask( dirRefDeforms,
                                   dirMasks, 
                                   rregRegTask,
                                   rregEvalOutFileBase )
        
        rregEevalTask.run()
        
    #
    # areg test section
    #
    if evalAREG :
    
    	# Registration 
        aregTarget = 'Y:\\testData\\S1i3AP10.gipl.Z'
        aregSource = 'Y:\\testData\\S1i1.gipl.Z'

        aregRegTask = aregRegistrationTask( aregTarget, aregSource, 'Y:\\testData\\outAREG', 'myID', 'Y:\\testData\\AffineRegn.params', 0 )
        aregRegTask.printInfo()
        aregRegTask.run()


		# Evaluation        
        aregEvalOutFileBase = 'Y:\\testData\\outAREG\\evalOut.txt'
        
        aregEevalTask = evaluationTask( dirRefDeforms,
                                   dirMasks, 
                                   aregRegTask,
                                   aregEvalOutFileBase )
        
        aregEevalTask.run()
    
    #
    # reg_aladin test section
    #
    if evalAladin :
        print( 'Starting test of reg_aladin' )
        
        aladinSource       = 'Y:\\testData\\S1i1.nii'
        aladinTarget       = 'Y:\\testData\\S1i3AP10.nii'
        aladinMask         = 'Y:\\testData\\S1i3AP10mask_breast.nii'
        aladinOutPath      = 'Y:\\testData\\outAladin\\'
        aladinEvalFileBase = 'Y:\\testData\\outAladin\\evalOut.txt'
        
        aladinRigOnly    = True
        aladinRegID      = 'rig001'
        
        regAladinTask = aladinRegistrationTask( aladinTarget, aladinSource, aladinMask, aladinOutPath, aladinRegID, aladinRigOnly )
        regAladinTask.printInfo()
        regAladinTask.run()
        
        evalAladinTask = evaluationTask( dirRefDeforms, dirMasks, regAladinTask, aladinEvalFileBase )
        evalAladinTask.run()
    

    
    
    
    
    
    