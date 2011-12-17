#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from registrationTask import RegistrationTask
from findExecutable import findExecutable
from fileCorrespondence import getFiles
import os, platform


class niftkFluidRegistrationTask( RegistrationTask ) :
    ''' Implementation of the registration task which is done by FEIR (PHILIPS).
        
        outDOF takes the dummy file name for the deformation vector field. This will be 
        replaced with the combined vector valued image, which is constructed for the
        evaluation. 
    '''
    
    def __init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID, 
                  numLevels=3, regInterpolator=2, finalInterpolator=3, 
                  similarityNum=4, forceStr='cc', stepSize=0.125, 
                  minDeformMagnitude=0.01, checkSimilarity=1, maxItPerLevel=300, 
                  regirdStepSize=1.0, minCostFuncChange=1e-12, 
                  lameMu=0.01, lameLambda=0.0, startLevel=None, stopLevel=None, iteratingStepSize=None, 
                  numDilations=None, affineInitialisationFile=None ) :
        ''' 
                
                Kelvin's parameters (brain), used as default...
                -ln 3          3 multi resolution levels
                -ri 2          linear registration interpolator
                -fi 3          b-spline interpolation for final image and regridding
                -sim 4         Normalised Cross Correlation as similarity measure
                -force cc      Correlation Coefficient as force
                -ls 0.125      Step size factor
                -md 0.01       Minimum change in deformation magnitude between iterations       
                -cs 1          Check similarity during optimisation      
                -mi 300        Maximum iterations per level
                -rs 1.0        Regridding step size factor
                -mc 1e-12      Minimmum change in cost function
                  
            Complete help of niftkFluid (v2.4):
            
            Copyright (C) 2008-2011 University College London
            NifTK, Version: 2.4.0
            
              Implements fluid registration, initially based on Christensen, IEEE TMI Vol. 5, No. 10, Oct 1996.
            
              niftkFluid -ti <filename> -si <filename> -xo <filename> [options]
            
            *** [mandatory] ***
            
                -ti <filename>                   Target/Fixed image
                -si <filename>                   Source/Moving image
                -to <filename> [fluid.dof]       Output tranformation
            
            *** [options]   ***
            
                -xo <filename>                     Output deformation field image
            
                -oi <filename>                     Output resampled image
            
                -adofin <filename>                 Initial affine dof
                -it <filename>                     Initial fluid transform
                -tm <filename>                     Target/Fixed mask image
                -sm <filename>                     Source/Moving mask image
                -ji <filaname>                     Output jacobian image filename
                -vi <filename>                     Output vector image base filename
                -mji <filaname>                    Output jacobian image filename in Midas format
                -mvi <filename>                    Output vector image base filename in Midas format
                -ln <int>         [1]              Number of multi-resolution levels
                -bn <int>         [64]             Histogram binning
                -mi <int>         [300]            Maximum number of iterations per level
                -mc <float>       [1.0e-9]         Minimum change in cost function (NMI),
                                                   below which we stop that resolution level.
                -md <float>       [0.05]           Minimum change in deformation magnitude between iterations,
                                                   below which we stop that resolution level.
                -mj <float>       [0.5]            Minimum jacobian threshold, below which we regrid
                -ts <float>       [350]            Time step size
                -ls <float>       [1.0]            Largest step size factor (voxel unit)
                -is <float>       [0.5]            Iterating step size factor
                -rs <float>       [1.0]            Regridding step size factor
                -js <float>       [0.5]            Jacobian below zero step size factor
                -fi <int>         [4]              Choose final and gridding reslicing interpolator
                                                   1. Nearest neighbour
                                                   2. Linear
                                                   3. BSpline
                                                   4. Sinc
                -ri <int>         [2]              Choose regristration interpolator
                                                   1. Nearest neighbour
                                                   2. Linear
                                                   3. BSpline
                                                   4. Sinc
                -sim <int>        [9]              Choose similarity measure
                                                   1. Sum Of Squared Differences (SSD)
                                                   2. Mean of Square Differences (MSD)
                                                   3. Sum Of Absolute Differences (SAD)
                                                   4. Normalised Cross Correlation (NCC)
                                                   5. Ratio Image Uniformity (RIU)
                                                   6. Partioned Image Uniformity (PIU)
                                                   7. Joint Entropy (JE)
                                                   8. Mutual Information (MI)
                                                   9. Normalized Mutual Information (NMI)
                -d   <int>        [0]              Number of dilations of masks (if -tm or -sm used)
                -mmin <float>     [0.5]            Mask minimum threshold (if -tm or -sm used)
                -mmax <float>     [max]            Mask maximum threshold (if -tm or -sm used)
                -mip <float>      [0]              Moving image pad value
                -fip <float>      [0]              Fixed image pad value
                -hfl <float>                       Fixed image lower intensity limit
                -hfu <float>                       Fixed image upper intensity limit
                -hml <float>                       Moving image lower intensity limit
                -hmu <float>                       Moving image upper intensity limit
                -cs <int>         [1]              Check similarity during optimisation, 1 to check, 2 to skip.
                -stl <int>        [0]              Start Level (starts at zero like C++)
                -spl <int>        [ln - 1 ]        Stop Level (default goes up to number of levels minus 1, like C++)
                -force <string>   [force]          Registration force type
                                                   ssd - Christensen's SSD derived force (default)
                                                   ssdn - Christensen's SSD derived force normalised by the mean intensity
                                                   nmi - Bill's normalised mutual information derived force
                                                   parzen_nmi - Marc's normalised mutual information derived force based on Parzen window
                                                   cc - Freeborough's cross correlation derived force
                -ssd_smooth <int> [0]              Smooth the SSD registration force
                -rescale <upper lower>             Rescale the input images to the specified intensity range
                -lambda <float>   [0]              Lame constant - lambda
                -mu <float>       [0.01]           Lame constant - mu
                -abs_output <int> [0]              Output absoulte intensity value in resliced image
                -resample <float> [-1.0]           Resample the input images to this isotropic voxel size
                -drop_off         [0 0 0 0]        Smoothly drop off the image intensity at the edge of the mask
                                                   by applying dilations to the mask and then a Gaussian filter
                                                   Parameters: first dilation, second dilation, FWHM, mask threshold.
                                                   DRC fluid uses 3 2 2 127.
                -crop <int>       [0 128 128 128]  Crop the image to be the size of the object in the mask
                                                   extended by the first given number and padded the image by 0 to the size given
                                                   by the last three given numbers. DRC fluid uses 8 128 128 128.
                -cf <filename>                     Output padded and cropped fixed image.
                -cm <filename>                     Output padded and cropped moving image.
                -mcf <filename>                    Output cropped fixed image for Midas.
                -mcm <filename>                    Output cropped moving image for Midas.
                -moi <filename>                    Output resliced image for  Midas.
                -fdj                               Forward difference Jacobian calculation.
                -ar                                Apply abs filter to the regridded image.
                -mdmi <double> <double> [-1 0.1]   Maximum number of iterations allowed for step size less than min deformation.
                -py <double> <double> ...          Multi-resolution pyramid scheme shrinking factors (specified after -ln).
                -bf                                Blur final image in mulit-resolution pyramid. 
        '''
        
        # prepare the base class
        # the mask is handled differently in FEIR, thus give an empty string to the super-class
        RegistrationTask.__init__( self, targetIn, sourceIn, maskIn, outputPath, registrationID )
        
        self.dofID    = 'dof'
        self.regAppID = '__niftkFluid__'
        
        self.constructOutputFileNames()
        self.constructRegistationCommand( numLevels, regInterpolator, finalInterpolator, 
                                          similarityNum, forceStr, stepSize, 
                                          minDeformMagnitude, checkSimilarity, maxItPerLevel, 
                                          regirdStepSize, minCostFuncChange, 
                                          lameMu, lameLambda, 
                                          startLevel, stopLevel, iteratingStepSize, numDilations, affineInitialisationFile )
        
        
        
    def run( self ) :
        self.runRegistrationTask()
        
        
        
        
    def constructRegistationCommand( self, numLevels, regInterpolator, finalInterpolator, 
                                     similarityNum, forceStr, stepSize, 
                                     minDeformMagnitude, checkSimilarity, maxItPerLevel, 
                                     regirdStepSize, minCostFuncChange, 
                                     lameMu, lameLambda,
                                     startLevel, stopLevel, iteratingStepSize, numDilations, affineInitialisationFile ) :
        ''' Put together the parameters in the fashion it is expected by feir 
        '''
        self.regCommand = 'niftkFluid'
        
        # Outputs: source, target, output image and output transformation
        # feir reference (target) and template (source)
        self.regParams  = ' -ti ' + self.target    # the target image
        self.regParams += ' -si ' + self.source    # the source image
        self.regParams += ' -oi ' + self.outImage  # the output image 
        
        self.regParams += ' -xo ' + self.outDOF    # the output transformation as an image
        
        # the output transformation
        # self.regParams += ' -to ' + os.path.splitext( self.outDOF )[0] + '.dof'  
        # self.tmpFiles = []
        # self.tmpFiles.append( os.path.splitext( self.outDOF )[0] + '.dof' )
    
        # the mask in target space (as in niftyReg)
        if len( self.mask ) != 0:
            self.regParams += ' -tm ' + self.mask
        
        self.regParams += ' -ln '     + str( numLevels          )
        self.regParams += ' -ri '     + str( regInterpolator    )
        self.regParams += ' -fi '     + str( finalInterpolator  ) 
        self.regParams += ' -sim '    + str( similarityNum      )
        self.regParams += ' -force '  + str( forceStr           )
        self.regParams += ' -ls '     + str( stepSize           )
        self.regParams += ' -md '     + str( minDeformMagnitude )
        self.regParams += ' -cs '     + str( checkSimilarity    )
        self.regParams += ' -mi '     + str( maxItPerLevel      )
        self.regParams += ' -rs '     + str( regirdStepSize     )
        self.regParams += ' -mc '     + str( minCostFuncChange  )
        self.regParams += ' -mu '     + str( lameMu             )
        self.regParams += ' -lambda ' + str( lameLambda         )
        
        if startLevel != None :
            self.regParams += ' -stl ' + str( startLevel )
        if stopLevel != None :
            self.regParams += ' -spl ' + str( stopLevel )
            
        if iteratingStepSize != None :
            self.regParams += ' -is ' + str( iteratingStepSize )
            
        if numDilations != None :
            self.regParams += ' -d ' + str( numDilations )
        
        if affineInitialisationFile != None :
            self.regParams += ' -adofin ' + affineInitialisationFile
            
        
        
        
        
    def constructOutputFileNames( self ) :
        ''' Generates the names of the registation outputs...

        '''
        RegistrationTask.constructOutputBaseFileName( self )
        
        # 
        self.outDOF     = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_dof.nii'  ) )
        self.outImage   = os.path.realpath( os.path.join( self.outPath,  self.outFileBaseName + '_out.nii'  ) )
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
    
    from evaluationTask import *
    
    print( 'Starting test of python handling of niftkFluid' )
    print( 'Step 1: Registration' )
    
    sourceIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i1.nii'
    targetIn   = 'X:\\NiftyRegValidationWithTannerData\\nii\\S1i3AP10full.nii'
    maskIn     = 'X:\\NiftyRegValidationWithTannerData\\nii\\masks\\S1i3AP10mask_breast.nii'
    outputPath = 'X:\\NiftyRegValidationWithTannerData\\testFluid\\'
    
    regID      = 'def'
    
    ''' 
                
                Kelvin's parameters (brain), used as default...
                -ln 3          3 multi resolution levels
                -ri 2          linear registration interpolator
                -fi 3          b-spline interpolation for final image and regridding
                -sim 4         Normalised Cross Correlation as similarity measure
                -force cc      Correlation Coefficient as force
                -ls 0.125      Step size factor
                -md 0.01       Minimum change in deformation magnitude between iterations       
                -cs 1          Check similarity during optimisation      
                -mi 300        Maximum iterations per level
                -rs 1.0        Regridding step size factor
                -mc 1e-12      Minimmum change in cost function
    '''
    
    numLevels          = 1
    regInterpolator    = 2
    finalInterpolator  = 3
    similarityNum      = 4
    forceStr           = 'cc'
    stepSize           = 0.125
    minDeformMagnitude = 0.01
    checkSimilarity    = 1
    maxItPerLevel      = 1
    regirdStepSize     = 1.0
    minCostFuncChange  = 1e-15
    lameMu             = 0.01
    lameLambda         = 0
    startLevel         = None
    stopLevel          = None
    iteratingStepSize  = 0.7
    numDilations       = 5
    
    affInitFile = 'X:/NiftyRegValidationWithTannerData/testDOFConversion/ucl.txt'
    
    
    
    regTask = niftkFluidRegistrationTask( targetIn, sourceIn, maskIn, outputPath, regID,
                                          numLevels, regInterpolator, finalInterpolator, 
                                          similarityNum, forceStr, stepSize, 
                                          minDeformMagnitude, checkSimilarity, maxItPerLevel, 
                                          regirdStepSize, minCostFuncChange, 
                                          lameMu, lameLambda,
                                          startLevel, stopLevel, iteratingStepSize, numDilations, affInitFile )
    
    regTask.printInfo()
    regTask.run()
    
    
    print( 'Step 2: Evaluation' )
    fluidEvalOutFile = outputPath + 'evalOut.txt'
    
    # Some standard file locations
    dirRefDeforms   = 'X:\\NiftyRegValidationWithTannerData\\nii\\deformationFields'
    dirMasks        = 'X:\\NiftyRegValidationWithTannerData\\nii'
        
    evalTask        = evaluationTask( dirRefDeforms,
                                      dirMasks, 
                                      regTask,
                                      fluidEvalOutFile )
    
    evalTask.run()
    
    print( 'Done...' )
    
    
    
    
    
    
    