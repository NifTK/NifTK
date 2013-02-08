/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkBSplineTransform.h"
#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkParzenWindowNMIDerivativeForceGenerator.h"
#include "itkSSDRegistrationForceFilter.h"
#include "itkBSplineSmoothVectorFieldFilter.h"
#include "itkBSplineBendingEnergyConstraint.h"
#include "itkSumLogJacobianDeterminantConstraint.h"
#include "itkInterpolateVectorFieldFilter.h"
#include "itkFFDGradientDescentOptimizer.h"
#include "itkFFDSteepestGradientDescentOptimizer.h"
#include "itkFFDConjugateGradientDescentOptimizer.h"
#include "itkFFDMultiResolutionMethod.h"
#include <string>
#include "itkTransformFileWriter.h"
#include "itkLinearlyInterpolatedDerivativeFilter.h"

/*!
 * \file niftkFFD.cxx
 * \page niftkFFD
 * \section niftkFFDSummary Implements FFD registration, initially based on Rueckert et. al., IEEE TMI Vol. 18, No. 8, Aug 1999.
 */

void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements FFD registration, initially based on Rueckert et. al., IEEE TMI Vol. 18, No. 8, Aug 1999." << std::endl;
  std::cout << "  Includes force generation via Crum et. al., IPMI 2003, 378-387" << std::endl;
  std::cout << "  Includes force generation via Modat et. al." << std::endl;
  std::cout << "  Includes incompressibility constraint via Rohlfing et. al., IEEE TMI Vol. 22, No. 6, Jun 2003." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -ti <filename> -si <filename> -xo <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -ti <filename>                 Target/Fixed image " << std::endl;
  std::cout << "    -si <filename>                 Source/Moving image " << std::endl;
  std::cout << "    -xo <filename>  [output.vtk]   Output control point image " << std::endl << std::endl;      
}

void EndUsage()
{
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -oi <filename>                 Output resampled image" << std::endl << std::endl;
  std::cout << "    -ot <filename>                 Output tranformation" << std::endl << std::endl;  
  std::cout << "    -adofin <filename>             Initial affine dof" << std::endl;
  std::cout << "    -invertAffine                  Inverts the affine dof" << std::endl << std::endl;
  std::cout << "    -tm <filename>                 Target/Fixed mask image" << std::endl;
  std::cout << "    -sm <filename>                 Source/Moving mask image" << std::endl;
  std::cout << "    -ji <filaname>                 Output jacobian image filename" << std::endl;
  std::cout << "    -vi <filename>                 Output vector image filename" << std::endl;
  std::cout << "    -sx <float>     [5.0]          Final spacing along x axis (in mm)" << std::endl;
  std::cout << "    -sy <float>     [5.0]          Final spacing along y axis (in mm)" << std::endl;
  std::cout << "    -sz <float>     [5.0]          Final spacing along z axis (in mm)" << std::endl;
  std::cout << "    -bn <int>       [64]           Number of histogram bins" << std::endl;
  std::cout << "    -mi <int>       [300]          Maximum number of iterations per level" << std::endl;
  std::cout << "    -mw <float>     [0.01]         Constraint weighting (0-1)" << std::endl;
  std::cout << "    -mc <float>     [0.00001]      Minimum change in cost function (NMI) " << std::endl;
  std::cout << "    -md <float>     [0.01]         Minimum change in control point displacement magnitude between iterations" << std::endl;  
  std::cout << "    -mf <float>     [0.0000001]    Minimum force vector magnitude" << std::endl;
  std::cout << "    -ls <float>     [1.0]          Largest step size factor (will be multiplied by max voxel dimension)" << std::endl;
  std::cout << "    -ss <float>     [0.01]         Smallest step size factor (will be multiplied by largest step size)" << std::endl;
  std::cout << "    -is <float>     [0.5]          Iterating step size factor" << std::endl;
  std::cout << "    -js <float>     [0.5]          Jacobian below zero step size factor" << std::endl;
  std::cout << "    -sv <float>     [0.01]         Gaussian smoothing sigma" << std::endl;
  std::cout << "    -fi <int>       [4]            Choose final reslicing interpolator" << std::endl;
  std::cout << "                                     1. Nearest neighbour" << std::endl;
  std::cout << "                                     2. Linear" << std::endl;
  std::cout << "                                     3. BSpline" << std::endl;
  std::cout << "                                     4. Sinc" << std::endl;
  std::cout << "    -gi <int>       [3]            Choose regridding interpolator" << std::endl;
  std::cout << "                                     1. Nearest neighbour" << std::endl;
  std::cout << "                                     2. Linear" << std::endl;
  std::cout << "                                     3. BSpline" << std::endl;
  std::cout << "                                     4. Sinc" << std::endl;
  std::cout << "    -ri <int>       [2]            Choose registration interpolator" << std::endl;
  std::cout << "                                     1. Nearest neighbour" << std::endl;
  std::cout << "                                     2. Linear" << std::endl;
  std::cout << "                                     3. BSpline" << std::endl;
  std::cout << "                                     4. Sinc" << std::endl; 
  std::cout << "    -opt <int>      [1]            Choose optimization strategy" << std::endl;
  std::cout << "                                     1. Conjugate gradient ascent" << std::endl;
  std::cout << "                                     2. Steepest gradient ascent" << std::endl;
  std::cout << "    -con <int>      [1]            Choose constraint" << std::endl;
  std::cout << "                                     1. BSpline smoothness (Rueckert et. al.)" << std::endl;
  std::cout << "                                     2. Incompressibility (Rohlfing et. al.)" << std::endl;
  std::cout << "    -d   <int>      [0]            Number of dilations of masks (if -tm or -sm used)" << std::endl;  
  std::cout << "    -mmin <float>   [0.5]          Mask minimum threshold (if -tm or -sm used)" << std::endl;
  std::cout << "    -mmax <float>   [max]          Mask maximum threshold (if -tm or -sm used)" << std::endl;
  std::cout << "    -cg                            Use constraint gradient (very expensive, and not valid if -con 2 specified)" << std::endl;                            
  std::cout << "    -bc                            Use Bill Crum's histogram derivative rather than Marc's Parzen Window derivative." << std::endl;
  std::cout << "    -ntc                           No deformation threshold check" << std::endl;
  std::cout << "    -nsc                           No similarity check" << std::endl;
  std::cout << "    -njc                           No Jacobian below zero check" << std::endl;
  std::cout << "    -wt                            Write transformed moving image after each iteration. Filename tmp.moving.<iteration>.nii" << std::endl;
  std::cout << "    -wf                            Write fixed image after each iteration. Filename tmp.fixed.<iteration>.nii" << std::endl;
  std::cout << "    -wj                            Write jacobian image after each resolution level. Filename tmp.jacobian.<level>.nii" << std::endl;
  std::cout << "    -wv                            Write vector image after each resolution level. Filename tmp.vector.<level>.vtk" << std::endl;
  std::cout << "    -wcp                           Write control point image after each resolution level. Filename tmp.cp.<level>.vtk" << std::endl;  
  std::cout << "    -wnf                           Write next deformation field image at each iteration. Filename tmp.deformation.<iteration>.vtk" << std::endl;
  std::cout << "    -wnp                           Write next control point image at each iteration. Filename tmp.next.<iteration>.vtk" << std::endl;
  std::cout << "    -wfi                           Write force image at each iteration. Filename tmp.force.<iteration>.vtk" << std::endl;
  std::cout << "    -wr                            Write regridded image, each time we regrid. Filename tmp.regridded.<iteration>.nii" << std::endl;
  std::cout << "    -sf                            Scale force vectors by gradient of transformed moving image" << std::endl;
  std::cout << "    -svm                           When doing -sf, scale by vector magnitude rather than componentwise" << std::endl;
  std::cout << "    -sbi                           Don't use BSpline smoothing of gradient field before interpolating from voxel to control point level" << std::endl;
  std::cout << "    -ln <int>       [3]            Number of multi-resolution levels" << std::endl;  
  std::cout << "    -stl <int>      [0]            Start Level (starts at zero like C++)" << std::endl;
  std::cout << "    -spl <int>      [ln - 1 ]      Stop Level (default goes up to number of levels minus 1, like C++)" << std::endl;
  std::cout << "    -rescale        [lower upper]  Rescale the input images to the specified intensity range (only valid if -bc set)" << std::endl;
  std::cout << "    -mip <float>    [0]            Moving image pad value" << std::endl;
  std::cout << "    -hfl <float>                   Similarity measure, fixed image lower intensity limit" << std::endl;
  std::cout << "    -hfu <float>                   Similarity measure, fixed image upper intensity limit" << std::endl;
  std::cout << "    -hml <float>                   Similarity measure, moving image lower intensity limit" << std::endl;
  std::cout << "    -hmu <float>                   Similarity measure, moving image upper intensity limit" << std::endl;  
}

/**
 * \brief Does FFD based registration.
 */
int main(int argc, char** argv)
{
  const    unsigned int   Dimension = 2;
  typedef  float          PixelType;
  typedef  short          OutputPixelType; 
  typedef  float          DeformableScalarType; 

  std::string fixedImage;
  std::string movingImage;
  std::string outputImage;
  std::string fixedMask;
  std::string movingMask;  
  std::string outputTransformation;
  std::string adofinFilename; 
  std::string outputControlPointImageFileName = "output";
  std::string outputControlPointImageFileExt = "vtk";
  std::string jacobianFile;
  std::string jacobianExt;
  std::string vectorFile;
  std::string vectorExt;  
  std::string tmpFilename;
  std::string tmpBaseName;
  std::string tmpFileExtension;  
  int bins = 64;
  int levels = 3;
  int startLevel = 0;
  int stopLevel = levels -1;
  int iters = 300;
  int finalInterp = 4;
  int regriddingInterp = 3;
  int registrationInterp = 2;
  int dilations = 0;
  int opt=1;
  int constraint=1;
  double spacing[Dimension];
  spacing[0] = 5.0;
  spacing[1] = 5.0;
  if (Dimension > 2)
    {
      spacing[2] = 5.0;    
    }
  double lowerIntensity = 0; 
  double higherIntensity = 0;      
  double dummyDefault = -987654321;
  double weight = 0.01;  
  double minCostTol   = 0.00001;
  double minDefChange = 0.01;
  double minVecMag    = 0.0000001;
  double minJac = 0.3;
  double iReduce = 0.5;
  double rReduce = 0.5;
  double jReduce = 0.5;
  double minStepFactor = 0.01;
  double maxStepFactor = 1.0;
  double sigma = 0.01;
  double maskMinimumThreshold = 0.5;
  double maskMaximumThreshold = std::numeric_limits<PixelType>::max();
  double intensityFixedLowerBound = dummyDefault;
  double intensityFixedUpperBound = dummyDefault;
  double intensityMovingLowerBound = dummyDefault;
  double intensityMovingUpperBound = dummyDefault;
  double movingImagePadValue = 0;
  bool cgrad = false;
  bool dumpCP = false;
  bool dumpJac = false;
  bool dumpVec = false;
  bool dumpTransformed = false;
  bool dumpFixed = false;
  bool dumpNextParam = false;
  bool dumpNextField = false;
  bool dumpForceImage = false;
  bool scaleForce = false;
  bool scaleByComponents = true;
  bool smoothBeforeInterpolating = true;
  bool useBC = false;
  bool dumpRegridded = false;
  bool isRescaleIntensity = false;
  bool similarityCheck = true;
  bool deformationThresholdCheck = true;
  bool jacobianBelowZeroCheck = true;
  bool userSetPadValue = false;
  bool invertAffine = false;
  

  // Parsing
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      StartUsage(argv[0]);
      EndUsage();
      return -1;
    }
    else if(strcmp(argv[i], "-ti") == 0){
      fixedImage=argv[++i];
      std::cout << "Set -ti=" << fixedImage << std::endl;
    }
    else if(strcmp(argv[i], "-si") == 0){
      movingImage=argv[++i];
      std::cout << "Set -si=" << movingImage << std::endl;
    }
    else if(strcmp(argv[i], "-oi") == 0){
      outputImage=argv[++i];
      std::cout << "Set -oi=" << outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-ot") == 0){
      outputTransformation = argv[++i];
      std::cout << "Set -ot=" << outputTransformation << std::endl;
    }    
    else if (strcmp(argv[i], "-adofin") == 0) {
      adofinFilename = argv[++i];
      std::cout << "Set -adofin=" << adofinFilename << std::endl;
    }    
    else if(strcmp(argv[i], "-tm") == 0){
      fixedMask=argv[++i];
      std::cout << "Set -tm=" << fixedMask << std::endl;
    }
    else if(strcmp(argv[i], "-sm") == 0){
      movingMask=argv[++i];
      std::cout << "Set -sm=" << movingMask << std::endl;
    }
    else if(strcmp(argv[i], "-sx") == 0){
      spacing[0]=atof(argv[++i]);
      std::cout << "Set -sx=" << niftk::ConvertToString(spacing[0]) << std::endl;
    }
    else if(strcmp(argv[i], "-sy") == 0){
      spacing[1]=atof(argv[++i]);
      std::cout << "Set -sy=" << niftk::ConvertToString(spacing[1]) << std::endl;
    }
    else if(strcmp(argv[i], "-sz") == 0){
      spacing[2]=atof(argv[++i]);
      std::cout << "Set -sz=" << niftk::ConvertToString(spacing[2]) << std::endl;
    }
    else if(strcmp(argv[i], "-bn") == 0){
      bins=atoi(argv[++i]);
      std::cout << "Set -bn=" << niftk::ConvertToString(bins) << std::endl;
    }
    else if(strcmp(argv[i], "-ln") == 0){
      levels=atoi(argv[++i]);
      std::cout << "Set -ln=" << niftk::ConvertToString(levels) << std::endl;
    }
    else if(strcmp(argv[i], "-stl") == 0){
      startLevel=atoi(argv[++i]);
      std::cout << "Set -stl=" << niftk::ConvertToString(startLevel) << std::endl;
    }
    else if(strcmp(argv[i], "-spl") == 0){
      stopLevel=atoi(argv[++i]);
      std::cout << "Set -spl=" << niftk::ConvertToString(stopLevel) << std::endl;
    }
    else if(strcmp(argv[i], "-mi") == 0){
      iters=atoi(argv[++i]);
      std::cout << "Set -mi=" << niftk::ConvertToString(iters) << std::endl;
    }
    else if(strcmp(argv[i], "-mw") == 0){
      weight=atof(argv[++i]);
      std::cout << "Set -mw=" << niftk::ConvertToString(weight) << std::endl;
    }
    else if(strcmp(argv[i], "-mc") == 0){
      minCostTol=atof(argv[++i]);
      std::cout << "Set -mc=" << niftk::ConvertToString(minCostTol) << std::endl;
    }
    else if(strcmp(argv[i], "-md") == 0){
      minDefChange=atof(argv[++i]);
      std::cout << "Set -md=" << niftk::ConvertToString(minDefChange) << std::endl;
    }
    else if(strcmp(argv[i], "-mf") == 0){
      minVecMag=atof(argv[++i]);
      std::cout << "Set -mf=" << niftk::ConvertToString(minVecMag) << std::endl;
    }
    else if(strcmp(argv[i], "-ls") == 0){
      maxStepFactor=atof(argv[++i]);
      std::cout << "Set -ls=" << niftk::ConvertToString(maxStepFactor) << std::endl;
    }
    else if(strcmp(argv[i], "-ss") == 0){
      minStepFactor=atof(argv[++i]);
      std::cout << "Set -ss=" << niftk::ConvertToString(minStepFactor) << std::endl;
    }
    else if(strcmp(argv[i], "-is") == 0){
      iReduce=atof(argv[++i]);
      std::cout << "Set -is=" << niftk::ConvertToString(iReduce) << std::endl;
    }
    else if(strcmp(argv[i], "-rs") == 0){
      rReduce=atof(argv[++i]);
      std::cout << "Set -rs=" << niftk::ConvertToString(rReduce) << std::endl;
    }
    else if(strcmp(argv[i], "-js") == 0){
      jReduce=atof(argv[++i]);
      std::cout << "Set -js=" << niftk::ConvertToString(jReduce) << std::endl;
    }
    else if(strcmp(argv[i], "-fi") == 0){
      finalInterp=atoi(argv[++i]);
      std::cout << "Set -fi=" << niftk::ConvertToString(finalInterp) << std::endl;
    }
    else if(strcmp(argv[i], "-gi") == 0){
      regriddingInterp=atoi(argv[++i]);
      std::cout << "Set -gi=" << niftk::ConvertToString(regriddingInterp) << std::endl;
    }
    else if(strcmp(argv[i], "-opt") == 0){
      opt=atoi(argv[++i]);
      std::cout << "Set -opt=" << niftk::ConvertToString(opt) << std::endl;
    }
    else if(strcmp(argv[i], "-con") == 0){
      constraint=atoi(argv[++i]);
      std::cout << "Set -con=" << niftk::ConvertToString(constraint) << std::endl;
    }    
    else if(strcmp(argv[i], "-ri") == 0){
      registrationInterp=atoi(argv[++i]);
      std::cout << "Set -ri=" << niftk::ConvertToString(registrationInterp) << std::endl;
    }    
    else if(strcmp(argv[i], "-sv") == 0){
      sigma=atof(argv[++i]);
      std::cout << "Set -sv=" << niftk::ConvertToString(sigma) << std::endl;
    }
    else if(strcmp(argv[i], "-cg") == 0){
      cgrad=true;
      std::cout << "Set -cg=" << niftk::ConvertToString(cgrad) << std::endl;
    }
    else if(strcmp(argv[i], "-wt") == 0){
      dumpTransformed=true;
      std::cout << "Set -wt=" << niftk::ConvertToString(dumpTransformed) << std::endl;
    }
    else if(strcmp(argv[i], "-wf") == 0){
      dumpFixed=true;
      std::cout << "Set -wf=" << niftk::ConvertToString(dumpFixed) << std::endl;
    }
    else if(strcmp(argv[i], "-wnp") == 0){
      dumpNextParam=true;
      std::cout << "Set -wf=" << niftk::ConvertToString(dumpNextParam) << std::endl;
    }        
    else if(strcmp(argv[i], "-wnf") == 0){
      dumpNextField=true;
      std::cout << "Set -wf=" << niftk::ConvertToString(dumpNextField) << std::endl;
    }
    else if(strcmp(argv[i], "-wfi") == 0){
      dumpForceImage=true;
      std::cout << "Set -wfi=" << niftk::ConvertToString(dumpForceImage) << std::endl;
    }
    else if(strcmp(argv[i], "-sf") == 0){
      scaleForce=true;
      std::cout << "Set -sf=" << niftk::ConvertToString(scaleForce) << std::endl;
    }
    else if(strcmp(argv[i], "-invertAffine") == 0){
      invertAffine=true;
      std::cout << "Set -invertAffine=" << niftk::ConvertToString(invertAffine) << std::endl;
    }            
    else if(strcmp(argv[i], "-svm") == 0){
      scaleByComponents=false;
      std::cout << "Set -svm=" << niftk::ConvertToString(scaleByComponents) << std::endl;
    }
    else if(strcmp(argv[i], "-sbi") == 0){
      smoothBeforeInterpolating=false;
      std::cout << "Set -sbi=" << niftk::ConvertToString(smoothBeforeInterpolating) << std::endl;
    }
    else if(strcmp(argv[i], "-bc") == 0){
      useBC=true;
      std::cout << "Set -bc=" << niftk::ConvertToString(useBC) << std::endl;
    }
    else if(strcmp(argv[i], "-ntc") == 0){
      deformationThresholdCheck=false;
      std::cout << "Set -ntc=" << niftk::ConvertToString(deformationThresholdCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-nsc") == 0){
      similarityCheck=false;
      std::cout << "Set -nsc=" << niftk::ConvertToString(similarityCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-njc") == 0){
      jacobianBelowZeroCheck=false;
      std::cout << "Set -njc=" << niftk::ConvertToString(jacobianBelowZeroCheck) << std::endl;
    }                                                                                                    
    else if(strcmp(argv[i], "-ji") == 0){
      tmpFilename = argv[++i];
      std::string::size_type idx = tmpFilename.rfind('.', tmpFilename.length());
      if (idx == std::string::npos)
        {
          std::cerr << argv[0] << ":\tIf you specify -ji you must have a file extension" << std::endl;
          return -1;          
        }
      jacobianFile = tmpFilename.substr(0, idx);
      jacobianExt = tmpFilename.substr(idx+1);
      if (jacobianExt.length() == 0)
        {
          std::cerr << argv[0] << ":\tIf you specify -ji i would expect an extension like .nii" << std::endl;  
          return -1;        
        }
      std::cout << "Set jacobianFile=" << jacobianFile << " and jacobianExt=" << jacobianExt << std::endl;
    }
    else if(strcmp(argv[i], "-vi") == 0){
      tmpFilename = argv[++i];
      std::string::size_type idx = tmpFilename.rfind('.', tmpFilename.length());
      if (idx == std::string::npos)
        {
          std::cerr << argv[0] << ":\tIf you specify -vi you must have a file extension" << std::endl;
          return -1;          
        }
      vectorFile = tmpFilename.substr(0, idx);
      vectorExt = tmpFilename.substr(idx+1);
      if (vectorExt.length() == 0)
        {
          std::cerr << argv[0] << ":\tIf you specify -vi i would expect an extension like .vtk" << std::endl;
          return -1;        
        }
      std::cout << "Set vectorFile=" << vectorFile << " and vectorExt=" << vectorExt << std::endl;
    }
    else if(strcmp(argv[i], "-xo") == 0){
      tmpFilename = argv[++i];
      std::string::size_type idx = tmpFilename.rfind('.', tmpFilename.length());
      if (idx == std::string::npos)
        {
          std::cerr << argv[0] << ":\tIf you specify -xo you must have a file extension" << std::endl;
          return -1;          
        }
      outputControlPointImageFileName = tmpFilename.substr(0, idx);
      outputControlPointImageFileExt = tmpFilename.substr(idx+1);
      if (outputControlPointImageFileExt.length() == 0)
        {
          std::cerr << argv[0] << ":\tIf you specify -xo i would expect an extension like .vtk" << std::endl;
          return -1;        
        }
      std::cout << "Set outputControlPointImageFileName=" << outputControlPointImageFileName << " and outputControlPointImageFileExt=" << outputControlPointImageFileExt << std::endl;
    }    
    
    else if(strcmp(argv[i], "-wv") == 0){
      dumpVec=true;
      std::cout << "Set -wv=" << niftk::ConvertToString(dumpVec) << std::endl;
    }
    else if(strcmp(argv[i], "-wj") == 0){
      dumpJac=true;
      std::cout << "Set -wj=" << niftk::ConvertToString(dumpJac) << std::endl;
    }
    else if(strcmp(argv[i], "-wcp") == 0){
      dumpCP=true;
      std::cout << "Set -wcp=" << niftk::ConvertToString(dumpCP) << std::endl;
    }    
    else if(strcmp(argv[i], "-wr") == 0){
      dumpRegridded=true;
      std::cout << "Set -wr=" << niftk::ConvertToString(dumpRegridded) << std::endl;
    }
    else if(strcmp(argv[i], "-d") == 0){
      dilations=atoi(argv[++i]);
      std::cout << "Set -d=" << niftk::ConvertToString(dilations) << std::endl;
    }    
    else if(strcmp(argv[i], "-mmin") == 0){
      maskMinimumThreshold=atof(argv[++i]);
      std::cout << "Set -mmin=" << niftk::ConvertToString(maskMinimumThreshold) << std::endl;
    }
    else if(strcmp(argv[i], "-mmax") == 0){
      maskMaximumThreshold=atof(argv[++i]);
      std::cout << "Set -mmax=" << niftk::ConvertToString(maskMaximumThreshold) << std::endl;
    }
    else if(strcmp(argv[i], "-hfl") == 0){
      intensityFixedLowerBound=atof(argv[++i]);
      std::cout << "Set -hfl=" << niftk::ConvertToString(intensityFixedLowerBound) << std::endl;
    }        
    else if(strcmp(argv[i], "-hfu") == 0){
      intensityFixedUpperBound=atof(argv[++i]);
      std::cout << "Set -hfu=" << niftk::ConvertToString(intensityFixedUpperBound) << std::endl;
    }        
    else if(strcmp(argv[i], "-hml") == 0){
      intensityMovingLowerBound=atof(argv[++i]);
      std::cout << "Set -hml=" << niftk::ConvertToString(intensityMovingLowerBound) << std::endl;
    }        
    else if(strcmp(argv[i], "-hmu") == 0){
      intensityMovingUpperBound=atof(argv[++i]);
      std::cout << "Set -hmu=" << niftk::ConvertToString(intensityMovingUpperBound) << std::endl;
    }
    else if(strcmp(argv[i], "-mip") == 0){
      movingImagePadValue=atof(argv[++i]);
      userSetPadValue=true;
      std::cout << "Set -mip=" << niftk::ConvertToString(movingImagePadValue) << std::endl;
    }   
    else if(strcmp(argv[i], "-rescale") == 0){
      isRescaleIntensity=true;
      lowerIntensity=atof(argv[++i]);
      higherIntensity=atof(argv[++i]);
      std::cout << "Set -rescale=" << niftk::ConvertToString(lowerIntensity) << "-"+niftk::ConvertToString(higherIntensity) << std::endl;
    }            
    else{
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validation
  if (fixedImage.length() <= 0 || movingImage.length() <= 0 )
    {
      StartUsage(argv[0]);
      std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
      return -1;
    }
  
  if(spacing[0] <= 0){
    std::cerr << argv[0] << "\tThe xSpacing must be > 0" << std::endl;
    return -1;
  }

  if(spacing[1] <= 0){
    std::cerr << argv[0] << "\tThe ySpacing must be > 0" << std::endl;
    return -1;
  }

  if(Dimension > 2 && spacing[2] <= 0){
    std::cerr << argv[0] << "\tThe zSpacing must be > 0" << std::endl;
    return -1;
  }

  if(bins <= 0){
    std::cerr << argv[0] << "\tThe number of bins must be > 0" << std::endl;
    return -1;
  }

  if(levels <= 0){
    std::cerr << argv[0] << "\tThe number of levels must be > 0" << std::endl;
    return -1;
  }

  if(iters <= 0){
    std::cerr << argv[0] << "\tThe number of iters must be > 0" << std::endl;
    return -1;
  }

  if(weight < 0 || weight > 1){
    std::cerr << argv[0] << "\tThe weight must be between 0 and 1" << std::endl;
    return -1;
  }

  if(minCostTol <= 0){
    std::cerr << argv[0] << "\tThe minCostTol must be > 0" << std::endl;
    return -1;
  }

  if(minDefChange <= 0){
    std::cerr << argv[0] << "\tThe minDefChange must be > 0" << std::endl;
    return -1;
  }

  if(minStepFactor <= 0){
    std::cerr << argv[0] << "\tThe minStepFactor must be > 0" << std::endl;
    return -1;
  }

  if(maxStepFactor <= 0){
    std::cerr << argv[0] << "\tThe maxStepFactor must be > 0" << std::endl;
    return -1;
  }

  if(minVecMag <= 0){
    std::cerr << argv[0] << "\tThe minVecMag must be > 0" << std::endl;
    return -1;
  }

  if(minJac <= 0){
    std::cerr << argv[0] << "\tThe minJac must be > 0" << std::endl;
    return -1;
  }

  if(iReduce <= 0){
    std::cerr << argv[0] << "\tThe iReduce must be > 0" << std::endl;
    return -1;
  }

  if(rReduce <= 0){
    std::cerr << argv[0] << "\tThe rReduce must be > 0" << std::endl;
    return -1;
  }

  if(jReduce <= 0){
    std::cerr << argv[0] << "\tThe jReduce must be > 0" << std::endl;
    return -1;
  }

  if(dilations < 0){
    std::cerr << argv[0] << "\tThe number of dilations must be >= 0" << std::endl;
    return -1;

  }

  if(constraint < 1 || constraint > 2){
    std::cerr << argv[0] << "\tThe constraint must be 1 or 2" << std::endl;
    return -1;
  }

  if(finalInterp < 1 || finalInterp > 4){
    std::cerr << argv[0] << "\tThe final interpolator must be 1,2,3 or 4" << std::endl;
    return -1;
  }

  if(regriddingInterp < 1 || regriddingInterp > 4){
    std::cerr << argv[0] << "\tThe regridding interpolator must be 1,2,3 or 4" << std::endl;
    return -1;
  }

  if(registrationInterp < 1 || registrationInterp > 4){
    std::cerr << argv[0] << "\tThe registration interpolator must be 1,2,3 or 4" << std::endl;
    return -1;
  }

  if(opt < 1 || opt > 2){
    std::cerr << argv[0] << "\tThe optimizer must be 1 or 2" << std::endl;
    return -1;
  }

  if (constraint == 2 && cgrad == true)
    {
      std::cerr << argv[0] << "If constraint==2, then constraint gradient must be off." << std::endl;
      return -1;      
    }

  if((intensityFixedLowerBound != dummyDefault && (intensityFixedUpperBound == dummyDefault ||
                                                   intensityMovingLowerBound == dummyDefault ||
                                                   intensityMovingUpperBound == dummyDefault))
    ||
     (intensityFixedUpperBound != dummyDefault && (intensityFixedLowerBound == dummyDefault ||
                                                   intensityMovingLowerBound == dummyDefault ||
                                                   intensityMovingUpperBound == dummyDefault))
    || 
     (intensityMovingLowerBound != dummyDefault && (intensityMovingUpperBound == dummyDefault ||
                                                    intensityFixedLowerBound == dummyDefault ||
                                                    intensityFixedUpperBound == dummyDefault))
    ||
     (intensityMovingUpperBound != dummyDefault && (intensityMovingLowerBound == dummyDefault || 
                                                    intensityFixedLowerBound == dummyDefault ||
                                                    intensityFixedUpperBound == dummyDefault))
                                                    )
  {
    std::cerr << argv[0] << "\tIf you specify any of -hfl, -hfu, -hml or -hmu you should specify all of them" << std::endl;
    return -1;
  }

  // A starter for 10. Here are some typedefs to get warmed up.
  typedef itk::Image< PixelType, Dimension >       InputImageType; 
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;  
  typedef itk::ImageFileReader< InputImageType  >  FixedImageReaderType;
  typedef itk::ImageFileReader< InputImageType >   MovingImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  OutputImageWriterType;
  typedef InputImageType::SpacingType              SpacingType;
  
  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  FixedImageReaderType::Pointer  fixedMaskReader  =  FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();
  
  // Load both images to be registered.
  try 
    { 
      std::cout << "Loading fixed image:" + fixedImage << std::endl;
      fixedImageReader->SetFileName(  fixedImage );
      fixedImageReader->Update();
      std::cout << "Done" << std::endl;
      
      std::cout << "Loading moving image:" + movingImage << std::endl;
      movingImageReader->SetFileName( movingImage );
      movingImageReader->Update();
      std::cout << "Done" << std::endl;

      if (fixedMask.length() > 0)
        {
          std::cout << "Loading fixed mask:" << fixedMask << std::endl;
          fixedMaskReader->SetFileName(fixedMask);
          fixedMaskReader->Update();
          std::cout << "Done" << std::endl;
        }
      
      if (movingMask.length() > 0)
        {
          std::cout << "Loading moving mask:" + movingMask << std::endl;
          movingMaskReader->SetFileName(movingMask);
          movingMaskReader->Update();
          std::cout << "Done" << std::endl;
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ExceptionObject caught !";
      return -2;
    }                

  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;  
  FactoryType::Pointer factory = FactoryType::New();

  typedef itk::BSplineTransform<InputImageType, double, Dimension, float > TransformType;
  TransformType::Pointer transform = TransformType::New();

  typedef itk::BSplineBendingEnergyConstraint<InputImageType, double, Dimension, float > BendingEnergyConstraintType;
  BendingEnergyConstraintType::Pointer smoothnessConstraint = BendingEnergyConstraintType::New();
  smoothnessConstraint->SetTransform(transform);
  
  typedef itk::SumLogJacobianDeterminantConstraint<InputImageType, double, Dimension, float > IncompressibilityConstraintType;
  IncompressibilityConstraintType::Pointer incompressibilityConstraint = IncompressibilityConstraintType::New();
  incompressibilityConstraint->SetTransform(transform);
  
  typedef itk::FFDSteepestGradientDescentOptimizer<InputImageType, InputImageType, double, float> SteepestOptimizerType;
  SteepestOptimizerType::Pointer steepestOptimizer = SteepestOptimizerType::New();

  typedef itk::FFDConjugateGradientDescentOptimizer<InputImageType, InputImageType, double, float> ConjugateOptimizerType;
  ConjugateOptimizerType::Pointer conjugateOptimizer = ConjugateOptimizerType::New();

  typedef itk::FFDGradientDescentOptimizer<InputImageType, InputImageType, double, float> OptimizerType;
  
  typedef itk::ParzenWindowNMIDerivativeForceGenerator<InputImageType, InputImageType, double, float> ParzenForceGeneratorFilterType;
  ParzenForceGeneratorFilterType::Pointer parzenForceFilter = ParzenForceGeneratorFilterType::New();
  
  typedef itk::NMILocalHistogramDerivativeForceFilter<InputImageType, InputImageType, float> HistogramForceGeneratorFilterType;
  HistogramForceGeneratorFilterType::Pointer histogramForceFilter = HistogramForceGeneratorFilterType::New();

  typedef itk::SSDRegistrationForceFilter<InputImageType, InputImageType, float> SSDForceFilterType;
  SSDForceFilterType::Pointer ssdForceFilter = SSDForceFilterType::New();

  typedef itk::LinearlyInterpolatedDerivativeFilter<InputImageType, InputImageType, double, float> GradientFilterType;
  GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
  
  typedef itk::BSplineSmoothVectorFieldFilter< float, Dimension> SmoothFilterType;
  SmoothFilterType::Pointer smoothFilter = SmoothFilterType::New();

  typedef itk::InterpolateVectorFieldFilter< float, Dimension> VectorInterpolatorType;
  VectorInterpolatorType::Pointer vectorInterpolator = VectorInterpolatorType::New();

  typedef itk::NMIImageToImageMetric<InputImageType, InputImageType> MetricType;
  MetricType::Pointer metric = MetricType::New();

  typedef itk::MaskedImageRegistrationMethod<InputImageType> SingleResImageRegistrationMethodType;
  SingleResImageRegistrationMethodType::Pointer singleResMethod = SingleResImageRegistrationMethodType::New();

  typedef itk::FFDMultiResolutionMethod<InputImageType, double, Dimension, float>   MultiResImageRegistrationMethodType;
  MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  typedef itk::IterationUpdateCommand CommandType;
  CommandType::Pointer command = CommandType::New();
  
  OptimizerType* optimizer;
  
  if (opt == 2)
    {
      optimizer = static_cast<OptimizerType*>(steepestOptimizer.GetPointer()); 
    }
  else
    {
      optimizer = static_cast<OptimizerType*>(conjugateOptimizer.GetPointer());  
    }
  
  SpacingType finalSpacing;
  for (unsigned int i = 0; i < Dimension; i++)
    {
      finalSpacing[i] = spacing[i];
    }

  // Setup transformation with affine transform if supplied.
  if (adofinFilename.length() > 0)
  {
    FactoryType::TransformType::Pointer globalTransform = factory->CreateTransform(adofinFilename);
    
    if (invertAffine)
    {
      itk::EulerAffineTransform<double, Dimension, Dimension>* affineTransform = dynamic_cast<itk::EulerAffineTransform<double, Dimension, Dimension>*>(globalTransform.GetPointer());
      affineTransform->InvertTransformationMatrix(); 
      std::cout << "inverted:" << std::endl << affineTransform->GetFullAffineMatrix() << std::endl; 
    }
    
    transform->SetGlobalTransform(globalTransform); 
  }

  if (constraint == 2)
    {
      metric->SetConstraint(incompressibilityConstraint);
        
    }
  else
    {
      metric->SetConstraint(smoothnessConstraint);
    }

  metric->SetUseConstraintGradient(cgrad);
  metric->SetUseDerivativeScaleArray(false);
  metric->ComputeGradientOff(); // internally to the itkImageToImageMetric, it calculates derivative of moving image, which we dont need here.
  metric->SetHistogramSize(bins, bins);

  optimizer->SetDeformableTransform(transform);
  optimizer->SetRegriddingInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)regriddingInterp));
  optimizer->SetMaximize(true);
  optimizer->SetMaximumNumberOfIterations(iters);
  optimizer->SetIteratingStepSizeReductionFactor(iReduce);
  optimizer->SetJacobianBelowZeroStepSizeReductionFactor(jReduce);
  optimizer->SetMinimumDeformationMagnitudeThreshold(minDefChange);
  optimizer->SetMinimumGradientVectorMagnitudeThreshold(minVecMag);
  optimizer->SetMinimumJacobianThreshold(minJac);
  optimizer->SetScaleForceVectorsByGradientImage(scaleForce);
  optimizer->SetScaleByComponents(scaleByComponents);
  optimizer->SetSmoothGradientVectorsBeforeInterpolatingToControlPointLevel(smoothBeforeInterpolating);
  optimizer->SetSmoothFilter(smoothFilter);
  optimizer->SetInterpolatorFilter(vectorInterpolator);
  optimizer->SetMinimumSimilarityChangeThreshold(minCostTol);
  optimizer->SetCheckMinDeformationMagnitudeThreshold(deformationThresholdCheck);
  optimizer->SetCheckSimilarityMeasure(similarityCheck);
  optimizer->SetCheckJacobianBelowZero(jacobianBelowZeroCheck);
  
  singleResMethod->SetMetric(metric);
  singleResMethod->SetTransform(transform.GetPointer());
  singleResMethod->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)registrationInterp));
  singleResMethod->SetOptimizer(optimizer);
  singleResMethod->SetIterationUpdateCommand(command);
  singleResMethod->SetSigma(sigma);
  singleResMethod->SetNumberOfDilations(dilations);
  singleResMethod->SetFixedMaskMinimum((PixelType)maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum((PixelType)maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum((PixelType)maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum((PixelType)maskMaximumThreshold);

  if (useBC)
    {

      metric->SetUseParzenFilling(false);
      singleResMethod->SetMaskImageDirectly(false);

      // If we are just rescaling, and not using thresholds below, we just set the rescale values.
      if (isRescaleIntensity)
        {
          singleResMethod->SetRescaleFixedImage(true);
          singleResMethod->SetRescaleFixedMinimum((PixelType)lowerIntensity);
          singleResMethod->SetRescaleFixedMaximum((PixelType)higherIntensity);
          singleResMethod->SetRescaleMovingImage(true);
          singleResMethod->SetRescaleMovingMinimum((PixelType)lowerIntensity);
          singleResMethod->SetRescaleMovingMaximum((PixelType)higherIntensity);
        }

      // However, if we are thresholding as well, we need to set rescaling properly.
      if (intensityFixedLowerBound != dummyDefault || 
          intensityFixedUpperBound != dummyDefault || 
          intensityMovingLowerBound != dummyDefault || 
          intensityMovingUpperBound != dummyDefault )
        {
          if (isRescaleIntensity)
            {
              singleResMethod->SetRescaleFixedImage(true);
              singleResMethod->SetRescaleFixedBoundaryValue((PixelType)lowerIntensity-1);
              singleResMethod->SetRescaleFixedLowerThreshold((PixelType)intensityFixedLowerBound);
              singleResMethod->SetRescaleFixedUpperThreshold((PixelType)intensityFixedUpperBound);
              singleResMethod->SetRescaleFixedMinimum((PixelType)lowerIntensity);
              singleResMethod->SetRescaleFixedMaximum((PixelType)higherIntensity);
              
              singleResMethod->SetRescaleMovingImage(true);
              singleResMethod->SetRescaleMovingBoundaryValue((PixelType)lowerIntensity-1);
              singleResMethod->SetRescaleMovingLowerThreshold((PixelType)intensityMovingLowerBound);
              singleResMethod->SetRescaleMovingUpperThreshold((PixelType)intensityMovingUpperBound);              
              singleResMethod->SetRescaleMovingMinimum((PixelType)lowerIntensity);
              singleResMethod->SetRescaleMovingMaximum((PixelType)higherIntensity);

              metric->SetIntensityBounds((PixelType)lowerIntensity, (PixelType)higherIntensity, (PixelType)lowerIntensity, (PixelType)higherIntensity);
              
              histogramForceFilter->SetFixedLowerPixelValue((PixelType)lowerIntensity);
              histogramForceFilter->SetFixedUpperPixelValue((PixelType)higherIntensity);
              histogramForceFilter->SetMovingLowerPixelValue((PixelType)lowerIntensity);
              histogramForceFilter->SetMovingUpperPixelValue((PixelType)higherIntensity);
              
            }
          else
            {
              metric->SetIntensityBounds((PixelType)intensityFixedLowerBound, (PixelType)intensityFixedUpperBound, (PixelType)intensityMovingLowerBound, (PixelType)intensityMovingUpperBound);
              
              histogramForceFilter->SetFixedLowerPixelValue((PixelType)intensityFixedLowerBound);
              histogramForceFilter->SetFixedUpperPixelValue((PixelType)intensityFixedUpperBound);
              histogramForceFilter->SetMovingLowerPixelValue((PixelType)intensityMovingLowerBound);
              histogramForceFilter->SetMovingUpperPixelValue((PixelType)intensityMovingUpperBound);              
            }
        }
      
      histogramForceFilter->SetMetric(metric);
      histogramForceFilter->SetScaleToSizeOfVoxelAxis(true);
      
      optimizer->SetForceFilter(histogramForceFilter);
    }  
  else
    {
      if (intensityFixedLowerBound != dummyDefault || 
          intensityFixedUpperBound != dummyDefault || 
          intensityMovingLowerBound != dummyDefault || 
          intensityMovingUpperBound != dummyDefault )
        {

          singleResMethod->SetRescaleFixedBoundaryValue(-1);
          singleResMethod->SetRescaleFixedLowerThreshold((PixelType)intensityFixedLowerBound);
          singleResMethod->SetRescaleFixedUpperThreshold((PixelType)intensityFixedUpperBound);

          singleResMethod->SetRescaleMovingBoundaryValue(-1);
          singleResMethod->SetRescaleMovingLowerThreshold((PixelType)intensityMovingLowerBound);
          singleResMethod->SetRescaleMovingUpperThreshold((PixelType)intensityMovingUpperBound);
        }

      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedMinimum(0);
      singleResMethod->SetRescaleFixedMaximum(bins-1);

      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingMinimum(0);
      singleResMethod->SetRescaleMovingMaximum(bins-1); 

      gradientFilter->SetMovingImageLowerPixelValue(0);
      gradientFilter->SetMovingImageUpperPixelValue(bins-1);

      metric->SetIntensityBounds(0, bins-1, 0, bins-1);
          
      parzenForceFilter->SetFixedLowerPixelValue(0);
      parzenForceFilter->SetFixedUpperPixelValue(bins-1);
      parzenForceFilter->SetMovingLowerPixelValue(0);
      parzenForceFilter->SetMovingUpperPixelValue(bins-1);              

      singleResMethod->SetMaskImageDirectly(true);
      singleResMethod->SetUseFixedMask(true);
      singleResMethod->SetUseMovingMask(true);
      singleResMethod->SetThresholdFixedMask(true);
      singleResMethod->SetThresholdMovingMask(true);

      metric->SetUseParzenFilling(true);
      
      gradientFilter->SetTransform(transform);
      
      parzenForceFilter->SetMetric(metric);
      parzenForceFilter->SetScalarImageGradientFilter(gradientFilter);
      parzenForceFilter->SetScaleToSizeOfVoxelAxis(false);
 
      optimizer->SetForceFilter(parzenForceFilter);
    }
  
  ssdForceFilter->SetMetric(metric);
  optimizer->SetForceFilter(ssdForceFilter);
  
  multiResMethod->SetInitialTransformParameters(transform->GetParameters());
  multiResMethod->SetSingleResMethod(singleResMethod);
  multiResMethod->SetTransform(transform);
  multiResMethod->SetMinStepSizeFactor(minStepFactor);
  multiResMethod->SetMaxStepSizeFactor(maxStepFactor);
  multiResMethod->SetFinalControlPointSpacing(finalSpacing);
  if (stopLevel > levels - 1)
    {
      stopLevel = levels - 1;
    }
  multiResMethod->SetNumberOfLevels(levels);
  multiResMethod->SetStartLevel(startLevel);
  multiResMethod->SetStopLevel(stopLevel);
  
  // Debug. I could provide file names as command line params, but there would be loads of them?
  multiResMethod->SetWriteJacobianImageAtEachLevel(dumpJac);
  if (dumpJac && (jacobianFile.length() == 0 || jacobianExt.length() == 0))
    {
      jacobianFile = "jacobian";
      jacobianExt = "nii";
    }
  multiResMethod->SetJacobianImageFileName(jacobianFile);
  multiResMethod->SetJacobianImageFileExtension(jacobianExt);
  
  multiResMethod->SetWriteVectorImageAtEachLevel(dumpVec);
  if (dumpVec && (vectorFile.length() == 0 || vectorExt.length() == 0))
    {
      vectorFile = "vector";
      vectorExt = "vtk";
    }
  multiResMethod->SetVectorImageFileName(vectorFile);
  multiResMethod->SetVectorImageFileExtension(vectorExt);  
  
  multiResMethod->SetWriteParametersAtEachLevel(dumpCP);
  if (dumpCP && (outputControlPointImageFileName.length() == 0 || outputControlPointImageFileExt.length() == 0))
    {
      // This shouldnt happen, as we set reasonable defaults above.
      outputControlPointImageFileName="tmp.cp";
      outputControlPointImageFileExt="vtk";
    }
  multiResMethod->SetParameterFileName(outputControlPointImageFileName);
  multiResMethod->SetParameterFileExt(outputControlPointImageFileExt);
  
  metric->SetWriteTransformedMovingImage(dumpTransformed);
  metric->SetTransformedMovingImageFileName("tmp.moving");
  metric->SetTransformedMovingImageFileExt("nii");
  metric->SetWriteFixedImage(dumpFixed);  
  metric->SetFixedImageFileName("tmp.fixed");
  metric->SetFixedImageFileExt("nii");

  optimizer->SetWriteDeformationField(dumpNextField);
  optimizer->SetDeformationFieldFileName("tmp.deformation");
  optimizer->SetDeformationFieldFileExt("vtk");
  optimizer->SetWriteNextParameters(dumpNextParam);
  optimizer->SetNextParametersFileName("tmp.next");
  optimizer->SetNextParametersFileExt("vtk");
  optimizer->SetWriteForceImage(dumpForceImage);
  optimizer->SetForceImageFileName("tmp.force");
  optimizer->SetForceImageFileExt("vtk");
  optimizer->SetWriteRegriddedImage(dumpRegridded);
  optimizer->SetRegriddedImageFileName("tmp.regridded");
  optimizer->SetRegriddedImageFileExt("nii");
  
  typedef itk::ImageRegistrationFilter<InputImageType, OutputImageType, Dimension, double, DeformableScalarType>  RegistrationFilterType;  
  RegistrationFilterType::Pointer regFilter = RegistrationFilterType::New();
  regFilter->SetMultiResolutionRegistrationMethod(multiResMethod);
  regFilter->SetFixedImage(fixedImageReader->GetOutput());
  regFilter->SetMovingImage(movingImageReader->GetOutput());
  if (fixedMask.length() > 0)
    {
      std::cout << "Using fixedMask" << std::endl;
      regFilter->SetFixedMask(fixedMaskReader->GetOutput());
    }
  if (movingMask.length() > 0)
    {
      std::cout << "Using movingMask" << std::endl;
      regFilter->SetMovingMask(movingMaskReader->GetOutput());
    }  

  // If we havent asked for output, turn off reslicing.
  if (outputImage.length() > 0)
    {
      regFilter->SetDoReslicing(true);
    }
  else
    {
      regFilter->SetDoReslicing(false);
    }  
  regFilter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterp));
  
  // Set the padding value
  if (!userSetPadValue)
    {
      InputImageType::IndexType index;
      for (unsigned int i = 0; i < Dimension; i++)
        {
          index[i] = 0;  
        }
      movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
      std::cout << "Set movingImagePadValue to:" << niftk::ConvertToString(movingImagePadValue) << std::endl;
    }  
  metric->SetTransformedMovingImagePadValue(movingImagePadValue);
  optimizer->SetRegriddedMovingImagePadValue(movingImagePadValue);
  regFilter->SetResampledMovingImagePadValue(movingImagePadValue);
  
  // Run the registration
  regFilter->Update();

  if (outputImage.length() > 0)
    {
      OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
      outputImageWriter->SetFileName(  outputImage );
      outputImageWriter->SetInput(regFilter->GetOutput());
      outputImageWriter->Update();      
    }

  if (outputControlPointImageFileName.length() > 0 && outputControlPointImageFileExt.length() > 0)
    {
      multiResMethod->WriteControlPointImage(outputControlPointImageFileName + "." + outputControlPointImageFileExt);
    }

  // Save the transform
  if (outputTransformation.length() > 0)
    {
      typedef itk::TransformFileWriter TransformFileWriterType;
      TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
      transformFileWriter->SetInput(singleResMethod->GetTransform());
      transformFileWriter->SetFileName(outputTransformation); 
      transformFileWriter->Update(); 
    }
  
  return 0;
}
