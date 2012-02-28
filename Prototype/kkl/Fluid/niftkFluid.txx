/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-16 13:25:04 +0100 (Mon, 16 May 2011) $
 Revision          : $Revision: 6173 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkFluidDeformableTransform.h"
#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkFluidPDEFilter.h"
#include "itkFluidGradientDescentOptimizer.h"
#include "itkMultiResolutionDeformableImageRegistrationMethod.h"
#include "itkFluidMultiResolutionMethod.h"
#include "itkTransformFileWriter.h"
#include "itkSSDRegistrationForceFilter.h"
#include "itkExtendedBrainMaskWithSmoothDropOffCompositeFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkLinearlyInterpolatedDerivativeFilter.h"
#include "itkParzenWindowNMIDerivativeForceGenerator.h"
#include "itkCrossCorrelationDerivativeForceFilter.h"
#include "ConversionUtils.h"
#include "itkUCLRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkAbsImageFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
using namespace std;


void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements fluid-based registration, initially based on Christensen, IEEE TMI Vol. 5, No. 10, Oct 1996." << std::endl;
  std::cout << std::endl; 
  std::cout << "  Symmetric option based on Avants et al. MEDIA, 2008" << std::endl; 
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -ti <filename> -si <filename> -xo <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -ti <filename>                   Target/Fixed image " << std::endl;
  std::cout << "    -si <filename>                   Source/Moving image " << std::endl;
  std::cout << "    -xo <filename>                   Output deformation field image " << std::endl << std::endl;    
  std::cout << "  Example for symmetric regsitration: " << std::endl; 
  std::cout << "  niftkFluid -ti bl.img -si rp.img -oi dummy.img -xo dummy.nii.gz -xof dummy.nii.gz " << std::endl; 
  std::cout << "             -sym bl_rp_warp.nii.gz bl_rp_inv.nii.gz bl_rp_warp.img bl_rp_warp_inv.img bl_rp_warp.str bl_rpwarp_inv.str " << std::endl; 
  std::cout << "             -ln 1 -ri 2 -fi 4 -sim 4 -force cc -ls 0.0625 -md 0.0025 -cs 0 -mi 500 -rs 1.0 -mc 1e-12 -abs_output 1 " << std::endl; 
  std::cout << "             -ar -asgd 1 1 -0.6 1e-10 1 1 -asgd_mask mask.img" << std::endl << std::endl; 
}

void EndUsage()
{
  std::cout << "    ///////////////////////////////////////////////////////////////////////////////////" << std::endl; 
  std::cout << "    // Some options may not work with symmetric regsitration.                          " << std::endl; 
  std::cout << "    ///////////////////////////////////////////////////////////////////////////////////" << std::endl; 
  std::cout << "    -oi <filename>                     Output resampled image" << std::endl << std::endl;  
  std::cout << "    -adofin <filename>                 Initial affine dof" << std::endl;  
  std::cout << "    -it <filename>                     Initial fluid transform" << std::endl; 
  std::cout << "    -tm <filename>                     Target/Fixed mask image" << std::endl;
  std::cout << "    -sm <filename>                     Source/Moving mask image" << std::endl;
  std::cout << "    -d   <int>        [0]              Number of dilations of masks (if -tm or -sm used)" << std::endl;  
  std::cout << "    -ji <filaname>                     Output jacobian image filename " << std::endl;
  std::cout << "    -vi <filename>                     Output vector image base filename" << std::endl;
  std::cout << "    -mji <filaname>                    Output jacobian image filename in Midas format" << std::endl;
  std::cout << "    -mvi <filename>                    Output vector image base filename in Midas format" << std::endl;
  std::cout << "    -ln <int>         [1]              Number of multi-resolution levels" << std::endl;
  std::cout << "    -bn <int>         [64]             Histogram binning" << std::endl;
  std::cout << "    -mi <int>         [500]            Maximum number of iterations per level" << std::endl;
  std::cout << "    -mc <float>       [1.0e-12]        Minimum change in cost function (NMI), " << std::endl;
  std::cout << "                                       below which we stop that resolution level." << std::endl;
  std::cout << "    -ls <float>       [0.0625]         Largest step size factor (voxel unit)" << std::endl;
  std::cout << "    -md <float>       [0.0025]         Minimum change in deformation magnitude between iterations, " << std::endl;
  std::cout << "                                       below which we stop that resolution level." << std::endl;  
  std::cout << "    -mj <float>       [0.5]            Minimum jacobian threshold, below which we regrid" << std::endl;  
  std::cout << "    -fi <int>         [4]              Choose final and gridding reslicing interpolator" << std::endl;
  std::cout << "                                       1. Nearest neighbour" << std::endl;
  std::cout << "                                       2. Linear" << std::endl;
  std::cout << "                                       3. BSpline" << std::endl;
  std::cout << "                                       4. Sinc" << std::endl;
  std::cout << "    -ri <int>         [2]              Choose regristration interpolator" << std::endl;
  std::cout << "                                       1. Nearest neighbour" << std::endl;
  std::cout << "                                       2. Linear" << std::endl;
  std::cout << "                                       3. BSpline" << std::endl;
  std::cout << "                                       4. Sinc" << std::endl;
  std::cout << "    -sim <int>        [4]              Choose similarity measure" << std::endl; 
  std::cout << "                                       1. Sum Of Squared Differences (SSD)" << std::endl; 
  std::cout << "                                       2. Mean of Square Differences (MSD)" << std::endl; 
  std::cout << "                                       3. Sum Of Absolute Differences (SAD)" << std::endl; 
  std::cout << "                                       4. Normalised Cross Correlation (NCC)"  << std::endl; 
  std::cout << "                                       5. Ratio Image Uniformity (RIU)"  << std::endl; 
  std::cout << "                                       6. Partioned Image Uniformity (PIU)"  << std::endl; 
  std::cout << "                                       7. Joint Entropy (JE)"  << std::endl; 
  std::cout << "                                       8. Mutual Information (MI)"  << std::endl; 
  std::cout << "                                       9. Normalized Mutual Information (NMI)"  << std::endl; 
  std::cout << "    -force <string>   [cc]             Registration force type" << std::endl; 
  std::cout << "                                       ssd - Christensen's SSD derived force" << std::endl;
  std::cout << "                                       ssdn - Christensen's SSD derived force normalised by the mean intensity" << std::endl;
  std::cout << "                                       nmi - Bill's normalised mutual information derived force" << std::endl;
  std::cout << "                                       parzen_nmi - Marc's normalised mutual information derived force based on Parzen window" << std::endl;
  std::cout << "                                       cc - Freeborough's cross correlation derived force" << std::endl;
  std::cout << "    -ar                                Apply abs filter to the regridded image." << std::endl; 
  std::cout << "    -abs_output <int> [1]              Output absoulte intensity value in resliced image" << std::endl;
  std::cout << "    -cs <int>         [0]              Check similarity during optimisation, 1 to check, 2 to skip. " << std::endl;
  std::cout << "    -sym <filename> <filename> <filename> <filename> <filename> <filename>" << std::endl;   
  std::cout << "                                       Symmetric? " << std::endl;   
  std::cout << "                                       movingFullTransformationName fixedFullTransformationName " << std::endl;   
  std::cout << "                                       movingFullTransformedImageName fixedFullTransformedImageName" << std::endl;   
  std::cout << "                                       movingComposedJacobian fixedComposedJacobian" << std::endl;   
  std::cout << "    -asgd A f_max f_min w f_min_factor w_factor    Adaptive step gradient descent parameters." << std::endl; 
  std::cout << "          [1 1 -0.6 1e-10 1 1]"         << std::endl; 
  std::cout << "    -asgd_mask <filename>              Adaptive step gradient descent mask." << std::endl; 
  std::cout << std::endl; 
  std::cout << "    ///////////////////////////////////////////////////////////////////////////////////" << std::endl; 
  std::cout << "    // The following options are not really used.                                      " << std::endl; 
  std::cout << "    ///////////////////////////////////////////////////////////////////////////////////" << std::endl; 
  std::cout << "    -ts <float>       [350]            Time step size" << std::endl;
  std::cout << "    -ssd_smooth <int> [0]              Smooth the SSD registration force" << std::endl; 
  std::cout << "    -rescale <upper lower>             Rescale the input images to the specified intensity range" << std::endl; 
  std::cout << "    -lambda <float>   [0]              Lame constant - lambda" << std::endl; 
  std::cout << "    -mu <float>       [0.01]           Lame constant - mu" << std::endl; 
  std::cout << "    -resample <float> [-1.0]           Resample the input images to this isotropic voxel size" << std::endl; 
  std::cout << "    -drop_off         [0 0 0 0]        Smoothly drop off the image intensity at the edge of the mask" << std::endl; 
  std::cout << "                                       by applying dilations to the mask and then a Gaussian filter" << std::endl;
  std::cout << "                                       Parameters: first dilation, second dilation, FWHM, mask threshold." << std::endl; 
  std::cout << "                                       DRC fluid uses 3 2 2 127." << std::endl; 
  std::cout << "    -crop <int>       [0 128 128 128]  Crop the image to be the size of the object in the mask" << std::endl; 
  std::cout << "                                       extended by the first given number and padded the image by 0 to the size given " << std::endl; 
  std::cout << "                                       by the last three given numbers. DRC fluid uses 8 128 128 128." << std::endl; 
  std::cout << "    -cf <filename>                     Output padded and cropped fixed image. " << std::endl; 
  std::cout << "    -cm <filename>                     Output padded and cropped moving image. " << std::endl; 
  std::cout << "    -mcf <filename>                    Output cropped fixed image for Midas. " << std::endl; 
  std::cout << "    -mcm <filename>                    Output cropped moving image for Midas. " << std::endl; 
  std::cout << "    -moi <filename>                    Output resliced image for  Midas. " << std::endl; 
  std::cout << "    -fdj                               Forward difference Jacobian calculation." << std::endl; 
  std::cout << "    -mdmi <double> <double> [-1 0.01]  Maximum number of iterations allowed for step size less than min deformation." << std::endl; 
  std::cout << "    -py <double> <double> ...          Multi-resolution pyramid scheme shrinking factors (specified after -ln)." << std::endl; 
  std::cout << "    -bf                                Blur final image in mulit-resolution pyramid." << std::endl; 
  std::cout << "    -is <float>       [0.5]            Iterating step size factor" << std::endl;
  std::cout << "    -rs <float>       [1.0]            Regridding step size factor" << std::endl;
  std::cout << "    -js <float>       [0.5]            Jacobian below zero step size factor" << std::endl;
  std::cout << "    -mmin <float>     [0.5]            Mask minimum threshold (if -tm or -sm used)" << std::endl;
  std::cout << "    -mmax <float>     [max]            Mask maximum threshold (if -tm or -sm used)" << std::endl; 
  std::cout << "    -mip <float>      [0]              Moving image pad value" << std::endl;  
  std::cout << "    -fip <float>      [0]              Fixed image pad value" << std::endl;  
  std::cout << "    -hfl <float>                       Fixed image lower intensity limit" << std::endl;
  std::cout << "    -hfu <float>                       Fixed image upper intensity limit" << std::endl;
  std::cout << "    -hml <float>                       Moving image lower intensity limit" << std::endl;
  std::cout << "    -hmu <float>                       Moving image upper intensity limit" << std::endl;    
  std::cout << "    -stl <int>        [0]              Start Level (starts at zero like C++)" << std::endl;
  std::cout << "    -spl <int>        [ln - 1 ]        Stop Level (default goes up to number of levels minus 1, like C++)" << std::endl;
      
  //std::cout << "    -wv                            Write vector image after each resolution level" << std::endl;
  //std::cout << "    -wj                            Write jacobian image after each resolution level" << std::endl;
}

/**
 * Smooth drop off the image around the mask. 
 * \param const InputImageType* inputFixedImage: Input image. 
 * \param const InputImageType* fixedMaskImage: Input mask. 
 * \param int dropoffThreshold: threshold value for the foreground/background value of the mask. 
 * \param int dropoffFirstDilation: first dilation. 
 * \param int dropoffSecondDilation: second dilation. 
 * \param int dropoffFWHM: width of the Gaussian kernel for dropping off the intensity. 
 */
template<typename InputImageType> 
typename InputImageType::Pointer SmoothDropOff(const InputImageType* inputFixedImage, 
                                               const InputImageType* fixedMaskImage, 
                                               int dropoffThreshold, 
                                               int dropoffFirstDilation, 
                                               int dropoffSecondDilation, 
                                               int dropoffFWHM)
{
  typedef itk::ExtendedBrainMaskWithSmoothDropOffCompositeFilter< InputImageType > SmoothDropOffCompositeFilterType;
  typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyFilterType;
  
  typename SmoothDropOffCompositeFilterType::Pointer maskFilter = SmoothDropOffCompositeFilterType::New();
  typename MultiplyFilterType::Pointer multiplyFilter = MultiplyFilterType::New();  
    
  maskFilter->SetInput(fixedMaskImage);
  maskFilter->SetInitialThreshold(dropoffThreshold);
  maskFilter->SetFirstNumberOfDilations(dropoffFirstDilation);
  maskFilter->SetSecondNumberOfDilations(dropoffSecondDilation);
  maskFilter->SetGaussianFWHM(dropoffFWHM);
  maskFilter->Update(); 
  multiplyFilter->SetInput1(inputFixedImage);
  multiplyFilter->SetInput2(maskFilter->GetOutput());
  multiplyFilter->Update(); 
  
  typename InputImageType::Pointer returnImage = multiplyFilter->GetOutput(); 
  
  returnImage->DisconnectPipeline(); 
  return returnImage; 
}

/**
 * Crop and pad an image before the registration. 
 * \param const InputImageType* inputFixedImage: Input image. 
 * \param typename InputImageType::RegionType desiredRegion: Region for cropping. 
 * \param int imagePadSize[InputImageType::ImageDimension]: Size to be increased/padded to. 
 * \param int fixedImagePadValue: Value used for padding. 
 * \param typename InputImageType::RegionType& paddedDesiredRegion: resulting/output region where the original region sits in the padded image. 
 */
template<typename InputImageType>
typename InputImageType::Pointer CropAndPadImage(const InputImageType* inputFixedImage, 
                                                 typename InputImageType::RegionType desiredRegion, 
                                                 unsigned int imagePadSize[InputImageType::ImageDimension],  
                                                 typename InputImageType::PixelType fixedImagePadValue, 
                                                 typename InputImageType::RegionType& paddedDesiredRegion)    
{
  typedef itk::RegionOfInterestImageFilter<InputImageType, InputImageType> RegionOfInterestImageFilterType;
  typename RegionOfInterestImageFilterType::Pointer roiFilter = RegionOfInterestImageFilterType::New();
  typedef itk::ConstantPadImageFilter<InputImageType, InputImageType> PadImageFilterType; 
  typename PadImageFilterType::Pointer imagePaddingFilter = PadImageFilterType::New(); 
  unsigned long padding[InputImageType::ImageDimension];
  typename InputImageType::SizeType imageRegionSize = desiredRegion.GetSize();
  typename InputImageType::RegionType largestPossibleRegion; 
  bool isPaddingOk = true; 
  typename InputImageType::Pointer returnImage; 
  
  paddedDesiredRegion = desiredRegion; 
  for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
  {
    if (imagePadSize[i] > imageRegionSize[i])
    {
      padding[i] = static_cast<unsigned long>(niftk::Round((imagePadSize[i]-imageRegionSize[i])/2.0)); 
      paddedDesiredRegion.SetIndex(i, padding[i]+1); 
    }
    else
    {
      isPaddingOk = false; 
    }
  }
  std::cout << "paddedDesiredRegion=" << paddedDesiredRegion << std::endl; 
  
  typename InputImageType::PointType oldFixedImageOrigin = inputFixedImage->GetOrigin(); 
  roiFilter->SetRegionOfInterest(desiredRegion);
  roiFilter->SetInput(inputFixedImage); 
  roiFilter->Update(); 
  returnImage = roiFilter->GetOutput(); 
  if (isPaddingOk)
  {
    imagePaddingFilter->SetInput(roiFilter->GetOutput());
    imagePaddingFilter->SetPadLowerBound(padding);
    imagePaddingFilter->SetPadUpperBound(padding);
    imagePaddingFilter->SetConstant(fixedImagePadValue);
    imagePaddingFilter->Update();
    returnImage = imagePaddingFilter->GetOutput(); 
  }
  
  returnImage->DisconnectPipeline(); 
  // The RegionOfInterestImageFilter may not be the correct filter to use here because 
  // it repositions the image to keep the ROI at the same physical positions after 
  // messing around with the origin and the regions. The registration codes don't really like this. 
  // So I am keeping the origin and setting the starting index of the regions to 0 to keep to 
  // the default/initial situation.  
  returnImage->SetOrigin(oldFixedImageOrigin); 
  largestPossibleRegion = returnImage->GetLargestPossibleRegion(); 
  for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
    largestPossibleRegion.SetIndex(i, 0); 
  returnImage->SetRegions(largestPossibleRegion); 
  
  return returnImage; 
}

/**
 * \brief Does Fluid based registration.
 */
template<int Dimension, typename OutputPixelType> 
int fluid_main(int argc, char** argv)
{
  typedef  float          PixelType;
  typedef  float          DeformableScalarType; 

  std::string fixedImage;
  std::string movingImage;
  std::string outputImage;
  std::string fixedMask;
  std::string movingMask;  
  std::string midasJacobianFilename; 
  std::string midasVectorFilename; 
  std::string outputDeformationFieldImage;
  std::string outputFixedImageDeformationFieldImage; 
  std::string outputTransformation = "fluid.dof";
  std::string croppedFixedImageName; 
  std::string croppedMovingImageName; 
  std::string midasCroppedFixedImageName; 
  std::string midasCroppedMovingImageName; 
  std::string midasCroppedOutputImageName; 
  int bins = 64;
  int levels = 1;
  int iters = 500;
  int finalInterp = 4;
  int regInterp = 2;
  int checkSim = 1; 
  int dilations = 0;
  double minCostTol = 1.0e-12;
  double minDefChange = 0.0025;
  double minJac = 0.5;
  double iReduce = 0.5;
  double rReduce = 1.0;
  double jReduce = 0.5;
  double maxStepSize = 0.0625;
  double maskMinimumThreshold = 0.5;
  double maskMaximumThreshold = std::numeric_limits<PixelType>::max();  
  double dummyDefault = -987654321;
  double intensityFixedLowerBound = dummyDefault;
  double intensityFixedUpperBound = dummyDefault;
  double intensityMovingLowerBound = dummyDefault;
  double intensityMovingUpperBound = dummyDefault;
  PixelType movingImagePadValue = 0;
  PixelType fixedImagePadValue = 0;
  std::string jacobianFile = "";
  std::string jacobianExt = "";
  std::string vectorFile = "";
  std::string vectorExt = "";
  std::string tmpFilename;
  std::string tmpBaseName;
  std::string tmpFileExtension;
  std::string adofinFilename; 
  int startLevel = 0;
  int stopLevel = 65535;
  bool isRescaleIntensity = false; 
  double lowerIntensity = 0; 
  double higherIntensity = 0;
  std::string registrationForceName = "cc";  
  int sim = 4; 
  int isSmoothForce = 0;
  double mu = 0.01; 
  double lambda = 0.0;  
  std::string initialFluidTransform; 
  std::string initialFluidVectorTransform; 
  int isOutputAbsIntensity = 1; 
  double isotropicVoxelSize = -1.0; 
  bool userSetPadValue = false;
  int dropoffFirstDilation = 0; 
  int dropoffSecondDilation = 0; 
  int dropoffFWHM = 0; 
  int crop = 0; 
  int dropoffThreshold = 0; 
  unsigned int imagePadSize[Dimension];     
  bool isForwardDifferenceJacobinaCalculation = false; 
  double timeStep = 350.0; 
  bool isAbsRegriddedImage = false; 
  int minimumDeformationMaximumIterations = -1;
  double minimumDeformationAllowedForIterations = 0.01; 
  typename std::vector<double> shrinkingFactors; 
  bool useOriginalImageAtFinalLevel = true; 
  bool isSymmetric = false; 
  std::string movingFullTransformationName; 
  std::string fixedFullTransformationName; 
  std::string movingFullTransformedImageName; 
  std::string fixedFullTransformedImageName; 
  double asgdFMax = 1.; 
  double asgdFMin = -0.6; 
  double asgdW = 0.001;
  double asgdFMinFudgeFactor = 1.; 
  double asgdWFudgeFactor = 1.; 
  double asgdA = 1.; 
  std::string asgdMaskName;  
  std::string movingFullJacobianName; 
  std::string fixedFullJacobianName; 

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
    else if(strcmp(argv[i], "-xo") == 0){
      outputDeformationFieldImage=argv[++i];
      std::cout << "Set -xo=" << outputDeformationFieldImage << std::endl; 
    }    
    else if(strcmp(argv[i], "-xof") == 0){
      outputFixedImageDeformationFieldImage=argv[++i];
      std::cout << "Set -xof=" << outputFixedImageDeformationFieldImage << std::endl; 
    }    
    else if(strcmp(argv[i], "-to") == 0){
      outputTransformation = argv[++i];
      std::cout << "Set -to=" << outputTransformation << std::endl; 
    }    
    else if(strcmp(argv[i], "-oi") == 0){
      outputImage=argv[++i];
      std::cout << "Set -oi=" << outputImage << std::endl; 
    }
    else if(strcmp(argv[i], "-tm") == 0){
      fixedMask=argv[++i];
      std::cout << "Set -tm=" << fixedMask << std::endl; 
    }
    else if(strcmp(argv[i], "-sm") == 0){
      movingMask=argv[++i];
      std::cout << "Set -sm=" << movingMask << std::endl; 
    }
    else if(strcmp(argv[i], "-bn") == 0){
      bins=atoi(argv[++i]);
      std::cout << "Set -bn=" << niftk::ConvertToString(bins) << std::endl; 
    }
    else if(strcmp(argv[i], "-ln") == 0){
      levels=atoi(argv[++i]);
      std::cout << "Set -ln=" << niftk::ConvertToString(levels) << std::endl; 
    }
    else if(strcmp(argv[i], "-mi") == 0){
      iters=atoi(argv[++i]);
      std::cout << "Set -mi=" << niftk::ConvertToString(iters) << std::endl; 
    }
    else if(strcmp(argv[i], "-mc") == 0){
      minCostTol=atof(argv[++i]);
      std::cout << "Set -mc=" << niftk::ConvertToString(minCostTol) << std::endl; 
    }
    else if(strcmp(argv[i], "-md") == 0){
      minDefChange=atof(argv[++i]);
      std::cout << "Set -md=" << niftk::ConvertToString(minDefChange) << std::endl; 
    }
    else if(strcmp(argv[i], "-mj") == 0){
      minJac=atof(argv[++i]);
      std::cout << "Set -mj=" << niftk::ConvertToString(minJac) << std::endl; 
    }
    else if(strcmp(argv[i], "-ls") == 0){
      maxStepSize=atof(argv[++i]);
      std::cout << "Set -ls=" << niftk::ConvertToString(maxStepSize) << std::endl; 
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
    else if(strcmp(argv[i], "-ri") == 0){
      regInterp=atoi(argv[++i]);
      std::cout << "Set -ri=" << niftk::ConvertToString(regInterp) << std::endl; 
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
      std::cout << "Set vectorFile=" << vectorFile << " and vectorExt=" + vectorExt << std::endl; 
    }    
    else if(strcmp(argv[i], "-mji") == 0){
      midasJacobianFilename = argv[++i]; 
      std::cout << "Set midas jacobianFile=" << midasJacobianFilename << std::endl; 
    }
    else if(strcmp(argv[i], "-mvi") == 0){
      midasVectorFilename = argv[++i]; 
      std::cout << "Set midas vectorFile=" << midasVectorFilename << std::endl; 
    }    
    else if (strcmp(argv[i], "-cs") == 0) {
      checkSim = atoi(argv[++i]);
      std::cout << "Set -cs=" << niftk::ConvertToString(checkSim) << std::endl; 
    }
    else if (strcmp(argv[i], "-adofin") == 0) {
      adofinFilename = argv[++i];
      std::cout << "Set -adofin=" << adofinFilename << std::endl; 
    }
    else if(strcmp(argv[i], "-stl") == 0){
      startLevel = atoi(argv[++i]);
      std::cout << "Set -stl=" << niftk::ConvertToString(startLevel) << std::endl; 
    }
    else if(strcmp(argv[i], "-spl") == 0){
      stopLevel = atoi(argv[++i]);
      std::cout << "Set -spl=" << niftk::ConvertToString(stopLevel) << std::endl; 
    }
    else if(strcmp(argv[i], "-force") == 0){
      registrationForceName=argv[++i];
      std::cout << "Set -force=" << registrationForceName << std::endl; 
    }    
    else if(strcmp(argv[i], "-sim") == 0){
      sim=atoi(argv[++i]);
      std::cout << "Set -sim=" << niftk::ConvertToString(sim) << std::endl; 
    }    
    else if(strcmp(argv[i], "-rescale") == 0){
      isRescaleIntensity=true;
      lowerIntensity=atof(argv[++i]);
      higherIntensity=atof(argv[++i]);
      std::cout << "Set -rescale=" << niftk::ConvertToString(lowerIntensity) << "-" << niftk::ConvertToString(higherIntensity) << std::endl; 
    }    
    else if(strcmp(argv[i], "-mip") == 0){
      movingImagePadValue=atof(argv[++i]);
      userSetPadValue=true;
      std::cout << "Set -mip=" << niftk::ConvertToString(movingImagePadValue) << std::endl; 
    }   
    else if(strcmp(argv[i], "-fip") == 0){
      fixedImagePadValue=atof(argv[++i]);
      std::cout << "Set -fip=" << niftk::ConvertToString(fixedImagePadValue) << std::endl; 
    }
    else if(strcmp(argv[i], "-ssd_smooth") == 0){
      isSmoothForce=atoi(argv[++i]);
      std::cout << "Set -ssd_smooth=" << niftk::ConvertToString(sim) << std::endl; 
    }    
    else if(strcmp(argv[i], "-lambda") == 0){
      lambda=atof(argv[++i]);
      std::cout << "Set -lambda=" << niftk::ConvertToString(lambda) << std::endl; 
    }    
    else if(strcmp(argv[i], "-mu") == 0){
      mu=atof(argv[++i]);
      std::cout << "Set -mu=" << niftk::ConvertToString(mu) << std::endl; 
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
    else if(strcmp(argv[i], "-it") == 0){
      initialFluidTransform=argv[++i];
      std::cout << "Set -it=" << initialFluidTransform << std::endl; 
    }                
    else if(strcmp(argv[i], "-itv") == 0){
      initialFluidVectorTransform=argv[++i];
      std::cout << "Set -itv=" << initialFluidVectorTransform << std::endl; 
    }                
    else if(strcmp(argv[i], "-abs_output") == 0){
      isOutputAbsIntensity = atoi(argv[++i]); 
      std::cout << "Set -abs_output=" << niftk::ConvertToString(isOutputAbsIntensity) << std::endl; 
    }
    else if(strcmp(argv[i], "-resample") == 0){
      isotropicVoxelSize = atof(argv[++i]); 
      std::cout << "Set -resample=" << niftk::ConvertToString(isotropicVoxelSize) << std::endl; 
    }
    else if(strcmp(argv[i], "-drop_off") == 0){
      dropoffFirstDilation = atoi(argv[++i]); 
      dropoffSecondDilation = atoi(argv[++i]); 
      dropoffFWHM = atoi(argv[++i]); 
      dropoffThreshold = atoi(argv[++i]); 
      std::cout << "Set -drop_off=" << niftk::ConvertToString(dropoffFirstDilation) << "," << niftk::ConvertToString(dropoffSecondDilation) << "," << niftk::ConvertToString(dropoffFWHM) << "," << niftk::ConvertToString(dropoffThreshold) << std::endl; 
    }
    else if(strcmp(argv[i], "-crop") == 0){
      crop = atoi(argv[++i]); 
      std::cout << "Set -crop=" << niftk::ConvertToString(crop) << std::endl; 
      for (unsigned int dimIndex = 0; dimIndex < Dimension; dimIndex++)
      {
        imagePadSize[dimIndex] = atoi(argv[++i]); 
        std::cout << "Set -crop padding=" << niftk::ConvertToString(static_cast<int>(imagePadSize[dimIndex])) << std::endl; 
      }
    }
    else if(strcmp(argv[i], "-cf") == 0) {
      croppedFixedImageName = argv[++i]; 
      std::cout << "Set -cf=" << croppedFixedImageName << std::endl; 
    }
    else if(strcmp(argv[i], "-cm") == 0) {
      croppedMovingImageName = argv[++i]; 
      std::cout << "Set -cm=" << croppedMovingImageName << std::endl; 
    }
    else if(strcmp(argv[i], "-mcf") == 0) {
      midasCroppedFixedImageName = argv[++i]; 
      std::cout << "Set -mcf=" << midasCroppedFixedImageName << std::endl; 
    }
    else if(strcmp(argv[i], "-mcm") == 0) {
      midasCroppedMovingImageName = argv[++i]; 
      std::cout << "Set -mcm=" << midasCroppedMovingImageName << std::endl; 
    }
    else if(strcmp(argv[i], "-moi") == 0) {
      midasCroppedOutputImageName = argv[++i]; 
      std::cout << "Set -moi=" << midasCroppedOutputImageName << std::endl; 
    }
    else if(strcmp(argv[i], "-fdj") == 0) {
      isForwardDifferenceJacobinaCalculation = true; 
      std::cout << "Set -fdj" << std::endl; 
    }
    else if(strcmp(argv[i], "-ts") == 0) {
      timeStep = atof(argv[++i]); 
      std::cout << "Set -ts=" << niftk::ConvertToString(timeStep) << std::endl; 
    }
    else if(strcmp(argv[i], "-ar") == 0) {
      isAbsRegriddedImage = true; 
      std::cout << "Set -ar" << std::endl; 
    }
    else if(strcmp(argv[i], "-mdmi") == 0) {
      minimumDeformationMaximumIterations = atoi(argv[++i]);  
      minimumDeformationAllowedForIterations = atof(argv[++i]);  
      std::cout << "Set -mkdi=" << niftk::ConvertToString(minimumDeformationMaximumIterations) << "," << niftk::ConvertToString(minimumDeformationAllowedForIterations)  << std::endl; 
    }
    //else if(strcmp(argv[i], "-wv") == 0){
    //  dumpVec=true;
    //  std::cout << "Set -wv=" << niftk::ConvertToString(dumpVec));
    //}
    //else if(strcmp(argv[i], "-wj") == 0){
    //  dumpJac=true;
    //  std::cout << "Set -wj=" << niftk::ConvertToString(dumpJac));
    //}
    else if (strcmp(argv[i], "-py") == 0) {
      for (int factorIndex = 0; factorIndex < levels; factorIndex++)
      {
        shrinkingFactors.push_back(atof(argv[++i])); 
        std::cout << "Set -py=" << niftk::ConvertToString(shrinkingFactors[factorIndex])  << std::endl; 
      }
    }
    else if (strcmp(argv[i], "-bf") == 0) {
        useOriginalImageAtFinalLevel = false; 
        std::cout << "Set -bf" << std::endl; 
    }
    else if (strcmp(argv[i], "-sym") == 0) {
        isSymmetric = true; 
        movingFullTransformationName = argv[++i]; 
        fixedFullTransformationName = argv[++i]; 
        movingFullTransformedImageName = argv[++i]; 
        fixedFullTransformedImageName = argv[++i]; 
        movingFullJacobianName = argv[++i]; 
        fixedFullJacobianName = argv[++i]; 
        std::cout << "Set -sym" << movingFullTransformationName << "," << fixedFullTransformationName << "," << movingFullTransformedImageName << "," << fixedFullTransformedImageName << std::endl; 
    }
    else if (strcmp(argv[i], "-asgd") == 0) {
        asgdA = atof(argv[++i]); 
        asgdFMax = atof(argv[++i]); 
        asgdFMin = atof(argv[++i]); 
        asgdW = atof(argv[++i]); 
        asgdFMinFudgeFactor = atof(argv[++i]); 
        asgdWFudgeFactor = atof(argv[++i]); 
        std::cout << "Set -asgd " << asgdA << "," << asgdFMax << "," << asgdFMin << "," << asgdW << "," << asgdFMinFudgeFactor << "," << asgdWFudgeFactor << std::endl; 
    }
    else if (strcmp(argv[i], "-asgd_mask") == 0) {
       asgdMaskName = argv[++i]; 
       std::cout << "Set -asgd_mask " << asgdMaskName << std::endl; 
    }
    else{
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  if (fixedImage.length() <= 0 || movingImage.length() <= 0)
  {
    StartUsage(argv[0]);
    std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
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

  if(minCostTol <= 0){
    std::cerr << argv[0] << "\tThe minCostTol must be > 0" << std::endl;
    return -1;
  }

  if(minDefChange <= 0){
    std::cerr << argv[0] << "\tThe minDefChange must be > 0" << std::endl;
    return -1;
  }

  if(maxStepSize <= 0){
    std::cerr << argv[0] << "\tThe maxStepSize must be > 0" << std::endl;
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
  typedef typename itk::Image< PixelType, Dimension >       InputImageType; 
  typedef typename itk::Image< OutputPixelType, Dimension > OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType  >  FixedImageReaderType;
  typedef typename itk::ImageFileReader< InputImageType >   MovingImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >  OutputImageWriterType;
  typedef typename InputImageType::SpacingType SpacingType;
  typedef typename itk::ExtendedBrainMaskWithSmoothDropOffCompositeFilter< InputImageType > SmoothDropOffCompositeFilterType;
  typedef typename itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyFilterType;
  
  typename InputImageType::Pointer inputFixedImage; 
  typename InputImageType::Pointer inputMovingImage; 
  typename InputImageType::Pointer fixedMaskImage; 
  typename InputImageType::Pointer movingMaskImage; 
  typename InputImageType::Pointer asgdMaskImage; 
  typename InputImageType::IndexType lowerCorner;
  typename InputImageType::IndexType upperCorner;
  typename InputImageType::RegionType desiredRegion;
  typename InputImageType::RegionType paddedDesiredRegion;
  
  // Load both images to be registered.
  try 
  { 
    typename FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
    std::cout << "Loading fixed image:" << fixedImage<< std::endl;
    fixedImageReader->SetFileName(  fixedImage );
    fixedImageReader->Update();
    inputFixedImage = fixedImageReader->GetOutput(); 
    inputFixedImage->DisconnectPipeline(); 
    std::cout << "Done" << std::endl;
    paddedDesiredRegion = inputFixedImage->GetLargestPossibleRegion();
    
    typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
    std::cout << "Loading moving image:" << movingImage<< std::endl;
    movingImageReader->SetFileName( movingImage );
    movingImageReader->Update();
    inputMovingImage = movingImageReader->GetOutput(); 
    inputMovingImage->DisconnectPipeline(); 
    std::cout << "Done" << std::endl;
    
    typename FixedImageReaderType::Pointer  fixedMaskReader  =  FixedImageReaderType::New();
    if (fixedMask.length() > 0)
    {
      std::cout << "Loading fixed mask:" << fixedMask<< std::endl;
      fixedMaskReader->SetFileName(fixedMask);
      fixedMaskReader->Update();
      fixedMaskImage = fixedMaskReader->GetOutput(); 
      fixedMaskImage->DisconnectPipeline(); 
      std::cout << "Done"<< std::endl;
      
      if (dropoffFirstDilation > 0)
      {
        // Create filter that generates the extended mask
        std::cout << "Smooth drop off..."<< std::endl;
        inputFixedImage = SmoothDropOff<InputImageType>(inputFixedImage, fixedMaskImage, dropoffThreshold, dropoffFirstDilation, dropoffSecondDilation, dropoffFWHM); 
        inputMovingImage = SmoothDropOff<InputImageType>(inputMovingImage, fixedMaskImage, dropoffThreshold, dropoffFirstDilation, dropoffSecondDilation, dropoffFWHM); 
      }
    }
      
    typename MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();
    if (movingMask.length() > 0)
    {
      std::cout << "Loading moving mask:" << movingMask << std::endl;
      movingMaskReader->SetFileName(movingMask);
      movingMaskReader->Update();
      movingMaskImage = movingMaskReader->GetOutput(); 
      movingMaskImage->DisconnectPipeline(); 
      std::cout << "Done"<< std::endl;
    }
    
    typename FixedImageReaderType::Pointer asgdMaskReader = FixedImageReaderType::New();
    if (asgdMaskName.length() > 0)
    {
      std::cout << "Loading asgd mask:"<< asgdMaskName << std::endl;
      asgdMaskReader->SetFileName(asgdMaskName); 
      asgdMaskReader->Update(); 
      asgdMaskImage = asgdMaskReader->GetOutput(); 
      asgdMaskImage->DisconnectPipeline(); 
      std::cout << "Done" << std::endl; 
    }
    
    if (crop > 0)
    {
      // Find the extent of the mask object. 
      typename itk::ImageRegionConstIteratorWithIndex<InputImageType> it(fixedMaskImage, fixedMaskImage->GetLargestPossibleRegion()); 
      
      for (unsigned int i = 0; i < Dimension; i++)
      {
        lowerCorner[i] = std::numeric_limits<typename InputImageType::IndexValueType>::max(); 
        upperCorner[i] = 0; 
      }
      for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
        if (it.Get())
        {
          for (unsigned int i = 0; i < Dimension; i++)
          {
            if (it.GetIndex()[i] < lowerCorner[i])
              lowerCorner[i] = it.GetIndex()[i]; 
            if (it.GetIndex()[i] > upperCorner[i])
              upperCorner[i] = it.GetIndex()[i]; 
          }
        }
      }
      typename InputImageType::IndexType start;
      for (unsigned int i = 0; i < Dimension; i++)
         start[i] = lowerCorner[i]-crop; 
      typename InputImageType::SizeType size;
      for (unsigned int i = 0; i < Dimension; i++)
        size[i] = upperCorner[i]-lowerCorner[i]+1+2*crop; 
      desiredRegion.SetSize(size);
      desiredRegion.SetIndex(start);
      // Crop and pad them. 
      inputFixedImage = CropAndPadImage<InputImageType>(inputFixedImage, desiredRegion, imagePadSize, fixedImagePadValue, paddedDesiredRegion); 
      inputMovingImage = CropAndPadImage<InputImageType>(inputMovingImage, desiredRegion, imagePadSize, movingImagePadValue, paddedDesiredRegion); 
      fixedMaskImage = CropAndPadImage<InputImageType>(fixedMaskImage, desiredRegion, imagePadSize, 0, paddedDesiredRegion); 
      if (!movingMaskImage.IsNull())
        movingMaskImage = CropAndPadImage<InputImageType>(movingMaskImage, desiredRegion, imagePadSize, 0, paddedDesiredRegion); 
      
      typedef typename itk::ImageFileWriter<InputImageType> CroppedOutputImageWriterType;
      typename CroppedOutputImageWriterType::Pointer imageWriter = CroppedOutputImageWriterType::New();
      if (croppedFixedImageName.length() > 0)
      {
        imageWriter->SetFileName(croppedFixedImageName);
        imageWriter->SetInput(inputFixedImage);
        imageWriter->Update(); 
      }
      if (croppedMovingImageName.length() > 0)
      {
        imageWriter->SetFileName(croppedMovingImageName);
        imageWriter->SetInput(inputMovingImage);
        imageWriter->Update(); 
      }
    }
  } 
  catch( typename itk::ExceptionObject & err ) 
  { 
    std::cerr <<"ExceptionObject caught !";
    std::cerr << err << std::endl; 
    return -2;
  }                

  // Setup objects to build registration.
  typedef typename itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;  
  typename FactoryType::Pointer factory = FactoryType::New();
  
  typedef typename itk::MaskedImageRegistrationMethod<InputImageType>  SingleResImageRegistrationMethodType;
  typename SingleResImageRegistrationMethodType::Pointer singleResMethod = SingleResImageRegistrationMethodType::New();
  
  typedef typename itk::UCLRecursiveMultiResolutionPyramidImageFilter<InputImageType, InputImageType, double> MultiResolutionPyramidImageFilterType; 

  typedef typename itk::FluidMultiResolutionMethod<InputImageType, double, Dimension, MultiResolutionPyramidImageFilterType>   MultiResImageRegistrationMethodType;
  //typedef itk::MultiResolutionDeformableImageRegistrationMethod<InputImageType, double, Dimension>   MultiResImageRegistrationMethodType;
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  typedef typename itk::FluidDeformableTransform<InputImageType, double, Dimension, float > TransformType;
  typename TransformType::Pointer transform = TransformType::New();
  typename TransformType::Pointer fixedImageTransform = TransformType::New();
  if (isForwardDifferenceJacobinaCalculation)
    transform->SetUseForwardDifferenceJacobianCalculation(); 

  typename FactoryType::MetricType::Pointer metric = factory->CreateMetric((itk::MetricTypeEnum)sim); 

  typedef typename itk::NMILocalHistogramDerivativeForceFilter<InputImageType, InputImageType, float> ForceGeneratorFilterType;
  typename ForceGeneratorFilterType::Pointer forceFilter = ForceGeneratorFilterType::New();
  
  typedef typename itk::SSDRegistrationForceFilter<InputImageType, InputImageType, float> SSDForceGeneratorFilterType;
  typename SSDForceGeneratorFilterType::Pointer ssdForceFilter = SSDForceGeneratorFilterType::New();

  typedef typename itk::FluidPDEFilter<float, Dimension > FluidPDEFilterType;
  typename FluidPDEFilterType::Pointer fluidPDEFilter = FluidPDEFilterType::New();
  
  typedef typename itk::FluidVelocityToDeformationFilter<float, Dimension > FluidAddVelocityToFieldFilterType;
  typename FluidAddVelocityToFieldFilterType::Pointer fluidAddVelocityFilter = FluidAddVelocityToFieldFilterType::New();
  typename FluidAddVelocityToFieldFilterType::Pointer fluidFixedImageAddVelocityFilter = FluidAddVelocityToFieldFilterType::New();

  typedef typename itk::FluidGradientDescentOptimizer<InputImageType, InputImageType, double, float> OptimizerType;
  typename OptimizerType::Pointer optimizer = OptimizerType::New();
  
  typedef typename itk::IterationUpdateCommand CommandType;
  typename CommandType::Pointer command = CommandType::New();
  
  typedef typename itk::Array2D<double> ScheduleType;
  
  typename TransformType::Pointer fixedImageInverseTransform = TransformType::New();
  typename TransformType::Pointer movingImageInverseTransform = TransformType::New();
  
  // Setup transformation
  transform->Initialize(inputFixedImage.GetPointer());
  transform->SetIdentity(); 
  if (adofinFilename.length() > 0)
  {
    transform->SetGlobalTransform(factory->CreateTransform(adofinFilename)); 
  }
  
  typedef typename itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  SimilarityMeasureType* similarityMeasureMetric = dynamic_cast<SimilarityMeasureType*>(metric.GetPointer()); 
  
  typedef typename itk::HistogramSimilarityMeasure<InputImageType, InputImageType> HistogramMetricType;
  HistogramMetricType* histogramMetric = dynamic_cast<HistogramMetricType*>(metric.GetPointer()); 
  
  if (histogramMetric)
  {
    histogramMetric->SetHistogramSize(bins, bins);
    histogramMetric->SetUseDerivativeScaleArray(false);
  }
  metric->ComputeGradientOff();
  forceFilter->SetMetric(histogramMetric);

  // for itkLocalSimilarityMeasureGradientDescentOptimizer
  optimizer->SetDeformableTransform(transform);
  optimizer->SetRegriddingInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)finalInterp));
  optimizer->SetMaximize(true);
  optimizer->SetMaximumNumberOfIterations(iters);
  optimizer->SetIteratingStepSizeReductionFactor(iReduce);
  optimizer->SetRegriddingStepSizeReductionFactor(rReduce);
  optimizer->SetJacobianBelowZeroStepSizeReductionFactor(jReduce);
  optimizer->SetMinimumJacobianThreshold(minJac);
  optimizer->SetCheckSimilarityMeasure(checkSim);
  optimizer->SetStepSize(maxStepSize);
  optimizer->SetMinimumDeformationMagnitudeThreshold(minDefChange);
  optimizer->SetMinimumStepSize(minDefChange); 
  optimizer->SetIsAbsRegriddedImage(isAbsRegriddedImage); 
  optimizer->SetMinimumDeformationAllowedForIterations(minimumDeformationAllowedForIterations); 
  optimizer->SetMinimumDeformationMaximumIterations(minimumDeformationMaximumIterations); 
  optimizer->SetIsSymmetric(isSymmetric); 
  optimizer->SetAsgdParameter(asgdA, asgdFMax, asgdFMin, asgdW, asgdFMinFudgeFactor, asgdWFudgeFactor); 
  if (!asgdMaskImage.IsNull())
  {
    optimizer->SetAsgdMask(asgdMaskImage); 
  }
  // optimizer->SetIsPropagateRegriddedMovingImage(true); 
  if (isSymmetric)
  {
    fixedImageTransform->Initialize(inputFixedImage.GetPointer());
    fixedImageTransform->SetIdentity(); 
    fixedImageInverseTransform->Initialize(inputFixedImage.GetPointer());
    fixedImageInverseTransform->SetIdentity(); 
    movingImageInverseTransform->Initialize(inputFixedImage.GetPointer());
    movingImageInverseTransform->SetIdentity(); 
    optimizer->SetFixedImageTransform(fixedImageTransform); 
    optimizer->SetFixedImageInverseTransform(fixedImageInverseTransform); 
    optimizer->SetMovingImageInverseTransform(movingImageInverseTransform); 
    optimizer->SetFluidVelocityToFixedImageDeformationFilter(fluidFixedImageAddVelocityFilter);
    metric->SetFixedImageTransform(fixedImageTransform); 
    metric->SetSymmetricMetric(SimilarityMeasureType::SYMMETRIC_METRIC_BOTH_FIXED_AND_MOVING_TRANSFORM); 
    metric->SetFixedImageInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)regInterp)); 
    metric->SetMovingImageInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)regInterp)); 
    metric->SetIsResampleWholeImage(true); 
  }
  
  fluidPDEFilter->SetMu(mu);
  fluidPDEFilter->SetLambda(lambda); 
  
  optimizer->SetFluidPDESolver(fluidPDEFilter);
  optimizer->SetFluidVelocityToDeformationFilter(fluidAddVelocityFilter);
  optimizer->SetMinimumSimilarityChangeThreshold(minCostTol);
  
  singleResMethod->SetMetric(metric);
  singleResMethod->SetTransform(transform);
  singleResMethod->SetInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)regInterp));
  singleResMethod->SetOptimizer(optimizer);
  singleResMethod->SetIterationUpdateCommand(command);
  singleResMethod->SetNumberOfDilations(dilations);
  singleResMethod->SetThresholdFixedMask(true); 
  singleResMethod->SetFixedMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(maskMaximumThreshold);
  
  // for itkFluidGradientDescentOptimizer
  if (registrationForceName == "nmi")
  {
    optimizer->SetForceFilter(forceFilter);
  }
  else if (registrationForceName == "ssd")
  {
    ssdForceFilter->SetSmoothing(isSmoothForce); 
    optimizer->SetForceFilter(ssdForceFilter); 
  }
  else if (registrationForceName == "ssdn")
  {
    ssdForceFilter->SetSmoothing(isSmoothForce); 
    ssdForceFilter->SetIsIntensityNormalised(true); 
    optimizer->SetForceFilter(ssdForceFilter); 
  }
  else if (registrationForceName == "parzen_nmi")
  {
    typedef typename itk::LinearlyInterpolatedDerivativeFilter<InputImageType, InputImageType, double, float> GradientFilterType;
    typename GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
    typedef typename itk::ParzenWindowNMIDerivativeForceGenerator<InputImageType, InputImageType, double, float> ParzenForceGeneratorFilterType;
    typename ParzenForceGeneratorFilterType::Pointer parzenForceFilter = ParzenForceGeneratorFilterType::New();
    
    if (histogramMetric == NULL)
    {
      std::cout << "Histogram based similarity measure must be used." << std::endl;
      return -1; 
    }
    
    std::cout << "Parzin window: forcing intensity rescaling......"<< std::endl;
    isRescaleIntensity = true; 
    lowerIntensity = 0; 
    higherIntensity = bins-1; 
    histogramMetric->SetIntensityBounds(0, bins-1, 0, bins-1);
    histogramMetric->SetUseParzenFilling(true);
    gradientFilter->SetMovingImageLowerPixelValue(0);
    gradientFilter->SetMovingImageUpperPixelValue(bins-1);
    gradientFilter->SetTransform(transform);
    parzenForceFilter->SetMetric(histogramMetric);
    parzenForceFilter->SetScalarImageGradientFilter(gradientFilter);
    parzenForceFilter->SetScaleToSizeOfVoxelAxis(false);
    parzenForceFilter->SetFixedLowerPixelValue(0);
    parzenForceFilter->SetFixedUpperPixelValue(bins-1);
    parzenForceFilter->SetMovingLowerPixelValue(0);
    parzenForceFilter->SetMovingUpperPixelValue(bins-1);              
    optimizer->SetForceFilter(parzenForceFilter);
  }
  else if (registrationForceName == "cc")
  {
    typedef typename itk::CrossCorrelationDerivativeForceFilter<InputImageType, InputImageType, float> CrossCorrelationDerivativeForceFilterType;
    typename CrossCorrelationDerivativeForceFilterType::Pointer ccForceFilter = CrossCorrelationDerivativeForceFilterType::New(); 
    optimizer->SetForceFilter(ccForceFilter); 
  }
      
  if (isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedMinimum((PixelType)lowerIntensity);
      singleResMethod->SetRescaleFixedMaximum((PixelType)higherIntensity);
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingMinimum((PixelType)lowerIntensity);
      singleResMethod->SetRescaleMovingMaximum((PixelType)higherIntensity);
    }

  if (intensityFixedLowerBound != dummyDefault || 
      intensityFixedUpperBound != dummyDefault || 
      intensityMovingLowerBound != dummyDefault || 
      intensityMovingUpperBound != dummyDefault)
  {
    if (isRescaleIntensity)
      {
        singleResMethod->SetRescaleFixedImage(true);
        singleResMethod->SetRescaleFixedBoundaryValue(lowerIntensity);
        singleResMethod->SetRescaleFixedLowerThreshold(intensityFixedLowerBound);
        singleResMethod->SetRescaleFixedUpperThreshold(intensityFixedUpperBound);
        singleResMethod->SetRescaleFixedMinimum((PixelType)lowerIntensity+1);
        singleResMethod->SetRescaleFixedMaximum((PixelType)higherIntensity);
        
        singleResMethod->SetRescaleMovingImage(true);
        singleResMethod->SetRescaleMovingBoundaryValue(lowerIntensity);
        singleResMethod->SetRescaleMovingLowerThreshold(intensityMovingLowerBound);
        singleResMethod->SetRescaleMovingUpperThreshold(intensityMovingUpperBound);              
        singleResMethod->SetRescaleMovingMinimum((PixelType)lowerIntensity+1);
        singleResMethod->SetRescaleMovingMaximum((PixelType)higherIntensity);

        similarityMeasureMetric->SetIntensityBounds(lowerIntensity+1, higherIntensity, lowerIntensity+1, higherIntensity);
      }
    else
      {
        similarityMeasureMetric->SetIntensityBounds(intensityFixedLowerBound, intensityFixedUpperBound, intensityMovingLowerBound, intensityMovingUpperBound);
      }
  }    
  
  if (initialFluidTransform.length() > 0)
  {
    multiResMethod->SetInitialTransformParameters(factory->CreateTransform(initialFluidTransform)->GetParameters()); 
  }
  else if (initialFluidVectorTransform.length() > 0)
  {
    typedef typename itk::ImageFileReader<typename TransformType::DeformationFieldType> DeformationReaderType; 
    typename DeformationReaderType::Pointer deformationReader = DeformationReaderType::New(); 
    deformationReader->SetFileName(initialFluidVectorTransform); 
    deformationReader->Update(); 
    transform->SetParametersFromField(deformationReader->GetOutput(), true); 
    multiResMethod->SetInitialTransformParameters(transform->GetParameters()); 
  }    
  

  multiResMethod->SetSingleResMethod(singleResMethod);
  multiResMethod->SetNumberOfLevels(levels);
  ScheduleType schedule(levels, Dimension);
  if (shrinkingFactors.size() > 0)
  {
    for (int levelIndex = 0; levelIndex < levels; levelIndex++)
    {
      for (unsigned int dimIndex = 0; dimIndex < Dimension; dimIndex++)
      {
        schedule[levelIndex][dimIndex] = shrinkingFactors[levelIndex];  
      }
    }
    multiResMethod->SetSchedule(&schedule);
  }
  multiResMethod->SetJacobianImageFileName(jacobianFile);
  multiResMethod->SetJacobianImageFileExtension(jacobianExt);
  multiResMethod->SetVectorImageFileName(vectorFile);
  multiResMethod->SetVectorImageFileExtension(vectorExt);  
  multiResMethod->SetTransform(transform);
  multiResMethod->SetMaxStepSize(maxStepSize);
  multiResMethod->SetMinDeformationSize(minDefChange);
  if (stopLevel > levels - 1)
  {
    stopLevel = levels - 1;
  }
  multiResMethod->SetStartLevel(startLevel);
  multiResMethod->SetStopLevel(stopLevel);
  multiResMethod->SetWriteJacobianImageAtEachLevel(false); 
  multiResMethod->SetWriteVectorImageAtEachLevel(false); 
  multiResMethod->SetWriteParametersAtEachLevel(false); 
  multiResMethod->SetUseOriginalImageAtFinalLevel(useOriginalImageAtFinalLevel); 
  
  typedef typename itk::ImageRegistrationFilter<InputImageType, OutputImageType, Dimension, double, DeformableScalarType, MultiResolutionPyramidImageFilterType>  RegistrationFilterType;  
  typename RegistrationFilterType::Pointer regFilter = RegistrationFilterType::New();
  regFilter->SetMultiResolutionRegistrationMethod(multiResMethod);
  regFilter->SetFixedImage(inputFixedImage);
  regFilter->SetMovingImage(inputMovingImage);
  regFilter->SetIsOutputAbsIntensity(isOutputAbsIntensity); 
  regFilter->SetIsotropicVoxelSize(isotropicVoxelSize); 
  regFilter->SetResampleImageInterpolation((typename itk::InterpolationTypeEnum)finalInterp); 
  regFilter->SetResampleMaskInterpolation(itk::LINEAR); 
  if (fixedMask.length() > 0)
    {
      std::cout << "Using fixedMask"<< std::endl;
      regFilter->SetFixedMask(fixedMaskImage);
      metric->SetInitialiseIntensityBoundsUsingMask(true); 
    }
  if (movingMask.length() > 0)
    {
      std::cout << "Using movingMask"<< std::endl;
      regFilter->SetMovingMask(movingMaskImage);
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
  regFilter->SetInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)finalInterp));
  
  // Set the padding values
  if (!userSetPadValue)
    {
      typename InputImageType::IndexType index;
      for (unsigned int i = 0; i < Dimension; i++)
        {
          index[i] = 0;  
        }
      movingImagePadValue = inputMovingImage->GetPixel(index);
      std::cout << "Set movingImagePadValue to:" << niftk::ConvertToString(movingImagePadValue)<< std::endl;
    }    
  metric->SetTransformedMovingImagePadValue(movingImagePadValue);
  optimizer->SetRegriddedMovingImagePadValue(movingImagePadValue);
  regFilter->SetResampledMovingImagePadValue(movingImagePadValue);
  regFilter->SetResampledFixedImagePadValue(fixedImagePadValue);
  
  // Run the registration
  regFilter->Update();

  unsigned int zeroSize[InputImageType::ImageDimension];
  for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
    zeroSize[i] = 0; 
  typename InputImageType::RegionType dummyRegion; 
  if (outputImage.length() > 0)
  {
    typedef typename itk::CastImageFilter<InputImageType, OutputImageType> InputToOutputImageCastFilterType; 
    //typename InputToOutputImageCastFilterType::Pointer castFilter = InputToOutputImageCastFilterType::New(); 
    //castFilter->SetInput(optimizer->GetRegriddedMovingImage()); 
    //castFilter->Update(); 
    
    typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
    outputImageWriter->SetFileName(outputImage);
    outputImageWriter->SetInput(regFilter->GetOutput());
    //outputImageWriter->SetInput(castFilter->GetOutput());
    outputImageWriter->Update();      
    if (crop > 0 && midasCroppedOutputImageName.length() > 0)
    {
      typename OutputImageType::Pointer outputRegisteredImage = CropAndPadImage<OutputImageType>(regFilter->GetOutput(), paddedDesiredRegion, zeroSize, 0, dummyRegion); 
      outputImageWriter->SetFileName(midasCroppedOutputImageName);
      outputImageWriter->SetInput(outputRegisteredImage);
      outputImageWriter->Update();      
    }
  }
  if (crop > 0)
  {
    typedef typename itk::ImageFileWriter<InputImageType> CroppedOutputImageWriterType;
    typename CroppedOutputImageWriterType::Pointer imageWriter = CroppedOutputImageWriterType::New();
    if (midasCroppedFixedImageName.length() > 0)
    {
      imageWriter->SetFileName(midasCroppedFixedImageName);
      inputFixedImage = CropAndPadImage<InputImageType>(inputFixedImage, paddedDesiredRegion, zeroSize, fixedImagePadValue, dummyRegion); 
      imageWriter->SetInput(inputFixedImage);
      imageWriter->Update(); 
    }
    if (midasCroppedMovingImageName.length() > 0)
    {
      imageWriter->SetFileName(midasCroppedMovingImageName);
      inputMovingImage = CropAndPadImage<InputImageType>(inputMovingImage, paddedDesiredRegion, zeroSize, fixedImagePadValue, dummyRegion); 
      imageWriter->SetInput(inputMovingImage);
      imageWriter->Update(); 
    }
  }
  
  if (outputDeformationFieldImage.length() > 0)
  {
    typedef typename itk::ImageFileWriter< typename TransformType::DeformationFieldType >  DeformationImageWriterType;
    typename DeformationImageWriterType::Pointer deformationImageWriter = DeformationImageWriterType::New(); 
    
    deformationImageWriter->SetInput(transform->GetDeformationField());
    deformationImageWriter->SetFileName(outputDeformationFieldImage);
    deformationImageWriter->Update(); 
    
    if (isSymmetric)
    {
      if (outputFixedImageDeformationFieldImage.length())
      {
        deformationImageWriter->SetInput(fixedImageTransform->GetDeformationField());
        deformationImageWriter->SetFileName(outputFixedImageDeformationFieldImage);
        deformationImageWriter->Update(); 
      }
      
      // Get the inverse transformations. 
      fixedImageInverseTransform->SetIdentity(); 
      movingImageInverseTransform->SetIdentity(); 
      fixedImageTransform->InvertUsingIterativeFixedPoint(fixedImageInverseTransform.GetPointer(), 30, 5, 0.005); 
      transform->InvertUsingIterativeFixedPoint(movingImageInverseTransform.GetPointer(), 30, 5, 0.005); 
      
      // Output the composed Jacobian. 
      typedef itk::ResampleImageFilter<typename OptimizerType::JacobianImageType, typename OptimizerType::JacobianImageType> JacobianResampleFilterType; 
      typename JacobianResampleFilterType::Pointer jacobianResampleFilter = JacobianResampleFilterType::New(); 
      typedef itk::LinearInterpolateImageFunction<typename OptimizerType::JacobianImageType, double > InterpolatorType; 
      typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
      typedef itk::MultiplyImageFilter<typename OptimizerType::JacobianImageType> MultiplyImageFilterType; 
      typename MultiplyImageFilterType::Pointer multiplyFilter = MultiplyImageFilterType::New(); 
      int origin[Dimension]; 
  
      for (unsigned int i = 0; i < Dimension; i++)
      {
        if (crop)
          origin[i] = lowerCorner[i] + 1 - crop; 
        else
          origin[i] = 0; 
      }
      jacobianResampleFilter->SetInput(optimizer->GetMovingImageTransformComposedJacobianForward());
      jacobianResampleFilter->SetTransform(fixedImageInverseTransform);
      jacobianResampleFilter->SetOutputParametersFromImage(optimizer->GetMovingImageTransformComposedJacobianForward());
      jacobianResampleFilter->SetDefaultPixelValue(0);
      jacobianResampleFilter->SetInterpolator(interpolator);      
      jacobianResampleFilter->Update();
      multiplyFilter->SetInput1(jacobianResampleFilter->GetOutput()); 
      multiplyFilter->SetInput2(optimizer->GetFixedImageTransformComposedJacobianBackward()); 
      multiplyFilter->Update(); 
      transform->WriteMidasStrImage(movingFullJacobianName, origin, paddedDesiredRegion, multiplyFilter->GetOutput()); 
      jacobianResampleFilter->SetInput(optimizer->GetFixedImageTransformComposedJacobianForward());
      jacobianResampleFilter->SetTransform(movingImageInverseTransform);
      jacobianResampleFilter->SetOutputParametersFromImage(optimizer->GetFixedImageTransformComposedJacobianForward());
      jacobianResampleFilter->SetDefaultPixelValue(0);
      jacobianResampleFilter->SetInterpolator(interpolator);      
      jacobianResampleFilter->Update();
      multiplyFilter->SetInput1(jacobianResampleFilter->GetOutput()); 
      multiplyFilter->SetInput2(optimizer->GetMovingImageTransformComposedJacobianBackward()); 
      multiplyFilter->Update(); 
      transform->WriteMidasStrImage(fixedFullJacobianName, origin, paddedDesiredRegion, multiplyFilter->GetOutput()); 
      
      // Compose the full transformation in each direction. 
      typename TransformType::DeformableParameterType::Pointer composedTransform = transform->GetDeformationField(); 
      transform->UpdateRegriddedDeformationParameters(composedTransform, fixedImageInverseTransform->GetDeformationField(), 1.); 
      transform->SetDeformableParameters(composedTransform); 
      deformationImageWriter->SetInput(composedTransform);
      deformationImageWriter->SetFileName(movingFullTransformationName);
      deformationImageWriter->Update(); 
      
      composedTransform = fixedImageTransform->GetDeformationField(); 
      fixedImageTransform->UpdateRegriddedDeformationParameters(composedTransform, movingImageInverseTransform->GetDeformationField(), 1.); 
      fixedImageTransform->SetDeformableParameters(composedTransform); 
      deformationImageWriter->SetInput(composedTransform);
      deformationImageWriter->SetFileName(fixedFullTransformationName);
      deformationImageWriter->Update(); 
      
      typedef itk::ResampleImageFilter<InputImageType, InputImageType> ResampleFilterType;
      typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New(); 
      typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
      typedef itk::AbsImageFilter<InputImageType, OutputImageType> AbsImageFilterType; 
      typename AbsImageFilterType::Pointer absImageFilter = AbsImageFilterType::New(); 
      
      resampleFilter->SetTransform(transform);
      resampleFilter->SetOutputParametersFromImage(inputFixedImage);
      resampleFilter->SetDefaultPixelValue(0);
      resampleFilter->SetInput(inputMovingImage);
      resampleFilter->SetInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)finalInterp));      
      resampleFilter->Update();
      imageWriter->SetFileName(movingFullTransformedImageName);
      absImageFilter->SetInput(resampleFilter->GetOutput()); 
      imageWriter->SetInput(absImageFilter->GetOutput());
      imageWriter->Update(); 
      
      resampleFilter->SetTransform(fixedImageTransform);
      resampleFilter->SetOutputParametersFromImage(inputFixedImage);
      resampleFilter->SetDefaultPixelValue(0);
      resampleFilter->SetInput(inputFixedImage);
      resampleFilter->SetInterpolator(factory->CreateInterpolator((typename itk::InterpolationTypeEnum)finalInterp));      
      resampleFilter->Update();
      imageWriter->SetFileName(fixedFullTransformedImageName);
      absImageFilter->SetInput(resampleFilter->GetOutput()); 
      imageWriter->SetInput(absImageFilter->GetOutput());
      imageWriter->Update(); 
    }
  }
  
  // Save midas jacobian file. 
  if (midasJacobianFilename.length() > 0)
  {
    int origin[Dimension]; 

    for (unsigned int i = 0; i < Dimension; i++)
    {
      if (crop)
        origin[i] = lowerCorner[i] + 1 - crop; 
      else
        origin[i] = 0; 
    }
    std::cout << "Saving Midas stretch file..."<< std::endl;
    transform->WriteMidasStrImage(midasJacobianFilename, origin, paddedDesiredRegion, optimizer->GetComposedJacobian()); 
    std::cout << "Saving Midas stretch file...done"<< std::endl;
  }
  
  // Save midas vector file. 
  if (midasVectorFilename.length() > 0)
  {
    int origin[Dimension]; 
    for (unsigned int i = 0; i < Dimension; i++)
    {
      if (crop)
        origin[i] = lowerCorner[i] + 1 - crop; 
      else
        origin[i] = 0; 
    }
    std::cout << "Saving Midas vector file..."<< std::endl;
    transform->WriteMidasVecImage(midasVectorFilename, origin, paddedDesiredRegion); 
    std::cout << "Saving Midas vector file...done"<< std::endl;
  }
  
  return 0;
}




