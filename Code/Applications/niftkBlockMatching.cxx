/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkImageRegistrationFilter.h"
#include "itkBlockMatchingMethod.h"
#include "itkPowellOptimizer.h"
#include "itkSimilarityMeasure.h"
#include "itkTransform.h"
#include "itkAbsoluteManhattanDistancePointMetric.h"
#include "itkSumOfSquaredDifferencePointMetric.h"
#include "itkMultiResolutionBlockMatchingMethod.h"
#include "itkTransformFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkContinuousIndex.h"
#include "itkArray.h"
#include "itkImageMomentsCalculator.h"
#include "itkAffineTransform.h"
#include "itkMatrix.h"
#include <string>

/*!
 * \file niftkBlockMatching.cxx
 * \page niftkBlockMatching
 * \section niftkBlockMatchingSummary Implements Block Matching, based on Ourselin et. al., Image and Vision Computing, 19 (2000) 25-31.
 *
 *
 * \li Dimensions: 2,3
 * \li Pixel type: Scalars only, images are converted to float on input.
 *
 * \section niftkBlockMatchingCaveat Caveats
 * \li 2D not widely used, use with caution.
 * \li In all likelihood, you should use NiftyReg on SourceForge: https://niftyreg.svn.sourceforge.net.
 */

void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements Block Matching, based on Ourselin et. al., Image and Vision Computing, 19 (2000) 25-31 and Includes modifications from Ourselin et. al. MICCAI 2002 pp 140-147." << std::endl;
  std::cout << "  However, you should probably use NiftyReg available on SourceForge: https://niftyreg.svn.sourceforge.net. " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -ti <filename> -si <filename> [-oitk <filename> | -otxt <filename] [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -ti  <filename>                 Target/Fixed image " << std::endl;
  std::cout << "    -si  <filename>                 Source/Moving image " << std::endl << std::endl;

  std::cout << "    -oitk  <filename>               Output transform file name (ITK format)" << std::endl;
  std::cout << "     or" << std::endl;
  std::cout << "    -otxt  <filename>               Output transform file name (plain text format)" << std::endl;
}

void EndUsage()
{
  std::cout << "*** [options]   ***" << std::endl << std::endl; 
  std::cout << "    -oi  <filename>                 Output resampled image" << std::endl;  
  std::cout << "    -iitk  <filename>               Initial transform file name (ITK format)" << std::endl;
  std::cout << "    -itxt  <filename>               Initial transform file name (plain text format)" << std::endl;
  std::cout << "    -tm  <filename>                 Target/Fixed mask image" << std::endl;
  std::cout << "    -fi  <int>       [4]            Choose final reslicing interpolator" << std::endl;
  std::cout << "                                      1. Nearest neighbour" << std::endl;
  std::cout << "                                      2. Linear" << std::endl;
  std::cout << "                                      3. BSpline" << std::endl;
  std::cout << "                                      4. Sinc" << std::endl;
  std::cout << "    -ri  <int>       [3]            Choose registration interpolator" << std::endl;
  std::cout << "                                      1. Nearest neighbour" << std::endl;
  std::cout << "                                      2. Linear" << std::endl;
  std::cout << "                                      3. BSpline" << std::endl;
  std::cout << "                                      4. Sinc" << std::endl;  
  std::cout << "    -s   <int>       [4]            Choose image similarity measure" << std::endl;
  std::cout << "                                      1. Sum Squared Difference" << std::endl;
  std::cout << "                                      2. Mean Squared Difference" << std::endl;
  std::cout << "                                      3. Sum Absolute Difference" << std::endl;
  std::cout << "                                      4. Normalized Cross Correlation" << std::endl;
  std::cout << "                                      5. Ratio Image Uniformity" << std::endl;
  std::cout << "                                      6. Partitioned Image Uniformity" << std::endl;
  std::cout << "                                      7. Joint Entropy" << std::endl;
  std::cout << "                                      8. Mutual Information" << std::endl;
  std::cout << "                                      9. Normalized Mutual Information" << std::endl;
  std::cout << "                                     10. Correlation Ratio (Moving|Fixed)" << std::endl;
  std::cout << "    -tr  <int>       [3]            Choose transformation" << std::endl;
  std::cout << "                                      1. Translation (mainly for testing)" << std::endl;
  std::cout << "                                      2. Rigid" << std::endl;
  std::cout << "                                      3. Rigid + Scale" << std::endl;
  std::cout << "                                      4. Full affine" << std::endl;  
  std::cout << "    -pm  <int>       [2]            Choose a point metric" << std::endl;
  std::cout << "                                      1. Sum of absolute Manhattan distance between corresponding points" << std::endl;
  std::cout << "                                      2. Sum of squared difference between corresponding points" << std::endl;
  std::cout << std::endl;
  std::cout << "    -ss  <int>       [1]            Super-sampling factor (multiplies number of voxels)" << std::endl;
  std::cout << "    -d   <int>       [0]            Number of dilations of masks (if -tm used)" << std::endl;
  std::cout << "    -mmin <float>    [0.5]          Mask minimum threshold (if -tm used)" << std::endl;
  std::cout << "    -mmax <float>    [max]          Mask maximum threshold (if -tm used)" << std::endl;  
  std::cout << "    -vrp <int>       [20%]          Initial percentage of blocks to keep" << std::endl;
  std::cout << "    -vrm <float>     [1.0]          When the resolution level changes, factor to multiply the % number of blocks to keep" << std::endl;
  std::cout << "    -vl  <int>       [20%]          When the resolution level changes, lower limit on the % number of blocks to keep" << std::endl;
  std::cout << "    -drp <int>       [50%]          Initial percentage of points to keep in LTS" << std::endl;
  std::cout << "    -drm <float>     [1.0]          When the resolution level changes, factor to multiply the % number of points in LTS" << std::endl;
  std::cout << "    -drl <int>       [20%]          When the resolution level changes, lower limit on the %  number of points in LTS" << std::endl;
  std::cout << "    -mi  <int>       [20]           Maximum number of iterations round main loop, per pyramid resolution level" << std::endl;
  std::cout << "    -e   <float>     [1]            Change in transformation in mm (Epsilon in ICV paper)" << std::endl;
  std::cout << "    -m   <float>     [4.0]          Minimum block size" << std::endl;  
  std::cout << "    -pi  <int>       [100]          Powell, maximum iterations" << std::endl;
  std::cout << "    -pl  <int.       [100]          Powell, maximum line iterations" << std::endl;
  std::cout << "    -ps  <float>     [1.0]          Powell, step length" << std::endl;
  std::cout << "    -pt  <float>     [0.00001]      Powell, step tolerance" << std::endl;
  std::cout << "    -pv  <float>     [0.00001]      Powell, value tolerance" << std::endl; 
  std::cout << "    -rescale         [lower upper]  Rescale the input images to the specified intensity range" << std::endl;
  std::cout << "    -mip <float>     [0]            Moving image pad value" << std::endl;  
  std::cout << "    -hfl <float>                    Similarity measure, fixed image lower intensity limit" << std::endl;
  std::cout << "    -hfu <float>                    Similarity measure, fixed image upper intensity limit" << std::endl;
  std::cout << "    -hml <float>                    Similarity measure, moving image lower intensity limit" << std::endl;
  std::cout << "    -hmu <float>                    Similarity measure, moving image upper intensity limit" << std::endl;    
  std::cout << "    -gv                             Use gradient magnitude image when checking for fixed image variance" << std::endl;
  std::cout << "    -mm                             Scale N, O, D1 and D2 by millimetres (to cope with anisotropic voxels)." << std::endl;
  std::cout << "    -pyramid         [x y z]        Pyramid sub sampling factors for each axis" << std::endl;
  std::cout << "    -wt                             Write transformed moving image after each iteration. Filename tmp.moving.<iteration>.nii" << std::endl;
  std::cout << "    -wps                            Write point set. Filename is always tmp.block.points.vtk" << std::endl;
  std::cout << "    -nozero                         Don't include point pairs with zero displacement" << std::endl;
  std::cout << "    -noaligncentre                  If neither -iitk nor -itxt is specified, this program will set the initial translation to align the centre of the volume. Default is on, this flag turns off this behaviour." << std::endl;
  std::cout << "    -alignaxes                      If neither -iitk nor -itxt is specified, this option will also try to align principal axes, to initialise rotations. Default is off, this flag turns it on." << std::endl;
  
  std::cout << std::endl;
  std::cout << "    -st  <int>       [2]            Choose an overall strategy" << std::endl;
  std::cout << "                                      1. As per ICV paper." << std::endl;
  std::cout << "                                         i.e. Guess N, Omega, Delta 1, Delta 2 as per section 2.3" << std::endl;
  std::cout << "                                         then '-ln 1 -stl 0 -spl 0 -r 0.5'" << std::endl;
  std::cout << "                                      2. As per MICCAI 2002 paper." << std::endl;
  std::cout << "                                         i.e. ' -N 4 -O 4 -D1 4 -D2 1 -r 1.0 -ln 3 -stl 0 -spl 2'" << std::endl;
  std::cout << "                                      3. User specifies all parameters below manually" << std::endl;
  std::cout << std::endl;
  std::cout << "    -N   <int>                      Block size in mm (N in paper)" << std::endl;
  std::cout << "    -O   <int>                      Block search size half width in mm (Omega in paper)" << std::endl;
  std::cout << "    -D1  <int>                      Block spacing in mm (Delta 1 in paper)" << std::endl;
  std::cout << "    -D2  <int>                      Block sub-sampling in mm (Delta 2 in paper)" << std::endl;
  std::cout << "    -r   <float>                    Multiplicative factor for N, Omega, Delta 1, Delta 2 when changing scale" << std::endl;
  std::cout << "    -ln  <int>      [3]             Number of multi-resolution levels" << std::endl;
  std::cout << "    -stl <int>      [0]             Start Level (starts at zero like C++)" << std::endl;
  std::cout << "    -spl <int>      [ln - 1 ]       Stop Level (default goes up to number of levels minus 1, like C++)" << std::endl;  
}

struct arguments
{
  std::string fixedImage;
  std::string movingImage;
  std::string outputImage;
  std::string outputITKTransformFile;
  std::string outputPlainTransformFile;
  std::string inputITKTransformFile;
  std::string inputPlainTransformFile;
  std::string fixedMask;   
  int finalInterpolator;
  int registrationInterpolator;
  int similarityMeasure;
  int transformation;
  int pointMetric;
  int strategy;
  int maxIterationRoundMainLoop;
  int powellMaxIters;
  int powellMaxLineIters;
  int dilations;
  int superSamplingFactor;
  int varianceInitialPercentage;
  int variancePercentageLowerLimit;
  int distanceInitialPercentage;
  int distancePercentageLowerLimit;
  int pyramidSubSampling[3];  
  double maskMinimumThreshold;
  double maskMaximumThreshold;  
  double lowerIntensity; 
  double higherIntensity;  
  double dummyDefault;
  double minimumBlockSize;
  double epsilon;
  double powellStepLength;
  double powellStepTolerance;
  double powellValueTolerance;
  double variancePercentageMultiplier;
  double distancePercentageMultiplier;
  double intensityFixedLowerBound;
  double intensityFixedUpperBound;
  double intensityMovingLowerBound;
  double intensityMovingUpperBound;
  double movingImagePadValue;
  bool dumpTransformed;
  bool scaleByMillimetres;
  bool gradientVariance;
  bool isRescaleIntensity; 
  bool writePointSet;
  bool noZero;
  bool userSpecifiedPyramid;
  bool userSetPadValue;
  bool alignCentres;
  bool alignAxes;
  
  // These are the main ones that control block sizes.
  int bigN;
  int bigO;
  int bigDelta1;
  int bigDelta2;
  int levels;
  int startLevel;
  int stopLevel;
  double resolutionReductionFactor;    
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef  short  PixelType;
  typedef  double ScalarType;
  typedef  float  DeformationScalarType; 

  typedef typename itk::Image< PixelType, Dimension >                                                   InputImageType; 
  typedef typename itk::Image< PixelType, Dimension >                                                   OutputImageType;  
  typedef typename itk::ImageFileReader< InputImageType  >                                              FixedImageReaderType;
  typedef typename itk::ImageFileReader< InputImageType >                                               MovingImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >                                              OutputImageWriterType;
  typedef typename itk::ImageRegistrationFactory<InputImageType,Dimension, ScalarType>                  FactoryType;
  typedef typename itk::SingleResolutionImageRegistrationBuilder<InputImageType, Dimension, ScalarType> BuilderType;
  typedef typename itk::MaskedImageRegistrationMethod<InputImageType>                                   ImageRegistrationMethodType;
  typedef typename itk::SimilarityMeasure<InputImageType, InputImageType>                               SimilarityMeasureType;
  typedef typename itk::BlockMatchingMethod<InputImageType, ScalarType>                                 BlockMatchingType;
  typedef BlockMatchingType*                                                                            BlockMatchingPointer;
  typedef typename BlockMatchingType::PointSetType                                                      PointSetType;
  typedef typename itk::SingleValuedNonLinearOptimizer                                                  OptimizerType;
  typedef typename itk::PowellOptimizer                                                                 PowellOptimizerType;
  typedef typename itk::AbsoluteManhattanDistancePointMetric<PointSetType, PointSetType>                ManhattanMetricType;  
  typedef typename itk::SumOfSquaredDifferencePointMetric<PointSetType, PointSetType>                   SumOfSquaredDifferencePointMetricType;
  typedef typename itk::MultiResolutionBlockMatchingMethod<InputImageType, ScalarType>                  MultiResImageRegistrationMethodType;
  typedef typename ImageRegistrationMethodType::ParametersType                                          ParametersType;
  typedef typename itk::ImageRegistrationFilter<InputImageType, OutputImageType, Dimension, ScalarType, DeformationScalarType> RegistrationFilterType;
  typedef typename itk::ResampleImageFilter< InputImageType, InputImageType >                           ResampleFilterType;
  typedef typename itk::Array2D<unsigned int>                                                           ScheduleType;
  typedef typename InputImageType::PointType                                                            InputImagePointType;
  typedef typename InputImageType::SizeType                                                             InputImageSizeType;
  typedef itk::ContinuousIndex<ScalarType, Dimension>                                                   ContinuousIndexType;
  typedef typename FactoryType::EulerAffineTransformType                                                EulerAffineTransformType;
  typedef itk::AffineTransform<ScalarType, Dimension>                                                   AffineTransformType;
  typedef itk::ImageMomentsCalculator<InputImageType>                                                   ImageMomentsCalculatorType;
  typename FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  typename FixedImageReaderType::Pointer  fixedMaskReader  =  FixedImageReaderType::New();
  typename ImageMomentsCalculatorType::Pointer fixedImageMoments = ImageMomentsCalculatorType::New();
  typename ImageMomentsCalculatorType::Pointer movingImageMoments = ImageMomentsCalculatorType::New();
  
  

  // Load both images to be registered.
  try 
    { 
      std::cout << "Loading fixed image:" + args.fixedImage<< std::endl;
      fixedImageReader->SetFileName(  args.fixedImage );
      fixedImageReader->Update();
      std::cout << "Done"<< std::endl;
      
      std::cout << "Loading moving image:" + args.movingImage<< std::endl;
      movingImageReader->SetFileName( args.movingImage );
      movingImageReader->Update();
      std::cout << "Done"<< std::endl;
      
      if (args.fixedMask.length() > 0)
        {
          std::cout << "Loading fixed mask:" + args.fixedMask<< std::endl;
          fixedMaskReader->SetFileName(args.fixedMask);
          fixedMaskReader->Update();
          std::cout << "Done"<< std::endl;
        }
      
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                
  
  typename FactoryType::Pointer factory = FactoryType::New();
  typename BuilderType::Pointer builder = BuilderType::New();
  builder->StartCreation(itk::SINGLE_RES_BLOCK_MATCH);                
  builder->CreateInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);

  if (args.inputITKTransformFile.length() > 0)
    {
      builder->CreateTransform(args.inputITKTransformFile);
    }
  else if (args.inputPlainTransformFile.length() > 0)
    {
      builder->CreateTransform(args.inputPlainTransformFile);
    }
  else
    {
      builder->CreateTransform((itk::TransformTypeEnum)args.transformation, fixedImageReader->GetOutput());  
    }
  
  typename SimilarityMeasureType::Pointer metric = builder->CreateMetric((itk::MetricTypeEnum)args.similarityMeasure);
  
  typename OptimizerType::Pointer optimizer = builder->CreateOptimizer(itk::POWELL); 
  PowellOptimizerType* powellOptimizer = static_cast<PowellOptimizerType*>(optimizer.GetPointer());    
  
  typename ImageRegistrationMethodType::Pointer method = builder->GetSingleResolutionImageRegistrationMethod();
  typename BlockMatchingType::Pointer blockMatchingMethod = static_cast<BlockMatchingPointer>(method.GetPointer());
  
  typename ManhattanMetricType::Pointer manhattanPointMetric = ManhattanMetricType::New();
  typename SumOfSquaredDifferencePointMetricType::Pointer squaredDifferencePointMetric = SumOfSquaredDifferencePointMetricType::New();

  // We have 3 options for setting up the algorithm.
  // 1. As per ICV 2000, we are doing 1 level, with varying block size.
  // 2. As per MICCAI 2002, we are doing multi-res pyramids, with fixed block size.
  // 3. User defines the lot.
  
  if (args.strategy == 1)
    {
      typename InputImageType::SizeType size = fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
      double minimumSize = size[0];
      for (unsigned int i = 1; i < Dimension; i++)
        {
          if (size[i] < minimumSize)
            {
              minimumSize = size[i];
            }
        }

      args.bigN       = (int)(minimumSize / 8);
      args.bigO       = args.bigN;      
      args.bigDelta1  = args.bigN / 4;     
      args.bigDelta2  = 4;
      args.levels     = 1;
      args.startLevel = 0;
      args.stopLevel  = 0;
      args.resolutionReductionFactor = 0.5;
      
      std::cout << "Strategy 1"<< std::endl;
      
    }
  else if (args.strategy == 2)
    {

      args.bigN       = 4;
      args.bigO       = 4;
      args.bigDelta1  = 4;
      args.bigDelta2  = 1;
      args.levels     = 3;
      args.startLevel = 0;
      args.stopLevel  = 2;
      args.resolutionReductionFactor = 1.0;
      
      std::cout << "Strategy 2"<< std::endl;
    }
  else
    {
      std::cout << "Strategy 3 - User should do set up."<< std::endl;
    }
  
  std::cout << "bigN=" << niftk::ConvertToString((int)args.bigN) \
      << ", bigO="                      << niftk::ConvertToString((int)args.bigO) \
      << ", bigDelta1="                 << niftk::ConvertToString((int)args.bigDelta1) \
      << ", bigDelta2="                 << niftk::ConvertToString((int)args.bigDelta2) \
      << ", levels="                    << niftk::ConvertToString((int)args.levels) \
      << ", startLevel="                << niftk::ConvertToString((int)args.startLevel) \
      << ", stopLevel="                 << niftk::ConvertToString((int)args.stopLevel) \
      << ", resolutionReductionFactor=" << niftk::ConvertToString((double)args.resolutionReductionFactor)<< std::endl;
  
  if(args.resolutionReductionFactor <= 0 || args.resolutionReductionFactor > 1){
    std::cerr << "\tThe resolutionReductionFactor must be > 0 and <= 1" << std::endl;
    return -1;
  }

  if(args.levels <= 0){
    std::cerr << "\tThe number of levels must be > 0" << std::endl;
    return -1;
  }

  if(args.startLevel < 0 || args.startLevel > args.levels -1 || args.startLevel > args.stopLevel){
    std::cerr << "\tThe startLevel must be >= 0, <= number levels - 1 and <= stopLevel" << std::endl;
    return -1;
  }

  if(args.stopLevel < 0 || args.stopLevel > args.levels -1 || args.stopLevel < args.startLevel){
    std::cerr << "\tThe stopLevel must be >= 0, <= number levels - 1 and >= startLevel" << std::endl;
    return -1;
  }

  if(args.bigN < 1){
    std::cerr << "\tThe bigN must be >= 1" << std::endl;
    return -1;
  }

  if(args.bigO < 1){
    std::cerr << "\tThe bigO must be >= 1" << std::endl;
    return -1;
  }

  if(args.bigDelta1 < 1){
    std::cerr << "\tThe bigDelta1 must be >= 1" << std::endl;
    return -1;
  }

  if(args.bigDelta2 < 1){
    std::cerr << "\tThe bigDelta2 must be >= 1" << std::endl;
    return -1;
  }

  // Set up the distance based optimizer.
  powellOptimizer->SetMaximumIteration(args.powellMaxIters);
  powellOptimizer->SetMaximumLineIteration(args.powellMaxLineIters);
  powellOptimizer->SetStepLength(args.powellStepLength);
  powellOptimizer->SetStepTolerance(args.powellStepTolerance);
  powellOptimizer->SetValueTolerance(args.powellValueTolerance);

  // SetPrintOutMetricEvaluation must be false, 
  // otherwise the whole algorithm grinds to a halt, as
  // the debugging output becomes the limiting factor.
  metric->SetPrintOutMetricEvaluation(false);  
  metric->SetWeightingFactor(0);  
  
  if (args.pointMetric == 1)
    {
      blockMatchingMethod->SetPointSetMetric(manhattanPointMetric);    
    }
  else
    {
      blockMatchingMethod->SetPointSetMetric(squaredDifferencePointMetric);
    }
  
  blockMatchingMethod->SetNoZero(args.noZero);
  blockMatchingMethod->SetWritePointSet(args.writePointSet);
  blockMatchingMethod->SetTransformedMovingImageFileName("tmp.moving");
  blockMatchingMethod->SetTransformedMovingImageFileExt("nii");
  blockMatchingMethod->SetWriteTransformedMovingImage(args.dumpTransformed);
  blockMatchingMethod->SetEpsilon(args.epsilon);
  blockMatchingMethod->SetMinimumBlockSize(args.minimumBlockSize);
  blockMatchingMethod->SetParameterReductionFactor(args.resolutionReductionFactor);
  blockMatchingMethod->SetBlockParameters(args.bigN, args.bigO, args.bigDelta1, args.bigDelta2);
  blockMatchingMethod->SetScaleByMillimetres(args.scaleByMillimetres);
  blockMatchingMethod->SetUseGradientMagnitudeVariance(args.gradientVariance);
  blockMatchingMethod->SetMaskImageDirectly(true);
  blockMatchingMethod->SetNumberOfDilations(args.dilations);
  blockMatchingMethod->SetThresholdFixedMask(true);
  blockMatchingMethod->SetFixedMaskMinimum((PixelType)args.maskMinimumThreshold);
  blockMatchingMethod->SetFixedMaskMaximum((PixelType)args.maskMaximumThreshold);
  blockMatchingMethod->SetMaximumNumberOfIterationsRoundMainLoop(args.maxIterationRoundMainLoop);
  
  if (args.isRescaleIntensity)
    {
      blockMatchingMethod->SetRescaleFixedImage(true);
      blockMatchingMethod->SetRescaleFixedMinimum((PixelType)args.lowerIntensity);
      blockMatchingMethod->SetRescaleFixedMaximum((PixelType)args.higherIntensity);
      blockMatchingMethod->SetRescaleMovingImage(true);
      blockMatchingMethod->SetRescaleMovingMinimum((PixelType)args.lowerIntensity);
      blockMatchingMethod->SetRescaleMovingMaximum((PixelType)args.higherIntensity);
    }
  
  if (args.intensityFixedLowerBound != args.dummyDefault || 
      args.intensityFixedUpperBound != args.dummyDefault || 
      args.intensityMovingLowerBound != args.dummyDefault || 
      args.intensityMovingUpperBound != args.dummyDefault)
    {
      if (args.isRescaleIntensity)
        {
          blockMatchingMethod->SetRescaleFixedImage(true);
          blockMatchingMethod->SetRescaleFixedBoundaryValue((PixelType)args.lowerIntensity);
          blockMatchingMethod->SetRescaleFixedLowerThreshold((PixelType)args.intensityFixedLowerBound);
          blockMatchingMethod->SetRescaleFixedUpperThreshold((PixelType)args.intensityFixedUpperBound);
          blockMatchingMethod->SetRescaleFixedMinimum((PixelType)args.lowerIntensity+1);
          blockMatchingMethod->SetRescaleFixedMaximum((PixelType)args.higherIntensity);
          
          blockMatchingMethod->SetRescaleMovingImage(true);
          blockMatchingMethod->SetRescaleMovingBoundaryValue((PixelType)args.lowerIntensity);
          blockMatchingMethod->SetRescaleMovingLowerThreshold((PixelType)args.intensityMovingLowerBound);
          blockMatchingMethod->SetRescaleMovingUpperThreshold((PixelType)args.intensityMovingUpperBound);              
          blockMatchingMethod->SetRescaleMovingMinimum((PixelType)args.lowerIntensity+1);
          blockMatchingMethod->SetRescaleMovingMaximum((PixelType)args.higherIntensity);

          metric->SetIntensityBounds((PixelType)args.lowerIntensity+1, (PixelType)args.higherIntensity, (PixelType)args.lowerIntensity+1, (PixelType)args.higherIntensity);
        }
      else
        {
          metric->SetIntensityBounds((PixelType)args.intensityFixedLowerBound, (PixelType)args.intensityFixedUpperBound, (PixelType)args.intensityMovingLowerBound, (PixelType)args.intensityMovingUpperBound);
        }      
    }

  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();  

  // Sort out initial transformation.
  if (args.inputITKTransformFile.length() > 0 || args.inputPlainTransformFile.length() > 0)
    {

      typename FactoryType::EulerAffineTransformType* eulerTransform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(blockMatchingMethod->GetTransform());

      if (args.transformation == 1)
        {
          eulerTransform->SetJustTranslation();
        }
      else if (args.transformation == 2)
        {
          eulerTransform->SetRigid();
        }
      else if (args.transformation == 3)
        {
          eulerTransform->SetRigidPlusScale();
        }
      else
        {
          eulerTransform->SetFullAffine();
        }
    }
  else
    {
      if (args.alignAxes)
        {
          fixedImageMoments->SetImage(fixedImageReader->GetOutput());
          fixedImageMoments->Compute();
          
          movingImageMoments->SetImage(movingImageReader->GetOutput());
          movingImageMoments->Compute();
          
          typedef itk::Matrix<ScalarType, Dimension, Dimension> MatrixType;
          
          MatrixType fixedPrincipalAxis = fixedImageMoments->GetPrincipalAxes();
          MatrixType movingPrincipalAxis = movingImageMoments->GetPrincipalAxes();
          
          MatrixType rotationMatrix = movingPrincipalAxis * fixedPrincipalAxis.GetInverse();
          
          itk::Array<double> transformParameters((Dimension+1) * (Dimension+1));
          for (unsigned int i = 0; i < Dimension; i++)
            {
              for (unsigned int j = 0; j < Dimension; j++)
                {
                  transformParameters[i*(Dimension+1)+j] = rotationMatrix[i][j];
                }
            }
          
          typename AffineTransformType::Pointer fullAffineTransform = AffineTransformType::New();
          fullAffineTransform->SetIdentity();
          fullAffineTransform->SetParameters(transformParameters);
          
          typename EulerAffineTransformType::Pointer eulerTransformOfRotations = EulerAffineTransformType::New();
          eulerTransformOfRotations->SetParametersFromTransform(fullAffineTransform);
          
          typename FactoryType::EulerAffineTransformType* eulerTransform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(method->GetTransform());
          eulerTransform->SetRotation(eulerTransformOfRotations->GetRotation());
          
        }
      
      if (args.alignCentres)
        {
          InputImageSizeType fixedImageSize = fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
          InputImageSizeType movingImageSize = movingImageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
          
          ContinuousIndexType fixedImageCentre;
          ContinuousIndexType movingImageCentre;
          
          for (unsigned int i = 0; i < Dimension; i++)
            {
              fixedImageCentre[i] = fixedImageSize[i]/2.0;
              movingImageCentre[i] = movingImageSize[i]/2.0;
            }
          
          InputImagePointType fixedImageCentreInMillimetres;
          InputImagePointType movingImageCentreInMillimetres;
          
          fixedImageReader->GetOutput()->TransformContinuousIndexToPhysicalPoint(fixedImageCentre, fixedImageCentreInMillimetres);
          movingImageReader->GetOutput()->TransformContinuousIndexToPhysicalPoint(movingImageCentre, movingImageCentreInMillimetres);

          itk::Array<double> initialTranslation(Dimension);
          
          for (unsigned int i = 0; i < Dimension; i++)
            {
              initialTranslation[i] = movingImageCentreInMillimetres[i] - fixedImageCentreInMillimetres[i]; 
            }
          
          typename FactoryType::EulerAffineTransformType* eulerTransform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(method->GetTransform());          
          eulerTransform->SetTranslation(initialTranslation);
        }
    }
  multiResMethod->SetInitialTransformParameters(method->GetTransform()->GetParameters());
  multiResMethod->SetSingleResMethod(blockMatchingMethod);
  
  if (args.stopLevel > args.levels - 1)
    {
      args.stopLevel = args.levels - 1;
    }  

  ScheduleType schedule(args.levels, Dimension);
    
  if (args.userSpecifiedPyramid)
    {
      std::cout << "Setting pyramids"<< std::endl;
      
      // Set finest resolution.
      for (unsigned int k = 0; k < Dimension; k++)
        {
          schedule[args.levels-1][k]=1;  
          std::cout << "Set[" << niftk::ConvertToString((int)(args.levels-1)) + "][" << niftk::ConvertToString((int)k) << "]=" << niftk::ConvertToString((int)1)<< std::endl;
        }
        
      // Work our way back up the array.
      for (int j = args.levels - 2; j >= 0; j--)
        {
          for (unsigned int k = 0; k < Dimension; k++)
            {
              unsigned int tmpVal = schedule[j+1][k] * args.pyramidSubSampling[k]; 
              schedule[j][k] =  tmpVal;
              std::cout << "Set[" << niftk::ConvertToString((int)j) << "][" << niftk::ConvertToString((int)k) << "]=" << niftk::ConvertToString((int)tmpVal)<< std::endl;
            }
        } 
      multiResMethod->SetSchedule(&schedule);
      std::cout << "Set pyramids"<< std::endl;
    }

  multiResMethod->SetNumberOfLevels(args.levels);
  multiResMethod->SetStartLevel(args.startLevel);
  multiResMethod->SetStopLevel(args.stopLevel);
  multiResMethod->SetVarianceRejectionInitialPercentage(args.varianceInitialPercentage);
  multiResMethod->SetVarianceRejectionPercentageMultiplier(args.variancePercentageMultiplier);
  multiResMethod->SetVarianceRejectionLowerPercentageLimit(args.variancePercentageLowerLimit);
  multiResMethod->SetDistanceRejectionInitialPercentage(args.distanceInitialPercentage);
  multiResMethod->SetDistanceRejectionPercentageMultiplier(args.distancePercentageMultiplier);
  multiResMethod->SetDistanceRejectionLowerPercentageLimit(args.distancePercentageLowerLimit);  
  
  typename RegistrationFilterType::Pointer regFilter = RegistrationFilterType::New();
  regFilter->SetMultiResolutionRegistrationMethod(multiResMethod);
  regFilter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)args.finalInterpolator));
  
  // For the main filter, the images are either supersampled or not.
  // This also means we also have to do the masks.
  typename itk::IdentityTransform<double, Dimension>::Pointer identityTransform = itk::IdentityTransform<double, Dimension>::New();
  typename ResampleFilterType::Pointer fixedImageResampler  = ResampleFilterType::New();
  typename ResampleFilterType::Pointer fixedMaskResampler   = ResampleFilterType::New();  
  typename ResampleFilterType::Pointer movingImageResampler = ResampleFilterType::New();
  
  if (args.superSamplingFactor > 1)
    {
      std::cout << "Super Sampling factor:" << niftk::ConvertToString((int)args.superSamplingFactor)<< std::endl;
      
      typename InputImageType::SizeType originalSize        = fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
      typename InputImageType::SpacingType originalSpacing  = fixedImageReader->GetOutput()->GetSpacing();
      
      typename InputImageType::SizeType newSize;
      typename InputImageType::SpacingType newSpacing;
      
      for (unsigned int i = 0; i < Dimension; i++)
        {
          newSize[i]     = (unsigned long int)(originalSize[i]   *(double)args.superSamplingFactor);
          newSpacing[i]  = originalSpacing[i]/(double)args.superSamplingFactor;
        }

      fixedImageResampler->SetInput(fixedImageReader->GetOutput());
      fixedImageResampler->SetTransform(identityTransform);
      fixedImageResampler->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)2)); // Linear
      fixedImageResampler->SetDefaultPixelValue(0);
      fixedImageResampler->SetOutputDirection(fixedImageReader->GetOutput()->GetDirection()); 
      fixedImageResampler->SetOutputOrigin(fixedImageReader->GetOutput()->GetOrigin()); 
      fixedImageResampler->SetOutputSpacing(newSpacing); 
      fixedImageResampler->SetSize(newSize); 
      fixedImageResampler->Update(); 

      std::cout << "Super sampled fixed image"<< std::endl;
      
      movingImageResampler->SetInput(movingImageReader->GetOutput());
      movingImageResampler->SetTransform(identityTransform);
      movingImageResampler->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)2)); // Linear
      movingImageResampler->SetDefaultPixelValue(0);
      movingImageResampler->SetOutputDirection(movingImageReader->GetOutput()->GetDirection()); 
      movingImageResampler->SetOutputOrigin(movingImageReader->GetOutput()->GetOrigin()); 
      movingImageResampler->SetOutputSpacing(newSpacing); 
      movingImageResampler->SetSize(newSize); 
      movingImageResampler->Update(); 

      std::cout << "Super sampled moving image"<< std::endl;
      
      regFilter->SetFixedImage(fixedImageResampler->GetOutput());
      regFilter->SetMovingImage(movingImageResampler->GetOutput());
      
      if (args.fixedMask.length() > 0)
        {
          fixedMaskResampler->SetInput(fixedMaskReader->GetOutput());
          fixedMaskResampler->SetTransform(identityTransform);
          fixedMaskResampler->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)2)); // Linear
          fixedMaskResampler->SetDefaultPixelValue(0);
          fixedMaskResampler->SetOutputDirection(fixedMaskReader->GetOutput()->GetDirection()); 
          fixedMaskResampler->SetOutputOrigin(fixedMaskReader->GetOutput()->GetOrigin()); 
          fixedMaskResampler->SetOutputSpacing(newSpacing); 
          fixedMaskResampler->SetSize(newSize); 
          fixedMaskResampler->Update(); 

          std::cout << "Using super sampled fixedMask"<< std::endl;
          regFilter->SetFixedMask(fixedMaskResampler->GetOutput());
        }
      
    }
  else
    {
      regFilter->SetFixedImage(fixedImageReader->GetOutput());
      regFilter->SetMovingImage(movingImageReader->GetOutput());
      
      if (args.fixedMask.length() > 0)
        {
          std::cout << "Using original fixedMask"<< std::endl;
          regFilter->SetFixedMask(fixedMaskReader->GetOutput());
        }
    }
  
  try 
    {

      // If we havent asked for output, turn off reslicing.
      if (args.outputImage.length() > 0)
        {
          regFilter->SetDoReslicing(true);
        }
      else
        {
          regFilter->SetDoReslicing(false);
        }

      // Set the padding value
      if (!args.userSetPadValue)
        {
          typename InputImageType::IndexType index;
          for (unsigned int i = 0; i < Dimension; i++)
            {
              index[i] = 0;  
            }
          args.movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
          std::cout << "Set movingImagePadValue to:" << niftk::ConvertToString((PixelType)args.movingImagePadValue)<< std::endl;
        }
      
      metric->SetTransformedMovingImagePadValue((PixelType)args.movingImagePadValue);
      blockMatchingMethod->SetTransformedMovingImagePadValue((PixelType)args.movingImagePadValue);
      regFilter->SetResampledMovingImagePadValue((PixelType)args.movingImagePadValue);

      // Run the registration
      regFilter->Update();

      if (args.outputImage.length() > 0)
        {
          typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
          outputImageWriter->SetFileName(args.outputImage);
          outputImageWriter->SetInput(regFilter->GetOutput());
          outputImageWriter->Update();      
        }

      typename FactoryType::EulerAffineTransformType* transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(blockMatchingMethod->GetTransform());

      // Save the transform (as 16 parameter ITK matrix transform).
      if (args.outputITKTransformFile.length() > 0)
        {
          std::cout << "Output ITK matrix to:" << args.outputITKTransformFile<< std::endl;
          typedef itk::TransformFileWriter TransformFileWriterType;
          TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
          transformFileWriter->SetInput(transform->GetFullAffineTransform());
          transformFileWriter->SetFileName(args.outputITKTransformFile);
          transformFileWriter->Update(); 
        }
      
      if (args.outputPlainTransformFile.length() > 0)
      {
    	  std::cout << "Output plain text matrix to:" << args.outputPlainTransformFile<< std::endl;
    	  transform->SaveFullAffineMatrix(args.outputPlainTransformFile);
      }

    }
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }                
    
  return EXIT_SUCCESS;   
}

/**
 * \brief Does Block Matching based registration.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  
  // Set defaults
  args.finalInterpolator = 4;
  args.registrationInterpolator = 3;
  args.similarityMeasure = 4;
  args.transformation = 3;
  args.pointMetric = 2;
  args.strategy = 2;
  args.maxIterationRoundMainLoop = 20;
  args.powellMaxIters = 100;
  args.powellMaxLineIters = 100;
  args.dilations = 0;
  args.superSamplingFactor = 1;
  args.varianceInitialPercentage = 20;
  args.variancePercentageMultiplier = 1.0;
  args.variancePercentageLowerLimit = 20;
  args.distanceInitialPercentage = 50;
  args.distancePercentageLowerLimit = 20;
  args.distancePercentageMultiplier = 1.0;
  args.maskMinimumThreshold = 1;
  args.maskMaximumThreshold = std::numeric_limits<double>::max();  
  args.lowerIntensity = 0; 
  args.higherIntensity = 0;  
  args.dummyDefault = -987654321;
  args.minimumBlockSize = 4;
  args.epsilon = 1;
  args.powellStepLength = 1;
  args.powellStepTolerance = 0.00001;
  args.powellValueTolerance = 0.00001;
  args.intensityFixedLowerBound = args.dummyDefault;
  args.intensityFixedUpperBound = args.dummyDefault;
  args.intensityMovingLowerBound = args.dummyDefault;
  args.intensityMovingUpperBound = args.dummyDefault;
  args.movingImagePadValue = 0;
  args.dumpTransformed = false;
  args.scaleByMillimetres = false;
  args.gradientVariance = false;
  args.isRescaleIntensity = false; 
  args.writePointSet = false;
  args.noZero = false;
  args.userSpecifiedPyramid = false;
  args.userSetPadValue = false;
  args.alignCentres = true;
  args.alignAxes = false;
  
  // These are the main ones that control block sizes.
  args.bigN=-1;
  args.bigO=-1;
  args.bigDelta1=-1;
  args.bigDelta2=-1;
  args.levels=-1;
  args.startLevel=-1;
  args.stopLevel=-1;
  args.resolutionReductionFactor=-1;
  

  // Start of command line parsing
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      StartUsage(argv[0]);
      EndUsage();
      return -1;
    }
    else if(strcmp(argv[i], "-ti") == 0){
      args.fixedImage=argv[++i];
      std::cout << "Set -ti=" << args.fixedImage<< std::endl;
    }
    else if(strcmp(argv[i], "-si") == 0){
      args.movingImage=argv[++i];
      std::cout << "Set -si=" << args.movingImage<< std::endl;
    }
    else if(strcmp(argv[i], "-oitk") == 0){
      args.outputITKTransformFile=argv[++i];
      std::cout << "Set -oitk=" << args.outputITKTransformFile<< std::endl;
    }
    else if(strcmp(argv[i], "-otxt") == 0){
      args.outputPlainTransformFile=argv[++i];
      std::cout << "Set -otxt=" << args.outputPlainTransformFile<< std::endl;
    }
  }

  // Validation
  if (args.fixedImage.length() <= 0 || args.movingImage.length() <= 0 || (args.outputITKTransformFile.length() == 0 && args.outputPlainTransformFile.length() == 0))
    {
      StartUsage(argv[0]);
      std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
      return -1;
    }

  unsigned int dims = itk::PeekAtImageDimension(args.fixedImage);
  if (dims != 3 && dims != 2)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  // Now we know image dimensions, parse the other command line args.
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-oi") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -oi=" << args.outputImage<< std::endl;
    }
    else if(strcmp(argv[i], "-iitk") == 0){
      args.inputITKTransformFile=argv[++i];
      std::cout << "Set -iitk=" << args.inputITKTransformFile<< std::endl;
    }
    else if(strcmp(argv[i], "-itxt") == 0){
      args.inputPlainTransformFile=argv[++i];
      std::cout << "Set -itxt=" << args.inputPlainTransformFile<< std::endl;
    }
    else if(strcmp(argv[i], "-tm") == 0){
      args.fixedMask=argv[++i];
      std::cout << "Set -tm=" << args.fixedMask<< std::endl;
    }
    else if(strcmp(argv[i], "-fi") == 0){
      args.finalInterpolator=atoi(argv[++i]);
      std::cout << "Set -fi=" << niftk::ConvertToString(args.finalInterpolator)<< std::endl;
    }
    else if(strcmp(argv[i], "-ri") == 0){
      args.registrationInterpolator=atoi(argv[++i]);
      std::cout << "Set -ri=" << niftk::ConvertToString(args.registrationInterpolator)<< std::endl;
    }
    else if(strcmp(argv[i], "-s") == 0){
      args.similarityMeasure=atoi(argv[++i]);
      std::cout << "Set -s=" << niftk::ConvertToString(args.similarityMeasure)<< std::endl;
    }
    else if(strcmp(argv[i], "-tr") == 0){
      args.transformation=atoi(argv[++i]);
      std::cout << "Set -tr=" << niftk::ConvertToString(args.transformation)<< std::endl;
    }
    else if(strcmp(argv[i], "-pm") == 0){
      args.pointMetric=atoi(argv[++i]);
      std::cout << "Set -pm=" << niftk::ConvertToString(args.pointMetric)<< std::endl;
    }
    else if(strcmp(argv[i], "-drp") == 0){
      args.distanceInitialPercentage=atoi(argv[++i]);
      std::cout << "Set -drp=" << niftk::ConvertToString(args.distanceInitialPercentage)<< std::endl;
    }
    else if(strcmp(argv[i], "-drm") == 0){
      args.distancePercentageMultiplier=atof(argv[++i]);
      std::cout << "Set -drm=" << niftk::ConvertToString(args.distancePercentageMultiplier)<< std::endl;
    }
    else if(strcmp(argv[i], "-drl") == 0){
      args.distancePercentageLowerLimit=atoi(argv[++i]);
      std::cout << "Set -drl=" << niftk::ConvertToString(args.distancePercentageLowerLimit)<< std::endl;
    }
    else if(strcmp(argv[i], "-vrp") == 0){
      args.varianceInitialPercentage=atoi(argv[++i]);
      std::cout << "Set -vrp=" << niftk::ConvertToString(args.varianceInitialPercentage)<< std::endl;
    }
    else if(strcmp(argv[i], "-vrm") == 0){
      args.variancePercentageMultiplier=atof(argv[++i]);
      std::cout << "Set -vrm=" << niftk::ConvertToString(args.variancePercentageMultiplier)<< std::endl;
    }    
    else if(strcmp(argv[i], "-vrl") == 0){
      args.variancePercentageLowerLimit=atoi(argv[++i]);
      std::cout << "Set -vrl=" << niftk::ConvertToString(args.variancePercentageLowerLimit)<< std::endl;
    }    
    else if(strcmp(argv[i], "-mi") == 0){
      args.maxIterationRoundMainLoop=atoi(argv[++i]);
      std::cout << "Set -mi=" << niftk::ConvertToString(args.maxIterationRoundMainLoop)<< std::endl;
    }
    else if(strcmp(argv[i], "-pi") == 0){
      args.powellMaxIters=atoi(argv[++i]);
      std::cout << "Set -pi=" << niftk::ConvertToString(args.powellMaxIters)<< std::endl;
    }
    else if(strcmp(argv[i], "-pl") == 0){
      args.powellMaxLineIters=atoi(argv[++i]);
      std::cout << "Set -pi=" << niftk::ConvertToString(args.powellMaxLineIters)<< std::endl;
    }
    else if(strcmp(argv[i], "-r") == 0){
      args.resolutionReductionFactor=atof(argv[++i]);
      std::cout << "Set -r=" << niftk::ConvertToString(args.resolutionReductionFactor)<< std::endl;
    }
    else if(strcmp(argv[i], "-m") == 0){
      args.minimumBlockSize=atof(argv[++i]);
      std::cout << "Set -m=" << niftk::ConvertToString(args.minimumBlockSize)<< std::endl;
    }
    else if(strcmp(argv[i], "-ps") == 0){
      args.powellStepLength=atof(argv[++i]);
      std::cout << "Set -ps=" << niftk::ConvertToString(args.powellStepLength)<< std::endl;
    }
    else if(strcmp(argv[i], "-pt") == 0){
      args.powellStepTolerance=atof(argv[++i]);
      std::cout << "Set -pt=" << niftk::ConvertToString(args.powellStepTolerance)<< std::endl;
    }
    else if(strcmp(argv[i], "-pv") == 0){
      args.powellValueTolerance=atof(argv[++i]);
      std::cout << "Set -pv=" << niftk::ConvertToString(args.powellValueTolerance)<< std::endl;
    }
    else if(strcmp(argv[i], "-N") == 0){
      args.bigN=atoi(argv[++i]);
      std::cout << "Set -N=" << niftk::ConvertToString(args.bigN)<< std::endl;
    }
    else if(strcmp(argv[i], "-O") == 0){
      args.bigO=atoi(argv[++i]);
      std::cout << "Set -O=" << niftk::ConvertToString(args.bigO)<< std::endl;
    }
    else if(strcmp(argv[i], "-D1") == 0){
      args.bigDelta1=atoi(argv[++i]);
      std::cout << "Set -D1=" << niftk::ConvertToString(args.bigDelta1)<< std::endl;
    }
    else if(strcmp(argv[i], "-D2") == 0){
      args.bigDelta2=atoi(argv[++i]);
      std::cout << "Set -D2=" << niftk::ConvertToString(args.bigDelta2)<< std::endl;
    }
    else if(strcmp(argv[i], "-e") == 0){
      args.epsilon=atof(argv[++i]);
      std::cout << "Set -e=" << niftk::ConvertToString(args.epsilon)<< std::endl;
    }
    else if(strcmp(argv[i], "-wt") == 0){
      args.dumpTransformed=true;
      std::cout << "Set -wt=" << niftk::ConvertToString(args.dumpTransformed)<< std::endl;
    }
    else if(strcmp(argv[i], "-wps") == 0){
      args.writePointSet=true;
      std::cout << "Set -wps=" << niftk::ConvertToString(args.writePointSet)<< std::endl;
    }
    else if(strcmp(argv[i], "-gv") == 0){
      args.gradientVariance=true;
      std::cout << "Set -gv=" << niftk::ConvertToString(args.gradientVariance)<< std::endl;
    }
    else if(strcmp(argv[i], "-mm") == 0){
      args.scaleByMillimetres=true;
      std::cout << "Set -mm=" << niftk::ConvertToString(args.scaleByMillimetres)<< std::endl;
    }
    else if(strcmp(argv[i], "-nozero") == 0){
      args.noZero=true;
      std::cout << "Set -nozero=" << niftk::ConvertToString(args.noZero)<< std::endl;
    }    
    else if(strcmp(argv[i], "-ln") == 0){
      args.levels=atoi(argv[++i]);
      std::cout << "Set -ln=" << niftk::ConvertToString(args.levels)<< std::endl;
    }
    else if(strcmp(argv[i], "-stl") == 0){
      args.startLevel=atoi(argv[++i]);
      std::cout << "Set -stl=" << niftk::ConvertToString(args.startLevel)<< std::endl;
    }
    else if(strcmp(argv[i], "-spl") == 0){
      args.stopLevel=atoi(argv[++i]);
      std::cout << "Set -spl=" << niftk::ConvertToString(args.stopLevel)<< std::endl;
    }
    else if(strcmp(argv[i], "-st") == 0){
      args.strategy=atoi(argv[++i]);
      std::cout << "Set -st=" << niftk::ConvertToString(args.strategy)<< std::endl;
    }    
    else if(strcmp(argv[i], "-d") == 0){
      args.dilations=atoi(argv[++i]);
      std::cout << "Set -d=" << niftk::ConvertToString(args.dilations)<< std::endl;
    }
    else if(strcmp(argv[i], "-ss") == 0){
      args.superSamplingFactor=atoi(argv[++i]);
      std::cout << "Set -ss=" << niftk::ConvertToString(args.superSamplingFactor)<< std::endl;
    }
    else if(strcmp(argv[i], "-mmin") == 0){
      args.maskMinimumThreshold=atof(argv[++i]);
      std::cout << "Set -mmin=" << niftk::ConvertToString(args.maskMinimumThreshold)<< std::endl;
    }
    else if(strcmp(argv[i], "-mmax") == 0){
      args.maskMaximumThreshold=atof(argv[++i]);
      std::cout << "Set -mmax=" << niftk::ConvertToString(args.maskMaximumThreshold)<< std::endl;
    }        
    else if(strcmp(argv[i], "-hfl") == 0){
      args.intensityFixedLowerBound=atof(argv[++i]);
      std::cout << "Set -hfl=" << niftk::ConvertToString(args.intensityFixedLowerBound)<< std::endl;
    }        
    else if(strcmp(argv[i], "-hfu") == 0){
      args.intensityFixedUpperBound=atof(argv[++i]);
      std::cout << "Set -hfu=" << niftk::ConvertToString(args.intensityFixedUpperBound)<< std::endl;
    }        
    else if(strcmp(argv[i], "-hml") == 0){
      args.intensityMovingLowerBound=atof(argv[++i]);
      std::cout << "Set -hml=" << niftk::ConvertToString(args.intensityMovingLowerBound)<< std::endl;
    }        
    else if(strcmp(argv[i], "-hmu") == 0){
      args.intensityMovingUpperBound=atof(argv[++i]);
      std::cout << "Set -hmu=" << niftk::ConvertToString(args.intensityMovingUpperBound)<< std::endl;
    }  
    else if(strcmp(argv[i], "-rescale") == 0){
      args.isRescaleIntensity=true;
      args.lowerIntensity=atof(argv[++i]);
      args.higherIntensity=atof(argv[++i]);
      std::cout << "Set -rescale=" << niftk::ConvertToString(args.lowerIntensity) << "-" << niftk::ConvertToString(args.higherIntensity)<< std::endl;
    }
    else if(strcmp(argv[i], "-mip") == 0){
      args.movingImagePadValue=atof(argv[++i]);
      args.userSetPadValue=true;
      std::cout << "Set -mip=" << niftk::ConvertToString(args.movingImagePadValue)<< std::endl;
    }    
    else if(strcmp(argv[i], "-pyramid") == 0){
      args.userSpecifiedPyramid=true;
      for (unsigned int j = 0; j < dims; j++)
        {
          args.pyramidSubSampling[j] = atoi(argv[++i]);
          std::cout << "Set -pyramid[" << niftk::ConvertToString((int)j) << "]=" << niftk::ConvertToString((double)args.pyramidSubSampling[j])<< std::endl;
        }
    }    
    else if(strcmp(argv[i], "-noaligncentre") == 0){
      args.alignCentres=false;
      std::cout << "Set -noaligncentre=" << niftk::ConvertToString(args.alignCentres)<< std::endl;
    }
    else if(strcmp(argv[i], "-alignaxes") == 0){
      args.alignAxes=true;
      std::cout << "Set -alignaxes=" << niftk::ConvertToString(args.alignAxes)<< std::endl;
    }            
  }

  if(args.finalInterpolator < 1 || args.finalInterpolator > 4){
    std::cerr << argv[0] << "\tThe finalInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.registrationInterpolator < 1 || args.registrationInterpolator > 4){
    std::cerr << argv[0] << "\tThe registrationInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.similarityMeasure < 1 || args.similarityMeasure > 10){
    std::cerr << argv[0] << "\tThe similarityMeasure must be >= 1 and <= 10" << std::endl;
    return -1;
  }

  if(args.transformation < 1 || args.transformation > 4){
    std::cerr << argv[0] << "\tThe transformation must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.pointMetric < 1 || args.pointMetric > 2){
    std::cerr << argv[0] << "\tThe pointMetric must be >= 1 and <= 2" << std::endl;
    return -1;
  }

  if(args.strategy < 1 || args.strategy > 3){
    std::cerr << argv[0] << "\tThe strategy must be >= 1 and <= 3" << std::endl;
    return -1;
  }

  if(args.distanceInitialPercentage <= 0 || args.distanceInitialPercentage > 100){
    std::cerr << argv[0] << "\tThe distancePercentage must be > 0 and <= 100" << std::endl;
    return -1;
  }

  if(args.distancePercentageMultiplier <= 0 ){
    std::cerr << argv[0] << "\tThe distancePercentage must be > 0" << std::endl;
    return -1;
  }

  if(args.distancePercentageLowerLimit <= 0 || args.distancePercentageLowerLimit > 100){
    std::cerr << argv[0] << "\tThe distancePercentageLowerLimit must be > 0 and <= 100" << std::endl;
    return -1;
  }

  if(args.varianceInitialPercentage <= 0 || args.varianceInitialPercentage > 100){
    std::cerr << argv[0] << "\tThe varianceInitialPercentage must be > 0 and <= 100" << std::endl;
    return -1;
  }

  if(args.variancePercentageMultiplier <= 0){
    std::cerr << argv[0] << "\tThe variancePercentageMultiplier must be > 0" << std::endl;
    return -1;
  }

  if(args.variancePercentageLowerLimit <= 0 || args.variancePercentageLowerLimit > 100){
    std::cerr << argv[0] << "\tThe variancePercentageLowerLimit must be > 0 and <= 100" << std::endl;
    return -1;
  }

  if(args.minimumBlockSize < 1){
    std::cerr << argv[0] << "\tThe minimumBlockSize must be >= 1" << std::endl;
    return -1;
  }

  if(args.maxIterationRoundMainLoop < 1){
    std::cerr << argv[0] << "\tThe maxIterationRoundMainLoop must be >= 1" << std::endl;
    return -1;
  }

  if(args.powellMaxIters < 1){
    std::cerr << argv[0] << "\tThe powellMaxIters must be >= 1" << std::endl;
    return -1;
  }

  if(args.powellMaxLineIters < 1){
    std::cerr << argv[0] << "\tThe powellMaxLineIters must be >= 1" << std::endl;
    return -1;
  }

  if(args.powellStepLength <= 0){
    std::cerr << argv[0] << "\tThe powellStepLength must be > 0" << std::endl;
    return -1;
  }

  if(args.powellStepTolerance <= 0){
    std::cerr << argv[0] << "\tThe powellStepTolerance must be > 0" << std::endl;
    return -1;
  }

  if(args.powellValueTolerance <= 0){
    std::cerr << argv[0] << "\tThe powellValueTolerance must be > 0" << std::endl;
    return -1;
  }

  if(args.epsilon <= 0){
    std::cerr << argv[0] << "\tThe epsilon must be > 0" << std::endl;
    return -1;
  }

  if(args.superSamplingFactor < 1){
    std::cerr << argv[0] << "\tThe superSamplingFactor must be >= 1" << std::endl;
    return -1;
  }

  if(args.dilations < 0){
    std::cerr << argv[0] << "\tThe number of dilations must be >= 0" << std::endl;
    return -1;
  }

  if((args.intensityFixedLowerBound != args.dummyDefault && (args.intensityFixedUpperBound == args.dummyDefault ||
      args.intensityMovingLowerBound == args.dummyDefault ||
      args.intensityMovingUpperBound == args.dummyDefault))
    ||
     (args.intensityFixedUpperBound != args.dummyDefault && (args.intensityFixedLowerBound == args.dummyDefault ||
         args.intensityMovingLowerBound == args.dummyDefault ||
         args.intensityMovingUpperBound == args.dummyDefault))
    || 
     (args.intensityMovingLowerBound != args.dummyDefault && (args.intensityMovingUpperBound == args.dummyDefault ||
         args.intensityFixedLowerBound == args.dummyDefault ||
         args.intensityFixedUpperBound == args.dummyDefault))
    ||
     (args.intensityMovingUpperBound != args.dummyDefault && (args.intensityMovingLowerBound == args.dummyDefault || 
         args.intensityFixedLowerBound == args.dummyDefault ||
         args.intensityFixedUpperBound == args.dummyDefault))
                                                    )
  {
    std::cerr << argv[0] << "\tIf you specify any of -hfl, -hfu, -hml or -hmu you should specify all of them" << std::endl;
    return -1;
  }
  
  int result;
  
  switch ( dims )
    {
      case 2:
        result = DoMain<2>(args);
        break;
      case 3:
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
  
}
