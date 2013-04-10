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
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkTransformFileWriter.h"
#include "itkImageMomentsCalculator.h"

/*!
 * \file niftkAffine.cxx
 * \page niftkAffine
 * \section niftkAffineSummary Using standard ITK filters, and the NifTK image registration library, implements a general purpose affine registration.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalar images only that are converted to float on input.
 *
 * \section niftkAffineCaveats Caveats
 * \li Rarely used in 2D, use with caution.
 */
void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Using standard ITK filters, and the NifTK image registration library, implements a general purpose affine registration." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -ti <filename> -si <filename> -ot <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -ti <filename>                 Target/Fixed image " << std::endl;
  std::cout << "    -si <filename>                 Source/Moving image " << std::endl;
  std::cout << "    -ot <filename>                 Output UCL tranformation" << std::endl << std::endl;      
}

void EndUsage()
{
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -om <filename>                 Output matrix transformation" << std::endl << std::endl;
  std::cout << "    -oi <filename>                 Output resampled image" << std::endl << std::endl;
  std::cout << "    -it <filename>                 Initial transform file name" << std::endl << std::endl;  
  std::cout << "    -tm <filename>                 Target/Fixed mask image" << std::endl;
  std::cout << "    -sm <filename>                 Source/Moving mask image" << std::endl;
  std::cout << "    -fi <int>       [4]            Choose final reslicing interpolator" << std::endl;
  std::cout << "                                      1. Nearest neighbour" << std::endl;
  std::cout << "                                      2. Linear" << std::endl;
  std::cout << "                                      3. BSpline" << std::endl;
  std::cout << "                                      4. Sinc" << std::endl;
  std::cout << "    -ri <int>       [2]            Choose registration interpolator" << std::endl;
  std::cout << "                                      1. Nearest neighbour" << std::endl;
  std::cout << "                                      2. Linear" << std::endl;
  std::cout << "                                      3. BSpline" << std::endl;
  std::cout << "                                      4. Sinc" << std::endl; 
  std::cout << "    -s   <int>      [4]            Choose image similarity measure" << std::endl;
  std::cout << "                                      1. Sum Squared Difference" << std::endl;
  std::cout << "                                      2. Mean Squared Difference" << std::endl;
  std::cout << "                                      3. Sum Absolute Difference" << std::endl;
  std::cout << "                                      4. Normalized Cross Correlation" << std::endl;
  std::cout << "                                      5. Ratio Image Uniformity" << std::endl;
  std::cout << "                                      6. Partitioned Image Uniformity" << std::endl;
  std::cout << "                                      7. Joint Entropy" << std::endl;
  std::cout << "                                      8. Mutual Information" << std::endl;
  std::cout << "                                      9. Normalized Mutual Information" << std::endl;
  std::cout << "    -tr  <int>      [3]            Choose transformation" << std::endl;
  std::cout << "                                      2. Rigid" << std::endl;
  std::cout << "                                      3. Rigid + Scale" << std::endl;
  std::cout << "                                      4. Full affine" << std::endl;
  std::cout << "    -rs  <int>      [1]            Choose registration strategy" << std::endl;
  std::cout << "                                      1. Normal (optimize transformation)" << std::endl;
  std::cout << "                                      2. Switching:Trans, Rotate" << std::endl;
  std::cout << "                                      3. Switching:Trans, Rotate, Scale" << std::endl;
  std::cout << "                                      4. Switching:Rigid, Scale" << std::endl;  
  std::cout << "    -o   <int>      [6]            Choose optimizer" << std::endl;
  std::cout << "                                      1. Simplex" << std::endl;
  std::cout << "                                      2. Gradient Descent" << std::endl;
  std::cout << "                                      3. Regular Step Size Gradient Descent" << std::endl;
  std::cout << "                                      5. Powell optimisation" << std::endl;  
  std::cout << "                                      6. Regular Step Size" << std::endl;
  std::cout << "                                      7. UCL Powell optimisation" << std::endl;
  std::cout << "    -bn <int>       [64]           Number of histogram bins" << std::endl;
  std::cout << "    -mi <int>       [300]          Maximum number of iterations per level" << std::endl;
  std::cout << "    -d   <int>      [0]            Number of dilations of masks (if -tm or -sm used)" << std::endl;  
  std::cout << "    -mmin <float>   [0.5]          Mask minimum threshold (if -tm or -sm used)" << std::endl;
  std::cout << "    -mmax <float>   [max]          Mask maximum threshold (if -tm or -sm used)" << std::endl;
  std::cout << "    -spt  <float>   [0.01]         Simplex: Parameter tolerance" << std::endl;
  std::cout << "    -sft  <float>   [0.01]         Simplex: Function tolerance" << std::endl;
  std::cout << "    -rmax <float>   [5.0]          Regular Step: Maximum step size" << std::endl;
  std::cout << "    -rmin <float>   [0.01]         Regular Step: Minimum step size" << std::endl;
  std::cout << "    -rgtol <float>  [0.01]         Regular Step: Gradient tolerance" << std::endl;
  std::cout << "    -rrfac <float>  [0.5]          Regular Step: Relaxation Factor" << std::endl;
  std::cout << "    -glr   <float>  [0.5]          Gradient: Learning rate" << std::endl;
  std::cout << "    -sym                           Symmetric metric" << std::endl;
  std::cout << "    -sym_midway                    Symmetric metric to the midway" << std::endl;
  std::cout << "    -ln  <int>      [3]            Number of multi-resolution levels" << std::endl;
  std::cout << "    -stl <int>      [0]            Start Level (starts at zero like C++)" << std::endl;
  std::cout << "    -spl <int>      [ln - 1 ]      Stop Level (default goes up to number of levels minus 1, like C++)" << std::endl;
  std::cout << "    -rescale        [lower upper]  Rescale the input images to the specified intensity range" << std::endl;
  std::cout << "    -mip <float>    [0]            Moving image pad value" << std::endl;  
  std::cout << "    -hfl <float>                   Similarity measure, fixed image lower intensity limit" << std::endl;
  std::cout << "    -hfu <float>                   Similarity measure, fixed image upper intensity limit" << std::endl;
  std::cout << "    -hml <float>                   Similarity measure, moving image lower intensity limit" << std::endl;
  std::cout << "    -hmu <float>                   Similarity measure, moving image upper intensity limit" << std::endl;  
  std::cout << "    -wsim <float>                  Try to use weighted similarity measure and specify the weighting distance threshold" << std::endl;  
  std::cout << "    -pptol <float>                 UCLPowell: Parameter tolerance" << std::endl;  
}

struct arguments
{
  std::string fixedImage;
  std::string movingImage;
  std::string outputImage;
  std::string outputMatrixTransformFile; 
  std::string outputUCLTransformFile;
  std::string inputTransformFile;
  std::string fixedMask;
  std::string movingMask;     
  int finalInterpolator;
  int registrationInterpolator;
  int similarityMeasure;
  int transformation;
  int registrationStrategy;
  int optimizer;
  int bins;
  int iterations;
  int dilations;
  int levels;
  int startLevel;
  int stopLevel;
  double lowerIntensity;
  double higherIntensity;
  double dummyDefault;
  double paramTol;
  double funcTol;
  double maxStep;
  double minStep;
  double gradTol;
  double relaxFactor;
  double learningRate;
  double maskMinimumThreshold;
  double maskMaximumThreshold;
  double intensityFixedLowerBound;
  double intensityFixedUpperBound;
  double intensityMovingLowerBound;
  double intensityMovingUpperBound;
  double movingImagePadValue;
  int symmetricMetric;
  bool isRescaleIntensity;
  bool userSetPadValue;
  bool useWeighting; 
  double weightingThreshold; 
  double parameterChangeTolerance; 
  bool useCogInitialisation; 
};

template <int Dimension>
int DoMain(arguments args)
{
  typedef  float           PixelType;
  typedef  double          ScalarType;
  typedef  short           OutputPixelType;
  typedef  float           DeformableScalarType;


  typedef typename itk::Image< PixelType, Dimension >  InputImageType;
  typedef typename itk::Image< OutputPixelType , Dimension >  OutputImageType;
  
  // Setup ojects to load images.  
  typedef typename itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef typename itk::ImageFileReader< InputImageType >  MovingImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  
  typename FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  typename FixedImageReaderType::Pointer  fixedMaskReader  = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();
  
  fixedImageReader->SetFileName(args.fixedImage);
  movingImageReader->SetFileName(args.movingImage);
  fixedMaskReader->SetFileName(args.fixedMask);
  movingMaskReader->SetFileName(args.movingMask);

  // Load both images to be registered.
  try 
    { 
      std::cout << "Loading fixed image:" << args.fixedImage<< std::endl;
      fixedImageReader->Update();
      std::cout << "Done"<< std::endl;
      
      std::cout << "Loading moving image:" << args.movingImage<< std::endl;
      movingImageReader->Update();
      std::cout << "Done"<< std::endl;
         
      if (args.fixedMask.length() > 0)
        {
          std::cout << "Loading fixed mask:" << args.fixedMask<< std::endl;
          fixedMaskReader->Update();  
          std::cout << "Done"<< std::endl;
        }
         
      if (args.movingMask.length() > 0)
        {
          std::cout << "Loading moving mask:" << args.movingMask<< std::endl;
          movingMaskReader->Update();  
          std::cout << "Done"<< std::endl;
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // Setup objects to build registration.
  typedef typename itk::ImageRegistrationFactory<InputImageType, Dimension, ScalarType> FactoryType;
  typedef typename itk::SingleResolutionImageRegistrationBuilder<InputImageType, Dimension, ScalarType> BuilderType;
  typedef typename itk::MaskedImageRegistrationMethod<InputImageType> SingleResImageRegistrationMethodType;
  typedef typename itk::MultiResolutionImageRegistrationWrapper<InputImageType> MultiResImageRegistrationMethodType;
  typedef typename itk::ImageRegistrationFilter<InputImageType, OutputImageType, Dimension, ScalarType, DeformableScalarType> RegistrationFilterType;
  typedef typename SingleResImageRegistrationMethodType::ParametersType ParametersType;
  typedef typename itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  typedef typename itk::ImageMomentsCalculator<InputImageType> ImageMomentCalculatorType;
  
  // The factory.
  typename FactoryType::Pointer factory = FactoryType::New();
  
  // Start building.
  typename BuilderType::Pointer builder = BuilderType::New();
  builder->StartCreation((itk::SingleResRegistrationMethodTypeEnum)args.registrationStrategy);
  builder->CreateInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
  typename SimilarityMeasureType::Pointer metric = builder->CreateMetric((itk::MetricTypeEnum)args.similarityMeasure);
  metric->SetSymmetricMetric(args.symmetricMetric);
  metric->SetUseWeighting(args.useWeighting); 
  if (args.useWeighting)
  {
    metric->SetWeightingDistanceThreshold(args.weightingThreshold); 
  }
  
  typename FactoryType::EulerAffineTransformType* transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(builder->CreateTransform((itk::TransformTypeEnum)args.transformation, fixedImageReader->GetOutput()).GetPointer());
  int dof = transform->GetNumberOfDOF(); 
  
  // Read in initial transform. 
  if (args.inputTransformFile.length() > 0)
  {
    transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(builder->CreateTransform(args.inputTransformFile).GetPointer());
    transform->SetNumberOfDOF(dof); 
  }
  
  typename ImageMomentCalculatorType::VectorType fixedImgeCOG; 
  typename ImageMomentCalculatorType::VectorType movingImgeCOG; 
  fixedImgeCOG.Fill(0.); 
  movingImgeCOG.Fill(0.); 
  
  // Calculate the CoG for the initialisation using CoG or for the symmetric transformation. 
  if (args.useCogInitialisation || args.symmetricMetric == 2)
  {
    typename ImageMomentCalculatorType::Pointer fixedImageMomentCalulator = ImageMomentCalculatorType::New(); 
    if (args.fixedMask.length() > 0)
      fixedImageMomentCalulator->SetImage(fixedMaskReader->GetOutput()); 
    else
      fixedImageMomentCalulator->SetImage(fixedImageReader->GetOutput()); 
    fixedImageMomentCalulator->Compute(); 
    fixedImgeCOG = fixedImageMomentCalulator->GetCenterOfGravity(); 
    typename ImageMomentCalculatorType::Pointer movingImageMomentCalulator = ImageMomentCalculatorType::New(); 
    if (args.movingMask.length() > 0)
      movingImageMomentCalulator->SetImage(movingMaskReader->GetOutput()); 
    else      
      movingImageMomentCalulator->SetImage(movingImageReader->GetOutput()); 
    movingImageMomentCalulator->Compute(); 
    movingImgeCOG = movingImageMomentCalulator->GetCenterOfGravity(); 
  }
  
  if (args.symmetricMetric == 2)
  {
    builder->CreateFixedImageInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
    builder->CreateMovingImageInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
    
    // Change the center of the transformation for the symmetric transform. 
    typename InputImageType::PointType centerPoint;
    for (unsigned int i = 0; i < Dimension; i++)
      centerPoint[i] = (fixedImgeCOG[i] + movingImgeCOG[i])/2.; 
    typename FactoryType::EulerAffineTransformType::FullAffineTransformType* fullAffineTransform = transform->GetFullAffineTransform();
    int dof = transform->GetNumberOfDOF();
    transform->SetCenter(centerPoint);
    // We need to keep the value of the initial transformation. 
    if (args.inputTransformFile.length() > 0)
    {
      transform->SetParametersFromTransform(fullAffineTransform);
    }
    transform->SetNumberOfDOF(dof);
  }
  
  // Initialise the transformation using the CoG. 
  if (args.useCogInitialisation)
  {
    if (args.symmetricMetric == 2)
      transform->InitialiseUsingCenterOfMass(fixedImgeCOG/2.0, movingImgeCOG/2.0); 
    else
      transform->InitialiseUsingCenterOfMass(fixedImgeCOG, movingImgeCOG); 
  }
  
  builder->CreateOptimizer((itk::OptimizerTypeEnum)args.optimizer);

  // Get the single res method.
  typename SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  // Sort out metric and optimizer  
  typedef typename itk::SimilarityMeasure<InputImageType, InputImageType>  SimilarityType;
  typedef SimilarityType*                                                  SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast<SimilarityPointer>(singleResMethod->GetMetric());
  
  if (args.optimizer == itk::SIMPLEX)
    {
      typedef typename itk::UCLSimplexOptimizer OptimizerType;
      typedef OptimizerType*                    OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumNumberOfIterations(args.iterations);
      op->SetParametersConvergenceTolerance(args.paramTol);
      op->SetFunctionConvergenceTolerance(args.funcTol);
      op->SetAutomaticInitialSimplex(true);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      typename OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
      {
        scales[6] = 100.0; 
        scales[7] = 100.0; 
        scales[8] = 100.0; 
      }
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
      {
        scales[9]  = 100.0; 
        scales[10] = 100.0; 
        scales[11] = 100.0; 
      }
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::GRADIENT_DESCENT)
    {
      typedef typename itk::GradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                         OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(args.iterations);
      op->SetLearningRate(args.learningRate);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      typename OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
      {
        scales[6] = 100.0; 
        scales[7] = 100.0; 
        scales[8] = 100.0; 
      }
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
      {
        scales[9]  = 100.0; 
        scales[10] = 100.0; 
        scales[11] = 100.0; 
      }
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::REGSTEP_GRADIENT_DESCENT)
    {
      typedef typename itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                                       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(args.iterations);
      op->SetMaximumStepLength(args.maxStep);
      op->SetMinimumStepLength(args.minStep);
      op->SetRelaxationFactor(args.relaxFactor);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
      {
        scales[6] = 100.0; 
        scales[7] = 100.0; 
        scales[8] = 100.0; 
      }
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
      {
        scales[9]  = 100.0; 
        scales[10] = 100.0; 
        scales[11] = 100.0; 
      }
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::POWELL)
    {
      typedef typename itk::PowellOptimizer OptimizerType;
      typedef OptimizerType*                OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumIteration(args.iterations);
      op->SetStepLength(args.maxStep);
      op->SetStepTolerance(args.minStep);
      op->SetMaximumLineIteration(10);
      op->SetValueTolerance(0.0001);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
      {
        scales[6] = 100.0; 
        scales[7] = 100.0; 
        scales[8] = 100.0; 
      }
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
      {
        scales[9]  = 100.0; 
        scales[10] = 100.0; 
        scales[11] = 100.0; 
      }
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::SIMPLE_REGSTEP)
    {
      typedef typename itk::UCLRegularStepOptimizer OptimizerType;
      typedef OptimizerType*                        OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(args.iterations);
      op->SetMaximumStepLength(args.maxStep);
      op->SetMinimumStepLength(args.minStep);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
      {
        scales[6] = 100.0; 
        scales[7] = 100.0; 
        scales[8] = 100.0; 
      }
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
      {
        scales[9]  = 100.0; 
        scales[10] = 100.0; 
        scales[11] = 100.0; 
      }
      op->SetScales(scales);      
    }
  else if (args.optimizer == itk::UCLPOWELL)
    {
      typedef itk::UCLPowellOptimizer OptimizerType;
      typedef OptimizerType*       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumIteration(args.iterations);
      op->SetStepLength(args.maxStep);
      op->SetStepTolerance(args.minStep);
      op->SetMaximumLineIteration(15);
      op->SetValueTolerance(1.0e-14);
      op->SetParameterTolerance(args.parameterChangeTolerance);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
      {
        scales[6] = 100.0; 
        scales[7] = 100.0; 
        scales[8] = 100.0; 
      }
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
      {
        scales[9]  = 100.0; 
        scales[10] = 100.0; 
        scales[11] = 100.0; 
      }
      op->SetScales(scales);
    }

  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(args.dilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);  
  singleResMethod->SetFixedMaskMinimum(args.maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(args.maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(args.maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(args.maskMaximumThreshold);
  
  if (args.isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedMinimum((PixelType)args.lowerIntensity);
      singleResMethod->SetRescaleFixedMaximum((PixelType)args.higherIntensity);
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingMinimum((PixelType)args.lowerIntensity);
      singleResMethod->SetRescaleMovingMaximum((PixelType)args.higherIntensity);
    }
  
  // Finish configuring multi-res object.
  multiResMethod->SetInitialTransformParameters( singleResMethod->GetTransform()->GetParameters() );
  multiResMethod->SetSingleResMethod(singleResMethod);
  if (args.stopLevel > args.levels - 1)
    {
    args.stopLevel = args.levels - 1;
    }  
  multiResMethod->SetNumberOfLevels(args.levels);
  multiResMethod->SetStartLevel(args.startLevel);
  multiResMethod->SetStopLevel(args.stopLevel);

  if (args.intensityFixedLowerBound != args.dummyDefault ||
      args.intensityFixedUpperBound != args.dummyDefault ||
      args.intensityMovingLowerBound != args.dummyDefault ||
      args.intensityMovingUpperBound != args.dummyDefault)
    {
      if (args.isRescaleIntensity)
        {
          singleResMethod->SetRescaleFixedImage(true);
          singleResMethod->SetRescaleFixedBoundaryValue(args.lowerIntensity);
          singleResMethod->SetRescaleFixedLowerThreshold(args.intensityFixedLowerBound);
          singleResMethod->SetRescaleFixedUpperThreshold(args.intensityFixedUpperBound);
          singleResMethod->SetRescaleFixedMinimum((PixelType)args.lowerIntensity+1);
          singleResMethod->SetRescaleFixedMaximum((PixelType)args.higherIntensity);
          
          singleResMethod->SetRescaleMovingImage(true);
          singleResMethod->SetRescaleMovingBoundaryValue(args.lowerIntensity);
          singleResMethod->SetRescaleMovingLowerThreshold(args.intensityMovingLowerBound);
          singleResMethod->SetRescaleMovingUpperThreshold(args.intensityMovingUpperBound);
          singleResMethod->SetRescaleMovingMinimum((PixelType)args.lowerIntensity+1);
          singleResMethod->SetRescaleMovingMaximum((PixelType)args.higherIntensity);

          metric->SetIntensityBounds(args.lowerIntensity+1, args.higherIntensity, args.lowerIntensity+1, args.higherIntensity);
        }
      else
        {
          metric->SetIntensityBounds(args.intensityFixedLowerBound, args.intensityFixedUpperBound, args.intensityMovingLowerBound, args.intensityMovingUpperBound);
        }
    }

  try
  {
    // The main filter.
    typename RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);
    std::cout << "Setting fixed image"<< std::endl;
    filter->SetFixedImage(fixedImageReader->GetOutput());
    std::cout << "Setting moving image"<< std::endl;
    filter->SetMovingImage(movingImageReader->GetOutput());

    if (args.fixedMask.length() > 0)
      {
        std::cout << "Setting fixed mask"<< std::endl;
        filter->SetFixedMask(fixedMaskReader->GetOutput());  
      }
      
    if (args.movingMask.length() > 0)
      {
        std::cout << "Setting moving mask"<< std::endl;
        filter->SetMovingMask(movingMaskReader->GetOutput());
      }

    // If we havent asked for output, turn off reslicing.
    if (args.outputImage.length() > 0)
      {
        filter->SetDoReslicing(true);
      }
    else
      {
        filter->SetDoReslicing(false);
      }
    
    filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)args.finalInterpolator));
    
    // Set the padding value
    if (!args.userSetPadValue)
      {
        typename InputImageType::IndexType index;
        for (unsigned int i = 0; i < Dimension; i++)
          {
            index[i] = 0;  
          }
        args.movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
        std::cout << "Set movingImagePadValue to:" + niftk::ConvertToString(args.movingImagePadValue)<< std::endl;
      }
    similarityPointer->SetTransformedMovingImagePadValue(args.movingImagePadValue);
    filter->SetResampledMovingImagePadValue(args.movingImagePadValue);
    
    // Run the registration
    filter->Update();
    
    // And Write the output.
    if (args.outputImage.length() > 0)
      {
        typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();
        outputImageWriter->SetFileName(args.outputImage);
        outputImageWriter->SetInput(filter->GetOutput());
        outputImageWriter->Update();        
      }
    
    // Make sure we get the final one.
    transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(singleResMethod->GetTransform());
    transform->SetFullAffine(); 
    
    // Save the transform (as 12 parameter UCLEulerAffine transform).
    typedef typename itk::TransformFileWriter TransformFileWriterType;
    typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(transform);
    transformFileWriter->SetFileName(args.outputUCLTransformFile);
    transformFileWriter->Update();         
    
    // Save the transform (as 16 parameter matrix transform).
    if (args.outputMatrixTransformFile.length() > 0)
      {
        transformFileWriter->SetInput(transform->GetFullAffineTransform());
        transformFileWriter->SetFileName(args.outputMatrixTransformFile);
        transformFileWriter->Update(); 
      }
    
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception caught:" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/**
 * \brief Does general purpose affine 3D image registration.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.finalInterpolator = 4;
  args.registrationInterpolator = 2;
  args.similarityMeasure = 4;
  args.transformation = 3;
  args.registrationStrategy = 1;
  args.optimizer = 6;
  args.bins = 64;
  args.iterations = 300;
  args.dilations = 0;
  args.levels = 3;
  args.startLevel = 0;
  args.stopLevel = args.levels -1;
  args.lowerIntensity = 0;
  args.higherIntensity = 0;
  args.dummyDefault = -987654321;
  args.paramTol = 0.01;
  args.funcTol = 0.01;
  args.maxStep = 5.0;
  args.minStep = 0.01;
  args.gradTol = 0.01;
  args.relaxFactor = 0.5;
  args.learningRate = 0.5;
  args.maskMinimumThreshold = 0.5;
  args.maskMaximumThreshold = 255;
  args.intensityFixedLowerBound = args.dummyDefault;
  args.intensityFixedUpperBound = args.dummyDefault;
  args.intensityMovingLowerBound = args.dummyDefault;
  args.intensityMovingUpperBound = args.dummyDefault;
  args.movingImagePadValue = 0;
  args.symmetricMetric = 0;
  args.isRescaleIntensity = false;
  args.userSetPadValue = false;
  args.useWeighting = false; 
  args.useCogInitialisation = false; 


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
    else if(strcmp(argv[i], "-ot") == 0){
      args.outputUCLTransformFile=argv[++i];
      std::cout << "Set -ot=" << args.outputUCLTransformFile<< std::endl;
    }
    else if(strcmp(argv[i], "-om") == 0){
      args.outputMatrixTransformFile=argv[++i];
      std::cout << "Set -om=" << args.outputMatrixTransformFile<< std::endl;
    }
    else if(strcmp(argv[i], "-oi") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -oi=" << args.outputImage<< std::endl;
    }
    else if(strcmp(argv[i], "-it") == 0){
      args.inputTransformFile=argv[++i];
      std::cout << "Set -it=" << args.inputTransformFile<< std::endl;
    }
    else if(strcmp(argv[i], "-tm") == 0){
      args.fixedMask=argv[++i];
      std::cout << "Set -tm=" << args.fixedMask<< std::endl;
    }
    else if(strcmp(argv[i], "-sm") == 0){
      args.movingMask=argv[++i];
      std::cout << "Set -sm=" << args.movingMask<< std::endl;
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
    else if(strcmp(argv[i], "-rs") == 0){
      args.registrationStrategy=atoi(argv[++i]);
      std::cout << "Set -rs=" << niftk::ConvertToString(args.registrationStrategy)<< std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.optimizer=atoi(argv[++i]);
      std::cout << "Set -o=" << niftk::ConvertToString(args.optimizer)<< std::endl;
    }
    else if(strcmp(argv[i], "-bn") == 0){
      args.bins=atoi(argv[++i]);
      std::cout << "Set -bn=" << niftk::ConvertToString(args.bins)<< std::endl;
    }
    else if(strcmp(argv[i], "-mi") == 0){
      args.iterations=atoi(argv[++i]);
      std::cout << "Set -mi=" << niftk::ConvertToString(args.iterations)<< std::endl;
    }
    else if(strcmp(argv[i], "-d") == 0){
      args.dilations=atoi(argv[++i]);
      std::cout << "Set -d=" << niftk::ConvertToString(args.dilations)<< std::endl;
    }
    else if(strcmp(argv[i], "-mmin") == 0){
      args.maskMinimumThreshold=atof(argv[++i]);
      std::cout << "Set -mmin=" << niftk::ConvertToString(args.maskMinimumThreshold)<< std::endl;
    }
    else if(strcmp(argv[i], "-mmax") == 0){
      args.maskMaximumThreshold=atof(argv[++i]);
      std::cout << "Set -mmax=" << niftk::ConvertToString(args.maskMaximumThreshold)<< std::endl;
    }
    else if(strcmp(argv[i], "-spt") == 0){
      args.paramTol=atof(argv[++i]);
      std::cout << "Set -spt=" << niftk::ConvertToString(args.paramTol)<< std::endl;
    }
    else if(strcmp(argv[i], "-sft") == 0){
      args.funcTol=atof(argv[++i]);
      std::cout << "Set -spt=" << niftk::ConvertToString(args.funcTol)<< std::endl;
    }
    else if(strcmp(argv[i], "-rmax") == 0){
      args.maxStep=atof(argv[++i]);
      std::cout << "Set -rmax=" << niftk::ConvertToString(args.maxStep)<< std::endl;
    }
    else if(strcmp(argv[i], "-rmin") == 0){
      args.minStep=atof(argv[++i]);
      std::cout << "Set -rmin=" << niftk::ConvertToString(args.minStep)<< std::endl;
    }
    else if(strcmp(argv[i], "-rgtol") == 0){
      args.gradTol=atof(argv[++i]);
      std::cout << "Set -rgtol=" << niftk::ConvertToString(args.gradTol)<< std::endl;
    }
    else if(strcmp(argv[i], "-rrfac") == 0){
      args.relaxFactor=atof(argv[++i]);
      std::cout << "Set -rrfac=" << niftk::ConvertToString(args.relaxFactor)<< std::endl;
    }
    else if(strcmp(argv[i], "-glr") == 0){
      args.learningRate=atof(argv[++i]);
      std::cout << "Set -glr=" << niftk::ConvertToString(args.learningRate)<< std::endl;
    }
    else if(strcmp(argv[i], "-sym") == 0){
      args.symmetricMetric=1;
      std::cout << "Set -sym=" << niftk::ConvertToString(args.symmetricMetric)<< std::endl;
    }
    else if(strcmp(argv[i], "-sym_midway") == 0){
      args.symmetricMetric=2;
      std::cout << "Set -sym_midway=" << niftk::ConvertToString(args.symmetricMetric)<< std::endl;
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
    else if(strcmp(argv[i], "-wsim") == 0){
      args.useWeighting=true;
      args.weightingThreshold=atof(argv[++i]);
      std::cout << "Set -wsim=" << niftk::ConvertToString(args.weightingThreshold)<< std::endl;
    }
    else if(strcmp(argv[i], "-pptol") == 0){
      args.parameterChangeTolerance=atof(argv[++i]);
      std::cout << "Set -pptol=" << niftk::ConvertToString(args.parameterChangeTolerance)<< std::endl;
    }
    else if(strcmp(argv[i], "-cog") == 0){
      args.useCogInitialisation=true; 
      std::cout << "Set -cog=" << niftk::ConvertToString(args.useCogInitialisation)<< std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validation
  if (args.fixedImage.length() <= 0 || args.movingImage.length() <= 0 || args.outputUCLTransformFile.length() <= 0)
    {
      StartUsage(argv[0]);
      std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
      return -1;
    }

  if(args.finalInterpolator < 1 || args.finalInterpolator > 4){
    std::cerr << argv[0] << "\tThe finalInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.registrationInterpolator < 1 || args.registrationInterpolator > 4){
    std::cerr << argv[0] << "\tThe registrationInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.similarityMeasure < 1 || args.similarityMeasure > 9){
    std::cerr << argv[0] << "\tThe similarityMeasure must be >= 1 and <= 9" << std::endl;
    return -1;
  }

  if(args.transformation < 2 || args.transformation > 4){
    std::cerr << argv[0] << "\tThe transformation must be >= 2 and <= 4" << std::endl;
    return -1;
  }

  if(args.registrationStrategy < 1 || args.registrationStrategy > 4){
    std::cerr << argv[0] << "\tThe registrationStrategy must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.optimizer < 1 || args.optimizer > 7){
    std::cerr << argv[0] << "\tThe optimizer must be >= 1 and <= 7" << std::endl;
    return -1;
  }

  if(args.bins <= 0){
    std::cerr << argv[0] << "\tThe number of bins must be > 0" << std::endl;
    return -1;
  }

  if(args.iterations <= 0){
    std::cerr << argv[0] << "\tThe number of iterations must be > 0" << std::endl;
    return -1;
  }

  if(args.dilations < 0){
    std::cerr << argv[0] << "\tThe number of dilations must be >= 0" << std::endl;
    return -1;
  }

  if(args.funcTol < 0){
    std::cerr << argv[0] << "\tThe funcTol must be >= 0" << std::endl;
    return -1;
  }

  if(args.maxStep <= 0){
    std::cerr << argv[0] << "\tThe maxStep must be > 0" << std::endl;
    return -1;
  }

  if(args.minStep <= 0){
    std::cerr << argv[0] << "\tThe minStep must be > 0" << std::endl;
    return -1;
  }

  if(args.maxStep < args.minStep){
    std::cerr << argv[0] << "\tThe maxStep must be > minStep" << std::endl;
    return -1;
  }

  if(args.gradTol < 0){
    std::cerr << argv[0] << "\tThe gradTol must be >= 0" << std::endl;
    return -1;
  }

  if(args.relaxFactor < 0 || args.relaxFactor > 1){
    std::cerr << argv[0] << "\tThe relaxFactor must be >= 0 and <= 1" << std::endl;
    return -1;
  }

  if(args.learningRate < 0 || args.learningRate > 1){
    std::cerr << argv[0] << "\tThe learningRate must be >= 0 and <= 1" << std::endl;
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

  unsigned int dims = itk::PeekAtImageDimension(args.fixedImage);
  if (dims != 3 && dims != 2)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
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
