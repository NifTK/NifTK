/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNMIGroupwiseImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkPowellOptimizer.h"
#include "itkAffineTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkTranslationTransform.h"
#include "itkApproximateMultivariateNMIImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"
#include "itkEulerAffineTransform.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkUCLRegularStepOptimizer.h"
#include "itkConjugateGradientOptimizer.h"
#include "itkLBFGSOptimizer.h"
#include "itkEuler3DTransform.h"
#include "itkCenteredEuler3DTransform.h"
#include "itkAmoebaOptimizer.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"

const unsigned int Dimension = 3;
//unsigned int maxNumberOfIterations = 3;
const unsigned int numberOfIterations = 2000;
const double stepLength = 10.0; 

typedef short PixelType;
typedef itk::Image< PixelType, Dimension > FixedImageType;
typedef itk::Image< unsigned char, Dimension > MaskImageType; 
typedef itk::ImageMaskSpatialObject< Dimension > ImageMaskSpatialObjectType;

typedef itk::ImageFileReader< FixedImageType > FixedImageReaderType;
typedef itk::ImageFileReader< MaskImageType > FixedMaskImageReaderType;

typedef itk::LinearInterpolateImageFunction< FixedImageType, double> InterpolatorType;

typedef itk::EulerAffineTransform< double,Dimension,Dimension > TransformType;

//typedef itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
typedef itk::UCLRegularStepOptimizer OptimizerType; 
//typedef itk::PowellOptimizer OptimizerType; 
//typedef itk::ConjugateGradientOptimizer OptimizerType;
//typedef itk::LBFGSOptimizer OptimizerType;
//typedef itk::AmoebaOptimizer OptimizerType;

typedef itk::ApproximateMultivariateNMIImageToImageMetric< FixedImageType > MetricType;  


//#define POWELL 1
//#define CONJUGATE 1
#define UCLOPTIMIZER 1
//#define LBFGS 1
//#define AMOEBA 1

/**
 * Initial registration. 
 */
void InitialRegistration(const FixedImageType* fixedImage, ImageMaskSpatialObjectType* fixedMask, const FixedImageType* movingImage, TransformType::ParametersType& bestParameters, FixedImageType::PointType centerPoint)
{
  typedef itk::MultiResolutionImageRegistrationMethod< FixedImageType, FixedImageType > RegistrationType;
  typedef itk::RecursiveMultiResolutionPyramidImageFilter< FixedImageType, FixedImageType >  FixedImagePyramidType;
  MetricType::InternalMetricType::Pointer metric = MetricType::InternalMetricType::New();
  TransformType::Pointer transform = TransformType::New();
  OptimizerType::Pointer optimizer = OptimizerType::New();
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  RegistrationType::Pointer registration = RegistrationType::New();
  FixedImagePyramidType::Pointer fixedImagePyramid = FixedImagePyramidType::New();
  FixedImagePyramidType::Pointer movingImagePyramid = FixedImagePyramidType::New();
  
  MetricType::HistogramSizeType histogramSize; 
  
  // Initialise the metirc group.
  histogramSize.Fill(64);
  metric->SetHistogramSize(histogramSize);
  transform->SetOptimiseTranslation(true);
  transform->SetOptimiseRotation(true);
  transform->SetOptimiseScale(false);
  transform->SetOptimiseSkew(false);
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetFixedImage(fixedImage);
  registration->SetFixedImageRegion(fixedImage->GetBufferedRegion());
  registration->SetMovingImage(movingImage);
  registration->SetInterpolator(interpolator);
  registration->SetFixedImagePyramid(fixedImagePyramid);
  registration->SetMovingImagePyramid(movingImagePyramid);

  OptimizerType::ScalesType scales(transform->GetNumberOfParameters());
  scales.Fill( 1.0 );
  //scales[0] = 57.0;
  //scales[1] = 57.0;
  //scales[2] = 57.0;
  //scales[3] = 0.2;
  //scales[4] = 0.2;
  //scales[5] = 0.2;
  //scales[6] = 100.0;
  //scales[7] = 100.0;
  //scales[8] = 100.0;
  optimizer->SetScales(scales);
#ifdef POWELL  
  optimizer->SetMaximumIteration(numberOfIterations);
  optimizer->SetMaximumLineIteration(10);
  //optimizer->SetMaximize(false);
  optimizer->SetMaximize(true);
  optimizer->SetStepLength(1.0);
  optimizer->SetStepTolerance(0.01);
  optimizer->SetValueTolerance(0.001);
  optimizer->SetScales(scales);
#endif   
#ifdef UCLOPTIMIZER
  optimizer->SetMaximumStepLength(2.0);
  optimizer->SetMinimumStepLength(0.01);
  optimizer->SetMinimize(false);
  optimizer->SetMaximize(true);
  optimizer->SetNumberOfIterations(100);
#endif  
  transform->SetIdentity();
  transform->SetCenter(centerPoint);
  registration->SetNumberOfLevels(3);
  registration->SetInitialTransformParameters(bestParameters);
  
  try
  {
    metric->SetFixedImageMask(fixedMask);
    registration->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << e << std::endl;
  }
  bestParameters  = registration->GetLastTransformParameters();
  transform->SetParameters(bestParameters);
}

/**
 * Transformation.
 */
void TransformImage(const FixedImageType* fixedImage, const FixedImageType* movingImage, const TransformType* transform, char* outputFilename)
{
  typedef itk::ResampleImageFilter< FixedImageType, FixedImageType > ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  typedef itk::BSplineInterpolateImageFunction< FixedImageType > BSplineInterpolatorType;
  BSplineInterpolatorType::Pointer bsplineInterpolator  = BSplineInterpolatorType::New();
  typedef unsigned char OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  typedef itk::RescaleIntensityImageFilter< FixedImageType, OutputImageType > RescalerType;
  RescalerType::Pointer intensityRescaler = RescalerType::New();
  WriterType::Pointer writer = WriterType::New();
  
  resampler->SetInput(movingImage);
  resampler->SetTransform(transform);
  resampler->SetInterpolator(bsplineInterpolator);
  resampler->SetDefaultPixelValue(0);
  resampler->SetReferenceImage(fixedImage);
  resampler->UseReferenceImageOn();
  intensityRescaler->SetInput(resampler->GetOutput());
  intensityRescaler->SetOutputMinimum(0);
  intensityRescaler->SetOutputMaximum(255);
  writer->SetInput(intensityRescaler->GetOutput());
  writer->SetFileName(outputFilename);
  writer->Update();
}


int main(int argc, char* argv[])
{
  
  MetricType::Pointer metric = MetricType::New();
  std::vector< FixedImageReaderType::Pointer > imageReaders;
  std::vector< TransformType::Pointer > transforms;
  std::vector< TransformType::ParametersType > parametersGroup;
  char* groupOutputFilenameFormat  = NULL;
  int numberOfImages = 0; 
  FixedImageType::SizeType regionSize; 
  const unsigned int dof = 9; 
  
  numberOfImages = atoi(argv[1]); 
  groupOutputFilenameFormat = argv[2];
  metric->SetNumberOfImages(numberOfImages);
  TransformType::ParametersType initialParameters((numberOfImages-1)*dof);
  
  unsigned int imageIndex = 0; 
  FixedImageReaderType::Pointer *fixedImageReader = new FixedImageReaderType::Pointer[numberOfImages]; 
  FixedMaskImageReaderType::Pointer *fixedMaskReader = new FixedMaskImageReaderType::Pointer[numberOfImages]; 
  ImageMaskSpatialObjectType::Pointer *maskObject = new ImageMaskSpatialObjectType::Pointer[numberOfImages]; 
  TransformType::Pointer *tempTransform = new TransformType::Pointer[numberOfImages]; 

  // Read in the images and their masks. 
  for (int argIndex = 3; argIndex < 3+3*numberOfImages-1; argIndex += 3)
  {
    fixedImageReader[imageIndex] = FixedImageReaderType::New();
    fixedMaskReader[imageIndex] = FixedMaskImageReaderType::New();
    maskObject[imageIndex] = ImageMaskSpatialObjectType::New();
    tempTransform[imageIndex] = TransformType::New();
    
    fixedImageReader[imageIndex]->SetFileName(argv[argIndex]);
    fixedImageReader[imageIndex]->Update();
    fixedMaskReader[imageIndex]->SetFileName(argv[argIndex+1]);
    fixedMaskReader[imageIndex]->Update();
    maskObject[imageIndex]->SetImage(fixedMaskReader[imageIndex]->GetOutput());
    maskObject[imageIndex]->Update();
    
    regionSize = fixedImageReader[imageIndex]->GetOutput()->GetLargestPossibleRegion().GetSize(); 
    FixedImageType::IndexType centerIndex; 
    centerIndex[0] = (regionSize[0]-1)/2; 
    centerIndex[1] = (regionSize[1]-1)/2; 
    centerIndex[2] = (regionSize[2]-1)/2; 
    FixedImageType::PointType centerPoint; 
    fixedImageReader[imageIndex]->GetOutput()->TransformIndexToPhysicalPoint(centerIndex, centerPoint);
    
    // Image and mask.     
    metric->SetImage(imageIndex, fixedImageReader[imageIndex]->GetOutput());
    metric->SetImageMask(imageIndex, maskObject[imageIndex]); 
    
    // Transform.         
    tempTransform[imageIndex]->SetNumberOfDOF(dof); 
    tempTransform[imageIndex]->SetIdentity();
    tempTransform[imageIndex]->SetCenter(centerPoint);
    if (strcmp("dummy", argv[argIndex+2]) != 0)
    {
      typedef itk::SingleResolutionImageRegistrationBuilder<FixedImageType, Dimension, double> BuilderType;
      BuilderType::Pointer builder = BuilderType::New(); 
      builder->StartCreation(itk::SINGLE_RES_RIGID_SCALE);
      tempTransform[imageIndex] = dynamic_cast<TransformType*>(builder->CreateTransform(argv[argIndex+2]).GetPointer());
    }
    if (argIndex >= 4)
    {
      for (unsigned int parameterIndex = 0; parameterIndex < dof; parameterIndex++)
        initialParameters[((argIndex-4)/3)*dof+parameterIndex] = tempTransform[imageIndex]->GetParameters()[parameterIndex]; 
    }
    
    metric->SetTransform(imageIndex, tempTransform[imageIndex]);
    imageIndex++; 
  }
    
  MetricType::HistogramSizeType histogramSize; 
  // Initialise the metirc group.
  histogramSize.Fill(64);
  metric->InitialiseMetricGroup(histogramSize);
  
  // Set up the optimizer.  
  OptimizerType::Pointer optimizer = OptimizerType::New();
  OptimizerType::ScalesType scales((numberOfImages-1)*dof);
  
  optimizer->DebugOn();
  optimizer->SetCostFunction(metric.GetPointer());
  
  scales.Fill(1.0);
  for (int index = 0; index < numberOfImages-1; index++)
  {
    int scaleIndex = index*dof;
    
    scales[scaleIndex+6] = 100.0; 
    scales[scaleIndex+7] = 100.0; 
    scales[scaleIndex+8] = 100.0; 
  }
  
#ifdef POWELL  
  optimizer->SetMaximumIteration(numberOfIterations);
  optimizer->SetMaximumLineIteration(10);
  //optimizer->SetMaximize(false);
  optimizer->SetMaximize(true);
  optimizer->SetStepLength(stepLength);
  optimizer->SetStepTolerance(0.01);
  optimizer->SetValueTolerance(0.001);
  optimizer->SetScales(scales);
#endif   
#ifdef CONJUGATE
  typedef  OptimizerType::InternalOptimizerType  vnlOptimizerType;
  vnlOptimizerType* vnlOptimizer = optimizer->GetOptimizer();

  const double F_Tolerance      = 1e-3;  // Function value tolerance
  const double G_Tolerance      = 1e-3;  // Gradient magnitude tolerance 
  const double X_Tolerance      = 1e-4;  // Search space tolerance
  const double Epsilon_Function = 1.0; // Step
  const int    Max_Iterations   = 1000; // Maximum number of iterations

  optimizer->SetMaximize(false);
  optimizer->SetScales(scales);
  vnlOptimizer->set_f_tolerance( F_Tolerance );
  vnlOptimizer->set_g_tolerance( G_Tolerance );
  vnlOptimizer->set_x_tolerance( X_Tolerance ); 
  vnlOptimizer->set_epsilon_function( Epsilon_Function );
  vnlOptimizer->set_max_function_evals( Max_Iterations );
  vnlOptimizer->set_check_derivatives(3);
#endif  
#ifdef LBFGS
  typedef  OptimizerType::InternalOptimizerType  vnlOptimizerType;
  vnlOptimizerType * vnlOptimizer = optimizer->GetOptimizer();
  
  optimizer->SetTrace( false );
  optimizer->SetScales(scales);
  optimizer->SetMaximumNumberOfFunctionEvaluations( 1000 );
  optimizer->SetGradientConvergenceTolerance( 1e-3 );
  optimizer->SetLineSearchAccuracy( 0.1 );
  optimizer->SetDefaultStepLength( 1000.0 );
  optimizer->SetCostFunction(metric.GetPointer());
  vnlOptimizer->set_check_derivatives( 0 );
#endif
#ifdef AMOEBA
  double xTolerance = 0.01;
  double fTolerance = 0.001;
  OptimizerType::ParametersType delta(numberOfMovingImage*transforms[0]->GetNumberOfParameters());
  
  optimizer->SetMaximize(false);
  //optimizer->SetScales(scales);
  optimizer->SetMaximumNumberOfIterations(100);
  optimizer->SetParametersConvergenceTolerance( xTolerance );
  optimizer->SetFunctionConvergenceTolerance( fTolerance );
  delta[0] = 0.5;
  delta[1] = 0.5;
  delta[2] = 0.5;
  delta[3] = 20.0;
  delta[4] = 20.0;
  delta[5] = 20.0;
  optimizer->SetInitialSimplexDelta(delta);
  optimizer->AutomaticInitialSimplexOff();
  optimizer->SetCostFunction(metric.GetPointer());
#endif
#ifdef UCLOPTIMIZER 
  optimizer->SetMaximumStepLength(0.1);
  optimizer->SetMinimumStepLength(0.01);
  optimizer->SetScales(scales);
  optimizer->SetMinimize(false);
  optimizer->SetMaximize(true);
#endif  
  
  try
  { 
    TransformType::ParametersType bestParameters((numberOfImages-1)*dof);

    optimizer->SetInitialPosition(initialParameters);
    optimizer->StartOptimization();
    bestParameters = optimizer->GetCurrentPosition();
  }
  catch (itk::ExceptionObject & err) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }
  
  
  return EXIT_SUCCESS;    
  
}
