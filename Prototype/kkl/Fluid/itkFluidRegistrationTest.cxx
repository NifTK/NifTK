/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-13 10:54:10 +0000 (Tue, 13 Dec 2011) $
 Revision          : $Revision: 8003 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkWindowedSincInterpolateImageFunction.h"
#include "itkNMIImageToImageMetric.h"
#include "itkTranslationTransform.h"
#include "itkNCCImageToImageMetric.h"
#include "itkSSDRegistrationForceFilter.h"
#include "itkCrossCorrelationDerivativeForceFilter.h"

// General registration
#include "itkMaskedImageRegistrationMethod.h"

// Fluid specific stuff.
#include "itkFluidDeformableTransform.h"
#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkFluidPDEFilter.h"
#include "itkFluidVelocityToDeformationFilter.h"
#include "itkFluidGradientDescentOptimizer.h" 
#include "itkVelocityFieldGradientDescentOptimizer.h"
#include "itkVelocityFieldDeformableTransform.h"

const    unsigned int    Dimension = 2;
//typedef  unsigned char   PixelType;
typedef  double PixelType;
typedef itk::Image< PixelType, Dimension >  FixedImageType;
typedef itk::Image< PixelType, Dimension >  MovingImageType;
typedef itk::Image< int, Dimension >  DiffImageType;


//int itkFluidRegistrationTest( int argc, char *argv[] )
int main( int argc, char *argv[] )
{
  if( argc < 4 )
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << "outputImagefile stretch mu lambda step_size iterations bin_size differenceImageAfter";
    std::cerr << "differenceImageBefore min_deformation_threshold sim force" << std::endl;
    return EXIT_FAILURE;
  }
  
  typedef itk::TranslationTransform<double, Dimension> GlobalTransformType;
  GlobalTransformType::Pointer globalTransform = GlobalTransformType::New();
  
  typedef itk::FluidDeformableTransform<FixedImageType, double, Dimension, float > TransformType;
  //typedef itk::VelocityFieldDeformableTransform<FixedImageType, double, Dimension, float > TransformType;
  TransformType::Pointer transform = TransformType::New();
  
  typedef itk::LinearInterpolateImageFunction< MovingImageType, double> LinearInterpolatorType;
  LinearInterpolatorType::Pointer registrationInterpolator  = LinearInterpolatorType::New();
  
  typedef itk::BSplineInterpolateImageFunction< FixedImageType, double>   BSplineInterpolatorType;
  BSplineInterpolatorType::Pointer regriddingInterpolator = BSplineInterpolatorType::New();
  //BSplineInterpolatorType::Pointer registrationInterpolator = BSplineInterpolatorType::New();
  
  typedef itk::NMIImageToImageMetric<FixedImageType, MovingImageType> MetricType;
  MetricType::Pointer metric = MetricType::New();
  typedef itk::NCCImageToImageMetric<FixedImageType, MovingImageType> NCCMetricType;
  NCCMetricType::Pointer nccMetric = NCCMetricType::New();
  
  typedef itk::NMILocalHistogramDerivativeForceFilter<FixedImageType, MovingImageType, float> ForceGeneratorFilterType;
  ForceGeneratorFilterType::Pointer forceFilter = ForceGeneratorFilterType::New();
  typedef itk::SSDRegistrationForceFilter<FixedImageType, MovingImageType, float> SSDForceGeneratorFilterType;
  SSDForceGeneratorFilterType::Pointer ssdForceFilter = SSDForceGeneratorFilterType::New();
  typedef itk::CrossCorrelationDerivativeForceFilter<FixedImageType, MovingImageType, float> CrossCorrelationDerivativeForceFilterType;
  CrossCorrelationDerivativeForceFilterType::Pointer ccForceFilter = CrossCorrelationDerivativeForceFilterType::New(); 

  typedef itk::FluidPDEFilter<float, Dimension > FluidPDEFilterType;
  FluidPDEFilterType::Pointer fluidPDEFilter = FluidPDEFilterType::New();
  
  typedef itk::FluidVelocityToDeformationFilter<float, Dimension > FluidVelocityToDeformationFilterType;
  FluidVelocityToDeformationFilterType::Pointer fluidAddVelocityFilter = FluidVelocityToDeformationFilterType::New();
 
  typedef itk::ConstantBoundaryCondition< FixedImageType >  BoundaryConditionType;
  const unsigned int WindowRadius = 5;
  typedef itk::Function::HammingWindowFunction<WindowRadius>  WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction< 
                                          FixedImageType, 
                                          WindowRadius,
                                          WindowFunctionType, 
                                          BoundaryConditionType, 
                                          double  >    SincInterpolatorType;
  SincInterpolatorType::Pointer sincInterpolator  = SincInterpolatorType::New();

  typedef itk::FluidGradientDescentOptimizer<FixedImageType, MovingImageType, double, float> OptimizerType;
  //typedef itk::VelocityFieldGradientDescentOptimizer<FixedImageType, FixedImageType, double, float> OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();

  typedef itk::IterationUpdateCommand CommandType;
  CommandType::Pointer command = CommandType::New();

  typedef itk::MaskedImageRegistrationMethod<FixedImageType> RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

  // Load images.
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;
  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  fixedImageReader->SetFileName(  argv[1] );
  fixedImageReader->Update();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  movingImageReader->SetFileName( argv[2] );
  movingImageReader->Update();

  // Setup transformation, as it needs to know how big the image is.
  globalTransform->SetIdentity();
  //transform->SetGlobalTransform(globalTransform);
  transform->Initialize(fixedImageReader->GetOutput());
  transform->SetIdentity();
  
  forceFilter->SetMetric(metric);

  // Command line args for PDE
  fluidPDEFilter->SetMu(atof(argv[5]));
  fluidPDEFilter->SetLambda(atof(argv[6])); 
  
  // Command line args for histogram  
  metric->SetHistogramSize(atoi(argv[9]), atoi(argv[9]));
  
  // for itkLocalSimilarityMeasureGradientDescentOptimizer
  optimizer->SetDeformableTransform(transform);
  optimizer->SetRegriddingInterpolator(regriddingInterpolator);
  optimizer->SetMaximize(true);
  optimizer->SetMaximumNumberOfIterations(atoi(argv[8]));
  optimizer->SetStepSize(atof(argv[7]));
  optimizer->SetMinimumJacobianThreshold(0.5); 
  //optimizer->SetFixedImageInterpolator(registrationInterpolator); 
  //optimizer->SetMovingImageInterpolator(registrationInterpolator); 
  optimizer->SetComposeTransformation(true); 

  // for itkFluidGradientDescentOptimizer
  int force = atoi(argv[14]); 
  if (force == 0)
  {
    optimizer->SetForceFilter(forceFilter);
  }
  else if (force == 1)
  {
    //optimizer->SetForceFilter(ssdForceFilter);
    optimizer->SetForceFilter(ccForceFilter); 
  }
  optimizer->SetFluidPDESolver(fluidPDEFilter);
  //optimizer->SetFluidVelocityToDeformationFilter(fluidAddVelocityFilter);
  optimizer->SetCheckSimilarityMeasure(false);
  optimizer->SetMinimumDeformationMagnitudeThreshold(atof(argv[12]));
  optimizer->SetRegriddingStepSizeReductionFactor(1.0);
  optimizer->SetMinimumStepSize(1.0e-20);
  optimizer->SetMinimumSimilarityChangeThreshold(1.0e-15); 
  optimizer->SetIteratingStepSizeReductionFactor(0.75); 
  optimizer->SetIsPropagateRegriddedMovingImage(true); 

  int sim = atoi(argv[13]); 
  if (sim == 0)
    registration->SetMetric(metric);
  else if (sim == 1)
    registration->SetMetric(nccMetric);
  registration->SetTransform(transform);
  registration->SetInterpolator(registrationInterpolator);
  registration->SetOptimizer(optimizer);
  registration->SetFixedImage(fixedImageReader->GetOutput());
  registration->SetMovingImage(movingImageReader->GetOutput());
  registration->SetInitialTransformParameters(transform->GetParameters());
  registration->SetIterationUpdateCommand(command);
  
  // Now run it.
  registration->Update();

  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double bestValue = optimizer->GetValue();
    
  std::cout << "Result = " << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;
  
  std::cout << "Min Jac=" << transform->ComputeMinJacobian() << std::endl; 

  typedef itk::ResampleImageFilter< 
                            MovingImageType, 
                            FixedImageType >    ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  typedef itk::BSplineInterpolateImageFunction< FixedImageType > BSplineInterpolatorType;
  
  BSplineInterpolatorType::Pointer bsplineInterpolator  = BSplineInterpolatorType::New();
  
  resampler->SetInput( movingImageReader->GetOutput() );
  resampler->SetTransform( transform );
  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
  resampler->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler->SetOutputDirection( fixedImage->GetDirection() );
  resampler->SetDefaultPixelValue( 100 );
  resampler->SetInterpolator(bsplineInterpolator);
  
  typedef unsigned char OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CastImageFilter< 
                        FixedImageType,
                        OutputImageType > CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  typedef itk::RescaleIntensityImageFilter< 
                                  FixedImageType, 
                                  OutputImageType >   RescalerType;
  RescalerType::Pointer intensityRescaler = RescalerType::New();
  
  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();

  writer->SetFileName( argv[3] );

  caster->SetInput( resampler->GetOutput() );
  //intensityRescaler->SetInput( resampler->GetOutput() );
  intensityRescaler->SetInput(optimizer->GetRegriddedMovingImage()); 
  intensityRescaler->SetOutputMinimum(   0 );
  intensityRescaler->SetOutputMaximum( 255 );
  
  writer->SetInput( intensityRescaler->GetOutput()   );
  writer->Update();
  
  
  MovingImageType::Pointer deformationField = MovingImageType::New();
  MovingImageType::SizeType regionSize = movingImageReader->GetOutput()->GetLargestPossibleRegion().GetSize(); 
   
  deformationField->SetRegions(regionSize);
  deformationField->Allocate();
  for (unsigned int x = 0; x < regionSize[0]; x++)
  {
    for (unsigned int y = 0; y < regionSize[1]; y++)
    {
      MovingImageType::IndexType index; 
      
      index[0] = x;
      index[1] = y;
      deformationField->SetPixel(index, 200);
      
      if (x % 5 == 0 || y % 5 == 0 || x % 5 == 1 || y % 5 == 1)
      {
        deformationField->SetPixel(index, 0);
      }
    }
  }
  resampler->SetInput( deformationField );
  resampler->Modified(); 
  intensityRescaler->SetInput(resampler->GetOutput()); 
  writer->SetFileName(argv[4]);
  writer->Update();
  
  
  typedef itk::SubtractImageFilter< 
                                  FixedImageType, 
                                  FixedImageType, 
                                  DiffImageType > DifferenceFilterType;
  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();
  WriterType::Pointer writer2 = WriterType::New();
  typedef itk::RescaleIntensityImageFilter< 
                                  DiffImageType, 
                                  OutputImageType >   DifferenceRescalerType;
  DifferenceRescalerType::Pointer intensityRescaler2 = DifferenceRescalerType::New();
  
  
  
  try
  {
    movingImageReader->Modified();
    resampler->SetInput( movingImageReader->GetOutput() );
    resampler->SetTransform( transform );
    difference->SetInput1( fixedImageReader->GetOutput() );
    difference->SetInput2( resampler->GetOutput() );
    
    intensityRescaler2->SetInput( difference->GetOutput() );
    intensityRescaler2->SetOutputMinimum(   0 );
    intensityRescaler2->SetOutputMaximum( 255 );
    resampler->SetDefaultPixelValue( 1 );
    writer2->SetInput( intensityRescaler2->GetOutput() );  
    if( argc > 10 )
    {
      writer2->SetFileName( argv[10] );
      writer2->Update();
    }
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }

  try
  {
    if( argc > 11 )
    {
      difference->SetInput1( fixedImageReader->GetOutput() );
      difference->SetInput2( movingImageReader->GetOutput() );
      difference->Modified();
      intensityRescaler2->SetInput( difference->GetOutput() );
      writer2->SetInput( intensityRescaler2->GetOutput() );  
      writer2->SetFileName( argv[11] );
      writer2->Update();
    }
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
