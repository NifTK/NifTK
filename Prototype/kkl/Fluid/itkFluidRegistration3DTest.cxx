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
#include "itkImageRegionIteratorWithIndex.h"

// General registration
#include "itkMaskedImageRegistrationMethod.h"

// Fluid specific stuff.
#include "itkFluidDeformableTransform.h"
#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkFluidPDEFilter.h"
#include "itkFluidVelocityToDeformationFilter.h"
#include "itkFluidGradientDescentOptimizer.h"
#include "itkConstantPadImageFilter.h"

const    unsigned int    Dimension = 3;
typedef  int PixelType;
typedef itk::Image< PixelType, Dimension >  FixedImageType;
typedef itk::Image< PixelType, Dimension >  MovingImageType;
typedef itk::Image< int, Dimension >  DiffImageType;


int itkFluidRegistration3DTest( int argc, char *argv[] )
{
  if( argc < 4 )
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << "outputImagefile stretch mu lambda step_size iterations bin_size differenceImageAfter";
    std::cerr << "differenceImageBefore min_deformation_threshold" << std::endl;
    return EXIT_FAILURE;
  }
  
  typedef itk::TranslationTransform<double, Dimension> GlobalTransformType;
  GlobalTransformType::Pointer globalTransform = GlobalTransformType::New();
  
  typedef itk::FluidDeformableTransform<FixedImageType, double, Dimension, float > TransformType;
  TransformType::Pointer transform = TransformType::New();
  
  typedef itk::LinearInterpolateImageFunction< MovingImageType, double> LinearInterpolatorType;
  LinearInterpolatorType::Pointer registrationInterpolator  = LinearInterpolatorType::New();
  
  typedef itk::BSplineInterpolateImageFunction< FixedImageType, double>   BSplineInterpolatorType;
  //BSplineInterpolatorType::Pointer regriddingInterpolator = BSplineInterpolatorType::New();
  
  typedef itk::ConstantBoundaryCondition< FixedImageType >  BoundaryConditionType;
  const unsigned int WindowRadius = 5;
  typedef itk::Function::HammingWindowFunction<WindowRadius>  WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction< 
      FixedImageType, 
      WindowRadius,
      WindowFunctionType, 
      BoundaryConditionType, 
      double  >    SincInterpolatorType;
  // SincInterpolatorType::Pointer sincInterpolator  = SincInterpolatorType::New();
  SincInterpolatorType::Pointer regriddingInterpolator  = SincInterpolatorType::New();
  
  typedef itk::NMIImageToImageMetric<FixedImageType, MovingImageType> MetricType;
  MetricType::Pointer metric = MetricType::New();
  
  typedef itk::NMILocalHistogramDerivativeForceFilter<FixedImageType, MovingImageType, float> ForceGeneratorFilterType;
  ForceGeneratorFilterType::Pointer forceFilter = ForceGeneratorFilterType::New();

  typedef itk::FluidPDEFilter<float, Dimension > FluidPDEFilterType;
  FluidPDEFilterType::Pointer fluidPDEFilter = FluidPDEFilterType::New();
  
  typedef itk::FluidVelocityToDeformationFilter<float, Dimension > FluidVelocityToDeformationFilterType;
  FluidVelocityToDeformationFilterType::Pointer fluidAddVelocityFilter = FluidVelocityToDeformationFilterType::New();
 

  typedef itk::FluidGradientDescentOptimizer<FixedImageType, MovingImageType, double, float> OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();

  typedef itk::IterationUpdateCommand CommandType;
  CommandType::Pointer command = CommandType::New();

  typedef itk::MaskedImageRegistrationMethod<FixedImageType> RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

#if 0
  MovingImageType::Pointer testImage = MovingImageType::New();
  MovingImageType::SizeType testImageSize; 
  typedef itk::ImageFileWriter< MovingImageType >  TestWriterType;
  double spacing[] = { 1.0, 1.0, 1.0 };
  double origin[] = { 0.0, 0.0, 0.0 };
   
  testImageSize[0] = 128; 
  testImageSize[1] = 128; 
  testImageSize[2] = 128; 
  testImage->SetRegions(testImageSize);
  testImage->SetSpacing(spacing);
  testImage->SetOrigin(origin);
  testImage->Allocate();
  int r = 40;
  for (unsigned int x = 0; x < testImageSize[0]; x++)
  {
    for (unsigned int y = 0; y < testImageSize[1]; y++)
    {
      for (unsigned int z = 0; z < testImageSize[2]; z++)
      {
	MovingImageType::IndexType index; 
	
	index[0] = x;
	index[1] = y;
	index[2] = z;
	testImage->SetPixel(index, 0.0);
	
	if (x > 10 &&  x < r && y > 10 && y < 2*r && z > 10 && z < 2*r)
	{
          testImage->SetPixel(index, 100+rand()%2);
	}
      }
    }
  }
  TestWriterType::Pointer testWriter = TestWriterType::New();
  testWriter->SetInput(testImage);
  testWriter->SetFileName("fluid-3d-test-1.nii");
  testWriter->Update();
  r = 43;
  for (unsigned int x = 0; x < testImageSize[0]; x++)
  {
    for (unsigned int y = 0; y < testImageSize[1]; y++)
    {
      for (unsigned int z = 0; z < testImageSize[2]; z++)
      {
	MovingImageType::IndexType index; 
	
	index[0] = x;
	index[1] = y;
	index[2] = z;
	testImage->SetPixel(index, 0.0);
	
        if (x > 9 &&  x < r && y > 9 && y < 2*r && z > 9 && z < 2*r)
        {
          testImage->SetPixel(index, 100+rand()%2);
	}
      }
    }
  }
  testWriter->SetInput(testImage);
  testWriter->SetFileName("fluid-3d-test-2.nii");
  testWriter->Update();

  return 0;
#endif

  // Load images.
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;
  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  fixedImageReader->SetFileName(argv[1]);
  fixedImageReader->Update();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  movingImageReader->SetFileName(argv[2]);
  movingImageReader->Update();
  FixedImageType* fixedImage = fixedImageReader->GetOutput();
  MovingImageType* movingImage = movingImageReader->GetOutput();
  
  // Setup transformation, as it needs to know how big the image is.
  globalTransform->SetIdentity();
  //transform->SetGlobalTransform(globalTransform);
  transform->Initialize(fixedImage);
  transform->SetIdentity();
  
  forceFilter->SetMetric(metric);

  // Command line args for PDE
  fluidPDEFilter->SetMu(atof(argv[5]));
  fluidPDEFilter->SetLambda(atof(argv[6])); 
  
  // Command line args for histogram  
  metric->SetHistogramSize(atoi(argv[9]), atoi(argv[9]));
  metric->ComputeGradientOff();
  if (argc > 14)
  {
    metric->SetIntensityBounds(atoi(argv[13]), 5000, atoi(argv[14]), 5000);
  }
  
  // for itkLocalSimilarityMeasureGradientDescentOptimizer
  optimizer->SetDeformableTransform(transform);
  optimizer->SetRegriddingInterpolator(regriddingInterpolator);
  optimizer->SetMaximize(true);
  optimizer->SetMaximumNumberOfIterations(atoi(argv[8]));
  optimizer->SetStepSize(atof(argv[7]));

  // for itkFluidGradientDescentOptimizer
  optimizer->SetForceFilter(forceFilter);
  optimizer->SetFluidPDESolver(fluidPDEFilter);
  optimizer->SetFluidVelocityToDeformationFilter(fluidAddVelocityFilter);
  optimizer->SetCheckSimilarityMeasure(true);
  optimizer->SetMinimumDeformationMagnitudeThreshold(atof(argv[12]));
  optimizer->SetRegriddingStepSizeReductionFactor(1.0);
  optimizer->SetMinimumSimilarityChangeThreshold(1.0e-9); 
      
  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);
  registration->SetMetric(metric);
  registration->SetTransform(transform);
  registration->SetOptimizer(optimizer);
  registration->SetInitialTransformParameters(transform->GetParameters());
  registration->SetIterationUpdateCommand(command);
  registration->SetInterpolator(registrationInterpolator);
  
  
#if 0  
  // interpolator test. 
  itk::ImageRegionIteratorWithIndex< FixedImageType > testIterator(fixedImageReader->GetOutput(), fixedImageReader->GetOutput()->GetLargestPossibleRegion());
  
  //registrationInterpolator->SetInputImage(fixedImageReader->GetOutput());
  registrationInterpolator->SetInputImage(movingImageReader->GetOutput());
  for (testIterator.GoToBegin(); !testIterator.IsAtEnd(); ++testIterator)
  {
    FixedImageType::IndexType index = testIterator.GetIndex();
    FixedImageType::PointType point; 
  
    fixedImageReader->GetOutput()->TransformIndexToPhysicalPoint(index, point);
    
    if (testIterator.Get() > 0)
      std::cout << index << "," << point << "," << testIterator.Get() << "," << registrationInterpolator->Evaluate(point) << std::endl;
  }
#endif  
  
  
  // Now run it.
  registration->Update();

  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double bestValue = optimizer->GetValue();
    
  std::cout << "Result = " << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;

  typedef itk::ResampleImageFilter< 
                            MovingImageType, 
                            FixedImageType >    ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  typedef itk::BSplineInterpolateImageFunction< FixedImageType > BSplineInterpolatorType;
  
  BSplineInterpolatorType::Pointer bsplineInterpolator  = BSplineInterpolatorType::New();
  SincInterpolatorType::Pointer sincInterpolator  = SincInterpolatorType::New();
  
  resampler->SetInput(movingImage);
  resampler->SetTransform( transform );
  resampler->SetOutputParametersFromImage(fixedImage);
  resampler->SetDefaultPixelValue( 100 );
  resampler->SetInterpolator(sincInterpolator);
  
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
  intensityRescaler->SetInput( resampler->GetOutput() );
  intensityRescaler->SetOutputMinimum(   0 );
  intensityRescaler->SetOutputMaximum( 255 );
  
  writer->SetInput( intensityRescaler->GetOutput()   );
  writer->Update();
  
  
  MovingImageType::Pointer deformationField = MovingImageType::New();
  MovingImageType::SizeType regionSize = movingImage->GetLargestPossibleRegion().GetSize(); 
  
  deformationField->SetRegions(regionSize);
  deformationField->Allocate();
  for (unsigned int x = 0; x < regionSize[0]; x++)
  {
    for (unsigned int y = 0; y < regionSize[1]; y++)
    {
      for (unsigned int z = 0; z < regionSize[2]; z++)
      {
        MovingImageType::IndexType index; 
        
        index[0] = x;
        index[1] = y;
        index[2] = z;
        deformationField->SetPixel(index, 200);
        
        if (x % 10 == 0 || y % 10 == 0 || z % 10 == 0)
        {
          deformationField->SetPixel(index, 0);
        }
      }
    }
  }
  resampler->SetInput( deformationField );
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
    movingImage->Modified();
    resampler->SetInput(movingImage);
    resampler->SetTransform( transform );
    difference->SetInput1(fixedImage);
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
      difference->SetInput1(fixedImage);
      difference->SetInput2(movingImage);
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
