/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkNMIImageToImageMetric.h>
#include <itkTranslationTransform.h>
#include <itkBSplineBendingEnergyConstraint.h>
#include <itkMaskedImageRegistrationMethod.h>
#include <itkFFDSteepestGradientDescentOptimizer.h>
#include <itkNMILocalHistogramDerivativeForceFilter.h>
#include <itkParzenWindowNMIDerivativeForceGenerator.h>
#include <itkBSplineDeformableTransform.h>
#include <itkBSplineSmoothVectorFieldFilter.h>
#include <itkInterpolateVectorFieldFilter.h>
#include <itkFFDMultiResolutionMethod.h>
#include <itkLinearlyInterpolatedDerivativeFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkTransformFileWriter.h>
#include <itkImageRegistrationFactory.h>

int FFDRegisterTest( int argc, char *argv[] )
{
  if( argc < 18 )
  {
    std::cerr << "Usage: FFDRegisterTest img1 img2 dx dy iterations levels scaleByGradient scaleComponentWise constraintGradient parzenWindows bins weighting outputImg outputGrid outputSub outputDiff outputTransform compareTransform" << std::endl;
    return EXIT_FAILURE;
  }

  std::string fixedImageName = argv[1];
  std::string movingImageName = argv[2];
  double dx = niftk::ConvertToDouble(argv[3]);
  double dy = niftk::ConvertToDouble(argv[4]);
  int iterations = niftk::ConvertToInt(argv[5]);
  int levels = niftk::ConvertToInt(argv[6]);
  std::string scaleByGradient = argv[7];
  std::string scaleComponentWise = argv[8];
  std::string constraintGradient = argv[9];
  std::string parzenWindows = argv[10];
  int bins = niftk::ConvertToInt(argv[11]);
  double weighting = niftk::ConvertToDouble(argv[12]);
  std::string outputImageName = argv[13];
  std::string outputGridName = argv[14];
  std::string outputSubName = argv[15];
  std::string outputDiffName = argv[16];
  std::string outputTransform = argv[17];
  
  std::string compareTransform;
  if (argc == 19)
    {
      compareTransform = argv[18];
    }
  
  
  const    unsigned int    Dimension = 2;
  typedef  float           PixelType;
  
  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::Image< PixelType, Dimension >  MovingImageType;

  typedef itk::TranslationTransform<double, Dimension> GlobalTransformType;
  GlobalTransformType::Pointer globalTransform = GlobalTransformType::New();
  globalTransform->SetIdentity();
  
  typedef itk::BSplineTransform<FixedImageType, double, Dimension, float> TransformType;
  TransformType::Pointer transform = TransformType::New();

  typedef itk::LinearInterpolateImageFunction< MovingImageType, double> LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction< FixedImageType, double>   BSplineInterpolatorType;
  
  LinearInterpolatorType::Pointer regriddingInterpolator = LinearInterpolatorType::New();
  LinearInterpolatorType::Pointer registrationInterpolator  = LinearInterpolatorType::New();
  LinearInterpolatorType::Pointer resamplingInterpolator = LinearInterpolatorType::New();
  
  typedef itk::NMIImageToImageMetric<FixedImageType, MovingImageType> MetricType;
  MetricType::Pointer metric = MetricType::New();

  typedef itk::ParzenWindowNMIDerivativeForceGenerator<FixedImageType, MovingImageType, double, float> ParzenForceGeneratorFilterType;
  ParzenForceGeneratorFilterType::Pointer parzenForceFilter = ParzenForceGeneratorFilterType::New();

  typedef itk::LinearlyInterpolatedDerivativeFilter<FixedImageType, MovingImageType, double, float> GradientFilterType;
  GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
  
  typedef itk::NMILocalHistogramDerivativeForceFilter<FixedImageType, MovingImageType, float> ForceGeneratorFilterType;
  ForceGeneratorFilterType::Pointer histogramForceFilter = ForceGeneratorFilterType::New();

  typedef itk::BSplineSmoothVectorFieldFilter< float, Dimension> SmoothFilterType;
  SmoothFilterType::Pointer smoothFilter = SmoothFilterType::New();
  
  typedef itk::InterpolateVectorFieldFilter< float, Dimension> VectorInterpolatorType;
  VectorInterpolatorType::Pointer vectorInterpolator = VectorInterpolatorType::New();
  
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

  typedef itk::FFDSteepestGradientDescentOptimizer<FixedImageType, MovingImageType, double, float> OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();

  typedef FixedImageType::SpacingType SpacingType;

  typedef itk::IterationUpdateCommand CommandType;
  CommandType::Pointer command = CommandType::New();
  
  typedef itk::MaskedImageRegistrationMethod<FixedImageType> RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

  typedef itk::FFDMultiResolutionMethod<FixedImageType, double, Dimension, float>   MultiResImageRegistrationMethodType;
  MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  // Load images.
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;
  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  fixedImageReader->SetFileName(  fixedImageName );
  fixedImageReader->Update();
  std::cerr << "Read:" << fixedImageName << std::endl;
  
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  movingImageReader->SetFileName( movingImageName );
  movingImageReader->Update();
  std::cerr << "Read:" << movingImageName << std::endl;
  
  // Start wiring it all together.

  // Setup transformation
  SpacingType spacing;
  spacing[0] = dx;
  spacing[1] = dy;
  globalTransform->SetIdentity();

  typedef itk::BSplineBendingEnergyConstraint<FixedImageType, double, Dimension, float> BendingEnergyConstraintType;
  BendingEnergyConstraintType::Pointer constraint = BendingEnergyConstraintType::New();
  constraint->SetTransform(transform);

  metric->SetWeightingFactor(weighting);
  metric->SetConstraint(constraint);
  metric->SetHistogramSize(bins, bins);
  metric->SetIntensityBounds(0, bins-1, 0, bins-1);
  metric->SetWriteFixedImage(true);
  metric->SetFixedImageFileName("tmp.similarity.fixed");
  metric->SetFixedImageFileExt("vtk");
  metric->SetWriteTransformedMovingImage(true);
  metric->SetTransformedMovingImageFileName("tmp.similarity.moving");
  metric->SetTransformedMovingImageFileExt("vtk");
  
  optimizer->SetDeformableTransform(transform);
  optimizer->SetRegriddingInterpolator(regriddingInterpolator);
  optimizer->SetMaximize(true);
  optimizer->SetMaximumNumberOfIterations(iterations);
  optimizer->SetIteratingStepSizeReductionFactor(0.5);
  optimizer->SetRegriddingStepSizeReductionFactor(0.5);
  optimizer->SetJacobianBelowZeroStepSizeReductionFactor(0.5);
  optimizer->SetMinimumDeformationMagnitudeThreshold(0.001);
  optimizer->SetMinimumJacobianThreshold(0.3);
  optimizer->SetSmoothFilter(smoothFilter);
  optimizer->SetInterpolatorFilter(vectorInterpolator);
  optimizer->SetMinimumSimilarityChangeThreshold(0.001);
  optimizer->SetWriteForceImage(true);
  optimizer->SetForceImageFileName("tmp.force");
  optimizer->SetForceImageFileExt("vtk");
  optimizer->SetSmoothGradientVectorsBeforeInterpolatingToControlPointLevel(true);
  optimizer->SetWriteNextParameters(true);
  optimizer->SetNextParametersFileName("tmp.next");
  optimizer->SetNextParametersFileExt("vtk");
  optimizer->SetWriteDeformationField(true);
  optimizer->SetDeformationFieldFileName("tmp.deformation");
  optimizer->SetDeformationFieldFileExt("vtk");
  optimizer->SetCheckMinDeformationMagnitudeThreshold(true);
  
  registration->SetMetric(metric);
  registration->SetTransform(transform);
  registration->SetInterpolator(registrationInterpolator);
  registration->SetOptimizer(optimizer);
  registration->SetIterationUpdateCommand(command);
  registration->SetRescaleFixedImage(true);
  
  registration->SetRescaleMovingImage(true);
  registration->SetRescaleFixedMinimum(0);
  registration->SetRescaleFixedMaximum(bins-1);
  registration->SetRescaleMovingMinimum(0);
  registration->SetRescaleMovingMaximum(bins-1);

  multiResMethod->SetFixedImage(fixedImageReader->GetOutput());
  multiResMethod->SetMovingImage(movingImageReader->GetOutput());  
  multiResMethod->SetSingleResMethod(registration);
  multiResMethod->SetTransform(transform);
  multiResMethod->SetWriteJacobianImageAtEachLevel(false);
  multiResMethod->SetJacobianImageFileName("tmp.jacobian");
  multiResMethod->SetJacobianImageFileExtension("nii");
  multiResMethod->SetWriteVectorImageAtEachLevel(false);
  multiResMethod->SetVectorImageFileName("tmp.vector");
  multiResMethod->SetVectorImageFileExtension("vtk");
  multiResMethod->SetWriteParametersAtEachLevel(false);
  multiResMethod->SetParameterFileName("tmp.cp");
  multiResMethod->SetParameterFileExt("vtk");
  
  multiResMethod->SetFinalControlPointSpacing(spacing);
  multiResMethod->SetNumberOfLevels(levels);

  if (constraintGradient == "TRUE")
    {
      metric->SetUseConstraintGradient(true);
    }
  else
    {
      metric->SetUseConstraintGradient(false);
    }
  
  if (parzenWindows == "TRUE")
    {
      gradientFilter->SetTransform(transform);
      
      metric->SetUseParzenFilling(true);
      metric->SetIntensityBounds(0, bins-1, 0, bins-1);

      parzenForceFilter->SetMetric(metric);
      parzenForceFilter->SetScalarImageGradientFilter(gradientFilter);
      parzenForceFilter->SetFixedLowerPixelValue(0);
      parzenForceFilter->SetFixedUpperPixelValue(bins-1);
      parzenForceFilter->SetMovingLowerPixelValue(0);
      parzenForceFilter->SetMovingUpperPixelValue(bins-1);              
      optimizer->SetForceFilter(parzenForceFilter);
    }
  else
    {
      histogramForceFilter->SetMetric(metric);
      optimizer->SetForceFilter(histogramForceFilter);
    }
  
  if (scaleByGradient == "TRUE")
    {
      optimizer->SetScaleForceVectorsByGradientImage(true);
    }
  else
    {
      optimizer->SetScaleForceVectorsByGradientImage(false);
    }
    
  if (scaleComponentWise == "TRUE")
    {
      optimizer->SetScaleByComponents(true);
    }
  else
    {
      optimizer->SetScaleByComponents(false);
    }
  
  // Now run it.
  multiResMethod->StartRegistration();
  
  typedef itk::ResampleImageFilter<MovingImageType, FixedImageType >           ResampleFilterType;
  typedef itk::RescaleIntensityImageFilter<FixedImageType, FixedImageType >    RescalerType;

  typedef unsigned char OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >                             OutputImageType;  
  typedef itk::CastImageFilter< FixedImageType, OutputImageType >              CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >                              WriterType;

  ResampleFilterType::Pointer resampler = ResampleFilterType::New();  
  RescalerType::Pointer intensityRescaler = RescalerType::New();
  CastFilterType::Pointer caster =  CastFilterType::New();
  WriterType::Pointer writer =  WriterType::New();
  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

  // Write the transformation 
  // Save the transform
  typedef itk::TransformFileWriter TransformFileWriterType;
  TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
  transformFileWriter->SetInput(transform);
  transformFileWriter->SetFileName(outputTransform); 
  transformFileWriter->Update(); 
  
  // Produce a transformed moving image
  resampler->SetTransform( transform );
  resampler->SetInput( movingImageReader->GetOutput() );
  resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
  resampler->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler->SetOutputDirection( fixedImage->GetDirection() );
  resampler->SetInterpolator(resamplingInterpolator);
  caster->SetInput( resampler->GetOutput() );
  writer->SetInput( caster->GetOutput()   );  
  writer->SetFileName( outputImageName );
  writer->Update();
  
  // Produce a grid, and resample that. So we can see which way the image is bending.
  
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
      
      if (x % 20 == 0 || y % 20 == 0 || x % 20 == 1 || y % 20 == 1)
      {
        deformationField->SetPixel(index, 0);
      }
    }
  }
  resampler->SetInput( deformationField );
  writer->SetFileName(outputGridName);
  resampler->Modified();  
  caster->Modified();
  writer->Modified();
  writer->Update();
  
  // Produce a difference image of registered images
  typedef itk::SubtractImageFilter<FixedImageType, FixedImageType, FixedImageType > DifferenceFilterType;
  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();
  resampler->SetInput( movingImageReader->GetOutput() );
  difference->SetInput1( fixedImageReader->GetOutput() );
  difference->SetInput2( resampler->GetOutput() );
  intensityRescaler->SetInput( difference->GetOutput() );
  intensityRescaler->SetOutputMinimum(   0 );
  intensityRescaler->SetOutputMaximum( 255 );
  caster->SetInput(intensityRescaler->GetOutput());
  writer->SetFileName( outputSubName );
  resampler->Modified();
  intensityRescaler->Modified();
  caster->Modified();
  writer->Update();

  // Produce a difference image of the initial input.
  difference->SetInput1( fixedImageReader->GetOutput() );
  difference->SetInput2( movingImageReader->GetOutput() );
  writer->SetFileName( outputDiffName );
  difference->Modified();
  resampler->Modified();
  intensityRescaler->Modified();
  caster->Modified();
  writer->Update();

  // If we have argument 19, it means we have a dof file to compare to.
  if (compareTransform.length() > 0)
    {
      std::cout << "Comparing dof files" << std::endl;
      typedef itk::ImageRegistrationFactory<FixedImageType, Dimension, double> FactoryType;
      FactoryType::Pointer factory = FactoryType::New();
      FactoryType::TransformType::Pointer resultTransform;
      FactoryType::TransformType::Pointer expectedTransform;
      
      resultTransform = factory->CreateTransform(outputTransform);
      expectedTransform = factory->CreateTransform(compareTransform);
      
      if (resultTransform->GetNumberOfParameters() != expectedTransform->GetNumberOfParameters())
        {
          std::cout << "Number of parameters differs, expected:" << expectedTransform->GetNumberOfParameters() << ", but result had:" << resultTransform->GetNumberOfParameters() << std::endl;
          return EXIT_FAILURE;
        }
      
      for (unsigned int i = 0; i < resultTransform->GetNumberOfParameters(); i++)
        {
          if (fabs(resultTransform->GetParameters()[i] - expectedTransform->GetParameters()[i]) > 0.5)
            {
              std::cout << "Parameter " << i << " differs. Expected:" << expectedTransform->GetParameters()[i] << ", but result had:" << resultTransform->GetParameters()[i] << std::endl;
              return EXIT_FAILURE;
            }
        }
      std::cout << "Comparing dof files...PASSED" << std::endl;
    }

  return EXIT_SUCCESS;
}
