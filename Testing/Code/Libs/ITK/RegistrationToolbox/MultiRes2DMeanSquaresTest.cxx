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
#include <iostream>
#include <memory>
#include <math.h>
#include <niftkConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegistrationFactory.h>
#include <itkMultiResolutionImageRegistrationWrapper.h>
#include <itkMaskedImageRegistrationMethod.h>
#include <itkSingleResolutionImageRegistrationBuilder.h>

/**
 * Aim of this test is simply to test the base class
 * SingleResolutionImageRegistrationMethod.
 * 
 * Note: While testing I found it very necessary to 
 * call registration->SetFixedImageRegion(fixedImageReader->GetOutput()->GetBufferedRegion() );
 * which I then added into itkSingleResolutionImageRegistrationMethod::GenerateData().
 */
int MultiRes2DMeanSquaresTest(int argc, char * argv[])
{
  if( argc < 8)
    {
    std::cerr << "Usage   : MultiRes2DMeanSquaresTest levels img1 img2 inputX inputY expectedX expectedY" << std::endl;
    return 1;
    }
 
  // Parse Input
  const    unsigned int    Dimension = 2;
  int levels = niftk::ConvertToInt(argv[1]);
  std::string fixedImage = argv[2];
  std::string movingImage = argv[3];
  double inputX = niftk::ConvertToDouble(argv[4]);
  double inputY = niftk::ConvertToDouble(argv[5]);
  double expectedX = niftk::ConvertToDouble(argv[6]);
  double expectedY = niftk::ConvertToDouble(argv[7]);

  // Load images.
  typedef itk::Image< float, Dimension>                                            RegImageType;
  typedef itk::ImageFileReader< RegImageType  >                                    ImageReaderType;

  ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
  ImageReaderType::Pointer movingImageReader = ImageReaderType::New();

  fixedImageReader->SetFileName(  fixedImage );
  fixedImageReader->Update();
  
  movingImageReader->SetFileName( movingImage );
  movingImageReader->Update();
  
  // Build Registration
  typedef itk::MultiResolutionImageRegistrationWrapper<RegImageType> MultiResImageRegistrationMethodType;
  typedef itk::MaskedImageRegistrationMethod<RegImageType> SingleResImageRegistrationMethodType;
  typedef itk::SingleResolutionImageRegistrationBuilder<RegImageType, Dimension, double> BuilderType;
  typedef itk::UCLRegularStepGradientDescentOptimizer RegularStepGradientDescentType;
  typedef RegularStepGradientDescentType* RegularStepGradientDescentPointer;
  typedef SingleResImageRegistrationMethodType::ParametersType ParametersType;
  
  BuilderType::Pointer builder = BuilderType::New();
  MultiResImageRegistrationMethodType::Pointer multires = MultiResImageRegistrationMethodType::New();
  
  // Accept all the defaults.
  builder->StartCreation(itk::SINGLE_RES_MASKED);
  builder->CreateInterpolator(itk::LINEAR);
  builder->CreateMetric(itk::MSD);  
  builder->CreateTransform(itk::TRANSLATION, fixedImageReader->GetOutput());
  builder->CreateOptimizer(itk::REGSTEP_GRADIENT_DESCENT);
  SingleResImageRegistrationMethodType::Pointer registration = builder->GetSingleResolutionImageRegistrationMethod();
  
  try  
    {
      RegularStepGradientDescentPointer optimizer = dynamic_cast<RegularStepGradientDescentPointer>(registration->GetOptimizer());
      optimizer->SetMaximumStepLength( 4.00 );
      optimizer->SetMinimumStepLength( 0.00001 );
      optimizer->SetNumberOfIterations( 200 );
      optimizer->SetMaximize(false);
      ParametersType initialParameters( registration->GetTransform()->GetNumberOfParameters() );
      initialParameters[0] = inputX;  // Initial offset in mm along X
      initialParameters[1] = inputY;  // Initial offset in mm along Y
      multires->SetInitialTransformParameters( initialParameters );
      multires->SetSingleResMethod(registration);
      multires->SetFixedImage(fixedImageReader->GetOutput());
      multires->SetMovingImage(movingImageReader->GetOutput());
      multires->SetNumberOfLevels(levels);
      multires->StartRegistration();
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }
  if (fabs(expectedX - registration->GetLastTransformParameters()[0]) > 0.0001) return EXIT_FAILURE;
  if (fabs(expectedY - registration->GetLastTransformParameters()[1]) > 0.0001) return EXIT_FAILURE;

  // These should default to OFF.
  if (registration->GetRescaleFixedImage()) return EXIT_FAILURE;
  if (registration->GetRescaleMovingImage()) return EXIT_FAILURE;

  // Mask sure the threshold gets passed to and from internal object. 
  registration->SetRescaleFixedMinimum(10);
  if (registration->GetRescaleFixedMinimum() != 10) return EXIT_FAILURE;
  registration->SetRescaleFixedMaximum(99);
  if (registration->GetRescaleFixedMaximum() != 99) return EXIT_FAILURE;
  
  registration->SetRescaleMovingMinimum(11);
  if (registration->GetRescaleMovingMinimum() != 11) return EXIT_FAILURE;
  registration->SetRescaleMovingMaximum(100);
  if (registration->GetRescaleMovingMaximum() != 100) return EXIT_FAILURE;
  
  // This is how the filter facade will eventually look.
  multires->SetFixedImage(fixedImageReader->GetOutput());
  multires->SetMovingImage(movingImageReader->GetOutput());
  registration->SetRescaleFixedImage(true);
  registration->SetRescaleFixedMinimum(1);
  registration->SetRescaleFixedMaximum(100);
  registration->SetRescaleMovingImage(true);
  registration->SetRescaleMovingMinimum(1);
  registration->SetRescaleMovingMaximum(100);


  return EXIT_SUCCESS;    
}
