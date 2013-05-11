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
#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegistrationFactory.h>
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
int SingleRes2DMeanSquaresTest(int argc, char * argv[])
{
  if( argc < 8)
    {
    std::cerr << "Usage   : SingleRes2DNCCTest img1 img2 inputX inputY expectedX expectedY tolerance" << std::endl;
    return 1;
    }
 
  // Parse Input
  const    unsigned int    Dimension = 2;
  std::string fixedImage = argv[1];
  std::string movingImage = argv[2];
  double inputX = niftk::ConvertToDouble(argv[3]);
  double inputY = niftk::ConvertToDouble(argv[4]);
  double expectedX = niftk::ConvertToDouble(argv[5]);
  double expectedY = niftk::ConvertToDouble(argv[6]);
  double tolerance = niftk::ConvertToDouble(argv[7]);
  
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
  typedef itk::MaskedImageRegistrationMethod<RegImageType> ImageRegistrationMethodType;
  typedef itk::ImageRegistrationFactory<RegImageType, Dimension, double> ImageRegistrationFactoryType;
  typedef itk::UCLRegularStepGradientDescentOptimizer RegularStepGradientDescentType;
  typedef RegularStepGradientDescentType* RegularStepGradientDescentPointer;
  typedef itk::MaskedImageRegistrationMethod<RegImageType> RegistrationType;
  typedef itk::SingleResolutionImageRegistrationBuilder<RegImageType, Dimension, double> BuilderType;
  typedef RegistrationType::ParametersType ParametersType;
  
  BuilderType::Pointer builder = BuilderType::New();
  
  // Accept all the defaults.
  builder->StartCreation(itk::SINGLE_RES_MASKED);
  builder->CreateInterpolator(itk::LINEAR);
  builder->CreateMetric(itk::MSD);  
  builder->CreateTransform(itk::TRANSLATION, fixedImageReader->GetOutput());
  builder->CreateOptimizer(itk::REGSTEP_GRADIENT_DESCENT);
  RegistrationType::Pointer registration = builder->GetSingleResolutionImageRegistrationMethod();
  
  try  
    {
      RegularStepGradientDescentPointer optimizer = dynamic_cast<RegularStepGradientDescentPointer>(registration->GetOptimizer());
      optimizer->SetMaximumStepLength( 4.00 );
      optimizer->SetMinimumStepLength( 0.0001 );
      optimizer->SetNumberOfIterations( 200 );
      ParametersType initialParameters( registration->GetTransform()->GetNumberOfParameters() );
      initialParameters[0] = inputX;  // Initial offset in mm along X
      initialParameters[1] = inputY;  // Initial offset in mm along Y
      registration->SetInitialTransformParameters( initialParameters );
      registration->SetFixedImage ( fixedImageReader->GetOutput() );
      registration->SetMovingImage ( movingImageReader->GetOutput() );
      registration->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  if (fabs(expectedX - registration->GetLastTransformParameters()[0]) > tolerance) return EXIT_FAILURE;
  if (fabs(expectedY - registration->GetLastTransformParameters()[1]) > tolerance) return EXIT_FAILURE;
  
  return EXIT_SUCCESS;    
}
