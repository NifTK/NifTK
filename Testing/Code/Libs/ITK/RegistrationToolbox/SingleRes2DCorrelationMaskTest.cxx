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
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkImageRegistrationFactory.h"
#include "itkSmartPointer.h"

/**
 * Aim of this test is simply to tesReg2D-Mask-1t the masking mechanism
 * in class MaskedImageRegistrationMethod.
 * 
 * So we are using a simple translation, with mean squares cost function.
 * 
 */
int SingleRes2DCorrelationMaskTest(int argc, char * argv[])
{
  if( argc < 13)
    {
    std::cerr << "Usage   : SingleRes2DCorrelationMaskTest mask img1 img2 mask1 mask2 levels dilations inputX inputY expectedX expectedY tolerance" << std::endl;
    return 1;
    }
 
  // Parse Input
  const    unsigned int    Dimension = 2;
  std::string mask = argv[1];
  std::string fixedImage = argv[2];
  std::string movingImage = argv[3];
  std::string fixedMask = argv[4];
  std::string movingMask = argv[5];
  int levels = niftk::ConvertToInt(argv[6]);
  int dilations = niftk::ConvertToInt(argv[7]);
  double inputX = niftk::ConvertToDouble(argv[8]);
  double inputY = niftk::ConvertToDouble(argv[9]);
  double expectedX = niftk::ConvertToDouble(argv[10]);
  double expectedY = niftk::ConvertToDouble(argv[11]);
  double tolerance = niftk::ConvertToDouble(argv[12]);
  
  // Load images.
  typedef itk::Image< float, Dimension>                                            RegImageType;
  typedef itk::ImageFileReader< RegImageType  >                                    ImageReaderType;

  ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
  fixedImageReader->SetFileName(  fixedImage );
  fixedImageReader->Update();

  ImageReaderType::Pointer movingImageReader = ImageReaderType::New();
  movingImageReader->SetFileName( movingImage );
  movingImageReader->Update();

  ImageReaderType::Pointer fixedMaskReader  = ImageReaderType::New();
  fixedMaskReader->SetFileName(  fixedMask );
  fixedMaskReader->Update();

  ImageReaderType::Pointer movingMaskReader  = ImageReaderType::New();
  movingMaskReader->SetFileName(  movingMask );
  movingMaskReader->Update();

  // Build Registration
  typedef itk::MultiResolutionImageRegistrationWrapper<RegImageType> MultiResImageRegistrationMethodType;  
  typedef itk::MaskedImageRegistrationMethod<RegImageType> RegistrationType;
  typedef RegistrationType* RegistrationPointer;
  typedef itk::ImageRegistrationFactory<RegImageType, Dimension, double> ImageRegistrationFactoryType;
  typedef itk::UCLRegularStepGradientDescentOptimizer RegularStepGradientDescentType;
  typedef RegularStepGradientDescentType* RegularStepGradientDescentPointer;  
  typedef itk::SingleResolutionImageRegistrationBuilder<RegImageType, Dimension, double> BuilderType;
  typedef RegistrationType::ParametersType ParametersType;
  
  BuilderType::Pointer builder = BuilderType::New();
  MultiResImageRegistrationMethodType::Pointer multires = MultiResImageRegistrationMethodType::New();
  
  // Accept all the defaults.
  builder->StartCreation(itk::SINGLE_RES_MASKED);
  builder->CreateInterpolator(itk::LINEAR);
  builder->CreateMetric(itk::NCC);  
  builder->CreateTransform(itk::RIGID, fixedImageReader->GetOutput());
  builder->CreateOptimizer(itk::REGSTEP_GRADIENT_DESCENT);
  
  RegistrationPointer singleResRegistration = dynamic_cast<RegistrationPointer>(builder->GetSingleResolutionImageRegistrationMethod().GetPointer());
  
  ParametersType initialParameters( singleResRegistration->GetTransform()->GetNumberOfParameters() );
  initialParameters[0] = inputX;  // Initial offset in mm along X
  initialParameters[1] = inputY;  // Initial offset in mm along Y
  
  multires->SetInitialTransformParameters( initialParameters );
  multires->SetSingleResMethod(singleResRegistration);
  multires->SetFixedImage(fixedImageReader->GetOutput());
  multires->SetMovingImage(movingImageReader->GetOutput());
  multires->SetNumberOfLevels(levels);

  singleResRegistration->SetNumberOfDilations(dilations);
  
  if (mask != "OFF")
    {
      multires->SetFixedMask( fixedMaskReader->GetOutput());
      multires->SetMovingMask( movingMaskReader->GetOutput());
      singleResRegistration->SetUseFixedMask(true);
      singleResRegistration->SetUseMovingMask(true);
    }
  
  singleResRegistration->SetFixedMaskMinimum(1);
  singleResRegistration->SetMovingMaskMinimum(1);
  
  try  
    {
      RegularStepGradientDescentPointer optimizer = dynamic_cast<RegularStepGradientDescentPointer>(singleResRegistration->GetOptimizer());
      optimizer->SetMaximumStepLength( 4.00 );
      optimizer->SetMinimumStepLength( 0.0001 );
      optimizer->SetNumberOfIterations( 200 );
      optimizer->SetMaximize(true);
      multires->SetInitialTransformParameters( singleResRegistration->GetTransform()->GetParameters() );
      multires->StartRegistration();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  if (fabs(expectedX - multires->GetLastTransformParameters()[0]) > tolerance) return EXIT_FAILURE;
  if (fabs(expectedY - multires->GetLastTransformParameters()[1]) > tolerance) return EXIT_FAILURE;
  
  return EXIT_SUCCESS;    
}
