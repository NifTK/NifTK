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
#include "itkMultiStageImageRegistrationMethod.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkImageRegistrationFactory.h"
#include "itkSmartPointer.h"
#include "itkAffineTransform.h"

/**
 * Aim of this test is simply to test the switching between Rigid and Scale.
 * 
 */
int SingleRes2DMultiStageMethodTest(int argc, char * argv[])
{
  if( argc < 13)
    {
    std::cerr << "Usage   : SingleRes2DMultiStageMethodTest Method mask img1 img2 mask1 mask2 inputX inputY inputZ expectedX expectedY expectedZ" << std::endl;
    return 1;
    }
 
  // Parse Input
  const    unsigned int    Dimension = 2;
  int method = niftk::ConvertToInt(argv[1]);
  std::string mask = argv[2];
  std::string fixedImage = argv[3];
  std::string movingImage = argv[4];
  std::string fixedMask = argv[5];
  std::string movingMask = argv[6];
  double inputX = niftk::ConvertToDouble(argv[7]);
  double inputY = niftk::ConvertToDouble(argv[8]);
  double inputZ = niftk::ConvertToDouble(argv[9]);
  double expectedX = niftk::ConvertToDouble(argv[10]);
  double expectedY = niftk::ConvertToDouble(argv[11]);
  double expectedZ = niftk::ConvertToDouble(argv[12]);
  
  // Load images.
  typedef itk::Image< short, Dimension>                                            RegImageType;
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
  typedef itk::MultiResolutionImageRegistrationWrapper<RegImageType>MultiResImageRegistrationMethodType;  
  typedef itk::ImageRegistrationFactory<RegImageType, Dimension, double>  ImageRegistrationFactoryType;
  typedef itk::UCLRegularStepGradientDescentOptimizer RegularStepGradientDescentType;
  typedef RegularStepGradientDescentType* RegularStepGradientDescentPointer;
  typedef itk::MultiStageImageRegistrationMethod<RegImageType> RegistrationType;
  typedef RegistrationType* RegistrationPointer;
  typedef itk::EulerAffineTransform<double, Dimension, Dimension> TransformType;
  typedef TransformType* TransformPointer;
  typedef itk::SingleResolutionImageRegistrationBuilder<RegImageType, Dimension, double> BuilderType;
  typedef RegistrationType::ParametersType ParametersType;

  BuilderType::Pointer builder = BuilderType::New();
  builder->StartCreation((itk::SingleResRegistrationMethodTypeEnum)method);
  builder->CreateInterpolator(itk::LINEAR);
  builder->CreateMetric(itk::NCC);
  builder->CreateTransform(itk::AFFINE, fixedImageReader->GetOutput()); // these should get overriden
  builder->CreateOptimizer(itk::SIMPLEX); // these should get overriden
  
  MultiResImageRegistrationMethodType::Pointer multires = MultiResImageRegistrationMethodType::New();  
  RegistrationPointer registration = dynamic_cast<RegistrationPointer>(builder->GetSingleResolutionImageRegistrationMethod().GetPointer());
  TransformPointer transform = dynamic_cast<TransformPointer>(registration->GetTransform());
  
  transform->SetIdentity();
  transform->SetRigid();
  
  ParametersType initialParameters( transform->GetNumberOfParameters() );
  std::cerr << "Initial Parameters:" << initialParameters << std::endl;
  initialParameters[0] = inputZ;  // Initial rotation about Z
  initialParameters[1] = inputX;  // Initial offset in mm along X
  initialParameters[2] = inputY;  // Initial offset in mm along Y  
  std::cerr << "Initial Parameters:" << initialParameters << std::endl;
  
  multires->SetInitialTransformParameters( initialParameters );
  multires->SetFixedImage ( fixedImageReader->GetOutput() );
  multires->SetMovingImage ( movingImageReader->GetOutput() );
  
  if (mask != "OFF")
    {
      multires->SetFixedMask( fixedMaskReader->GetOutput());
      multires->SetMovingMask( movingMaskReader->GetOutput());
      registration->SetFixedMaskMinimum(1);
      registration->SetMovingMaskMinimum(1);
      registration->SetNumberOfDilations(2);      
    }

  try  
    {
      RegularStepGradientDescentPointer optimizer = dynamic_cast<RegularStepGradientDescentPointer>(registration->GetOptimizer());
      optimizer->SetMaximumStepLength( 2.00 );
      optimizer->SetMinimumStepLength( 0.01 );
      optimizer->SetNumberOfIterations( 200 );
      optimizer->SetMaximize(true);
      multires->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  if (fabs(expectedX - multires->GetLastTransformParameters()[0]) > 0.1) return EXIT_FAILURE;
  if (fabs(expectedY - multires->GetLastTransformParameters()[1]) > 0.1) return EXIT_FAILURE;
  if (fabs(expectedZ - multires->GetLastTransformParameters()[2]) > 0.1) return EXIT_FAILURE;
  
  return EXIT_SUCCESS;    
}
