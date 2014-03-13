/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegistrationFactory.h>
#include <itkImageRegistrationFilter.h>
#include <itkImageRegistrationFactory.h>
#include <itkGradientDescentOptimizer.h>
#include <itkUCLSimplexOptimizer.h>
#include <itkUCLRegularStepGradientDescentOptimizer.h>
#include <itkSingleResolutionImageRegistrationBuilder.h>
#include <itkMaskedImageRegistrationMethod.h>
#include <itkTransformFileWriter.h>

/*!
 * \file niftkInvertTransformation.cxx
 * \page niftkInvertTransformation
 * \section niftkInvertTransformationSummary Program to invert transformation".
 * 
 * Program to invert transformation:. 
 * 
 */


/**
 * \brief 
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image< PixelType, Dimension >  InputImageType; 
  
  if (argc < 3)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Inverts an affine transform" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << argv[0] << " inputTransform outputTransform" << std::endl; 
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    inputTransform       Input transformation." << std::endl;
    std::cout << "    outputTransform      Output transformation." << std::endl << std::endl;      
    return -1; 
  }
  
  char* inputTransformName = argv[1]; 
  char* outputTransformName = argv[2]; 
  
  // Setup objects to build registration.
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  
  // The factory.
  FactoryType::Pointer factory = FactoryType::New();
  
  // Save the transform
  typedef itk::TransformFileWriter TransformFileWriterType;
  TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
  
  
  // Compute average transform. 
  try
  {
    FactoryType::TransformType::Pointer genericTransform = factory->CreateTransform(inputTransformName);
    FactoryType::FluidDeformableTransformType* fluidTransform = dynamic_cast<FactoryType::FluidDeformableTransformType*>(genericTransform.GetPointer());
    FactoryType::FluidDeformableTransformType* inverseTransform = dynamic_cast<FactoryType::FluidDeformableTransformType*>(genericTransform.GetPointer());
    
    FactoryType::EulerAffineTransformType* affineTransform = dynamic_cast<FactoryType::EulerAffineTransformType*>(genericTransform.GetPointer());
    FactoryType::EulerAffineTransformType::Pointer inverseAffineTransform = FactoryType::EulerAffineTransformType::New();

    if (fluidTransform != NULL)
    {
      std::cout << "Inverting fluid transform..." << std::endl; 
      fluidTransform->SetInverseSearchRadius(5); 
      fluidTransform->GetInverse(inverseTransform); 
      transformFileWriter->SetInput(inverseTransform);
    }
    else if (affineTransform != NULL)
    {
      std::cout << "Inverting affine transform..." << std::endl; 
      affineTransform->Print(std::cout);
      inverseAffineTransform->SetFixedParameters(affineTransform->GetFixedParameters());
      inverseAffineTransform->SetCenter(inverseAffineTransform->GetCenter()); 
      affineTransform->GetInverse(inverseAffineTransform);
      std::cout << inverseAffineTransform << std::endl;
      inverseAffineTransform->SetParametersFromTransform(inverseAffineTransform->GetFullAffineTransform()); 
      transformFileWriter->SetInput(inverseAffineTransform);
    }
      
    transformFileWriter->SetFileName(outputTransformName); 
    transformFileWriter->Update(); 
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed to load deformableTransform tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  return EXIT_SUCCESS;   
}


