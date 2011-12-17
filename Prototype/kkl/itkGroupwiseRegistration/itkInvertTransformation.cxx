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
#include "ConversionUtils.h"
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
    std::cout << argv[0] << " inputTransform outputTransform" << std::endl; 
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
    FactoryType::FluidDeformableTransformType* fluidTransform = dynamic_cast<FactoryType::FluidDeformableTransformType*>(factory->CreateTransform(inputTransformName).GetPointer()); 
    FactoryType::FluidDeformableTransformType* inverseTransform = dynamic_cast<FactoryType::FluidDeformableTransformType*>(factory->CreateTransform(inputTransformName).GetPointer()); 
    
    FactoryType::EulerAffineTransformType* affineTransform = dynamic_cast<FactoryType::EulerAffineTransformType*>(factory->CreateTransform(inputTransformName).GetPointer()); 
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
      inverseAffineTransform->SetFixedParameters(affineTransform->GetFixedParameters()); 
      inverseAffineTransform->SetCenter(inverseAffineTransform->GetCenter()); 
      affineTransform->GetInverse(inverseAffineTransform); 
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


