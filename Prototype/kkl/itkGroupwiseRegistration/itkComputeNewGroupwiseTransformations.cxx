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
  
  if (argc < 4)
  {
    std::cout << argv[0] << " outputTransformNameFormat maxDifferenceName inputTransform1 inputTransform2 ... " << std::endl; 
    return -1; 
  }
  
  char* outputTransformNameFormat = argv[1]; 
  char* maxDifferenceName = argv[2]; 
  int startingArgIndex = 3; 
  
  // Setup objects to build registration.
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  
  // The factory.
  FactoryType::Pointer factory = FactoryType::New();
  FactoryType::TransformType::Pointer *deformableTransform = new FactoryType::TransformType::Pointer[argc-startingArgIndex]; 
  FactoryType::TransformType::ParametersType averageParameters; 
  
  // Save the transform
  typedef itk::TransformFileWriter TransformFileWriterType;
  TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
  
  // Compute average transform. 
  try
  {
    deformableTransform[0] = factory->CreateTransform(argv[startingArgIndex]);
    averageParameters = deformableTransform[0]->GetParameters(); 
  
    for (int i = startingArgIndex+1; i < argc; i++)
    {
      deformableTransform[i-startingArgIndex] = factory->CreateTransform(argv[i]);
      FactoryType::TransformType::ParametersType parameters = deformableTransform[i-startingArgIndex]->GetParameters();
      
      for (unsigned int parameterIndex = 0; parameterIndex < averageParameters.GetSize(); parameterIndex++)
      {
        averageParameters[parameterIndex] += parameters[parameterIndex]; 
      }
    }
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed to load deformableTransform tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  double maxDifference = 0.0; 
  double totalDifference = 0.0; 
  for (unsigned int parameterIndex = 0; parameterIndex < averageParameters.GetSize(); parameterIndex++)
  {
    averageParameters[parameterIndex] /= static_cast<double>(argc-startingArgIndex); 
    
    double diff = fabs(averageParameters[parameterIndex]); 
    totalDifference += diff; 
    if (diff > maxDifference)
      maxDifference = diff; 
  }
  std::cout << "maxDifference=" << maxDifference << std::endl; 
  
  // Update the individual transforms using the average transform, so that the total transform is zero. 
  for (int i = startingArgIndex; i < argc; i++)
  {
    FactoryType::TransformType::ParametersType parameters = deformableTransform[i-startingArgIndex]->GetParameters();
    
    for (unsigned int parameterIndex = 0; parameterIndex < averageParameters.GetSize(); parameterIndex++)
    {
      parameters[parameterIndex] -= averageParameters[parameterIndex]; 
    }
    deformableTransform[i-startingArgIndex]->SetParameters(parameters); 
    
    char outputName[255]; 
    
    sprintf(outputName, outputTransformNameFormat, i-startingArgIndex); 
    transformFileWriter->SetFileName(outputName); 
    transformFileWriter->SetInput(deformableTransform[i-startingArgIndex]);
    transformFileWriter->Update(); 
  }
  

  std::ofstream maxDifferenceFileStream(maxDifferenceName); 
  maxDifferenceFileStream << maxDifference << "," << totalDifference << std::endl; 
  
  if (deformableTransform != NULL) delete deformableTransform;

  return EXIT_SUCCESS;   
}


