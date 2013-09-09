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
#include <itkImageRegistrationFactory.h>
#include <itkImageRegistrationFilter.h>
#include <itkTransformFileWriter.h>

/*!
 * \file niftkComputeMeanTransformation.cxx
 * \page niftkComputeMeanTransformation
 * \section niftkCombineSegmentationsSummary Compute the geometric mean of the transformations. 
 * 
 * Compute the geometric mean of the transformations
 *
 */

void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Compute the geometric mean of the transformation." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << exec << " outputTransform tol inputTransform1 weight1 inputTransform2 weight2 ... " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    outputTransform         Output transformation" << std::endl;
  std::cout << "    tol                     Tolerance for matrix exp and log, typically 1e-8" << std::endl; 
  std::cout << "    inputTransform1         Input transformation 1" << std::endl; 
  std::cout << "    weight1                 Weight of the transformation 1" << std::endl; 
  std::cout << "    ...                                                   " << std::endl << std::endl;      
}


/**
 * \brief Compute mean transformaitons. 
 */
int main(int argc, char** argv)
{
  if (argc < 7)
  {
    Usage(argv[0]); 
    return EXIT_FAILURE; 
  }
  
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image< PixelType, Dimension >  InputImageType; 
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  typedef itk::EulerAffineTransform<double, Dimension, Dimension> AffineTransformType; 
  typedef itk::MatrixLinearCombinationFunctions<AffineTransformType::FullAffineMatrixType::InternalMatrixType> MatrixLinearCombinationFunctionsType; 
  typedef itk::TransformFileWriter TransformFileWriterType;
  itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1); 
  
  int startingArgIndex = 3; 
  const char* outputName = argv[1]; 
  double tolerance = atof(argv[2]);   // suggest 1e-7. 
  double numberOfTransformations = static_cast<double>(argc-startingArgIndex)/2; 
  std::cout << "numberOfTransformations=" << numberOfTransformations << std::endl; 
  
  try
  {
    FactoryType::Pointer factory = FactoryType::New();
    AffineTransformType::Pointer averageTransform = AffineTransformType::New(); 
    AffineTransformType::FullAffineMatrixType averageMatrix; 
    std::cout << "averageMatrix=" << averageMatrix << std::endl; 
    
    for (int i = 0; i < numberOfTransformations; i++)
    {
      std::cout << "----------------------------------------" << std::endl; 
      std::cout << "Reading tranform: " << i << " " << argv[startingArgIndex+2*i] << std::endl; 
      AffineTransformType* currentTransform = dynamic_cast<AffineTransformType*>(factory->CreateTransform(argv[startingArgIndex+2*i]).GetPointer());
      double weight = atof(argv[startingArgIndex+2*i+1]); 
      std::cout << "weight=" << weight << std::endl; 
      AffineTransformType::FullAffineMatrixType currentMatrix = currentTransform->GetFullAffineMatrix();  
      currentMatrix = MatrixLinearCombinationFunctionsType::ComputeMatrixLogarithm(currentMatrix.GetVnlMatrix(), tolerance); 
      currentMatrix *= weight; 
      currentMatrix /= numberOfTransformations; 
      averageMatrix += currentMatrix; 
      std::cout << "averageMatrix=" << averageMatrix << std::endl; 
      std::cout << "----------------------------------------" << std::endl; 
    }
      
    averageMatrix = MatrixLinearCombinationFunctionsType::ComputeMatrixExponential(averageMatrix.GetVnlMatrix()); 
    averageTransform->SetFullAffineMatrix(averageMatrix); 
    averageTransform->SetParametersFromTransform(averageTransform->GetFullAffineTransform());     
    std::cout << "averageMatrix=" << averageMatrix << std::endl; 
    
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(averageTransform);
    transformFileWriter->SetFileName(outputName); 
    transformFileWriter->Update(); 
  }    
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed:" << exceptionObject << std::endl;
    return 2; 
  }
  
  return 0; 
}


