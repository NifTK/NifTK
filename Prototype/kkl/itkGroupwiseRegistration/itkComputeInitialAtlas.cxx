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
#include "itkResampleImageFilter.h"
#include "itkImageRegionIterator.h"

/**
 * \brief 
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image< PixelType, Dimension >  InputImageType; 
  // Setup objects to build registration.
  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, double> FactoryType;
  typedef itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  if (argc < 6) 
  {
    std::cout << argv[0] << " atlas outputTransformNameFormat fixedImage finalInterp movingImage1 dof1 movingImage2 dof2 ..." << std::endl; 
    return 1; 
  }
  
  InputImageType::Pointer atlas; 
  
  // The factory.
  FactoryType::Pointer factory = FactoryType::New();
  FixedImageReaderType::Pointer fixedImageReader  = FixedImageReaderType::New();
  char* atlasName = argv[1]; 
  char* outputTransformNameFormat = argv[2]; 
  char* fixedImageName = argv[3]; 
  fixedImageReader->SetFileName(fixedImageName); 
  std::cout << "Reading fixed image: " << fixedImageName << std::endl; 
  fixedImageReader->Update(); 
  int finalInterp = atoi(argv[4]); 
  
  int startingArgIndex = 5; 
  int numberOfMovingImages = (argc-startingArgIndex)/2; 
  FixedImageReaderType::Pointer *movingImageReader = new FixedImageReaderType::Pointer[numberOfMovingImages];
  FactoryType::TransformType::Pointer averageTransform; 
  FactoryType::TransformType::Pointer *deformableTransform = new FactoryType::TransformType::Pointer[numberOfMovingImages]; 
  FactoryType::TransformType::ParametersType averageParameters; 
  FactoryType::InterpolatorType::Pointer interpolator; 
  interpolator = factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterp);
  
  // Read in the moving images and compute an average transformation. 
  try
  {
    movingImageReader[0] = FixedImageReaderType::New(); 
    movingImageReader[0]->SetFileName(argv[startingArgIndex]); 
    std::cout << "Reading moving image: 0 " << argv[startingArgIndex] << std::endl; 
    movingImageReader[0]->Update(); 
    std::cout << "Reading tranform: 0 " << argv[startingArgIndex+1] << std::endl; 
    averageTransform = factory->CreateTransform(argv[startingArgIndex+1]);
    deformableTransform[0] = factory->CreateTransform(argv[startingArgIndex+1]);
    averageParameters = deformableTransform[0]->GetParameters(); 
    for (int i = 1; i < numberOfMovingImages; i++)
    {
      movingImageReader[i] = FixedImageReaderType::New(); 
      movingImageReader[i]->SetFileName(argv[startingArgIndex+2*i]); 
      std::cout << "Reading moving image: " << i << " " << argv[startingArgIndex+2*i] << std::endl; 
      movingImageReader[i]->Update(); 
      
      std::cout << "Reading transform: " << i << " " << argv[startingArgIndex+2*i+1] << std::endl; 
      deformableTransform[i] = factory->CreateTransform(argv[startingArgIndex+2*i+1]);
      FactoryType::TransformType::ParametersType parameters = deformableTransform[i]->GetParameters();
      
      for (unsigned int parameterIndex = 0; parameterIndex < averageParameters.GetSize(); parameterIndex++)
      {
        averageParameters[parameterIndex] += parameters[parameterIndex]; 
      }
    }
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed to load deformableTransform tranform:" << exceptionObject << std::endl;
    return 2; 
  }
  std::cout << "Averaging deformable parameters..." << std::endl; 
  for (unsigned int parameterIndex = 0; parameterIndex < averageParameters.GetSize(); parameterIndex++)
  {
    averageParameters[parameterIndex] /= static_cast<double>(numberOfMovingImages+1); 
  }
  averageTransform->SetParameters(averageParameters); 
  averageParameters.SetSize(1); 
  
  if (strcmp(deformableTransform[0]->GetNameOfClass(),"BSplineDeformableTransform") == 0)
  {
    // Invert the average transform.   
    FactoryType::BSplineDeformableTransformType* bSplineTransform = dynamic_cast<FactoryType::BSplineDeformableTransformType*>(averageTransform.GetPointer()); 
    typedef itk::BSplineTransform<InputImageType, double, Dimension, float> BSplineTransformType;
    BSplineTransformType::Pointer inverseTransform = BSplineTransformType::New();
    inverseTransform->Initialize(fixedImageReader->GetOutput(), 1.0, 1); 
    bSplineTransform->SetInverseSearchRadius(5); 
    bSplineTransform->GetInverse(inverseTransform); 
    
    // Starting building the atlas. 
    typedef itk::ResampleImageFilter< InputImageType, InputImageType >   ResampleFilterType;
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New(); 
    
    resampleFilter->SetUseReferenceImage(true); 
    resampleFilter->SetInput(fixedImageReader->GetOutput());
    resampleFilter->SetDefaultPixelValue(static_cast<InputImageType::PixelType>(0)); 
    resampleFilter->SetReferenceImage(fixedImageReader->GetOutput()); 
      
    // Fixed image is transformed using the inverse average transform. 
    resampleFilter->SetTransform(inverseTransform);
    resampleFilter->Update(); 
    atlas = resampleFilter->GetOutput(); 
    atlas->DisconnectPipeline(); 
    
    // Concatenate the inverse transform with the individual transform. 
    for (int i = 0; i < numberOfMovingImages; i++)    
    {
      resampleFilter->SetInput(movingImageReader[i]->GetOutput());
      FactoryType::BSplineDeformableTransformType* currentTransform = dynamic_cast<FactoryType::BSplineDeformableTransformType*>(deformableTransform[i].GetPointer()); 
      currentTransform->ConcatenateAfterGivenTransform(inverseTransform); 
      resampleFilter->SetTransform(currentTransform);
      resampleFilter->Update(); 
      
      itk::ImageRegionIterator< InputImageType > atlasIterator(atlas, atlas->GetLargestPossibleRegion()); 
      itk::ImageRegionIterator< InputImageType > imageIterator(resampleFilter->GetOutput(), resampleFilter->GetOutput()->GetLargestPossibleRegion()); 
      for (atlasIterator.GoToBegin(), imageIterator.GoToBegin();
           !atlasIterator.IsAtEnd(); 
           ++atlasIterator, ++imageIterator)
      {
        atlasIterator.Set(atlasIterator.Get() + imageIterator.Get()); 
      }
    }
    
  }
  else if (strcmp(deformableTransform[0]->GetNameOfClass(),"FluidDeformableTransform") == 0)
  {
    FactoryType::FluidDeformableTransformType* fluidTransform = dynamic_cast<FactoryType::FluidDeformableTransformType*>(averageTransform.GetPointer()); 
    std::cout << "fluidTransform=" << fluidTransform << std::endl; 
    typedef itk::FluidDeformableTransform<InputImageType, double, Dimension, float > FluidDeformableTransformType;
    FluidDeformableTransformType::Pointer inverseTransform = FluidDeformableTransformType::New();
    inverseTransform->Initialize(fixedImageReader->GetOutput()); 
    fluidTransform->SetInverseSearchRadius(5); 
    std::cout << "Inverting transform..." << std::endl; 
    fluidTransform->GetInverse(inverseTransform); 
    
    // Starting building the atlas. 
    typedef itk::ResampleImageFilter<InputImageType, InputImageType >   ResampleFilterType;
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New(); 
    
    resampleFilter->SetUseReferenceImage(true); 
    resampleFilter->SetInput(fixedImageReader->GetOutput());
    resampleFilter->SetDefaultPixelValue(static_cast<InputImageType::PixelType>(0)); 
    resampleFilter->SetReferenceImage(fixedImageReader->GetOutput()); 
    resampleFilter->SetInterpolator(interpolator);
      
    // Fixed image is transformed using the inverse average transform. 
    resampleFilter->SetTransform(inverseTransform);
    std::cout << "Transforming fixed image..." << std::endl; 
    resampleFilter->Update(); 
    atlas = resampleFilter->GetOutput(); 
    atlas->DisconnectPipeline(); 
    
    // Save the transform
    typedef itk::TransformFileWriter TransformFileWriterType;
    char outputTransformName[255]; 
    int outputTransformNameIndex = 0; 
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    sprintf(outputTransformName, outputTransformNameFormat, outputTransformNameIndex); 
    outputTransformNameIndex++; 
    inverseTransform->SetParametersFromField(inverseTransform->GetDeformationField(), true); 
    transformFileWriter->SetInput(inverseTransform);
    transformFileWriter->SetFileName(outputTransformName); 
    transformFileWriter->Update(); 
    
    // Concatenate the inverse transform with the individual transform. 
    for (int i = 0; i < numberOfMovingImages; i++)    
    {
      FactoryType::FluidDeformableTransformType* currentTransform = dynamic_cast<FactoryType::FluidDeformableTransformType*>(deformableTransform[i].GetPointer()); 
      std::cout << "currentTransform=" << currentTransform << std::endl; 
      resampleFilter->SetInput(movingImageReader[i]->GetOutput());
      std::cout << "Concatenating transform..." << i << " ..." << std::endl; 
      currentTransform->ConcatenateAfterGivenTransform(inverseTransform); 
      std::cout << "Transforming moving image " << i << " ..." << std::endl; 
      resampleFilter->SetTransform(currentTransform);
      resampleFilter->Update(); 
      
      // Save the transform
      sprintf(outputTransformName, outputTransformNameFormat, outputTransformNameIndex); 
      outputTransformNameIndex++; 
      currentTransform->SetParametersFromField(currentTransform->GetDeformationField(), true); 
      transformFileWriter->SetInput(currentTransform);
      transformFileWriter->SetFileName(outputTransformName); 
      transformFileWriter->Update(); 
      
      itk::ImageRegionIterator< InputImageType > atlasIterator(atlas, atlas->GetLargestPossibleRegion()); 
      itk::ImageRegionIterator< InputImageType > imageIterator(resampleFilter->GetOutput(), resampleFilter->GetOutput()->GetLargestPossibleRegion()); 
      for (atlasIterator.GoToBegin(), imageIterator.GoToBegin();
           !atlasIterator.IsAtEnd(); 
           ++atlasIterator, ++imageIterator)
      {
        atlasIterator.Set(atlasIterator.Get() + imageIterator.Get()); 
      }
    }
    
  }
  
  std::cout << "Averaging the atlas intensity..." << std::endl; 
  itk::ImageRegionIterator< InputImageType > atlasIterator(atlas, atlas->GetLargestPossibleRegion()); 
  for (atlasIterator.GoToBegin(); !atlasIterator.IsAtEnd(); ++atlasIterator)
  {
    atlasIterator.Set(atlasIterator.Get()/(numberOfMovingImages+1)); 
  }
  
  OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  outputImageWriter->SetFileName(atlasName);
  outputImageWriter->SetInput(atlas);
  std::cout << "Saving atlas..." << std::endl; 
  outputImageWriter->Update();
  
  std::cout << "You've got an atlas." << std::endl; 

  if (movingImageReader != NULL) delete movingImageReader;
  if (deformableTransform != NULL) delete deformableTransform;

  return 0; 
}      
  
  
  
    
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
