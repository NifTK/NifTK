/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-15 12:05:31 +0100 (Thu, 15 Sep 2011) $
 Revision          : $Revision: 7313 $
 Last modified by  : $Author: kkl $

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
#include "itkShiftScaleImageFilter.h"

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
  typedef itk::EulerAffineTransform<double, Dimension, Dimension> AffineTransformType; 
  typedef itk::MatrixLinearCombinationFunctions<AffineTransformType::FullAffineMatrixType::InternalMatrixType> MatrixLinearCombinationFunctionsType; 
  
  itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1); 

  if (argc < 7) 
  {
    std::cout << argv[0] << " outputFormat fixedImage finalInterp isPropBack ajc movingImage1 dof1 movingImage2 dof2 ..." << std::endl; 
    return 1; 
  }
  
  InputImageType::Pointer atlas; 
  
  // The factory.
  FactoryType::Pointer factory = FactoryType::New();
  FixedImageReaderType::Pointer fixedImageReader  = FixedImageReaderType::New();
  char* outputFormat = argv[1]; 
  char* fixedImageName = argv[2]; 
  fixedImageReader->SetFileName(fixedImageName); 
  std::cout << "Reading fixed image: " << fixedImageName << std::endl; 
  fixedImageReader->Update(); 
  int finalInterp = atoi(argv[3]); 
  int isPropBack = atoi(argv[4]); 
  int isAffineJacobianCorrection = atoi(argv[5]); 
  
  int startingArgIndex = 6; 
  int numberOfMovingImages = (argc-startingArgIndex)/2; 
  FixedImageReaderType::Pointer* movingImageReader = new FixedImageReaderType::Pointer[numberOfMovingImages];
  AffineTransformType::Pointer averageTransform; 
  FactoryType::TransformType::Pointer* transform = new FactoryType::TransformType::Pointer[numberOfMovingImages]; 
  AffineTransformType** affineTransform = new AffineTransformType*[numberOfMovingImages]; 
  for (int i = 0; i < numberOfMovingImages; i++)
  {
    affineTransform[i] = AffineTransformType::New();
  }
  AffineTransformType::FullAffineMatrixType averageMatrix; 
  FactoryType::InterpolatorType::Pointer interpolator; 
  interpolator = factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterp);
  
  InputImageType::SpacingType averageSpacing = fixedImageReader->GetOutput()->GetSpacing();
  std::cout << "Fixed image spacing=" << averageSpacing << std::endl; 
  
  // Read in the moving images and compute an average transformation. 
  try
  {
    movingImageReader[0] = FixedImageReaderType::New(); 
    movingImageReader[0]->SetFileName(argv[startingArgIndex]); 
    std::cout << "Reading moving image: 0 " << argv[startingArgIndex] << std::endl; 
    movingImageReader[0]->Update(); 
    std::cout << "Reading tranform: 0 " << argv[startingArgIndex+1] << std::endl; 
    averageTransform = dynamic_cast<AffineTransformType*>(factory->CreateTransform(argv[startingArgIndex+1]).GetPointer());
    transform[0] = factory->CreateTransform(argv[startingArgIndex+1]);
    affineTransform[0] = dynamic_cast<AffineTransformType*>(transform[0].GetPointer()); 
    averageMatrix = affineTransform[0]->GetFullAffineMatrix(); 
    averageMatrix = MatrixLinearCombinationFunctionsType::ComputeMatrixLogarithm(averageMatrix.GetVnlMatrix(), 0.001); 
    InputImageType::SpacingType spacing = movingImageReader[0]->GetOutput()->GetSpacing();
    for (unsigned int j = 0; j < Dimension; j++)
      averageSpacing[j] += spacing[j]; 
    std::cout << "Total image spacing=" << averageSpacing << std::endl; 
    
    for (int i = 1; i < numberOfMovingImages; i++)
    {
      movingImageReader[i] = FixedImageReaderType::New(); 
      movingImageReader[i]->SetFileName(argv[startingArgIndex+2*i]); 
      std::cout << "Reading moving image: " << i << " " << argv[startingArgIndex+2*i] << std::endl; 
      movingImageReader[i]->Update(); 
      
      std::cout << "Reading transform: " << i << " " << argv[startingArgIndex+2*i+1] << std::endl; 
      transform[i] = factory->CreateTransform(argv[startingArgIndex+2*i+1]);
      affineTransform[i] = dynamic_cast<AffineTransformType*>(transform[i].GetPointer()); 
      
      AffineTransformType::FullAffineMatrixType currentMatrix = affineTransform[i]->GetFullAffineMatrix(); 
      currentMatrix = MatrixLinearCombinationFunctionsType::ComputeMatrixLogarithm(currentMatrix.GetVnlMatrix(), 0.001); 
      averageMatrix += currentMatrix; 
      
      spacing = movingImageReader[i]->GetOutput()->GetSpacing();
      for (unsigned int j = 0; j < Dimension; j++)
        averageSpacing[j] += spacing[j]; 
      std::cout << "Total image spacing=" << averageSpacing << std::endl; 
    }
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed to load affine tranform:" << exceptionObject << std::endl;
    return 2; 
  }
  std::cout << "Averaging affine parameters..." << std::endl; 
  averageMatrix /= (numberOfMovingImages+1.0); 
  averageMatrix = MatrixLinearCombinationFunctionsType::ComputeMatrixExponential(averageMatrix.GetVnlMatrix()); 
  averageTransform->SetFullAffineMatrix(averageMatrix); 
  for (unsigned int j = 0; j < Dimension; j++)
    averageSpacing[j] /= (numberOfMovingImages+1.0); 
  std::cout << "Averaging image spacing=" << averageSpacing << std::endl; 
  
  // Invert the average transform.   
  AffineTransformType::Pointer inverseTransform = AffineTransformType::New();
  std::cout << "Inverting transform..." << std::endl; 
  averageTransform->GetInverse(inverseTransform); 
    
  // Starting building the atlas. 
  typedef itk::ResampleImageFilter<InputImageType, InputImageType >   ResampleFilterType;
  ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New(); 
  
  resampleFilter->SetInput(fixedImageReader->GetOutput());
  resampleFilter->SetDefaultPixelValue(static_cast<InputImageType::PixelType>(0)); 
  resampleFilter->SetInterpolator(interpolator);
  resampleFilter->SetOutputSpacing(averageSpacing); 
  resampleFilter->SetOutputDirection(fixedImageReader->GetOutput()->GetDirection()); 
  resampleFilter->SetOutputOrigin(fixedImageReader->GetOutput()->GetOrigin()); 
  resampleFilter->SetSize(fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize()); 
      
  // Fixed image is transformed using the inverse average transform. 
  double fixedImageAffineScalingFactor = 1.0; 
  std::cout << "Transforming fixed image..." << std::endl; 
  if (isPropBack == 0)
  {
    resampleFilter->SetTransform(inverseTransform);
    resampleFilter->Update(); 
    std::cout << "inverseTransform:" << std::endl << inverseTransform->GetFullAffineMatrix() << std::endl; 
    inverseTransform->SetParametersFromTransform(inverseTransform->GetFullAffineTransform()); 
    for (unsigned int j = 0; j < Dimension; j++)
      fixedImageAffineScalingFactor *= inverseTransform->GetScale()[j]; 
    std::cout << "Scaling factor = " << fixedImageAffineScalingFactor << std::endl; 
    std::cout << "inverseTransform:" << std::endl << inverseTransform->GetFullAffineMatrix() << std::endl; 
  }
  else
  {
    resampleFilter->SetTransform(averageTransform);
    resampleFilter->Update(); 
    averageTransform->SetParametersFromTransform(averageTransform->GetFullAffineTransform()); 
    for (unsigned int j = 0; j < Dimension; j++)
      fixedImageAffineScalingFactor *= averageTransform->GetScale()[j]; 
    std::cout << "Scaling factor = " << fixedImageAffineScalingFactor << std::endl; 
  }
  atlas = resampleFilter->GetOutput(); 
  atlas->DisconnectPipeline(); 
  
  char outputFilename[4096]; 
  int outputFileIndex = 0; 
  OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
  sprintf(outputFilename, outputFormat, outputFileIndex); 
  outputFileIndex++; 
  outputImageWriter->SetFileName(outputFilename);
  outputImageWriter->SetInput(atlas);
  
  typedef itk::ShiftScaleImageFilter<InputImageType, InputImageType> ShiftScaleImageFilterType; 
  ShiftScaleImageFilterType::Pointer shiftScaleImageFilter = ShiftScaleImageFilterType::New(); 

  if (isAffineJacobianCorrection != 0)
  {
    shiftScaleImageFilter->SetInput(atlas); 
    shiftScaleImageFilter->SetScale(fixedImageAffineScalingFactor); 
    outputImageWriter->SetInput(shiftScaleImageFilter->GetOutput());
  }
  
  std::cout << "Saving outputs..." << std::endl; 
  outputImageWriter->Update();
    
  std::cout << "inverseTransform:" << std::endl << inverseTransform->GetFullAffineMatrix() << std::endl; 
  // Concatenate the inverse transform with the individual transform. 
  for (int i = 0; i < numberOfMovingImages; i++)    
  {
    resampleFilter->SetInput(movingImageReader[i]->GetOutput());
    std::cout << "Concatenating transform..." << i << " ..." << std::endl; 
    AffineTransformType::FullAffineMatrixType currentMatrix = affineTransform[i]->GetFullAffineMatrix(); 
    currentMatrix *= inverseTransform->GetFullAffineMatrix(); 
    if (isPropBack != 0)
    {
      currentMatrix = currentMatrix.GetInverse(); 
      // Need to resample to the moving image when doing inverse. 
      resampleFilter->SetReferenceImage(movingImageReader[i]->GetOutput()); 
    }
    affineTransform[i]->SetFullAffineMatrix(currentMatrix); 
    std::cout << "Transforming moving image " << i << " ..." << std::endl; 
    resampleFilter->SetTransform(affineTransform[i]);
    resampleFilter->Update(); 
    
    double affineScalingFactor = 1.0; 
    std::cout << affineTransform[i]->GetFullAffineMatrix() << std::endl; 
    affineTransform[i]->SetParametersFromTransform(affineTransform[i]->GetFullAffineTransform()); 
    std::cout << "paramters: " << affineTransform[i]->GetParameters() << std::endl; 
    for (unsigned int j = 0; j < Dimension; j++)
      affineScalingFactor *= affineTransform[i]->GetScale()[j]; 
    std::cout << "Scaling factor = " << affineScalingFactor << std::endl; 
    std::cout << affineTransform[i]->GetFullAffineMatrix() << std::endl; 
    
    sprintf(outputFilename, outputFormat, outputFileIndex); 
    outputFileIndex++;
    outputImageWriter->SetFileName(outputFilename);
    outputImageWriter->SetInput(resampleFilter->GetOutput());
    
    if (isAffineJacobianCorrection != 0)
    {
      shiftScaleImageFilter->SetInput(resampleFilter->GetOutput()); 
      shiftScaleImageFilter->SetScale(affineScalingFactor); 
      outputImageWriter->SetInput(shiftScaleImageFilter->GetOutput());
    }
    
    std::cout << "Saving outputs..." << std::endl; 
    outputImageWriter->Update();
    
    itk::ImageRegionIterator< InputImageType > atlasIterator(atlas, atlas->GetLargestPossibleRegion()); 
    itk::ImageRegionIterator< InputImageType > imageIterator(resampleFilter->GetOutput(), resampleFilter->GetOutput()->GetLargestPossibleRegion()); 
    for (atlasIterator.GoToBegin(), imageIterator.GoToBegin();
          !atlasIterator.IsAtEnd(); 
          ++atlasIterator, ++imageIterator)
    {
      atlasIterator.Set(atlasIterator.Get() + imageIterator.Get()); 
    }
  }
    
  if (isPropBack == 0)
  {
    std::cout << "Averaging the atlas intensity..." << std::endl; 
    itk::ImageRegionIterator< InputImageType > atlasIterator(atlas, atlas->GetLargestPossibleRegion()); 
    for (atlasIterator.GoToBegin(); !atlasIterator.IsAtEnd(); ++atlasIterator)
    {
      atlasIterator.Set(atlasIterator.Get()/(numberOfMovingImages+1)); 
    }
    
    sprintf(outputFilename, outputFormat, 999); 
    outputImageWriter->SetFileName(outputFilename);
    outputImageWriter->SetInput(atlas);
    std::cout << "Saving atlas..." << std::endl; 
    outputImageWriter->Update();
    
    std::cout << "You've got an atlas." << std::endl; 
  }

  if (movingImageReader != NULL) 
    delete [] movingImageReader;
  if (transform != NULL) 
    delete [] transform;
  if (affineTransform != NULL) 
    delete [] affineTransform;

  return 0; 
}      
  
  
  
    
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
