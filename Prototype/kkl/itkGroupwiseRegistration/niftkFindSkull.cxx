/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author:  $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkGradientToMagnitudeImageFilter.h"
#include "itkOtsuThresholdImageCalculator.h"
#include "itkOtsuMultipleThresholdsImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkSimpleKMeansClusteringImageFilter.h"

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3; 
  typedef short PixelType; 
  typedef itk::Image<PixelType, Dimension> InputImageType;   
  typedef itk::ImageFileReader<InputImageType> InputImageReaderType;
  typedef itk::ImageFileWriter<InputImageType> OutputImageWriterType;
  typedef itk::ImageRegionIteratorWithIndex<InputImageType> InputImageIteratorType; 
  typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdImageFilterType; 
  
  char* inputFilename = argv[1]; 
  char* inputMaskFilename = argv[2]; 
  char* outputFilename = argv[3]; 
  char* outputFilenameOtsu = argv[4]; 
  char* outputFilenameOtsu2 = argv[5]; 
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageReader->SetFileName(inputFilename);
  imageReader->Update(); 
  maskReader->SetFileName(inputMaskFilename); 
  maskReader->Update(); 
  imageWriter->SetFileName(outputFilename);
  
#if 0  
  
  typedef itk::GradientRecursiveGaussianImageFilter<InputImageType> GradientRecursiveGaussianImageFilterType; 
  GradientRecursiveGaussianImageFilterType::Pointer gradientRecursiveGaussianImageFilter = GradientRecursiveGaussianImageFilterType::New(); 
  
  gradientRecursiveGaussianImageFilter->SetInput(imageReader->GetOutput()); 
  gradientRecursiveGaussianImageFilter->SetSigma(1); 
  gradientRecursiveGaussianImageFilter->SetUseImageDirection(true); 
  gradientRecursiveGaussianImageFilter->Update(); 
  
  typedef itk::GradientToMagnitudeImageFilter<GradientRecursiveGaussianImageFilterType::OutputImageType, InputImageType> GradientToMagnitudeImageFilterType; 
  GradientToMagnitudeImageFilterType::Pointer gradientToMagnitudeImageFilter = GradientToMagnitudeImageFilterType::New(); 
  
  gradientToMagnitudeImageFilter->SetInput(gradientRecursiveGaussianImageFilter->GetOutput()); 
  gradientToMagnitudeImageFilter->Update(); 
  
  imageWriter->SetInput(gradientToMagnitudeImageFilter->GetOutput()); 
  imageWriter->Update(); 
  
  typedef itk::OtsuMultipleThresholdsImageFilter<InputImageType, InputImageType> OtsuMultipleThresholdsImageFilterType; 
  OtsuMultipleThresholdsImageFilterType::Pointer otsuMultipleThresholdsImageFilter = OtsuMultipleThresholdsImageFilterType::New(); 
  otsuMultipleThresholdsImageFilter->SetInput(gradientToMagnitudeImageFilter->GetOutput()); 
  otsuMultipleThresholdsImageFilter->SetNumberOfHistogramBins(128); 
  otsuMultipleThresholdsImageFilter->SetNumberOfThresholds(2); 
  
  typedef itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdImageFilterType; 
  BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter = BinaryThresholdImageFilterType::New(); 
  binaryThresholdImageFilter->SetInput(otsuMultipleThresholdsImageFilter->GetOutput()); 
  binaryThresholdImageFilter->SetLowerThreshold(1); 
  binaryThresholdImageFilter->SetUpperThreshold(2); 
  binaryThresholdImageFilter->SetInsideValue(1); 
  binaryThresholdImageFilter->SetOutsideValue(0); 
  
  imageWriter->SetFileName(outputFilenameOtsu); 
  imageWriter->SetInput(binaryThresholdImageFilter->GetOutput()); 
  imageWriter->Update(); 
  
  typedef itk::OtsuMultipleThresholdsImageFilter<InputImageType, InputImageType> OtsuMultipleThresholdsImageFilterType; 
  OtsuMultipleThresholdsImageFilterType::Pointer otsuMultipleThresholdsImageFilter2 = OtsuMultipleThresholdsImageFilterType::New(); 
  otsuMultipleThresholdsImageFilter2->SetInput(imageReader->GetOutput()); 
  otsuMultipleThresholdsImageFilter2->SetNumberOfHistogramBins(128); 
  otsuMultipleThresholdsImageFilter2->SetNumberOfThresholds(3); 
  
  BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter2 = BinaryThresholdImageFilterType::New(); 
  binaryThresholdImageFilter2->SetInput(otsuMultipleThresholdsImageFilter2->GetOutput()); 
  binaryThresholdImageFilter2->SetLowerThreshold(2); 
  binaryThresholdImageFilter2->SetUpperThreshold(3); 
  binaryThresholdImageFilter2->SetInsideValue(1); 
  binaryThresholdImageFilter2->SetOutsideValue(0); 
  
  imageWriter->SetFileName(outputFilenameOtsu2); 
  imageWriter->SetInput(binaryThresholdImageFilter2->GetOutput()); 
  imageWriter->Update(); 
  
#else
  InputImageIteratorType maskIterator(maskReader->GetOutput(), maskReader->GetOutput()->GetLargestPossibleRegion()); 
  InputImageIteratorType imageIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion()); 
  
  double mean = 0.; 
  int count = 0; 
  int lowerLeft[Dimension]; 
  int upperRight[Dimension]; 
  
  for (unsigned int i = 0; i < Dimension; i++)
  {
    lowerLeft[i] = std::numeric_limits<int>::max(); 
    upperRight[i] = 0; 
  }
  
  for (maskIterator.GoToBegin(), imageIterator.GoToBegin(); 
       !imageIterator.IsAtEnd(); 
       ++maskIterator, ++imageIterator)
  {
    if (maskIterator.Get() > 0)
    {
      mean += imageIterator.Get(); 
      count++; 
      InputImageType::IndexType index = imageIterator.GetIndex(); 
      for (unsigned int i = 0; i < Dimension; i++)
      {
        if (index[i] < lowerLeft[i])
          lowerLeft[i] = index[i]; 
        if (index[i] > upperRight[i])
          upperRight[i] = index[i]; 
      }
    }
  }
  mean /= (double)count; 
  std::cout << "mean=" << mean << std::endl; 
  
  std::cout << "lowerLeft=" << lowerLeft[0] << "," << lowerLeft[1] << "," << lowerLeft[2] << std::endl; 
  std::cout << "upperRight=" << upperRight[0] << "," << upperRight[1] << "," << upperRight[2] << std::endl; 
  
  for (imageIterator.GoToBegin(); 
       !imageIterator.IsAtEnd(); 
       ++imageIterator)
  {
    InputImageType::IndexType index = imageIterator.GetIndex(); 
    if (index[1] > upperRight[1]+8) 
    {
      imageIterator.Set(0);   
    }
  }
  
  typedef itk::SimpleKMeansClusteringImageFilter< InputImageType, InputImageType, InputImageType > SimpleKMeansClusteringImageFilterType;
  SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
  const unsigned int numberOfInitialClasses = 3;
  SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(numberOfInitialClasses);
  SimpleKMeansClusteringImageFilterType::ParametersType finalMeans(numberOfInitialClasses);
  SimpleKMeansClusteringImageFilterType::ParametersType finalStds(numberOfInitialClasses);
  
  initialMeans[0] = mean*0.3; 
  initialMeans[1] = mean*0.7; 
  initialMeans[2] = mean*1.2; 
  simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
  simpleKMeansClusteringImageFilter->SetInput(imageReader->GetOutput());
  simpleKMeansClusteringImageFilter->SetInputMask(maskReader->GetOutput());
  simpleKMeansClusteringImageFilter->Update();
  finalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
  finalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
  
  for ( unsigned int i = 0 ; i < numberOfInitialClasses ; ++i )
  {
    std::cout << finalMeans[i] << " " << finalStds[i] << " "; 
  }
  std::cout << std::endl;
  
  BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter = BinaryThresholdImageFilterType::New(); 
  binaryThresholdImageFilter->SetInput(imageReader->GetOutput()); 
  binaryThresholdImageFilter->SetLowerThreshold(0); 
  binaryThresholdImageFilter->SetUpperThreshold(finalMeans[1]); 
  binaryThresholdImageFilter->SetInsideValue(0); 
  binaryThresholdImageFilter->SetOutsideValue(1); 
  
  imageWriter->SetFileName(outputFilename); 
  imageWriter->SetInput(binaryThresholdImageFilter->GetOutput()); 
  imageWriter->Update(); 
  
  
#endif
  
  
      
  return 0; 
}








