/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkMultipleDilateImageFilter.h"
#include "itkSimpleFuzzyCMeansClusteringImageFilter.h"
#include "itkIndent.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
  try
  {
    typedef itk::Image<float, 3> DoubleImageType;
    typedef itk::Image<int, 3> IntImageType;
    typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
    typedef itk::ImageFileReader<IntImageType> IntReaderType;
    typedef itk::ImageFileWriter<DoubleImageType>  WriterType;
    typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
    typedef itk::SimpleFuzzyCMeansClusteringImageFilter< DoubleImageType, IntImageType > SimpleFuzzyCMeansClusteringImageFilterType;
    typedef itk::BinariseUsingPaddingImageFilter<IntImageType,IntImageType> BinariseUsingPaddingImageFilterType;
    typedef itk::MultipleDilateImageFilter<IntImageType> MultipleDilateImageFilterType;
    
    DoubleReaderType::Pointer baselineNormalisationImageReader = DoubleReaderType::New();
    DoubleReaderType::Pointer repeatNormalisationImageReader = DoubleReaderType::New();
    IntReaderType::Pointer baselineNormalisationMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatNormalisationMaskReader = IntReaderType::New();
    WriterType::Pointer imageWriter = WriterType::New();
    // Calculate mean brain intensity. 
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();
    
    baselineNormalisationImageReader->SetFileName(argv[1]);
    baselineNormalisationMaskReader->SetFileName(argv[2]);
    repeatNormalisationImageReader->SetFileName(argv[3]);
    repeatNormalisationMaskReader->SetFileName(argv[4]);

    normalisationCalculator->SetInputImage1(repeatNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputImage2(baselineNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputMask1(repeatNormalisationMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(baselineNormalisationMaskReader->GetOutput());
    normalisationCalculator->Compute();
                                     
    // Calculate the intensity window.                                      
    SimpleFuzzyCMeansClusteringImageFilterType::Pointer SimpleFuzzyCMeansClusteringImageFilter = SimpleFuzzyCMeansClusteringImageFilterType::New();
    SimpleFuzzyCMeansClusteringImageFilterType::ParametersType initialMeans(3);
    BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
    MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
    
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
    binariseImageFilter->SetPaddingValue(0);
    binariseImageFilter->SetInput(baselineNormalisationMaskReader->GetOutput());
    binariseImageFilter->Update();
    multipleDilateImageFilter->SetNumberOfDilations(3);
    multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
    multipleDilateImageFilter->Update();
    SimpleFuzzyCMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    SimpleFuzzyCMeansClusteringImageFilter->SetInput(baselineNormalisationImageReader->GetOutput());
    SimpleFuzzyCMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    SimpleFuzzyCMeansClusteringImageFilter->Update();
    
    SimpleFuzzyCMeansClusteringImageFilterType::ParametersType finalMeans(3);
    SimpleFuzzyCMeansClusteringImageFilterType::ParametersType finalStds(3);
    
    finalMeans = SimpleFuzzyCMeansClusteringImageFilter->GetFinalMeans();
    finalStds = SimpleFuzzyCMeansClusteringImageFilter->GetFinalStds();
    
    std::cout << "means=" << finalMeans[0] << "," << finalMeans[1] << "," << finalMeans[2] << std::endl; 
    std::cout << "stds=" << finalStds[0] << "," << finalStds[1] << "," << finalStds[2] << std::endl; 
    
    imageWriter->SetInput(SimpleFuzzyCMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName(argv[5]);
    imageWriter->Update();
    
    std::cout << "Done" << std::endl; 
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS; 
}





