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
#include "itkLogHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkSimpleKMeansClusteringImageFilter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkIndent.h"
#include <stdio.h>

/*!
 * \file niftkKMeansWindowBSI.cxx
 * \page niftkKMeansWindowBSI
 * \section niftkKMeansWindowBSISummary Program to calculate BSI with automatic window".
 * 
 * Program to calculate BSI with automatic window, based on the papers:
 * Freeborough PA and Fox NC, The boundary shift integral: an accurate and robust measure of cerebral volume changes from registered repeat MRI,
 * IEEE Trans Med Imaging. 1997 Oct;16(5):623-9.
 * 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 *
 * \section niftkKMeansWindowNBSICaveat Caveats
 * \li Notice that all the images and masks for intensity normalisation must have the SAME voxel sizes and image dimensions. The same applies to the images and masks for BSI.
 */

int main(int argc, char* argv[])
{
  if (argc < 14)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
    std::cerr << std::endl;
    std::cerr << "K-BSI: Program to calculate the boundary shift integral, based on the paper" << std::endl;
    std::cerr << "  Freeborough PA and Fox NC, The boundary shift integral: an accurate and" << std::endl; 
    std::cerr << "  robust measure of cerebral volume changes from registered repeat MRI," << std::endl; 
    std::cerr << "  IEEE Trans Med Imaging. 1997 Oct;16(5):623-9." << std::endl << std::endl;
    std::cerr << "Added automatic window selection" << std::endl << std::endl;
    std::cerr << "Usage: " << argv[0] << std::endl;
    std::cerr << "         <baseline image for intensity normalisation>" << std::endl; 
    std::cerr << "         <baseline mask for intensity normalisation>" << std::endl; 
    std::cerr << "         <repeat image for intensity normalisation>" << std::endl; 
    std::cerr << "         <repeat mask for intensity normalisation>" << std::endl; 
    std::cerr << "         <baseline image for BSI>" << std::endl;
    std::cerr << "         <baseline mask for BSI>" << std::endl; 
    std::cerr << "         <repeat image for BSI>" << std::endl;
    std::cerr << "         <repeat mask for BSI>" << std::endl;
    std::cerr << "         <number of erosion>" << std::endl;
    std::cerr << "         <number of dilation>" << std::endl;
    std::cerr << "         <number of dilation for the K-means classification> (recommand 3)" << std::endl;
    std::cerr << "         <baseline image classification>" << std::endl;
    std::cerr << "         <repeat image classification>" << std::endl;
    std::cerr << "Notice that all the images and masks for intensity normalisation must " << std::endl;
    std::cerr << "have the SAME voxel sizes and image dimensions. The same applies to the " << std::endl;
    std::cerr << "images and masks for BSI." << std::endl;
    return EXIT_FAILURE;
  }
  
  try
  {
    typedef itk::Image<double, 3> DoubleImageType;
    typedef itk::Image<int, 3> IntImageType;

    typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
    typedef itk::ImageFileReader<IntImageType> IntReaderType;
    typedef itk::ImageFileWriter<IntImageType> WriterType;
    typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
    typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;
    typedef itk::SimpleKMeansClusteringImageFilter< DoubleImageType, IntImageType, IntImageType > SimpleKMeansClusteringImageFilterType;
    typedef itk::MultipleDilateImageFilter<IntImageType> MultipleDilateImageFilterType;
    typedef itk::BinariseUsingPaddingImageFilter<IntImageType,IntImageType> BinariseUsingPaddingImageFilterType;

    DoubleReaderType::Pointer baselineNormalisationImageReader = DoubleReaderType::New();
    DoubleReaderType::Pointer repeatNormalisationImageReader = DoubleReaderType::New();
    IntReaderType::Pointer baselineNormalisationMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatNormalisationMaskReader = IntReaderType::New();
    WriterType::Pointer imageWriter = WriterType::New();


    baselineNormalisationImageReader->SetFileName(argv[1]);
    baselineNormalisationMaskReader->SetFileName(argv[2]);
    repeatNormalisationImageReader->SetFileName(argv[3]);
    repeatNormalisationMaskReader->SetFileName(argv[4]);
    std::cout << argv[1] << "," << argv[3] << ",";

    // Calculate mean brain intensity. 
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();

    normalisationCalculator->SetInputImage1(baselineNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputImage2(repeatNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputMask1(baselineNormalisationMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(repeatNormalisationMaskReader->GetOutput());
    normalisationCalculator->Compute();
    std::cout << "mean intensities," << normalisationCalculator->GetNormalisationMean1() << "," 
                                     << normalisationCalculator->GetNormalisationMean2() << ",";
                                     
    // Calculate the intensity window.                                      
    SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
    SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalStds(3);
    BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
    MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
    
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
    binariseImageFilter->SetPaddingValue(0);
    binariseImageFilter->SetInput(baselineNormalisationMaskReader->GetOutput());
    binariseImageFilter->Update();
    multipleDilateImageFilter->SetNumberOfDilations(atoi(argv[11]));
    multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
    multipleDilateImageFilter->Update();
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(baselineNormalisationImageReader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->Update();
    baselineFinalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    baselineFinalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    imageWriter->SetInput (simpleKMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName (argv[12]);
    imageWriter->Update();
    
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
    binariseImageFilter->SetPaddingValue(0);
    binariseImageFilter->SetInput(repeatNormalisationMaskReader->GetOutput());
    binariseImageFilter->Update();
    multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
    multipleDilateImageFilter->Update();
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(repeatNormalisationImageReader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->Update();
    repeatFinalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    repeatFinalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    imageWriter->SetInput (simpleKMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName (argv[13]);
    imageWriter->Update();
    
    double lowerWindow = ((baselineFinalMeans[0]+baselineFinalStds[0])/normalisationCalculator->GetNormalisationMean1()+ 
                          (repeatFinalMeans[0]+repeatFinalStds[0])/normalisationCalculator->GetNormalisationMean2())/2.0;
    double upperWindow = ((baselineFinalMeans[1]-baselineFinalStds[1])/normalisationCalculator->GetNormalisationMean1()+ 
                          (repeatFinalMeans[1]-repeatFinalStds[1])/normalisationCalculator->GetNormalisationMean2())/2.0;
    
    std::cout << "baseline means," << baselineFinalMeans[0] << "," << baselineFinalMeans[1] << "," << baselineFinalMeans[2] << ",";  
    std::cout << "repeat means," << repeatFinalMeans[0] << "," << repeatFinalMeans[1] << "," << repeatFinalMeans[2] << ",";  
    std::cout << "baseline std," << baselineFinalStds[0] << "," << baselineFinalStds[1] << "," << baselineFinalStds[2] << ",";  
    std::cout << "repeat std," << repeatFinalStds[0] << "," << repeatFinalStds[1] << "," << repeatFinalStds[2] << ",";  
    std::cout << "window," << lowerWindow << "," << upperWindow << ",";                                  

    BoundaryShiftIntegralFilterType::Pointer bsiFilter = BoundaryShiftIntegralFilterType::New();
    WriterType::Pointer writer = WriterType::New();
    DoubleReaderType::Pointer baselineBSIImageReader = DoubleReaderType::New();
    DoubleReaderType::Pointer repeatBSIImageReader = DoubleReaderType::New();
    IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();
    IntReaderType::Pointer subROIMaskReader = IntReaderType::New();

    baselineBSIImageReader->SetFileName(argv[5]);
    baselineBSIMaskReader->SetFileName(argv[6]);
    repeatBSIImageReader->SetFileName(argv[7]);
    repeatBSIMaskReader->SetFileName(argv[8]);
    
    bsiFilter->SetBaselineImage(baselineBSIImageReader->GetOutput());
    bsiFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
    bsiFilter->SetRepeatImage(repeatBSIImageReader->GetOutput());
    bsiFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
    bsiFilter->SetBaselineIntensityNormalisationFactor(normalisationCalculator->GetNormalisationMean1());
    bsiFilter->SetRepeatIntensityNormalisationFactor(normalisationCalculator->GetNormalisationMean2());
    bsiFilter->SetNumberOfErosion(atoi(argv[9]));
    bsiFilter->SetNumberOfDilation(atoi(argv[10]));
    bsiFilter->SetLowerCutoffValue(lowerWindow);
    bsiFilter->SetUpperCutoffValue(upperWindow);
    bsiFilter->Compute();
    std::cout << "BSI," << bsiFilter->GetBoundaryShiftIntegral() << std::endl;
    
    
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


