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
#pragma warning ( disable : 4996 )
#endif
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkSimpleKMeansClusteringImageFilter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkIndent.h"
#include <stdio.h>
#include "itkCastImageFilter.h"
#include "itkSubtractImageFilter.h"

/*!
 * \file niftkKMeansWindowWithLinearRegressionNormalisationBSI.cxx
 * \page niftkKMeansWindowWithLinearRegressionNormalisationBSI
 * \section niftkKMeansWindowWithLinearRegressionNormalisationBSISummary Program to calculate KN-BSI".
 * 
 * Program to calculate the KN-BSI, based on the papers:
 * Freeborough PA and Fox NC, The boundary shift integral: an accurate and robust measure of cerebral volume changes from registered repeat MRI,
 * IEEE Trans Med Imaging. 1997 Oct;16(5):623-9.
 * 
 * Leung et al, Robust atrophy rate measurement in Alzheimer's disease using multi-site serial MRI: Tissue-specific intensity normalization and parameter selection, 
 * NeuroImage. 2010. 50 (2) 516 - 523.
 * 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 *
 * \section niftkKMeansWindowNBSIWithLinearRegressionNormalisationCaveat Caveats
 * \li Notice that all the images and masks for intensity normalisation must have the SAME voxel sizes and image dimensions. The same applies to the images and masks for BSI.
 */


/**
 * Roughly estimate the CSF intensity using the dilated mask. 
 */
void estimateCSFGMWMIntensityFromDilatedMask(char* imageName, char* maskName, double& csfMean, double& csfSd, double& gmMean, double& wmMean)
{
  typedef itk::Image<double, 3> DoubleImageType;
  typedef itk::Image<int, 3> IntImageType;
  typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
  typedef itk::ImageFileReader<IntImageType> IntReaderType;
  
  DoubleReaderType::Pointer imageReader = DoubleReaderType::New();
  IntReaderType::Pointer maskReader = IntReaderType::New();
  
  imageReader->SetFileName(imageName);
  imageReader->Update();
  maskReader->SetFileName(maskName);
  maskReader->Update();
  
  typedef itk::BinariseUsingPaddingImageFilter<IntImageType, IntImageType> BinariseUsingPaddingType; 
  
  BinariseUsingPaddingType::Pointer binariseUsingPadding = BinariseUsingPaddingType::New(); 
  binariseUsingPadding->SetInput(maskReader->GetOutput()); 
  binariseUsingPadding->SetPaddingValue(0); 
  binariseUsingPadding->Update(); 
  
  typedef itk::MultipleDilateImageFilter<IntImageType> MultipleDilateImageFilterType;
  MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
  
  // Dilate multiple times. 
  multipleDilateImageFilter->SetNumberOfDilations(3);
  multipleDilateImageFilter->SetInput(binariseUsingPadding->GetOutput());
  multipleDilateImageFilter->Update();
  
  // Rough estimate for CSF mean by taking the mean values of the (dilated region - brain region).
  typedef itk::SubtractImageFilter<IntImageType, IntImageType> SubtractImageFilterType; 
  SubtractImageFilterType::Pointer subtractImageFilter = SubtractImageFilterType::New(); 
  subtractImageFilter->SetInput1(multipleDilateImageFilter->GetOutput());
  subtractImageFilter->SetInput2(binariseUsingPadding->GetOutput()); 
  subtractImageFilter->Update(); 

  typedef itk::MultipleErodeImageFilter<IntImageType> MultipleErodeImageFilterType;
  MultipleErodeImageFilterType::Pointer multipleErodeImageFilterFilter = MultipleErodeImageFilterType::New();

  // Erode multiple times.
  multipleErodeImageFilterFilter->SetNumberOfErosions(3);
  multipleErodeImageFilterFilter->SetInput(binariseUsingPadding->GetOutput());
  multipleErodeImageFilterFilter->Update();

  // Rough estimate for GM mean by taking the mean values of the (brain region - eroded region).
  SubtractImageFilterType::Pointer gmSubtractImageFilter = SubtractImageFilterType::New();
  gmSubtractImageFilter->SetInput1(binariseUsingPadding->GetOutput());
  gmSubtractImageFilter->SetInput2(multipleErodeImageFilterFilter->GetOutput());
  gmSubtractImageFilter->Update();
  
  itk::ImageRegionIterator<DoubleImageType> imageIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<IntImageType> maskIterator(subtractImageFilter->GetOutput(), subtractImageFilter->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<IntImageType> gmMaskIterator(gmSubtractImageFilter->GetOutput(), gmSubtractImageFilter->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<IntImageType> wmMaskIterator(multipleErodeImageFilterFilter->GetOutput(), multipleErodeImageFilterFilter->GetOutput()->GetLargestPossibleRegion());

  csfMean = 0.0;
  int count = 0; 
  gmMean = 0.0;
  int gmCount = 0;
  wmMean = 0.0;
  int wmCount = 0;

  for (imageIterator.GoToBegin(), maskIterator.GoToBegin(), gmMaskIterator.GoToBegin(), wmMaskIterator.GoToBegin();
       !imageIterator.IsAtEnd(); 
       ++imageIterator, ++maskIterator, ++gmMaskIterator, ++wmMaskIterator)
  {
    if (maskIterator.Get() > 0)
    {
      csfMean += imageIterator.Get();
      count++; 
    }
    if (gmMaskIterator.Get() > 0)
    {
      gmMean += imageIterator.Get();
      gmCount++;
    }
    if (wmMaskIterator.Get() > 0)
    {
      wmMean += imageIterator.Get();
      wmCount++;
    }
  }
  csfMean /= count;
  gmMean /= gmCount;
  wmMean /= wmCount;
  csfSd = 0.0;
  for (imageIterator.GoToBegin(), maskIterator.GoToBegin();
       !imageIterator.IsAtEnd(); 
       ++imageIterator, ++maskIterator)
  {
    if (maskIterator.Get() > 0)
    {
      double diff = imageIterator.Get()-csfMean;
      csfSd += diff*diff;
    }
    
  }
  csfSd = sqrt(csfSd/count);

  // std::cerr << "csfMean=" << csfMean << ", gmMean=" << gmMean << ", wmMean=" << wmMean << std::endl;

} 


/**
 * Calculate BSI using the linear regression results. 
 */
double calculateBSI(char* baselineImageName, char* repeatImageName, 
                    char* baselineMaskName, char* repeatMaskName, char* subROIMaskName,
                    int numberOfErosion, int numberOfDilation, 
                    double userLowerWindow, double userUpperWindow, 
                    const itk::Array<double>& baselineFinalMeans, const itk::Array<double>& repeatFinalMeans, 
                    const itk::Array<double>& baselineFinalStds, const itk::Array<double>& repeatFinalStds, 
                    double slope, double intercept, int numberOfClasses, double numberOfWmSdForIntensityExclusion)
{
  typedef itk::Image<double, 3> DoubleImageType;
  typedef itk::Image<int, 3> IntImageType;
  typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
  typedef itk::ImageFileReader<IntImageType> IntReaderType;
  typedef itk::ImageFileWriter<IntImageType> WriterType;
  typedef itk::ImageFileWriter<DoubleImageType> DoublerWriterType;
  typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;
  
  DoubleReaderType::Pointer baselineBSIImageReader = DoubleReaderType::New();
  DoubleReaderType::Pointer repeatBSIImageReader = DoubleReaderType::New();
  IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
  IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();
  IntReaderType::Pointer subroiMaskReader = IntReaderType::New();
  // Normalise the intensity using the linear regression of CSF, GM and WM intensities, mapping to the baseline scan. 
  baselineBSIImageReader->SetFileName(baselineImageName);
  baselineBSIImageReader->Update();
  repeatBSIImageReader->SetFileName(repeatImageName);
  repeatBSIImageReader->Update();
  
  itk::ImageRegionIterator<DoubleImageType> baselineImageIterator(baselineBSIImageReader->GetOutput(), baselineBSIImageReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<DoubleImageType> repeatImageIterator(repeatBSIImageReader->GetOutput(), repeatBSIImageReader->GetOutput()->GetLargestPossibleRegion());

  if (numberOfWmSdForIntensityExclusion >= 0)
  {
    double baselineMaxIntensity = baselineFinalMeans[1]+numberOfWmSdForIntensityExclusion*baselineFinalStds[1];
    double repeatMaxIntensity = repeatFinalMeans[1]+numberOfWmSdForIntensityExclusion*repeatFinalStds[1];
    if (numberOfClasses >= 3)
    {
      baselineMaxIntensity = baselineFinalMeans[2]+numberOfWmSdForIntensityExclusion*baselineFinalStds[2];
      repeatMaxIntensity = repeatFinalMeans[2]+numberOfWmSdForIntensityExclusion*repeatFinalStds[2];
    }
    for (baselineImageIterator.GoToBegin(); !baselineImageIterator.IsAtEnd(); ++baselineImageIterator)
    {
      if (baselineImageIterator.Get() > baselineMaxIntensity)
        baselineImageIterator.Set(0);
    }
    for (repeatImageIterator.GoToBegin(); !repeatImageIterator.IsAtEnd(); ++repeatImageIterator)
    {
      if (repeatImageIterator.Get() > repeatMaxIntensity)
        repeatImageIterator.Set(0);
    }
  }
  
  double baselineNormalisationFactorRepeatBaseline = baselineFinalMeans[1];
  double repeatNormalisationFactorRepeatBaseline = baselineNormalisationFactorRepeatBaseline;
  
  if (numberOfClasses >= 3)
  {
    baselineNormalisationFactorRepeatBaseline = baselineFinalMeans[2];
    repeatNormalisationFactorRepeatBaseline = baselineNormalisationFactorRepeatBaseline;
  }
  
  baselineImageIterator.GoToBegin();
  repeatImageIterator.GoToBegin();
  
  for (; !baselineImageIterator.IsAtEnd(); ++baselineImageIterator, ++repeatImageIterator)
  {
    double baselineValue = baselineImageIterator.Get();
    double repeatValue = repeatImageIterator.Get();
    
    baselineValue = baselineValue/baselineNormalisationFactorRepeatBaseline; 
    baselineImageIterator.Set(baselineValue);
    
    repeatValue = (slope*repeatValue+intercept)/repeatNormalisationFactorRepeatBaseline; 
    repeatImageIterator.Set(repeatValue);
  }
  
  double baselineLowerWindow = (baselineFinalMeans[0]+baselineFinalStds[0])/baselineNormalisationFactorRepeatBaseline;
  double repeatLowerWindow = (slope*(repeatFinalMeans[0]+repeatFinalStds[0])+intercept)/repeatNormalisationFactorRepeatBaseline;
  double kmeansLowerWindow = (baselineLowerWindow+repeatLowerWindow)/2.0;
  
  double baselineUpperWindow = (baselineFinalMeans[1]-baselineFinalStds[1])/baselineNormalisationFactorRepeatBaseline; 
  double repeatUpperWindow = (slope*(repeatFinalMeans[1]-repeatFinalStds[1])+intercept)/repeatNormalisationFactorRepeatBaseline; 
  double kmeansUpperWindow = (baselineUpperWindow+repeatUpperWindow)/2.0;
  
  double lowerWindow = userLowerWindow;
  double upperWindow = userUpperWindow;
  
  if (lowerWindow < 0.0)
    lowerWindow = kmeansLowerWindow;
  if (upperWindow < 0.0)
    upperWindow = kmeansUpperWindow;
  
  std::cout << "lowerWindow," << baselineLowerWindow << "," << repeatLowerWindow << "," << lowerWindow << ",";
  std::cout << "upperWindow," << baselineUpperWindow << "," << repeatUpperWindow << "," << upperWindow << ",";

  double windowWidth = upperWindow-lowerWindow;
  if (windowWidth < 0.0)
  {
    std::cerr << "Error: window width is less than 0:" << windowWidth << std::endl;
    exit(1);
  }
  
  BoundaryShiftIntegralFilterType::Pointer bsiFilter = BoundaryShiftIntegralFilterType::New();
  IntReaderType::Pointer subROIMaskReader = IntReaderType::New();
  
  baselineBSIMaskReader->SetFileName(baselineMaskName);
  repeatBSIMaskReader->SetFileName(repeatMaskName);
  if (subROIMaskName != NULL)
  {
    std::cout << "subROI," << subROIMaskName << ",";
    subroiMaskReader->SetFileName(subROIMaskName);
    bsiFilter->SetSubROIMask(subroiMaskReader->GetOutput());
  }
  
  bsiFilter->SetBaselineImage(baselineBSIImageReader->GetOutput());
  bsiFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
  bsiFilter->SetRepeatImage(repeatBSIImageReader->GetOutput());
  bsiFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
  bsiFilter->SetBaselineIntensityNormalisationFactor(1.0);
  bsiFilter->SetRepeatIntensityNormalisationFactor(1.0);
  bsiFilter->SetNumberOfErosion(numberOfErosion);
  bsiFilter->SetNumberOfDilation(numberOfDilation);
  bsiFilter->SetLowerCutoffValue(lowerWindow);
  bsiFilter->SetUpperCutoffValue(upperWindow);
  bsiFilter->Compute();
  
  return bsiFilter->GetBoundaryShiftIntegral();
}

/**
 * Save normalised repeat image. 
 */
void saveNormalisedImage(char* repeatImageName, char* outputImageName, double slope, double intercept)
{
  typedef itk::Image<short, 3> ShortImageType;
  typedef itk::ImageFileReader<ShortImageType> ReaderType;
  typedef itk::ImageFileWriter<ShortImageType> WriterType;
  
  ReaderType::Pointer repeatBSIImageReader = ReaderType::New();
  repeatBSIImageReader->SetFileName(repeatImageName);
  repeatBSIImageReader->Update();
  
  itk::ImageRegionIterator<ShortImageType> repeatImageIterator(repeatBSIImageReader->GetOutput(), repeatBSIImageReader->GetOutput()->GetLargestPossibleRegion());
  
  for (repeatImageIterator.GoToBegin(); !repeatImageIterator.IsAtEnd(); ++repeatImageIterator)
  {
    short repeatValue = repeatImageIterator.Get(); 
        
    repeatImageIterator.Set(static_cast<short>(round(fabs(slope*repeatValue+intercept))));
  }
  
  WriterType::Pointer writer = WriterType::New(); 
  
  writer->SetInput(repeatBSIImageReader->GetOutput()); 
  writer->SetFileName(outputImageName); 
  writer->Update(); 
}


int main(int argc, char* argv[])
{
  if (argc < 10)
  {
    std::cerr << std::endl;
    std::cerr << "Program to calculate the boundary shift integral, based on the paper" << std::endl; 
    std::cerr << "  Freeborough PA and Fox NC, The boundary shift integral: an accurate and" << std::endl; 
    std::cerr << "  robust measure of cerebral volume changes from registered repeat MRI," << std::endl; 
    std::cerr << "  IEEE Trans Med Imaging. 1997 Oct;16(5):623-9." << std::endl << std::endl;
    std::cerr << "Added intensity normalisation and automatic window selection" << std::endl;
    std::cerr << "  Leung KK, Clarkson MJ, Bartlett JW, Clegg S, Jack CR Jr, Weiner MW, " << std::endl;
    std::cerr << "  Fox NC, Ourselin S; the Alzheimer's Disease Neuroimaging Initiative. " << std::endl; 
    std::cerr << "  Robust atrophy rate measurement in Alzheimer's disease using multi-site serial MRI: " <<std::endl;
    std::cerr << "  Tissue-specific intensity normalization and parameter selection. Neuroimage. 2009 Dec 23. " << std::endl << std::endl; 
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
    std::cerr << "         <number of dilation for the K-means classification>" << std::endl;
    std::cerr << "         <lower window: -1=automatic>" << std::endl;
    std::cerr << "         <upper window: -1=automatic>" << std::endl;
    std::cerr << "         <baseline image classification>" << std::endl;
    std::cerr << "         <repeat image classification>" << std::endl;
    std::cerr << "         <output normalised repeat image>" << std::endl;
    std::cerr << "         <subroi:optional sub region of interest>" << std::endl;
    std::cerr << "         <number of classes for k-means clustering>" << std::endl;
    std::cerr << "         <number of SD of WM intensity for the exclusion of high intensity voxel>" << std::endl;
    std::cerr << "         <1 for using kmeans auto init from image>" << std::endl;
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
    typedef itk::ImageFileWriter<DoubleImageType> DoublerWriterType;
    typedef itk::CastImageFilter<DoubleImageType, IntImageType> DoubleToIntImageFilterType;
    typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
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
    
    int numberOfClasses = 3; 
    if (argc > 18 && strlen(argv[18]) > 0)
    {
      numberOfClasses = atoi(argv[18]);
    }
    double numberOfWmSdForIntensityExclusion = -1.0;
    if (argc > 19 && strlen(argv[19]) > 0)
    {
      numberOfWmSdForIntensityExclusion = atof(argv[19]);
    }
    bool isUseKMeanAutoInit = false;
    if (argc > 20 && strlen(argv[20]) > 0)
    {
      if (atoi(argv[20]) == 1)
      {
        isUseKMeanAutoInit = true;
        // std::cerr << "isUseKMeanAutoInit = true" << std::endl;
      }
    }

    // Calculate mean brain intensity. 
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();

    normalisationCalculator->SetInputImage1(repeatNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputImage2(baselineNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputMask1(repeatNormalisationMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(baselineNormalisationMaskReader->GetOutput());
    normalisationCalculator->Compute();
    std::cout << "mean intensities," << normalisationCalculator->GetNormalisationMean2() << "," 
                                     << normalisationCalculator->GetNormalisationMean1() << ",";
    
    double baselineCsfMean; 
    double baselineCsfSd; 
    double baselineGmMean;
    double baselineWmMean;
    estimateCSFGMWMIntensityFromDilatedMask(argv[1], argv[2], baselineCsfMean, baselineCsfSd, baselineGmMean, baselineWmMean);
    std::cout << "baseline csf," << baselineCsfMean << "," << baselineCsfSd << ","; 
    double repeatCsfMean; 
    double repeatCsfSd; 
    double repeatGmMean;
    double repeatWmMean;
    estimateCSFGMWMIntensityFromDilatedMask(argv[3], argv[4], repeatCsfMean, repeatCsfSd, repeatGmMean, repeatWmMean);
    std::cout << "repeat csf," << repeatCsfMean << "," << repeatCsfSd << ","; 
                                     
    // Calculate the intensity window.                                      
    SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
    SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalMeans(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalStds(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalMeans(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalStds(numberOfClasses);
    BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
    MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
    
    if (isUseKMeanAutoInit == false)
    {
      initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
      initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
      if (numberOfClasses >= 3)
        initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
    }
    else
    {
      initialMeans[0] = baselineCsfMean;
      initialMeans[1] = baselineGmMean;
      if (numberOfClasses >= 3)
        initialMeans[2] = baselineWmMean;
    }
    binariseImageFilter->SetPaddingValue(0);
    binariseImageFilter->SetInput(baselineNormalisationMaskReader->GetOutput());
    binariseImageFilter->Update();
    multipleDilateImageFilter->SetNumberOfDilations(atoi(argv[11]));
    multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
    multipleDilateImageFilter->Update();
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(baselineNormalisationImageReader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->SetNumberOfClasses(numberOfClasses); 
    simpleKMeansClusteringImageFilter->Update();
    baselineFinalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    baselineFinalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    imageWriter->SetInput (simpleKMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName (argv[14]);
    imageWriter->Update();
    
    if (isUseKMeanAutoInit == false)
    {
      initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
      initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
      if (numberOfClasses >= 3)
        initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
    }
    else
    {
      initialMeans[0] = repeatCsfMean;
      initialMeans[1] = repeatGmMean;
      if (numberOfClasses >= 3)
        initialMeans[2] = repeatWmMean;
    }
    binariseImageFilter->SetPaddingValue(0);
    binariseImageFilter->SetInput(repeatNormalisationMaskReader->GetOutput());
    binariseImageFilter->Update();
    multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
    multipleDilateImageFilter->Update();
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(repeatNormalisationImageReader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->SetNumberOfClasses(numberOfClasses); 
    simpleKMeansClusteringImageFilter->Update();
    repeatFinalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    repeatFinalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    imageWriter->SetInput (simpleKMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName (argv[15]);
    imageWriter->Update();
    
    std::cout << "baseline means,";
    for (int i = 0; i < numberOfClasses; i++)
      std::cout << baselineFinalMeans[i] << ",";  
    std::cout << "repeat means,";
    for (int i = 0; i < numberOfClasses; i++)
      std::cout << repeatFinalMeans[i] << ",";  
    std::cout << "baseline std,"; 
    for (int i = 0; i < numberOfClasses; i++)
      std::cout << baselineFinalStds[i] << ",";  
    std::cout << "repeat std,"; 
    for (int i = 0; i < numberOfClasses; i++)
      std::cout << repeatFinalStds[i] << ",";

    std::vector<double> baselineMeans;
    std::vector<double> repeatMeans;
    double slopeRepeatBaseline = 0.0;
    double interceptRepeatBaseline = 0.0;
    double slopeBaselineRepeat = 0.0;
    double interceptBaselineRepeat = 0.0;
    
    baselineMeans.push_back(normalisationCalculator->GetNormalisationMean2());
    repeatMeans.push_back(normalisationCalculator->GetNormalisationMean1());
    for (int index = 0; index < numberOfClasses; index++)
    {
      baselineMeans.push_back(baselineFinalMeans[index]);
      repeatMeans.push_back(repeatFinalMeans[index]);
    }

    char* baselineBSIImageName = argv[5];
    char* baselineBSIMaskName = argv[6];
    char* repeatBSIImageName = argv[7];
    char* repeatBSIMaskName = argv[8];
    int numberOfErosion = atoi(argv[9]);
    int numberOfDilation = atoi(argv[10]);
    double userLowerWindow = atof(argv[12]);
    double userUpperWindow = atof(argv[13]);
    char* outputNormalisedImageName = argv[16]; 
    char* subROIMaskName = NULL;
    
    if (argc > 17 && strlen(argv[17]) > 0 && strcmp(argv[17], "dummy") != 0)
    {
      subROIMaskName = argv[17];
    }
    
    // Repeat (x) regress on baseline (y).  
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(repeatMeans, baselineMeans, &slopeRepeatBaseline, &interceptRepeatBaseline);
    
    double forwardBSI = calculateBSI(baselineBSIImageName, repeatBSIImageName, baselineBSIMaskName, repeatBSIMaskName, subROIMaskName, 
                                     numberOfErosion, numberOfDilation, userLowerWindow, userUpperWindow, 
                                     baselineFinalMeans, repeatFinalMeans, 
                                     baselineFinalStds, repeatFinalStds, 
                                     slopeRepeatBaseline, interceptRepeatBaseline, numberOfClasses, numberOfWmSdForIntensityExclusion);
    
    // Baseline (x) regress on repeat (y).  
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(baselineMeans, repeatMeans, &slopeBaselineRepeat, &interceptBaselineRepeat);

    double backwardBSI = calculateBSI(repeatBSIImageName, baselineBSIImageName, repeatBSIMaskName, baselineBSIMaskName, subROIMaskName, 
                                      numberOfErosion, numberOfDilation, userLowerWindow, userUpperWindow, 
                                      repeatFinalMeans, baselineFinalMeans, 
                                      repeatFinalStds, baselineFinalStds, 
                                      slopeBaselineRepeat, interceptBaselineRepeat, numberOfClasses, numberOfWmSdForIntensityExclusion);
    
    std::cout << "slopeRepeatBaseline," << slopeRepeatBaseline << ",interceptRepeatBaseline," << interceptRepeatBaseline << ",";
    std::cout << "slopeBaselineRepeat," << slopeBaselineRepeat << ",interceptBaselineRepeat," << interceptBaselineRepeat << ",";
    std::cout << "BSI," << forwardBSI << "," << -backwardBSI << "," << (forwardBSI-backwardBSI)/2.0 << ",";
    
    std::cout << "baseline csf," << baselineCsfMean << "," << baselineCsfSd << ","; 
    std::cout << "repeat csf," << repeatCsfMean << "," << repeatCsfSd; 
    
    // Save the normalised repeat image. 
    saveNormalisedImage(repeatBSIImageName, outputNormalisedImageName, slopeRepeatBaseline, interceptRepeatBaseline); 
    
    if (fabs(baselineCsfMean-baselineFinalMeans[0]) > baselineFinalStds[0])
    {
      SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(2);
      SimpleKMeansClusteringImageFilterType::ParametersType finalMeans(2);
      SimpleKMeansClusteringImageFilterType::ParametersType finalStds(2);
      
      initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
      initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
      binariseImageFilter->SetPaddingValue(0);
      binariseImageFilter->SetInput(baselineNormalisationMaskReader->GetOutput());
      binariseImageFilter->Update();
      multipleDilateImageFilter->SetNumberOfDilations(atoi(argv[11]));
      multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
      multipleDilateImageFilter->Update();
      simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
      simpleKMeansClusteringImageFilter->SetInput(baselineNormalisationImageReader->GetOutput());
      simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
      simpleKMeansClusteringImageFilter->SetNumberOfClasses(2); 
      simpleKMeansClusteringImageFilter->Update();
      finalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
      finalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
      std::cout << "baseline 2-class k-means csf," << finalMeans[0] << "," << finalStds[0]; 
      std::cout << ",inconsistency between baseline CSF intensity estimated from " << numberOfClasses << "-class k-means and from dilated and undilated brain regions"; 
      if (numberOfClasses == 3 && fabs(baselineCsfMean-baselineFinalMeans[0]) > fabs(baselineCsfMean-finalMeans[0]))
        std::cout << ",CSF intensity from 2-class k-means is closer to the CSF intensity from dilated and undilated brain regions than 3-class k-means"; 
    }
    
    if (fabs(repeatCsfMean-repeatFinalMeans[0]) > repeatFinalStds[0])
    {
      SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(2);
      SimpleKMeansClusteringImageFilterType::ParametersType finalMeans(2);
      SimpleKMeansClusteringImageFilterType::ParametersType finalStds(2);
      
      initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
      initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
      binariseImageFilter->SetPaddingValue(0);
      binariseImageFilter->SetInput(repeatNormalisationMaskReader->GetOutput());
      binariseImageFilter->Update();
      multipleDilateImageFilter->SetNumberOfDilations(atoi(argv[11]));
      multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
      multipleDilateImageFilter->Update();
      simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
      simpleKMeansClusteringImageFilter->SetInput(repeatNormalisationImageReader->GetOutput());
      simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
      simpleKMeansClusteringImageFilter->SetNumberOfClasses(2); 
      simpleKMeansClusteringImageFilter->Update();
      finalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
      finalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
      std::cout << ",repeat 2-class k-means csf," << finalMeans[0] << "," << finalStds[0]; 
      std::cout << ",inconsistency between repeat CSF intensity estimated from " << numberOfClasses << "-class k-means and from dilated and undilated brain regions"; 
      if (numberOfClasses == 3 && fabs(repeatCsfMean-repeatFinalMeans[0]) > fabs(repeatCsfMean-finalMeans[0]))
        std::cout << ",CSF intensity from 2-class k-means is closer to the CSF intensity from dilated and undilated brain regions than 3-class k-means"; 
    }
    
    std::cout <<  std::endl; 
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


