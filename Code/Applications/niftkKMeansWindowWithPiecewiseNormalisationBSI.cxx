/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-06-04 15:11:19 +0100 (Fri, 04 Jun 2010) $
 Revision          : $Revision: 3349 $
 Last modified by  : $Author: ma $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#pragma warning ( disable : 4996 )
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
#include "itkCastImageFilter.h"

/**
 * Calculate BSI using the linear regression results. 
 */
double calculateBSI(char* baselineImageName, char* repeatImageName, 
                    char* baselineMaskName, char* repeatMaskName, char* subROIMaskName,
                    int numberOfErosion, int numberOfDilation, 
                    double userLowerWindow, double userUpperWindow, 
                    const itk::Array<double>& baselineFinalMeans, const itk::Array<double>& repeatFinalMeans, 
                    const itk::Array<double>& baselineFinalStds, const itk::Array<double>& repeatFinalStds, 
                    double slope, double intercept)
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
  
  double baselineNormalisationFactorRepeatBaseline = baselineFinalMeans[2];
  double repeatNormalisationFactorRepeatBaseline = baselineNormalisationFactorRepeatBaseline;
  
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
 * Calculate BSI using the linear regression results. 
 */
double calculateBSIPiecewise(char* baselineImageName, char* repeatImageName, 
                    char* baselineMaskName, char* repeatMaskName, char* subROIMaskName,
                    int numberOfErosion, int numberOfDilation, 
                    double userLowerWindow, double userUpperWindow, 
                    const itk::Array<double>& baselineFinalMeans, const itk::Array<double>& repeatFinalMeans, 
                    const itk::Array<double>& baselineFinalStds, const itk::Array<double>& repeatFinalStds, 
                    double slopeCsfGm, double interceptCsfGm,
                    double slopeGmWm, double interceptGmWm)
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
  
  double baselineNormalisationFactorRepeatBaseline = baselineFinalMeans[2];
  double repeatNormalisationFactorRepeatBaseline = baselineNormalisationFactorRepeatBaseline;
  
  baselineImageIterator.GoToBegin();
  repeatImageIterator.GoToBegin();
  
  for (; !baselineImageIterator.IsAtEnd(); ++baselineImageIterator, ++repeatImageIterator)
  {
    double baselineValue = baselineImageIterator.Get();
    double repeatValue = repeatImageIterator.Get();
    
    baselineValue = baselineValue/baselineNormalisationFactorRepeatBaseline; 
    baselineImageIterator.Set(baselineValue);
    
    if (repeatValue > repeatFinalMeans[1])
      repeatValue = (slopeGmWm*repeatValue+interceptGmWm)/repeatNormalisationFactorRepeatBaseline; 
    else
      repeatValue = (slopeCsfGm*repeatValue+interceptCsfGm)/repeatNormalisationFactorRepeatBaseline; 
    repeatImageIterator.Set(repeatValue);
  }
  
  double baselineLowerWindow = (baselineFinalMeans[0]+baselineFinalStds[0])/baselineNormalisationFactorRepeatBaseline;
  double repeatLowerWindow = (slopeCsfGm*(repeatFinalMeans[0]+repeatFinalStds[0])+interceptCsfGm)/repeatNormalisationFactorRepeatBaseline;
  double kmeansLowerWindow = (baselineLowerWindow+repeatLowerWindow)/2.0;
  
  double baselineUpperWindow = (baselineFinalMeans[1]-baselineFinalStds[1])/baselineNormalisationFactorRepeatBaseline; 
  double repeatUpperWindow = (slopeGmWm*(repeatFinalMeans[1]-repeatFinalStds[1])+interceptGmWm)/repeatNormalisationFactorRepeatBaseline; 
  double kmeansUpperWindow = (baselineUpperWindow+repeatUpperWindow)/2.0;
  
  double lowerWindow = userLowerWindow;
  double upperWindow = userUpperWindow;
  
  if (lowerWindow < 0.0)
    lowerWindow = kmeansLowerWindow;
  if (upperWindow < 0.0)
    upperWindow = kmeansUpperWindow;
  
  std::cout << "lowerWindow," << baselineLowerWindow << "," << repeatLowerWindow << "," << lowerWindow << ",";
  std::cout << "upperWindow," << baselineUpperWindow << "," << repeatUpperWindow << "," << upperWindow << ",";
  
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
        
    repeatImageIterator.Set(static_cast<short>(itk::Math::Round(fabs(slope*repeatValue+intercept))));
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
    niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
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

    // Calculate mean brain intensity. 
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();

    normalisationCalculator->SetInputImage1(repeatNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputImage2(baselineNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputMask1(repeatNormalisationMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(baselineNormalisationMaskReader->GetOutput());
    normalisationCalculator->Compute();
    std::cout << "mean intensities," << normalisationCalculator->GetNormalisationMean2() << "," 
                                     << normalisationCalculator->GetNormalisationMean1() << ",";
                                     
    // Calculate the intensity window.                                      
    SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
    SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalStds(3);
    BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
    MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
    
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
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
    imageWriter->SetFileName (argv[14]);
    imageWriter->Update();
    
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
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
    imageWriter->SetFileName (argv[15]);
    imageWriter->Update();
    
    std::cout << "baseline means," << baselineFinalMeans[0] << "," << baselineFinalMeans[1] << "," << baselineFinalMeans[2] << ",";  
    std::cout << "repeat means," << repeatFinalMeans[0] << "," << repeatFinalMeans[1] << "," << repeatFinalMeans[2] << ",";  
    std::cout << "baseline std," << baselineFinalStds[0] << "," << baselineFinalStds[1] << "," << baselineFinalStds[2] << ",";  
    std::cout << "repeat std," << repeatFinalStds[0] << "," << repeatFinalStds[1] << "," << repeatFinalStds[2] << ",";
    
    std::vector<double> baselineMeans;
    std::vector<double> repeatMeans;
    std::vector<double> baselineCsfGmMeans;
    std::vector<double> repeatCsfGmMeans;
    std::vector<double> baselineGmWmMeans;
    std::vector<double> repeatGmWmMeans;
    double slopeRepeatBaseline = 0.0;
    double interceptRepeatBaseline = 0.0;
    double slopeBaselineRepeat = 0.0;
    double interceptBaselineRepeat = 0.0;
    double slopeRepeatBaselineCsfGm = 0.0;
    double interceptRepeatBaselineCsfGm = 0.0;
    double slopeBaselineRepeatCsfGm = 0.0;
    double interceptBaselineRepeatCsfGm = 0.0;
    double slopeRepeatBaselineGmWm = 0.0;
    double interceptRepeatBaselineGmWm = 0.0;
    double slopeBaselineRepeatGmWm = 0.0;
    double interceptBaselineRepeatGmWm = 0.0;
    
    baselineMeans.push_back(normalisationCalculator->GetNormalisationMean2());
    repeatMeans.push_back(normalisationCalculator->GetNormalisationMean1());
    for (int index = 0; index < 3; index++)
    {
      baselineMeans.push_back(baselineFinalMeans[index]);
      repeatMeans.push_back(repeatFinalMeans[index]);
    }
    baselineCsfGmMeans.push_back(baselineFinalMeans[0]);
    baselineCsfGmMeans.push_back(baselineFinalMeans[1]);
    repeatCsfGmMeans.push_back(repeatFinalMeans[0]);
    repeatCsfGmMeans.push_back(repeatFinalMeans[1]);
    baselineGmWmMeans.push_back(baselineFinalMeans[1]);
    baselineGmWmMeans.push_back(baselineFinalMeans[2]);
    repeatGmWmMeans.push_back(repeatFinalMeans[1]);
    repeatGmWmMeans.push_back(repeatFinalMeans[2]);

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
    
    if (argc > 17 && strlen(argv[17]) > 0)
    {
      subROIMaskName = argv[17];
    }
    
    // Repeat (x) regress on baseline (y).  
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(repeatMeans, baselineMeans, &slopeRepeatBaseline, &interceptRepeatBaseline);
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(repeatCsfGmMeans, baselineCsfGmMeans, &slopeRepeatBaselineCsfGm, &interceptRepeatBaselineCsfGm);
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(repeatGmWmMeans, baselineGmWmMeans, &slopeRepeatBaselineGmWm, &interceptRepeatBaselineGmWm);
    
    double forwardBSI = calculateBSIPiecewise(baselineBSIImageName, repeatBSIImageName, baselineBSIMaskName, repeatBSIMaskName, subROIMaskName, 
                                     numberOfErosion, numberOfDilation, userLowerWindow, userUpperWindow, 
                                     baselineFinalMeans, repeatFinalMeans, 
                                     baselineFinalStds, repeatFinalStds, 
                                     slopeRepeatBaselineCsfGm, interceptRepeatBaselineCsfGm,
                                     slopeRepeatBaselineGmWm, interceptRepeatBaselineGmWm);    
    
    // Baseline (x) regress on repeat (y).  
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(baselineMeans, repeatMeans, &slopeBaselineRepeat, &interceptBaselineRepeat);
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(baselineCsfGmMeans, repeatCsfGmMeans, &slopeBaselineRepeatCsfGm, &interceptBaselineRepeatCsfGm);
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(baselineGmWmMeans, repeatGmWmMeans, &slopeBaselineRepeatGmWm, &interceptBaselineRepeatGmWm);

    double backwardBSI = calculateBSIPiecewise(repeatBSIImageName, baselineBSIImageName, repeatBSIMaskName, baselineBSIMaskName, subROIMaskName, 
                                      numberOfErosion, numberOfDilation, userLowerWindow, userUpperWindow, 
                                      repeatFinalMeans, baselineFinalMeans, 
                                      repeatFinalStds, baselineFinalStds, 
                                      slopeBaselineRepeatCsfGm, interceptBaselineRepeatCsfGm,
                                      slopeBaselineRepeatGmWm, interceptBaselineRepeatGmWm);
    
    std::cout << "slopeRepeatBaseline," << slopeRepeatBaseline << ",interceptRepeatBaseline," << interceptRepeatBaseline << ",";
    std::cout << "slopeBaselineRepeat," << slopeBaselineRepeat << ",interceptBaselineRepeat," << interceptBaselineRepeat << ",";
    std::cout << "BSI," << forwardBSI << "," << -backwardBSI << "," << (forwardBSI-backwardBSI)/2.0 << std::endl;
    
    // Save the normalised repeat image. 
    saveNormalisedImage(repeatBSIImageName, outputNormalisedImageName, slopeRepeatBaseline, interceptRepeatBaseline); 
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


