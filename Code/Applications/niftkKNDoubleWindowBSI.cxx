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
#ifndef OPEN_SOURCE_BSI
#include "itkLogHelper.h"
#endif
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkDoubleWindowBoundaryShiftIntegralCalculator.h"
#include "itkSimpleKMeansClusteringImageFilter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkIndent.h"
#include <stdio.h>

/*!
 * \file niftkKNDoubleWindowBSI.cxx
 * \page niftkKNDoubleWindowBSI
 * \section niftkKNDoubleWindowBSISummary Program to calculate the double window boundary shift integral (with linear regression normalisation).
 * 
 * This program calculates the double window boundary shift integral (with linear regression normalisation). 
 * The double window BSI aims to capture the boundary change between CSF and GM as well as the boundary change 
 * between GM and WM. This is mainly used to calculate the caudate BSI because one bounday of 
 * caudate is CSF/GM and the other side is GM/WM.
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 *
 * \section niftkKNDoubleWindowBSICaveat Caveats
 * \li File sizes not checked.
 * \li Image headers not checked. By "voxel by voxel basis" we mean that the image geometry, origin, orientation is not checked.
 */


/**
 * Typedefs. 
 */
typedef itk::Image<double, 3> DoubleImageType;
typedef itk::Image<float, 3> FloatImageType;
typedef itk::Image<int, 3> IntImageType;

typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
typedef itk::ImageFileReader<IntImageType> IntReaderType;
typedef itk::ImageFileWriter<IntImageType> WriterType;
typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;
typedef itk::DoubleWindowBoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,FloatImageType> DoubleWindowBoundaryShiftIntegralFilterType;
typedef itk::SimpleKMeansClusteringImageFilter< DoubleImageType, IntImageType, IntImageType > SimpleKMeansClusteringImageFilterType;
typedef itk::MultipleDilateImageFilter<IntImageType> MultipleDilateImageFilterType;
typedef itk::BinariseUsingPaddingImageFilter<IntImageType,IntImageType> BinariseUsingPaddingImageFilterType;

/**
 * Use K-means clustering to get the means of the tissues. 
 */
void KMeansClassification(SimpleKMeansClusteringImageFilterType::ParametersType& means, 
  SimpleKMeansClusteringImageFilterType::ParametersType& stds, 
  const DoubleImageType* image, 
  const IntImageType* mask, 
  int numberOfDilations,
  int numberOfClasses, 
  const char* outputImageName)  
{
  SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
  BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
  MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
  IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();
  WriterType::Pointer imageWriter = WriterType::New();
    
  binariseImageFilter->SetPaddingValue(0);
  binariseImageFilter->SetInput(mask);
  binariseImageFilter->Update();
  multipleDilateImageFilter->SetNumberOfDilations(numberOfDilations);
  multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
  multipleDilateImageFilter->Update();
  simpleKMeansClusteringImageFilter->SetInitialMeans(means);
  simpleKMeansClusteringImageFilter->SetInput(image);
  simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
  simpleKMeansClusteringImageFilter->SetNumberOfClasses(numberOfClasses);
  simpleKMeansClusteringImageFilter->Update();
  means = simpleKMeansClusteringImageFilter->GetFinalMeans();
  stds = simpleKMeansClusteringImageFilter->GetFinalStds();
  if (outputImageName != NULL && strlen(outputImageName) > 0)
  {
    imageWriter->SetInput(simpleKMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName(outputImageName);
    imageWriter->Update();
  }
}

/**
 * Use K-means clustering to get the local means of the tissues. 
 */
void LocalKMeansClassification(SimpleKMeansClusteringImageFilterType::ParametersType& means, 
  SimpleKMeansClusteringImageFilterType::ParametersType& stds, 
  const DoubleImageType* image, 
  const IntImageType* localGMNormalisationMask, 
  int numberOfDilations,
  int numberOfClasses)  
{
  double gmRegionMeanIntensity = 0.0; 
  double gmRegionStdIntensity = 0.0; 

  itk::ImageRegionConstIterator<DoubleImageType> baselineImageIterator(image, image->GetLargestPossibleRegion());                                       
  itk::ImageRegionConstIterator<IntImageType> maskImageIterator(localGMNormalisationMask, localGMNormalisationMask->GetLargestPossibleRegion());
  double numberOfVoxelInGMRegion = 0.0; 

  // Get the mean and std intensity of the GM region in the baseline image. 
  baselineImageIterator.GoToBegin();
  maskImageIterator.GoToBegin();
  for (; !baselineImageIterator.IsAtEnd(); ++baselineImageIterator, ++maskImageIterator)
  {
    if (maskImageIterator.Get() > 0)
    {
      gmRegionMeanIntensity += baselineImageIterator.Get();
      numberOfVoxelInGMRegion += 1.0;
    }                            
  }
  gmRegionMeanIntensity /= numberOfVoxelInGMRegion;
  baselineImageIterator.GoToBegin();
  maskImageIterator.GoToBegin();
  for (; !baselineImageIterator.IsAtEnd(); ++baselineImageIterator, ++maskImageIterator)
  {
    if (maskImageIterator.Get() > 0)
    {
      gmRegionStdIntensity += (gmRegionMeanIntensity-baselineImageIterator.Get())*(gmRegionMeanIntensity-baselineImageIterator.Get());
    }                            
  }
  gmRegionStdIntensity = sqrt(gmRegionStdIntensity/(numberOfVoxelInGMRegion-1.0));

  // Take out the GM region and do 3-class k-means classification.         
  MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
  BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();

  binariseImageFilter->SetPaddingValue(0);
  binariseImageFilter->SetInput(localGMNormalisationMask);
  binariseImageFilter->Update();
  multipleDilateImageFilter->SetNumberOfDilations(numberOfDilations);
  multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
  multipleDilateImageFilter->Update();
  itk::ImageRegionConstIterator<IntImageType> dilatedMaskImageIterator(multipleDilateImageFilter->GetOutput(), multipleDilateImageFilter->GetOutput()->GetLargestPossibleRegion());
  SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
  IntImageType::Pointer noGMRegionMask = IntImageType::New();
  
  noGMRegionMask->SetRegions(multipleDilateImageFilter->GetOutput()->GetLargestPossibleRegion());
  noGMRegionMask->Allocate();
  itk::ImageRegionIterator<IntImageType> noGMRegionMaskIterator(noGMRegionMask, noGMRegionMask->GetLargestPossibleRegion()); 
  maskImageIterator.GoToBegin();
  dilatedMaskImageIterator.GoToBegin();
  noGMRegionMaskIterator.GoToBegin();
  for (; !maskImageIterator.IsAtEnd(); ++dilatedMaskImageIterator, ++maskImageIterator, ++noGMRegionMaskIterator)
  {
    noGMRegionMaskIterator.Set(0);
    if (dilatedMaskImageIterator.Get() > 0 && maskImageIterator.Get() == 0)
    {
      noGMRegionMaskIterator.Set(1);
    }
  }

  simpleKMeansClusteringImageFilter->SetNumberOfClasses(3);
  simpleKMeansClusteringImageFilter->SetInitialMeans(means);
  simpleKMeansClusteringImageFilter->SetInput(image);
  simpleKMeansClusteringImageFilter->SetInputMask(noGMRegionMask);
  simpleKMeansClusteringImageFilter->Update();
  
  // Use the CSF intensity from k-means and GM intensity from caudate. 
  means = simpleKMeansClusteringImageFilter->GetFinalMeans();
  means[1] = gmRegionMeanIntensity; 
  stds = simpleKMeansClusteringImageFilter->GetFinalStds();
  stds[1] = gmRegionStdIntensity; 
}

/**
 * Calculate BSI using the linear regression results. 
 * 
 * int windowAverageMode: 0 - baseline and repeat, 1 - use baseline only, 2 - use repeat only.
 */
double calculateKNBSI(const char* baselineImageName, const char* repeatImageName, 
  const char* baselineMaskName, const char* repeatMaskName, const char* subROIMaskName, const char* weightImageName, 
  const char* bsiMaskName, 
  double csfGreyWindowFactor, double greyWhiteWindowFactor, 
  int numberOfErosion, int numberOfDilation, 
  const itk::Array<double>& baselineFinalMeans, const itk::Array<double>& repeatFinalMeans, 
  const itk::Array<double>& baselineFinalStds, const itk::Array<double>& repeatFinalStds, 
  double slope, double intercept, int windowAverageMode, double minSecondWindowWidth, 
  const char* firstBSIMapName, const char* secondBSIMapName, 
  double userLowerCSFGMWindowValue, double userUpperCSFGMWindowValue, double userLowerGMWMWindowValue, double userUpperGMWMWindowValue)
{
  typedef itk::Image<double, 3> DoubleImageType;
  typedef itk::Image<float, 3> FloatImageType;
  typedef itk::Image<int, 3> IntImageType;
  typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
  typedef itk::ImageFileReader<IntImageType> IntReaderType;
  typedef itk::ImageFileWriter<IntImageType> WriterType;
  typedef itk::ImageFileWriter<DoubleImageType> DoublerWriterType;
  DoubleWindowBoundaryShiftIntegralFilterType::Pointer doubleWindowBSIFilter = DoubleWindowBoundaryShiftIntegralFilterType::New();
  typedef itk::ImageFileReader<DoubleWindowBoundaryShiftIntegralFilterType::WeightImageType> WeightImageReaderType; 
  WeightImageReaderType::Pointer weightImageReader = WeightImageReaderType::New(); 
  typedef itk::ImageFileWriter<FloatImageType> FloatWriterType;
  FloatWriterType::Pointer bsiMapWriter = FloatWriterType::New(); 
  
  DoubleReaderType::Pointer baselineBSIImageReader = DoubleReaderType::New();
  DoubleReaderType::Pointer repeatBSIImageReader = DoubleReaderType::New();
  IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
  IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();
  // Normalise the intensity using the linear regression of CSF, GM and WM intensities, mapping to the baseline scan. 
  baselineBSIImageReader->SetFileName(baselineImageName);
  baselineBSIImageReader->Update();
  repeatBSIImageReader->SetFileName(repeatImageName);
  repeatBSIImageReader->Update();
  
  itk::ImageRegionIterator<DoubleImageType> baselineImageIterator(baselineBSIImageReader->GetOutput(), baselineBSIImageReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<DoubleImageType> repeatImageIterator(repeatBSIImageReader->GetOutput(), repeatBSIImageReader->GetOutput()->GetLargestPossibleRegion());
  
  double baselineNormalisationFactorRepeatBaseline = baselineFinalMeans[2];
  if (windowAverageMode == 2) 
    baselineNormalisationFactorRepeatBaseline = repeatFinalMeans[2];
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
  double lowerWindow = 0.0; 
  double upperWindow = 0.0;
  double lowerGreyWhiteWindow = 0.0;
  double upperGreyWhiteWindow = 0.0;
  
  // CSF-GM boundray intensity window = [ CSF+CSF_sd, local GM-local GM_sd ]. 
  double baselineLowerWindow = (baselineFinalMeans[0]+csfGreyWindowFactor*baselineFinalStds[0])/baselineNormalisationFactorRepeatBaseline;
  double repeatLowerWindow = (slope*(repeatFinalMeans[0]+csfGreyWindowFactor*repeatFinalStds[0])+intercept)/repeatNormalisationFactorRepeatBaseline;
  double baselineUpperWindow = (baselineFinalMeans[1]-csfGreyWindowFactor*baselineFinalStds[1])/baselineNormalisationFactorRepeatBaseline; 
  double repeatUpperWindow = (slope*(repeatFinalMeans[1]-csfGreyWindowFactor*repeatFinalStds[1])+intercept)/repeatNormalisationFactorRepeatBaseline; 
  
  // GM-WM boundray intensity window = [ local GM+local GM_sd, WM-WM_sd ]. 
  double baselineLowerGreyWhiteWindow = (baselineFinalMeans[1]+greyWhiteWindowFactor*baselineFinalStds[1])/baselineNormalisationFactorRepeatBaseline;
  double repeatLowerGreyWhiteWindow = (slope*(repeatFinalMeans[1]+greyWhiteWindowFactor*repeatFinalStds[1])+intercept)/repeatNormalisationFactorRepeatBaseline;
  double baselineUpperGreyWhiteWindow = (baselineFinalMeans[2]-greyWhiteWindowFactor*baselineFinalStds[2])/baselineNormalisationFactorRepeatBaseline;
  double repeatUpperGreyWhiteWindow = (slope*(repeatFinalMeans[2]-greyWhiteWindowFactor*repeatFinalStds[2])+intercept)/repeatNormalisationFactorRepeatBaseline;
  
  switch (windowAverageMode)
  {
    case 0: 
      lowerWindow = (baselineLowerWindow+repeatLowerWindow)/2.0;
      upperWindow = (baselineUpperWindow+repeatUpperWindow)/2.0;
      lowerGreyWhiteWindow = (baselineLowerGreyWhiteWindow+repeatLowerGreyWhiteWindow)/2.0;
      upperGreyWhiteWindow = (baselineUpperGreyWhiteWindow+repeatUpperGreyWhiteWindow)/2.0;
      break; 
    case 1: 
      lowerWindow = baselineLowerWindow;
      upperWindow = baselineUpperWindow;
      lowerGreyWhiteWindow = baselineLowerGreyWhiteWindow;
      upperGreyWhiteWindow = baselineUpperGreyWhiteWindow;
      break; 
    case 2: 
      lowerWindow = repeatLowerWindow;
      upperWindow = repeatUpperWindow;
      lowerGreyWhiteWindow = repeatLowerGreyWhiteWindow;
      upperGreyWhiteWindow = repeatUpperGreyWhiteWindow;
      break; 
    default:
      assert(false); 
  }
  
  if (userLowerCSFGMWindowValue > 0.)
  {
    baselineLowerWindow = userLowerCSFGMWindowValue;
    repeatLowerWindow = userLowerCSFGMWindowValue; 
    lowerWindow = userLowerCSFGMWindowValue; 
  }
  if (userUpperCSFGMWindowValue > 0.)
  {
    baselineUpperWindow = userUpperCSFGMWindowValue;
    repeatUpperWindow = userUpperCSFGMWindowValue; 
    upperWindow = userUpperCSFGMWindowValue; 
  }
  if (userLowerGMWMWindowValue > 0.)
  {
    baselineLowerGreyWhiteWindow = userLowerGMWMWindowValue; 
    repeatLowerGreyWhiteWindow = userLowerGMWMWindowValue; 
    lowerGreyWhiteWindow = userLowerGMWMWindowValue; 
  }
  if (userUpperGMWMWindowValue > 0.)
  {
    baselineUpperGreyWhiteWindow = userUpperGMWMWindowValue; 
    repeatUpperGreyWhiteWindow = userUpperGMWMWindowValue; 
    upperGreyWhiteWindow = userUpperGMWMWindowValue; 
  }
  
  std::cout << "lowerWindow," << baselineLowerWindow << "," << repeatLowerWindow << "," << lowerWindow << ",";
  std::cout << "upperWindow," << baselineUpperWindow << "," << repeatUpperWindow << "," << upperWindow << ",";
  std::cout << "lowerGreyWhiteWindow," << baselineLowerGreyWhiteWindow << "," << repeatLowerGreyWhiteWindow << "," << lowerGreyWhiteWindow << ",";
  std::cout << "upperGreyWhiteWindow," << baselineUpperGreyWhiteWindow << "," << repeatUpperGreyWhiteWindow << "," << upperGreyWhiteWindow << ",";
  
  BoundaryShiftIntegralFilterType::Pointer bsiFilter = BoundaryShiftIntegralFilterType::New();
  IntReaderType::Pointer subROIMaskReader = IntReaderType::New();
  
  baselineBSIMaskReader->SetFileName(baselineMaskName);
  repeatBSIMaskReader->SetFileName(repeatMaskName);
  
  // Double window BSI without double counting. 
  doubleWindowBSIFilter->SetBaselineImage(baselineBSIImageReader->GetOutput());
  doubleWindowBSIFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
  doubleWindowBSIFilter->SetRepeatImage(repeatBSIImageReader->GetOutput());
  doubleWindowBSIFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
  doubleWindowBSIFilter->SetBaselineIntensityNormalisationFactor(1.0);
  doubleWindowBSIFilter->SetRepeatIntensityNormalisationFactor(1.0);
  doubleWindowBSIFilter->SetNumberOfErosion(numberOfErosion);
  doubleWindowBSIFilter->SetNumberOfDilation(numberOfDilation);
  doubleWindowBSIFilter->SetLowerCutoffValue(lowerWindow);
  doubleWindowBSIFilter->SetUpperCutoffValue(upperWindow);
  doubleWindowBSIFilter->SetSecondLowerCutoffValue(lowerGreyWhiteWindow);
  doubleWindowBSIFilter->SetSecondUpperCutoffValue(upperGreyWhiteWindow);
  doubleWindowBSIFilter->SetMinSecondWindowWidth(minSecondWindowWidth); 
  if (subROIMaskName != NULL)
  {
    std::cout << "subROI," << subROIMaskName << ",";
    subROIMaskReader->SetFileName(subROIMaskName);
    doubleWindowBSIFilter->SetSubROIMask(subROIMaskReader->GetOutput()); 
  }
  if (weightImageName != NULL)
  {
    std::cout << "weight," << weightImageName << ","; 
    weightImageReader->SetFileName(weightImageName); 
    doubleWindowBSIFilter->SetWeightImage(weightImageReader->GetOutput()); 
  }
  
  doubleWindowBSIFilter->Compute(); 
  
  if (bsiMaskName != NULL && strlen(bsiMaskName) > 0)
  {
    WriterType::Pointer imageWriter = WriterType::New(); 
    imageWriter->SetInput(doubleWindowBSIFilter->GetBSIMask());
    imageWriter->SetFileName(bsiMaskName);
    imageWriter->Update();
  }
  
  std::cout << "CSF-GM BSI," << doubleWindowBSIFilter->GetFirstBoundaryShiftIntegral() << "," 
            << "GM-WM BSI," << doubleWindowBSIFilter->GetSecondBoundaryShiftIntegral() << ","
            << "DW BSI," << doubleWindowBSIFilter->GetBoundaryShiftIntegral() << ","; 
  
  // Save the BSI maps. 
  if (firstBSIMapName != NULL && strlen(firstBSIMapName) > 0)
  {
    bsiMapWriter->SetInput(doubleWindowBSIFilter->GetBSIMap()); 
    bsiMapWriter->SetFileName(firstBSIMapName); 
    bsiMapWriter->Update(); 
  }
  if (secondBSIMapName != NULL && strlen(secondBSIMapName) > 0)
  {
    bsiMapWriter->SetInput(doubleWindowBSIFilter->GetSecondBSIMap()); 
    bsiMapWriter->SetFileName(secondBSIMapName); 
    bsiMapWriter->Update(); 
  }
  
  return doubleWindowBSIFilter->GetBoundaryShiftIntegral();
}


/**
 * Main program.
 */
int main(int argc, char* argv[])
{
  if (argc < 21)
  {
#ifndef OPEN_SOURCE_BSI
    niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
#endif
    std::cerr << std::endl;
    std::cerr << "Program to calculate the k-means normalised double window boundary shift integral, based on the paper" << std::endl; 
    std::cerr << "  Freeborough PA and Fox NC, The boundary shift integral: an accurate and" << std::endl; 
    std::cerr << "  robust measure of cerebral volume changes from registered repeat MRI," << std::endl; 
    std::cerr << "  IEEE Trans Med Imaging. 1997 Oct;16(5):623-9." << std::endl << std::endl;
    std::cerr << "The double window BSI aims to capture the boundary change between CSF and GM " << std::endl; 
    std::cerr << "as well as the boundary change between GM and WM. This is mainly used to calculate " << std::endl; 
    std::cerr << "the caudate BSI because one bounday of caudate is CSF/GM and the other side is GM/WM." << std::endl; 
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
    std::cerr << "         <output baseline image classification>" << std::endl;
    std::cerr << "         <output repeat image classification>" << std::endl;
    std::cerr << "         <sub-ROI mask name>" << std::endl;
    std::cerr << "         <output XOR mask name>" << std::endl;
    std::cerr << "         <baseline local GM intensity normalisation mask>" << std::endl;
    std::cerr << "         <repeat local GM intensity normalisation mask>" << std::endl;
    std::cerr << "         <csf-grey window factor> (recommanded value 1)" << std::endl; 
    std::cerr << "         <grey-white window factor> (recommanded value 1)" << std::endl; 
    std::cerr << "         <weight image> ('dummy' for not used)" << std::endl;
    std::cerr << "         <min. GM-WM window width> (-1 for not used)" << std::endl; 
    std::cerr << "         <CSF-GM BSI map image name> " << std::endl; 
    std::cerr << "         <GM-WM BSI map image name> " << std::endl; 
    std::cerr << "         <Lower CSF-GM window value> (-1 for automatic)" << std::endl; 
    std::cerr << "         <Upper CSF-GM window value> (-1 for automatic)" << std::endl; 
    std::cerr << "         <Lower GM-WM window value> (-1 for automatic)" << std::endl; 
    std::cerr << "         <Upper CSF-GM window value> (-1 for automatic)" << std::endl; 
    std::cerr << "Notice that all the images and masks for intensity normalisation must " << std::endl;
    std::cerr << "have the SAME voxel sizes and image dimensions. The same applies to the " << std::endl;
    std::cerr << "images and masks for BSI." << std::endl;
    return EXIT_FAILURE;
  }
  
  try
  {
    const char* baselineNormalisationImageName = argv[1]; 
    const char* baselineNormalisationMaskName = argv[2]; 
    const char* repeatNormalisationImageName = argv[3]; 
    const char* repeatNormalisationMaskName = argv[4]; 
    const char* baselineBSIImageName = argv[5]; 
    const char* baselineBSIMaskName = argv[6]; 
    const char* repeatBSIImageName = argv[7]; 
    const char* repeatBSIMaskName = argv[8]; 
    int numberOfBSIErosion = atoi(argv[9]); 
    int numberOfBSIDilation = atoi(argv[10]); 
    int numberOfDilationsForKMeans = atoi(argv[11]); 
    const char* baselineKMeansOutputName = argv[12]; 
    const char* repeatKMeansOutputName = argv[13]; 
    const char* subROIMaskName = NULL;
    double minSecondWindowWidth = -1.0; 
    const char* firstBSIMapName = NULL; 
    const char* secondBSIMapName = NULL; 
    double userLowerCSFGMWindowValue = -1.;
    double userUpperCSFGMWindowValue = -1.;
    double userLowerGMWMWindowValue = -1.;
    double userUpperGMWMWindowValue = -1.;
    if (argv[14] != NULL && strlen(argv[14]) > 0 && strcmp(argv[14], "dummy") != 0)
    {
      subROIMaskName = argv[14];
    }
    // Output BSI mask name. 
    const char* bsiMaskName = argv[15];
    const char* baselineLocalGMNormalisationMaskName = NULL; 
    if (argv[16] != NULL && strcmp(argv[16], "dummy") != 0)
    {
      baselineLocalGMNormalisationMaskName = argv[16]; 
    }
    const char* repeatLocalGMNormalisationMaskName = NULL; 
    if (argv[17] != NULL && strcmp(argv[17], "dummy") != 0)
    {
      repeatLocalGMNormalisationMaskName = argv[17]; 
    }
    double csfGreyWindowFactor = atof(argv[18]); 
    std::cerr << "csfGreyWindowFactor=" << csfGreyWindowFactor << std::endl; 
    double greyWhiteWindowFactor = atof(argv[19]); 
    std::cerr << "greyWhiteWindowFactor=" << greyWhiteWindowFactor << std::endl; 
    const char* weightImageName = NULL; 
    if (argc > 20 && strcmp(argv[20], "dummy") != 0)
    {
      weightImageName = argv[20]; 
    }
    if (argc > 21)
    {
      minSecondWindowWidth = atof(argv[21]); 
    }
    if (argc > 22)
    {
      firstBSIMapName = argv[22];  
    }
    if (argc > 23)
    {
      secondBSIMapName = argv[23];  
    }
    if (argc > 24) 
    {
      userLowerCSFGMWindowValue = atof(argv[24]); 
    }
    if (argc > 25) 
    {
      userUpperCSFGMWindowValue = atof(argv[25]); 
    }
    if (argc > 26) 
    {
      userLowerGMWMWindowValue = atof(argv[26]); 
    }
    if (argc > 27) 
    {
      userUpperGMWMWindowValue = atof(argv[27]); 
    }
    
    DoubleReaderType::Pointer baselineNormalisationImageReader = DoubleReaderType::New();
    DoubleReaderType::Pointer repeatNormalisationImageReader = DoubleReaderType::New();
    IntReaderType::Pointer baselineNormalisationMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatNormalisationMaskReader = IntReaderType::New();

    baselineNormalisationImageReader->SetFileName(baselineNormalisationImageName);
    baselineNormalisationMaskReader->SetFileName(baselineNormalisationMaskName);
    repeatNormalisationImageReader->SetFileName(repeatNormalisationImageName);
    repeatNormalisationMaskReader->SetFileName(repeatNormalisationMaskName);
    std::cout << baselineNormalisationImageName << "," << repeatNormalisationImageName << ",";

    // Calculate mean brain intensity. 
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();

    normalisationCalculator->SetInputImage1(baselineNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputImage2(repeatNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputMask1(baselineNormalisationMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(repeatNormalisationMaskReader->GetOutput());
    normalisationCalculator->Compute();
    std::cout << "mean intensities," << normalisationCalculator->GetNormalisationMean1() << "," 
                                     << normalisationCalculator->GetNormalisationMean2() << ",";
                                     
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localBaselineCSFWMMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localBaselineCSFWMStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localRepeatCSFWMMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localRepeatCSFWMStds(3);
    double slopeRepeatBaseline = 0.0; 
    double interceptRepeatBaseline = 0.0;      
    double slopeBaselineRepeat = 0.0; 
    double interceptBaselineRepeat = 0.0; 
    std::vector<double> baselineMeans;
    std::vector<double> repeatMeans;
    
    baselineFinalMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
    baselineFinalMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
    baselineFinalMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
    KMeansClassification(baselineFinalMeans, baselineFinalStds, 
                          baselineNormalisationImageReader->GetOutput(),
                          baselineNormalisationMaskReader->GetOutput(),
                          numberOfDilationsForKMeans, 3, baselineKMeansOutputName);  
    
    repeatFinalMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
    repeatFinalMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
    repeatFinalMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
    KMeansClassification(repeatFinalMeans, repeatFinalStds, 
                          repeatNormalisationImageReader->GetOutput(),
                          repeatNormalisationMaskReader->GetOutput(),
                          numberOfDilationsForKMeans, 3, repeatKMeansOutputName);  
    
    baselineMeans.push_back(normalisationCalculator->GetNormalisationMean1());
    repeatMeans.push_back(normalisationCalculator->GetNormalisationMean2());
    for (int index = 0; index < 3; index++)
    {
      baselineMeans.push_back(baselineFinalMeans[index]);
      repeatMeans.push_back(repeatFinalMeans[index]);
    }
    
    // Repeat (x) regress on baseline (y).  
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(repeatMeans, baselineMeans, &slopeRepeatBaseline, &interceptRepeatBaseline);      
    
    // Baseline (x) regress on repeat (y).  
    itk::BoundaryShiftIntegralCalculator<IntImageType,IntImageType,IntImageType>::PerformLinearRegression(baselineMeans, repeatMeans, &slopeBaselineRepeat, &interceptBaselineRepeat);
  
    std::cout << "baseline means," << baselineFinalMeans[0] << "," << baselineFinalMeans[1] << "," << baselineFinalMeans[2] << ",";  
    std::cout << "repeat means," << repeatFinalMeans[0] << "," << repeatFinalMeans[1] << "," << repeatFinalMeans[2] << ",";  
    std::cout << "baseline std," << baselineFinalStds[0] << "," << baselineFinalStds[1] << "," << baselineFinalStds[2] << ",";  
    std::cout << "repeat std," << repeatFinalStds[0] << "," << repeatFinalStds[1] << "," << repeatFinalStds[2] << ",";  
  
    // Use the local GM mask to get the GM intensity. 
    // Dilate the mask by 3 and the k-means again. 
    IntReaderType::Pointer baselineLocalGMBaselineNormalisationMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatLocalGMBaselineNormalisationMaskReader = IntReaderType::New();
    
    if (baselineLocalGMNormalisationMaskName != NULL)
    {
      baselineLocalGMBaselineNormalisationMaskReader->SetFileName(baselineLocalGMNormalisationMaskName); 
      baselineLocalGMBaselineNormalisationMaskReader->Update(); 
      // Baseline. 
      localBaselineCSFWMMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
      localBaselineCSFWMMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
      localBaselineCSFWMMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
      LocalKMeansClassification(localBaselineCSFWMMeans, localBaselineCSFWMStds, 
          baselineNormalisationImageReader->GetOutput(), baselineLocalGMBaselineNormalisationMaskReader->GetOutput(), numberOfDilationsForKMeans, 3); 
    }
    else
    {
      localBaselineCSFWMMeans[0] = baselineFinalMeans[0]; 
      localBaselineCSFWMMeans[1] = baselineFinalMeans[1]; 
      localBaselineCSFWMMeans[2] = baselineFinalMeans[2]; 
      localBaselineCSFWMStds[0] = baselineFinalStds[0]; 
      localBaselineCSFWMStds[1] = baselineFinalStds[1]; 
      localBaselineCSFWMStds[2] = baselineFinalStds[2]; 
    }
    if (repeatLocalGMNormalisationMaskName != NULL)
    {
      repeatLocalGMBaselineNormalisationMaskReader->SetFileName(repeatLocalGMNormalisationMaskName); 
      repeatLocalGMBaselineNormalisationMaskReader->Update();
      // Repeat.                                   
      localRepeatCSFWMMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
      localRepeatCSFWMMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
      localRepeatCSFWMMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
      LocalKMeansClassification(localRepeatCSFWMMeans, localRepeatCSFWMStds, 
          repeatNormalisationImageReader->GetOutput(), repeatLocalGMBaselineNormalisationMaskReader->GetOutput(), numberOfDilationsForKMeans, 3); 
    }
    else
    {
      localRepeatCSFWMMeans[0] = repeatFinalMeans[0]; 
      localRepeatCSFWMMeans[1] = repeatFinalMeans[1]; 
      localRepeatCSFWMMeans[2] = repeatFinalMeans[2]; 
      localRepeatCSFWMStds[0] = repeatFinalStds[0]; 
      localRepeatCSFWMStds[1] = repeatFinalStds[1]; 
      localRepeatCSFWMStds[2] = repeatFinalStds[2]; 
    }
    
    std::cout << "localBaselineCSFWMMeans," << localBaselineCSFWMMeans[0] << "," << localBaselineCSFWMMeans[1] << "," << localBaselineCSFWMMeans[2] << ",";
    std::cout << "localBaselineCSFWMStds," << localBaselineCSFWMStds[0] << "," << localBaselineCSFWMStds[1] << "," << localBaselineCSFWMStds[2] << ",";
    std::cout << "localRepeatCSFWMMeans," << localRepeatCSFWMMeans[0] << "," << localRepeatCSFWMMeans[1] << "," << localRepeatCSFWMMeans[2] << ",";
    std::cout << "localRepeatCSFWMStds," << localRepeatCSFWMStds[0] << "," << localRepeatCSFWMStds[1] << "," << localRepeatCSFWMStds[2] << ",";
    std::cout << "slopeRepeatBaseline," << slopeRepeatBaseline << ",interceptRepeatBaseline," << interceptRepeatBaseline << ",";
    std::cout << "slopeBaselineRepeat," << slopeBaselineRepeat << ",interceptBaselineRepeat," << interceptBaselineRepeat << ",";
    
    double forwardBSI = 0.0; 
    double backwardBSI = 0.0; 
    if (repeatLocalGMNormalisationMaskName != NULL)
    {
      // Forward BSI. 
      forwardBSI = calculateKNBSI(baselineBSIImageName, repeatBSIImageName, baselineBSIMaskName, repeatBSIMaskName, subROIMaskName, weightImageName, 
                                  bsiMaskName, csfGreyWindowFactor, greyWhiteWindowFactor, numberOfBSIErosion, numberOfBSIDilation, localBaselineCSFWMMeans,
                                  localRepeatCSFWMMeans, localBaselineCSFWMStds, localRepeatCSFWMStds, slopeRepeatBaseline, interceptRepeatBaseline, 0,
                                  minSecondWindowWidth, firstBSIMapName, secondBSIMapName, 
                                  userLowerCSFGMWindowValue, userUpperCSFGMWindowValue, userLowerGMWMWindowValue, userUpperGMWMWindowValue); 
      
      // Backward BSI. 
      backwardBSI = calculateKNBSI(repeatBSIImageName, baselineBSIImageName, repeatBSIMaskName, baselineBSIMaskName, subROIMaskName, weightImageName, 
                                   bsiMaskName, csfGreyWindowFactor, greyWhiteWindowFactor, numberOfBSIErosion, numberOfBSIDilation, localRepeatCSFWMMeans,
                                   localBaselineCSFWMMeans, localRepeatCSFWMStds, localBaselineCSFWMStds, slopeBaselineRepeat, interceptBaselineRepeat, 0,
                                   minSecondWindowWidth, NULL, NULL, 
                                   userLowerCSFGMWindowValue, userUpperCSFGMWindowValue, userLowerGMWMWindowValue, userUpperGMWMWindowValue); 
    }
    else
    {
      // Forward BSI. 
      forwardBSI = calculateKNBSI(baselineBSIImageName, repeatBSIImageName, baselineBSIMaskName, repeatBSIMaskName, subROIMaskName, weightImageName, 
                                  bsiMaskName, csfGreyWindowFactor, greyWhiteWindowFactor, numberOfBSIErosion, numberOfBSIDilation, localBaselineCSFWMMeans,
                                  localRepeatCSFWMMeans, localBaselineCSFWMStds, localRepeatCSFWMStds, slopeRepeatBaseline, interceptRepeatBaseline, 1, 
                                  minSecondWindowWidth, firstBSIMapName, secondBSIMapName, 
                                  userLowerCSFGMWindowValue, userUpperCSFGMWindowValue, userLowerGMWMWindowValue, userUpperGMWMWindowValue); 
      
      // Backward BSI. 
      backwardBSI = calculateKNBSI(repeatBSIImageName, baselineBSIImageName, repeatBSIMaskName, baselineBSIMaskName, subROIMaskName, weightImageName, 
                                   bsiMaskName, csfGreyWindowFactor, greyWhiteWindowFactor, numberOfBSIErosion, numberOfBSIDilation, localRepeatCSFWMMeans,
                                   localBaselineCSFWMMeans, localRepeatCSFWMStds, localBaselineCSFWMStds, slopeBaselineRepeat, interceptBaselineRepeat, 2, 
                                   minSecondWindowWidth, NULL, NULL, 
                                   userLowerCSFGMWindowValue, userUpperCSFGMWindowValue, userLowerGMWMWindowValue, userUpperGMWMWindowValue); 
    }
    
    std::cout << "BSI," << forwardBSI << "," << -backwardBSI << "," << (forwardBSI-backwardBSI)/2.0 << std::endl;
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


