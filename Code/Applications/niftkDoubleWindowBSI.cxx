#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include "itkLogHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkDoubleWindowBoundaryShiftIntegralCalculator.h"
#include "itkSimpleKMeansClusteringImageFilter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkIndent.h"
#include <stdio.h>

/**
 * Typedefs. 
 */
typedef itk::Image<double, 3> DoubleImageType;
typedef itk::Image<int, 3> IntImageType;

typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
typedef itk::ImageFileReader<IntImageType> IntReaderType;
typedef itk::ImageFileWriter<IntImageType> WriterType;
typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;
typedef itk::DoubleWindowBoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> DoubleWindowBoundaryShiftIntegralFilterType;
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
 * Main program.
 */
int main(int argc, char* argv[])
{
  if (argc < 21)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
    std::cerr << std::endl;    
    std::cerr << "Program to calculate the double window boundary shift integral, based on the paper" << std::endl; 
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
    std::cerr << "         <CSF-GM lower window>" << std::endl;
    std::cerr << "         <CSF-GM upper window>" << std::endl;
    std::cerr << "         <GM-WM lower window>" << std::endl;
    std::cerr << "         <GM-WM upper window>" << std::endl;
    std::cerr << "         <sub-ROI mask name>" << std::endl;
    std::cerr << "         <output XOR mask name>" << std::endl;
    std::cerr << "         <baseline local GM intensity normalisation mask>" << std::endl;
    std::cerr << "         <csf-grey window factor> (recommanded value 1)" << std::endl; 
    std::cerr << "         <grey-white window factor> (recommanded value 1)" << std::endl; 
    std::cerr << "Notice that all the images and masks for intensity normalisation must " << std::endl;
    std::cerr << "have the SAME voxel sizes and image dimensions. The same applies to the " << std::endl;
    std::cerr << "images and masks for BSI." << std::endl;
    return EXIT_FAILURE;
  }
  
  try
  {
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
                                     
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localBaselineFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localBaselineFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localRepeatFinalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localRepeatFinalStds(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localBaselineCSFWMMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType localBaselineCSFWMStds(3);
    
    double lowerWindow = atof(argv[14]);
    double upperWindow = atof(argv[15]);
    double lowerGreyWhiteWindow = atof(argv[16]); 
    double upperGreyWhiteWindow = atof(argv[17]);
    const char* localGMNormalisationMaskName = argv[20];
    double csfGreyWindowFactor = atof(argv[21]); 
    double greyWhiteWindowFactor = atof(argv[22]); 
    bool isDoKmeans = false;
    double caudateMeanIntensity = 0.0;
    double caudateStdIntensity = 0.0;
    
    if (lowerWindow < 0.0 || upperWindow < 0.0 || lowerGreyWhiteWindow < 0.0 || upperGreyWhiteWindow < 0.0)
      isDoKmeans = true; 
      
    // Calculate the intensity window.                                      
    if (isDoKmeans)
    {
      baselineFinalMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
      baselineFinalMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
      baselineFinalMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
      KMeansClassification(baselineFinalMeans, baselineFinalStds, 
                           baselineNormalisationImageReader->GetOutput(),
                           baselineNormalisationMaskReader->GetOutput(),
                           atoi(argv[11]), 3, argv[12]);  
      
      repeatFinalMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
      repeatFinalMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
      repeatFinalMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
      KMeansClassification(repeatFinalMeans, repeatFinalStds, 
                           repeatNormalisationImageReader->GetOutput(),
                           repeatNormalisationMaskReader->GetOutput(),
                           atoi(argv[11]), 3, argv[13]);  
      
      lowerWindow = ((baselineFinalMeans[0]+baselineFinalStds[0])/normalisationCalculator->GetNormalisationMean1()+ 
                            (repeatFinalMeans[0]+repeatFinalStds[0])/normalisationCalculator->GetNormalisationMean2())/2.0;
      upperWindow = ((baselineFinalMeans[1]-baselineFinalStds[1])/normalisationCalculator->GetNormalisationMean1()+ 
                            (repeatFinalMeans[1]-repeatFinalStds[1])/normalisationCalculator->GetNormalisationMean2())/2.0;
    
      // Use the local GM mask to get the GM intensity. 
      // Dilate the mask by 3 and the k-means again. 
      if (localGMNormalisationMaskName != NULL && strlen(localGMNormalisationMaskName) > 0)
      {
        IntReaderType::Pointer localGMBaselineNormalisationMaskReader = IntReaderType::New();
        
        localGMBaselineNormalisationMaskReader->SetFileName(localGMNormalisationMaskName);
        localGMBaselineNormalisationMaskReader->Update();
        
        localBaselineFinalMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
        localBaselineFinalMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
        localBaselineFinalMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
                             
        localRepeatFinalMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
        localRepeatFinalMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
        localRepeatFinalMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
                             
        itk::ImageRegionConstIterator < DoubleImageType > baselineImageIterator(baselineNormalisationImageReader->GetOutput(), baselineNormalisationImageReader->GetOutput()->GetLargestPossibleRegion());                                       
        itk::ImageRegionConstIterator < IntImageType > maskImageIterator(localGMBaselineNormalisationMaskReader->GetOutput(), localGMBaselineNormalisationMaskReader->GetOutput()->GetLargestPossibleRegion());
        double numberOfVoxelInCaudate = 0.0; 

        // Get the mean and std intensity of the caudate in the baseline image. 
        baselineImageIterator.GoToBegin();
        maskImageIterator.GoToBegin();
        for (; !baselineImageIterator.IsAtEnd(); ++baselineImageIterator, ++maskImageIterator)
        {
          if (maskImageIterator.Get() > 0)
          {
            caudateMeanIntensity += baselineImageIterator.Get();
            numberOfVoxelInCaudate += 1.0;
          }                            
        }
        caudateMeanIntensity /= numberOfVoxelInCaudate;
        baselineImageIterator.GoToBegin();
        maskImageIterator.GoToBegin();
        for (; !baselineImageIterator.IsAtEnd(); ++baselineImageIterator, ++maskImageIterator)
        {
          if (maskImageIterator.Get() > 0)
          {
            caudateStdIntensity += (caudateMeanIntensity-baselineImageIterator.Get())*(caudateMeanIntensity-baselineImageIterator.Get());
          }                            
        }
        caudateStdIntensity = sqrt(caudateStdIntensity/(numberOfVoxelInCaudate-1.0));

        // Take out the caudate and do 3-class k-means classification.         
        MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
        BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
    
        binariseImageFilter->SetPaddingValue(0);
        binariseImageFilter->SetInput(localGMBaselineNormalisationMaskReader->GetOutput());
        binariseImageFilter->Update();
        multipleDilateImageFilter->SetNumberOfDilations(atoi(argv[11]));
        multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
        multipleDilateImageFilter->Update();
        itk::ImageRegionConstIterator < IntImageType > dilatedMaskImageIterator(multipleDilateImageFilter->GetOutput(), multipleDilateImageFilter->GetOutput()->GetLargestPossibleRegion());
        SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
        IntImageType::Pointer noCaudateMask = IntImageType::New();
        
        noCaudateMask->SetRegions(multipleDilateImageFilter->GetOutput()->GetLargestPossibleRegion());
        noCaudateMask->Allocate();
        itk::ImageRegionIterator < IntImageType > noCaudateMaskIterator(noCaudateMask, noCaudateMask->GetLargestPossibleRegion()); 
        maskImageIterator.GoToBegin();
        dilatedMaskImageIterator.GoToBegin();
        noCaudateMaskIterator.GoToBegin();
        for (; !maskImageIterator.IsAtEnd(); ++dilatedMaskImageIterator, ++maskImageIterator, ++noCaudateMaskIterator)
        {
          noCaudateMaskIterator.Set(0);
          if (dilatedMaskImageIterator.Get() > 0 && maskImageIterator.Get() == 0)
          {
            noCaudateMaskIterator.Set(1);
          }
        }
  
        localBaselineCSFWMMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
        localBaselineCSFWMMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
        localBaselineCSFWMMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
        simpleKMeansClusteringImageFilter->SetNumberOfClasses(3);
        simpleKMeansClusteringImageFilter->SetInitialMeans(localBaselineCSFWMMeans);
        simpleKMeansClusteringImageFilter->SetInput(baselineNormalisationImageReader->GetOutput());
        simpleKMeansClusteringImageFilter->SetInputMask(noCaudateMask);
        simpleKMeansClusteringImageFilter->Update();
        
        // Use the CSF intensity from k-means and GM intensity from caudate. 
        localBaselineCSFWMMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
        localBaselineCSFWMStds = simpleKMeansClusteringImageFilter->GetFinalStds();
        lowerWindow = (localBaselineCSFWMMeans[0]+csfGreyWindowFactor*localBaselineCSFWMStds[0])/normalisationCalculator->GetNormalisationMean1();  
        upperWindow = (caudateMeanIntensity-csfGreyWindowFactor*caudateStdIntensity)/normalisationCalculator->GetNormalisationMean1();
      }
    }
                                  
    std::cout << "baseline means," << baselineFinalMeans[0] << "," << baselineFinalMeans[1] << "," << baselineFinalMeans[2] << ",";  
    std::cout << "repeat means," << repeatFinalMeans[0] << "," << repeatFinalMeans[1] << "," << repeatFinalMeans[2] << ",";  
    std::cout << "baseline std," << baselineFinalStds[0] << "," << baselineFinalStds[1] << "," << baselineFinalStds[2] << ",";  
    std::cout << "repeat std," << repeatFinalStds[0] << "," << repeatFinalStds[1] << "," << repeatFinalStds[2] << ",";  
    //std::cout << "baseline local means," << localBaselineFinalMeans[0] << "," << localBaselineFinalMeans[1] << "," << localBaselineFinalMeans[2] << ",";  
    //std::cout << "repeat local means," << localRepeatFinalMeans[0] << "," << localRepeatFinalMeans[1] << "," << localRepeatFinalMeans[2] << ",";  
    //std::cout << "baseline local std," << localBaselineFinalStds[0] << "," << localBaselineFinalStds[1] << "," << localBaselineFinalStds[2] << ",";  
    //std::cout << "repeat local std," << localRepeatFinalStds[0] << "," << localRepeatFinalStds[1] << "," << localRepeatFinalStds[2] << ",";
    std::cout << "localBaselineCSFWMMeans," << localBaselineCSFWMMeans[0] << "," << localBaselineCSFWMMeans[1] << "," << localBaselineCSFWMMeans[2] << ",";
    std::cout << "localBaselineCSFWMStds," << localBaselineCSFWMStds[0] << "," << localBaselineCSFWMStds[1] << "," << localBaselineCSFWMStds[2] << ",";
    std::cout << "caudate mean and std," << caudateMeanIntensity << "," << caudateStdIntensity << ",";
    std::cout << "CSF-GM window," << lowerWindow << "," << upperWindow << ",";                                  

    BoundaryShiftIntegralFilterType::Pointer bsiFilter = BoundaryShiftIntegralFilterType::New();
    DoubleWindowBoundaryShiftIntegralFilterType::Pointer doubleWindowBSIFilter = DoubleWindowBoundaryShiftIntegralFilterType::New();
    DoubleReaderType::Pointer baselineBSIImageReader = DoubleReaderType::New();
    DoubleReaderType::Pointer repeatBSIImageReader = DoubleReaderType::New();
    IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();
    IntReaderType::Pointer subROIMaskReader = IntReaderType::New();

    baselineBSIImageReader->SetFileName(argv[5]);
    baselineBSIMaskReader->SetFileName(argv[6]);
    repeatBSIImageReader->SetFileName(argv[7]);
    repeatBSIMaskReader->SetFileName(argv[8]);
    char* subROIMaskName = NULL;
    if (argc > 18)
    {
      if (argv[18] != NULL && strlen(argv[18]) > 0 && strcmp(argv[18], "dummy") != 0)
      {
        subROIMaskName = argv[18];
        subROIMaskReader->SetFileName(subROIMaskName); 
        subROIMaskReader->Update(); 
      }
    }
    
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
    if (subROIMaskName != NULL)
      bsiFilter->SetSubROIMask(subROIMaskReader->GetOutput()); 
    bsiFilter->Compute();
    
    
    char* bsiMaskName = NULL;
    
    if (argc > 19)
    {
      bsiMaskName = argv[19];
    }
    double csfGMBSI = bsiFilter->GetBoundaryShiftIntegral();
    
    std::cout << "CSF-GM BSI," << csfGMBSI << ",";
    
    if (isDoKmeans)
    {
      lowerGreyWhiteWindow = ((baselineFinalMeans[1]+baselineFinalStds[1])/normalisationCalculator->GetNormalisationMean1()+ 
                                   (repeatFinalMeans[1]+repeatFinalStds[1])/normalisationCalculator->GetNormalisationMean2())/2.0;
      upperGreyWhiteWindow = ((baselineFinalMeans[2]-baselineFinalStds[2])/normalisationCalculator->GetNormalisationMean1()+ 
                                   (repeatFinalMeans[2]-repeatFinalStds[2])/normalisationCalculator->GetNormalisationMean2())/2.0;
                                   
      if (localGMNormalisationMaskName != NULL && strlen(localGMNormalisationMaskName) > 0)
      {
        // Use the GM intensity from caudate and WM intensity from k-means. 
        lowerGreyWhiteWindow = (caudateMeanIntensity+greyWhiteWindowFactor*caudateStdIntensity)/normalisationCalculator->GetNormalisationMean1();
        upperGreyWhiteWindow = (localBaselineCSFWMMeans[2]-greyWhiteWindowFactor*localBaselineCSFWMStds[2])/normalisationCalculator->GetNormalisationMean1();  
      }
                                   
    }
    
    double gmWMBSI = 0.0; 
      
    if (lowerGreyWhiteWindow < upperGreyWhiteWindow)
    {
      bsiFilter->SetLowerCutoffValue(lowerGreyWhiteWindow);
      bsiFilter->SetUpperCutoffValue(upperGreyWhiteWindow);
      if (subROIMaskName != NULL)
        bsiFilter->SetSubROIMask(subROIMaskReader->GetOutput()); 
      bsiFilter->Compute();
      gmWMBSI = bsiFilter->GetBoundaryShiftIntegral();
    }
    else
    {
      std::cerr << "Warning: lower GM-WM window is higher than upper GM-WM window: " << lowerGreyWhiteWindow << "-" << upperGreyWhiteWindow << std::endl; 
      std::cerr << "Setting GM-WM BSI to 0" << std::endl;  
    }
    
    if (bsiMaskName != NULL && strlen(bsiMaskName) > 0)
    {
      imageWriter->SetInput(bsiFilter->GetBSIMask());
      imageWriter->SetFileName(bsiMaskName);
      imageWriter->Update();
    }
    
    
    std::cout << "GM-WM window," << lowerGreyWhiteWindow << "," << upperGreyWhiteWindow << ",";
    std::cout << "GM-WM BSI," << gmWMBSI << ",";
    
    double totalBSI = csfGMBSI-gmWMBSI;
    
    std::cout << "old total BSI," << totalBSI << ","; 
    
    typedef itk::ImageFileReader<DoubleWindowBoundaryShiftIntegralFilterType::WeightImageType> WeightImageReaderType; 
    WeightImageReaderType::Pointer weightImageReader = WeightImageReaderType::New(); 
    
    if (argc > 23 && strcmp(argv[23], "dummy") != 0)
    {
      weightImageReader->SetFileName(argv[23]); 
      weightImageReader->Update(); 
      doubleWindowBSIFilter->SetWeightImage(weightImageReader->GetOutput()); 
    }
    
    // Double window BSI without double counting. 
    doubleWindowBSIFilter->SetBaselineImage(baselineBSIImageReader->GetOutput());
    doubleWindowBSIFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
    doubleWindowBSIFilter->SetRepeatImage(repeatBSIImageReader->GetOutput());
    doubleWindowBSIFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
    doubleWindowBSIFilter->SetBaselineIntensityNormalisationFactor(normalisationCalculator->GetNormalisationMean1());
    doubleWindowBSIFilter->SetRepeatIntensityNormalisationFactor(normalisationCalculator->GetNormalisationMean2());
    doubleWindowBSIFilter->SetNumberOfErosion(atoi(argv[9]));
    doubleWindowBSIFilter->SetNumberOfDilation(atoi(argv[10]));
    doubleWindowBSIFilter->SetLowerCutoffValue(lowerWindow);
    doubleWindowBSIFilter->SetUpperCutoffValue(upperWindow);
    doubleWindowBSIFilter->SetSecondLowerCutoffValue(lowerGreyWhiteWindow);
    doubleWindowBSIFilter->SetSecondUpperCutoffValue(upperGreyWhiteWindow);
    if (subROIMaskName != NULL)
      doubleWindowBSIFilter->SetSubROIMask(subROIMaskReader->GetOutput()); 
    doubleWindowBSIFilter->Compute(); 
    
    std::cout << "CSF-GM BSI," << doubleWindowBSIFilter->GetFirstBoundaryShiftIntegral() << "," 
              << "GM-WM BSI," << doubleWindowBSIFilter->GetSecondBoundaryShiftIntegral() << ","
              << "total BSI," << doubleWindowBSIFilter->GetBoundaryShiftIntegral() << std::endl; 
    
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


