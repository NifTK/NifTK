/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
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

#include <iostream>
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkBilateralImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkSimpleKMeansClusteringImageFilter.h"
#include "itkBinariseUsingPaddingImageFilter.h"


void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Bilteral filter from ITK. See itk::BilateralImageFilter." << std::endl;
  std::cout << "  " << std::endl; 
  std::cout << "  -i <input> -o <output> -ds <domain/spatial sigma> -rs <range/intensity sigma>" << std::endl; 
  std::cout << "  " << std::endl;
}

  

int main(int argc, char* argv[])
{
  std::string inputImage1Name;
  std::string inputImage2Name; 
  std::string inputMaskName; 
  std::string outputImageName; 
  double domainSigma = 0.0; 
  double rangeSigma = 0.0; 
  int numberOfDilations = 3; 
  
  if (argc < 9)
  {
    StartUsage(argv[0]); 
    return EXIT_FAILURE; 
  }
  
  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0)
    {
      StartUsage(argv[0]);
      return -1;
    }
    else if (strcmp(argv[i], "-i1") == 0)
    {
      inputImage1Name = argv[++i];
    }
    else if (strcmp(argv[i], "-i2") == 0)
    {
      inputImage2Name = argv[++i];
    }
    else if (strcmp(argv[i], "-o") == 0)
    {
      outputImageName = argv[++i];
    }
    else if (strcmp(argv[i], "-ds") == 0)
    {
      domainSigma = atof(argv[++i]);
    }
    else if (strcmp(argv[i], "-rs") == 0)
    {
      rangeSigma = atof(argv[++i]);
    }
    else if (strcmp(argv[i], "-d") == 0)
    {
      numberOfDilations = atof(argv[++i]);
    }
    else if (strcmp(argv[i], "-m") == 0)
    {
      inputMaskName = argv[++i];
    }
    else
    {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      StartUsage(argv[0]); 
      return -1;
    }
  }
  
  const unsigned int Dimension = 3;
  typedef itk::Image<short, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  typedef itk::BilateralImageFilter<ImageType, ImageType> BilateralImageFilterType;
  typedef itk::Image<double, Dimension> DoubleImageType;
  typedef itk::Image<short, Dimension> ShortImageType;
  typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
  typedef itk::ImageFileReader<ShortImageType> ShortReaderType;
  typedef itk::SimpleKMeansClusteringImageFilter<DoubleImageType, ShortImageType, ShortImageType> SimpleKMeansClusteringImageFilterType;
  typedef itk::MultipleDilateImageFilter<ShortImageType> MultipleDilateImageFilterType;
  typedef itk::BinariseUsingPaddingImageFilter<ShortImageType,ShortImageType> BinariseUsingPaddingImageFilterType;
  typedef itk::IntensityNormalisationCalculator<DoubleImageType,ShortImageType> IntensityNormalisationCalculatorType;

  try
  {
    DoubleReaderType::Pointer doubleImage1Reader = DoubleReaderType::New();
    doubleImage1Reader->SetFileName(inputImage1Name); 
    doubleImage1Reader->Update(); 
    DoubleReaderType::Pointer doubleImage2Reader = DoubleReaderType::New();
    doubleImage2Reader->SetFileName(inputImage2Name); 
    doubleImage2Reader->Update(); 
    ShortReaderType::Pointer shortMaskReader = ShortReaderType::New();
    shortMaskReader->SetFileName(inputMaskName); 
    shortMaskReader->Update(); 
    
    // Estimate the CSF/GM/WM mean and SD intensity. 
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();
    normalisationCalculator->SetInputImage1(doubleImage1Reader->GetOutput());
    normalisationCalculator->SetInputImage2(doubleImage2Reader->GetOutput());
    normalisationCalculator->SetInputMask1(shortMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(shortMaskReader->GetOutput());
    normalisationCalculator->Compute();
    int numberOfClasses = 3; 
    std::cout << "mean brain intensity=" << normalisationCalculator->GetNormalisationMean1() << std::endl; 
    SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
    SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalMeans(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalStds(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType baselineFinalSizes(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalMeans(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalStds(numberOfClasses);
    SimpleKMeansClusteringImageFilterType::ParametersType repeatFinalSizes(numberOfClasses);
    BinariseUsingPaddingImageFilterType::Pointer binariseImageFilter = BinariseUsingPaddingImageFilterType::New();
    MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean1();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean1();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean1();
    binariseImageFilter->SetPaddingValue(0);
    binariseImageFilter->SetInput(shortMaskReader->GetOutput());
    binariseImageFilter->Update();
    multipleDilateImageFilter->SetNumberOfDilations(numberOfDilations);
    multipleDilateImageFilter->SetInput(binariseImageFilter->GetOutput());
    multipleDilateImageFilter->Update();
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(doubleImage1Reader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->SetNumberOfClasses(numberOfClasses); 
    simpleKMeansClusteringImageFilter->Update();
    baselineFinalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    baselineFinalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    baselineFinalSizes = simpleKMeansClusteringImageFilter->GetFinalClassSizes(); 
    std::cout << "baselineFinalStds=" 
              << baselineFinalStds[0] << "," 
              << baselineFinalStds[1] << "," 
              << baselineFinalStds[2] << "," << std::endl; 
    initialMeans[0] = 0.3*normalisationCalculator->GetNormalisationMean2();
    initialMeans[1] = 0.7*normalisationCalculator->GetNormalisationMean2();
    initialMeans[2] = 1.1*normalisationCalculator->GetNormalisationMean2();
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(doubleImage2Reader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->SetNumberOfClasses(numberOfClasses); 
    simpleKMeansClusteringImageFilter->Update();
    repeatFinalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    repeatFinalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    repeatFinalSizes = simpleKMeansClusteringImageFilter->GetFinalClassSizes(); 
    std::cout << "repeatFinalStds=" 
              << repeatFinalStds[0] << "," 
              << repeatFinalStds[1] << "," 
              << repeatFinalStds[2] << "," << std::endl; 
    
    if (rangeSigma < 0.0)
    {
      double baselineRangeSigma = vcl_sqrt((baselineFinalSizes[0]*baselineFinalStds[0]*baselineFinalStds[0]+
                             baselineFinalSizes[1]*baselineFinalStds[1]*baselineFinalStds[1]+
                             baselineFinalSizes[2]*baselineFinalStds[2]*baselineFinalStds[2])/(baselineFinalSizes[0]+baselineFinalSizes[1]+baselineFinalSizes[2])); 
      std::cout << "baselineRangeSigma=" << baselineRangeSigma << std::endl; 
      
      rangeSigma = vcl_sqrt((baselineFinalSizes[0]*baselineFinalStds[0]*baselineFinalStds[0]+
                             baselineFinalSizes[1]*baselineFinalStds[1]*baselineFinalStds[1]+
                             baselineFinalSizes[2]*baselineFinalStds[2]*baselineFinalStds[2]+
                             repeatFinalSizes[0]*repeatFinalStds[0]*repeatFinalStds[0]+
                             repeatFinalSizes[1]*repeatFinalStds[1]*repeatFinalStds[1]+
                             repeatFinalSizes[2]*repeatFinalStds[2]*repeatFinalStds[2])
                            /(baselineFinalSizes[0]+baselineFinalSizes[1]+baselineFinalSizes[2]+repeatFinalSizes[0]+repeatFinalSizes[1]+repeatFinalSizes[2])); 
    }
    std::cout << "rangeSigma=" << rangeSigma << std::endl; 
    
    ImageReaderType::Pointer reader1 = ImageReaderType::New(); 
    reader1->SetFileName(inputImage1Name); 
    reader1->Update(); 
    ImageReaderType::Pointer reader2 = ImageReaderType::New(); 
    reader2->SetFileName(inputImage2Name); 
    reader2->Update(); 
    
    // Clip the intensity as well. 
    itk::ImageRegionIterator<ImageType> iterator1(reader1->GetOutput(), reader1->GetOutput()->GetLargestPossibleRegion()); 
    itk::ImageRegionIterator<ImageType> iterator2(reader2->GetOutput(), reader2->GetOutput()->GetLargestPossibleRegion()); 
    short maxBaselineIntensity = static_cast<short>(baselineFinalMeans[2]+3.0*baselineFinalStds[2]); 
    short maxRepeatIntensity = static_cast<short>(repeatFinalMeans[2]+3.0*repeatFinalStds[2]); 
    for (iterator1.GoToBegin(),iterator2.GoToBegin(); 
         !iterator1.IsAtEnd(); 
         ++iterator1, ++iterator2)
    {
      if (iterator1.Get() > maxBaselineIntensity)
        iterator1.Set(maxBaselineIntensity); 
    }
    
    BilateralImageFilterType::Pointer filter = BilateralImageFilterType::New();
    filter->SetDomainSigma(domainSigma);
    filter->SetRangeSigma(rangeSigma);
    filter->SetInput(reader1->GetOutput()); 
    
    ImageWriterType::Pointer writer = ImageWriterType::New(); 
    writer->SetInput(filter->GetOutput()); 
    writer->SetFileName(outputImageName); 
    writer->Update(); 
  }
  catch(itk::ExceptionObject &err)
  {
    (&err)->Print(std::cerr);
    return EXIT_FAILURE;
  } 
  return EXIT_SUCCESS;   
}


