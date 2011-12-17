#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkMultipleDilateImageFilter.h"
#include "itkSimpleKMeansClusteringImageFilter.h"
#include "itkIndent.h"
#include <stdio.h>

int itkSimpleKMeansClusteringImageFilterTest(int argc, char* argv[])
{
  try
  {
    const double DoubleTolerance = 0.001;
    typedef itk::Image<double, 3> DoubleImageType;
    typedef itk::Image<int, 3> IntImageType;
    typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
    typedef itk::ImageFileReader<IntImageType> IntReaderType;
    typedef itk::ImageFileWriter<IntImageType>  WriterType;
    typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
    typedef itk::SimpleKMeansClusteringImageFilter< DoubleImageType, IntImageType, IntImageType > SimpleKMeansClusteringImageFilterType;
    typedef itk::BinariseUsingPaddingImageFilter<IntImageType,IntImageType> BinariseUsingPaddingImageFilterType;
    typedef itk::MultipleDilateImageFilter<IntImageType> MultipleDilateImageFilterType;
    typedef itk::ImageFileWriter<IntImageType> WriterType;
    
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
    SimpleKMeansClusteringImageFilterType::Pointer simpleKMeansClusteringImageFilter = SimpleKMeansClusteringImageFilterType::New();
    SimpleKMeansClusteringImageFilterType::ParametersType initialMeans(3);
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
    simpleKMeansClusteringImageFilter->SetInitialMeans(initialMeans);
    simpleKMeansClusteringImageFilter->SetInput(baselineNormalisationImageReader->GetOutput());
    simpleKMeansClusteringImageFilter->SetInputMask(multipleDilateImageFilter->GetOutput());
    simpleKMeansClusteringImageFilter->Update();
    
    SimpleKMeansClusteringImageFilterType::ParametersType finalMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType finalStds(3);
    
    finalMeans = simpleKMeansClusteringImageFilter->GetFinalMeans();
    finalStds = simpleKMeansClusteringImageFilter->GetFinalStds();
    
    std::cout << "means=" << finalMeans[0] << "," << finalMeans[1] << "," << finalMeans[2] << std::endl; 
    std::cout << "stds=" << finalStds[0] << "," << finalStds[1] << "," << finalStds[2] << std::endl; 
    
    imageWriter->SetInput (simpleKMeansClusteringImageFilter->GetOutput());
    imageWriter->SetFileName(argv[5]);
    imageWriter->Update();
    
    SimpleKMeansClusteringImageFilterType::ParametersType trueMeans(3);
    SimpleKMeansClusteringImageFilterType::ParametersType trueStds(3);

    trueMeans[0] = atof(argv[6]); 
    trueMeans[1] = atof(argv[7]); 
    trueMeans[2] = atof(argv[8]); 
    trueStds[0] = atof(argv[9]); 
    trueStds[1] = atof(argv[10]); 
    trueStds[2] = atof(argv[11]); 
    
    if ((fabs(finalMeans[0]-trueMeans[0]) > DoubleTolerance) || 
        (fabs(finalMeans[1]-trueMeans[1]) > DoubleTolerance) ||
        (fabs(finalMeans[2]-trueMeans[2]) > DoubleTolerance) ||
        (fabs(finalStds[0]-trueStds[0]) > DoubleTolerance) || 
        (fabs(finalStds[1]-trueStds[1]) > DoubleTolerance) ||
        (fabs(finalStds[2]-trueStds[2]) > DoubleTolerance))
    {
      return EXIT_FAILURE; 
    }
    std::cout << "Passed" << std::endl; 
    
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS; 
}





