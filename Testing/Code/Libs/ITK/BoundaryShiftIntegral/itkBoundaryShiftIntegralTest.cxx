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

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkIndent.h"
#include <stdio.h>

// Command line arguments:
// 1. baseline image for intensity normalisation.
// 2. baseline mask for intensity normalisation.
// 3. repeat image for intensity normalisation.
// 4. repeat mask for intensity normalisation.
// 5. baseline image for BSI.
// 6. baseline mask for BSI.
// 7. repeat image for BSI.
// 8. repeat mask for BSI.
// 9. number of erosion.
// 10. number of dilation.
// 11. lower intensity in the BSI window.
// 12. upper intensity in the BSI window.
// 13. Sub ROI to intersect with the BSI XOR region. 
// 14. gold standard BSI to be compared.


int itkBoundaryShiftIntegralTest(int argc, char* argv[])
{
  try
  {
    const double DoubleTolerance = 0.00001;
    
    typedef itk::Image<double, 3> DoubleImageType;
    typedef itk::Image<int, 3> IntImageType;
  
    typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
    typedef itk::ImageFileReader<IntImageType> IntReaderType;
    typedef itk::ImageFileWriter<IntImageType>  WriterType;
    typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
    typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;
    
    DoubleReaderType::Pointer baselineNormalisationImageReader = DoubleReaderType::New();
    DoubleReaderType::Pointer repeatNormalisationImageReader = DoubleReaderType::New();
    IntReaderType::Pointer baselineNormalisationMaskReader = IntReaderType::New();
    IntReaderType::Pointer repeatNormalisationMaskReader = IntReaderType::New();
    
    baselineNormalisationImageReader->SetFileName(argv[1]);
    baselineNormalisationMaskReader->SetFileName(argv[2]);
    repeatNormalisationImageReader->SetFileName(argv[3]);
    repeatNormalisationMaskReader->SetFileName(argv[4]);
    
    IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();
    
    normalisationCalculator->SetInputImage1(baselineNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputImage2(repeatNormalisationImageReader->GetOutput());
    normalisationCalculator->SetInputMask1(baselineNormalisationMaskReader->GetOutput());
    normalisationCalculator->SetInputMask2(repeatNormalisationMaskReader->GetOutput());
    normalisationCalculator->Compute();
    std::cout.precision(16);
    std::cout << "mean1=" << normalisationCalculator->GetNormalisationMean1() <<
                 ",mean2=" << normalisationCalculator->GetNormalisationMean2() << std::endl;
  
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
    bsiFilter->SetLowerCutoffValue(atof(argv[11]));
    bsiFilter->SetUpperCutoffValue(atof(argv[12]));
    if (strlen(argv[13]) > 0)
    {
      subROIMaskReader->SetFileName(argv[13]);
      bsiFilter->SetSubROIMask(subROIMaskReader->GetOutput());
    }
    bsiFilter->Compute();
    std::cout << "BSI=" << bsiFilter->GetBoundaryShiftIntegral() << std::endl;
    
    double trueBSI = atof(argv[14]);
    
    if (fabs(bsiFilter->GetBoundaryShiftIntegral()-trueBSI) > fabs(DoubleTolerance*trueBSI))
    {
      std::cout << "Expected: " << trueBSI 
                << ", Actual: " << bsiFilter->GetBoundaryShiftIntegral() 
                << ", Diff: " << bsiFilter->GetBoundaryShiftIntegral()-trueBSI 
                << ", Tol: " << DoubleTolerance*trueBSI << std::endl;
      return EXIT_FAILURE;
    }
    
    std::cout << "Test PASSED !" << std::endl;
  }
  catch (itk::ExceptionObject& itkException)
  {
    std::cerr << "Error: " << itkException << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}


