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

int itkIntensityNormalisationCalculatorTest(int argc, char* argv[]) 
{
  typedef itk::Image<double, 3> DoubleImageType;
  typedef itk::Image<int, 3> IntImageType;
  typedef itk::ImageFileReader<DoubleImageType>  DoubleReaderType;
  typedef itk::ImageFileReader<IntImageType>  IntReaderType;
  const double DoubleTolerance = 0.0000000001;
  
  DoubleReaderType::Pointer doubleReader1 = DoubleReaderType::New();
  DoubleReaderType::Pointer doubleReader2 = DoubleReaderType::New();
  IntReaderType::Pointer maskReader1 = IntReaderType::New();
  IntReaderType::Pointer maskReader2 = IntReaderType::New();
  
  doubleReader1->SetFileName(argv[1]);
  doubleReader2->SetFileName(argv[2]);
  maskReader1->SetFileName(argv[3]);
  maskReader2->SetFileName(argv[4]);
  
  typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
  IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();
  
  normalisationCalculator->SetInputImage1(doubleReader1->GetOutput());
  normalisationCalculator->SetInputImage2(doubleReader2->GetOutput());
  normalisationCalculator->SetInputMask1(maskReader1->GetOutput());
  normalisationCalculator->SetInputMask2(maskReader2->GetOutput());
  normalisationCalculator->SetPaddingValue(0);
  normalisationCalculator->Compute();
  std::cout << "mean1=" << normalisationCalculator->GetNormalisationMean1() <<
               ",mean2=" << normalisationCalculator->GetNormalisationMean2() << std::endl;

  double trueMean1 = atof(argv[5]);
  double trueMean2 = atof(argv[6]);
  
  if (fabs(normalisationCalculator->GetNormalisationMean1()-trueMean1) > fabs(DoubleTolerance*trueMean1) ||
      fabs(normalisationCalculator->GetNormalisationMean2()-trueMean2) > fabs(DoubleTolerance*trueMean2))
  {
    return EXIT_FAILURE;
  }
  std::cout << "Test PASSED !" << std::endl;
  
  return  EXIT_SUCCESS;
}

