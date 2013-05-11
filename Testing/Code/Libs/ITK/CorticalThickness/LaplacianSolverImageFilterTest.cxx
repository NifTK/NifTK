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
#include <iostream>
#include <memory>
#include <math.h>
#include <itkImage.h>
#include <itkLaplacianSolverImageFilter.h>
#include <ConversionUtils.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>

/**
 * Basic tests for LaplacianSolverImageFilter
 */
int LaplacianSolverImageFilterTest(int argc, char * argv[])
{

  if( argc < 11)
    {
      std::cerr << "Usage   : LaplacianSolverImageFilterTest inputImg outputImg gm wm csf low high epsilon max expectedIterations" << std::endl;
      return 1;
    }

  // Define the dimension of the images
  const unsigned int Dimension = 2;
  typedef double PixelType;
  
  // Read args
  std::string inputImage = argv[1];
  std::string outputImage = argv[2];
  PixelType gmThreshold = (PixelType) niftk::ConvertToInt(argv[3]);
  PixelType wmThreshold = (PixelType) niftk::ConvertToInt(argv[4]);
  PixelType csfThreshold = (PixelType) niftk::ConvertToInt(argv[5]);
  float lowVoltage = (float) niftk::ConvertToDouble(argv[6]);
  float highVoltage = (float) niftk::ConvertToDouble(argv[7]);
  float epsilonThreshold = (float) niftk::ConvertToDouble(argv[8]);
  unsigned long int maxIters = (int) niftk::ConvertToInt(argv[9]);
  unsigned long int expectedIters = (int) niftk::ConvertToInt(argv[10]);
  typedef itk::Image< PixelType, Dimension >   ImageType;
  typedef itk::ImageFileReader< ImageType >    ReaderType;
  ReaderType::Pointer reader  = ReaderType::New();
  reader->SetFileName( inputImage );
  reader->Update();
  
  typedef itk::LaplacianSolverImageFilter<ImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(reader->GetOutput());
  filter->SetLowVoltage(lowVoltage);
  filter->SetHighVoltage(highVoltage);
  filter->SetMaximumNumberOfIterations(maxIters);
  filter->SetEpsilonConvergenceThreshold(epsilonThreshold);
  filter->SetLabelThresholds(gmThreshold, wmThreshold, csfThreshold);
  filter->Update();
  
  // Get an output image.
  typedef unsigned char OutputPixelType;  
  typedef itk::Image<OutputPixelType, Dimension>                OutputImageType; 
  typedef itk::RescaleIntensityImageFilter<ImageType,ImageType> RescaleFilterType;
  typedef itk::CastImageFilter<ImageType, OutputImageType>      CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >               WriterType;
  
  RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();
  WriterType::Pointer writer = WriterType::New();
  
  rescaler->SetInput(filter->GetOutput());
  rescaler->SetOutputMinimum(0);
  rescaler->SetOutputMaximum(255);
  caster->SetInput(rescaler->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->SetFileName(outputImage);
  writer->Update();
  
  // Better actually check stuff, or else it isnt really a unit test, or even a regression test.
  if (filter->GetGreyMatterLabel() != gmThreshold) return EXIT_FAILURE;
  if (filter->GetWhiteMatterLabel() != wmThreshold) return EXIT_FAILURE;
  if (filter->GetExtraCerebralMatterLabel() != csfThreshold) return EXIT_FAILURE;
  if (filter->GetLowVoltage() != lowVoltage) return EXIT_FAILURE;
  if (filter->GetHighVoltage() != highVoltage) return EXIT_FAILURE;
  if (filter->GetMaximumNumberOfIterations() != maxIters) return EXIT_FAILURE;
  if (filter->GetCurrentIteration() != expectedIters) 
    {
    	std::cerr << "Expected:" << expectedIters << " iterations, but it actually took:" << filter->GetCurrentIteration() << std::endl;
    	return EXIT_FAILURE;
    }
  if (filter->GetEpsilonConvergenceThreshold() != epsilonThreshold) return EXIT_FAILURE;
  
  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
