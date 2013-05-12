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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMIDASRethresholdingFilter.h>

/**
 * Basic tests for MIDASRethresholdingFilterTest
 */
int itkMIDASRethresholdingFilterTest(int argc, char * argv[])
{
  if (argc != 4)
  {
    std::cerr << "Useage: MIDASRethresholdingFilterTest inImage.nii threshold outImage" << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < argc; i++)
  {
    std::cerr << "argv[" << i << "]=" << argv[i] << std::endl;
  }

  // Declare the types of the images
  const unsigned int Dimension = 3;
  typedef int PixelType;

  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType>  ImageFileReaderType;
  typedef itk::ImageFileWriter<ImageType>  ImageFileWriterType;
  typedef itk::BinaryThresholdImageFilter<ImageType, ImageType> ThresholdingFilterType;
  typedef itk::MIDASRethresholdingFilter<ImageType, ImageType, ImageType> RethresholdingFilterType;

  ImageFileReaderType::Pointer reader = ImageFileReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  ThresholdingFilterType::Pointer thresholdingFilter = ThresholdingFilterType::New();
  thresholdingFilter->SetInput(reader->GetOutput());
  thresholdingFilter->SetInsideValue(1);
  thresholdingFilter->SetOutsideValue(0);
  thresholdingFilter->SetLowerThreshold(atoi(argv[2]));
  thresholdingFilter->SetUpperThreshold(std::numeric_limits<int>::max());
  thresholdingFilter->Update();

  RethresholdingFilterType::Pointer rethresholdingFilter = RethresholdingFilterType::New();
  rethresholdingFilter->SetGreyScaleImageInput(reader->GetOutput());
  rethresholdingFilter->SetBinaryImageInput(thresholdingFilter->GetOutput());
  rethresholdingFilter->SetThresholdedImageInput(thresholdingFilter->GetOutput());
  rethresholdingFilter->SetDownSamplingFactor(4);

  ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
  writer->SetInput(rethresholdingFilter->GetOutput());
  writer->SetFileName(argv[3]);
  writer->Update();

  return EXIT_SUCCESS;
}
