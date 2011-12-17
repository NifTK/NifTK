/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkMIDASRethresholdingFilter.h"

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
  rethresholdingFilter->SetDownSamplingFactor(4);

  ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
  writer->SetInput(rethresholdingFilter->GetOutput());
  writer->SetFileName(argv[3]);
  writer->Update();

  return EXIT_SUCCESS;
}
