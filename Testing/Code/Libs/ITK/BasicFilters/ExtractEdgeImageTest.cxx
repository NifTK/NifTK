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
#include <time.h>
#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkResampleImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMultipleDilateImageFilter.h>
#include <itkMultipleErodeImageFilter.h>
#include <itkImageFileWriter.h>

/**
 * Basic tests to compare ITK and VTK pipelines to extract a slice, and draw a region.
 * Mainly done for performance testing.
 */
int ExtractEdgeImageTest(int argc, char * argv[])
{

  // Declare the types of the images
  const unsigned int InputDimension = 3;
  const unsigned int OutputDimension = 2;
  typedef double PixelType;
  typedef itk::Image<PixelType, InputDimension>   InputImageType;
  typedef itk::Image<PixelType, OutputDimension>  OutputImageType;
  typedef itk::ImageFileReader<InputImageType>    ReaderType;
  typedef itk::ResampleImageFilter<InputImageType, InputImageType, double> ResampleType;
  typedef itk::ExtractImageFilter<InputImageType, OutputImageType> ExtractType;
  typedef itk::BinaryThresholdImageFilter<OutputImageType, OutputImageType> ThresholdType;
  typedef itk::MultipleDilateImageFilter<OutputImageType> DilateType;
  typedef itk::MultipleErodeImageFilter<OutputImageType> ErodeType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  ResampleType::Pointer resampleFilter = ResampleType::New();
  ExtractType::Pointer extractType = ExtractType::New();
  ThresholdType::Pointer thresholdFilter = ThresholdType::New();
  DilateType::Pointer dilateFilter = DilateType::New();
  ErodeType::Pointer erodeFilter = ErodeType::New();
  WriterType::Pointer writer = WriterType::New();

  clock_t tStart;    // The start clock time
  clock_t tFinish;     // The finish clock time

  tStart = clock();
  reader->SetFileName(argv[1]);
  reader->Update();
  tFinish = clock();

  std::cerr << "Read time: " << tFinish - tStart << std::endl;

  InputImageType::SizeType inputSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  InputImageType::SpacingType inputSpacing = reader->GetOutput()->GetSpacing();

  InputImageType::SizeType outputSize;
  for (unsigned int i = 0; i < InputDimension; i++)
  {
    outputSize[i] = inputSize[i] * 5;
  }
  outputSize[0] = 1;
  std::cerr << "outputSize=" << outputSize << std::endl;

  InputImageType::SpacingType outputSpacing = inputSpacing / 5.0;
  inputSpacing[0] = 1;
  std::cerr << "outputSpacing=" << outputSpacing << std::endl;

  InputImageType::IndexType outputOriginVoxel;
  InputImageType::PointType outputOrigin;

  outputOriginVoxel.Fill(0);
  outputOriginVoxel[0] = inputSize[0]/5;
  reader->GetOutput()->TransformIndexToPhysicalPoint(outputOriginVoxel, outputOrigin);
  std::cerr << "outputOrigin=" << outputOrigin << std::endl;

  InputImageType::DirectionType outputDirection = reader->GetOutput()->GetDirection();
  std::cerr << "outputDirection=\n" << outputDirection << std::endl;

  resampleFilter->SetInput(reader->GetOutput());
  extractType->SetInput(resampleFilter->GetOutput());
  thresholdFilter->SetInput(extractType->GetOutput());
  dilateFilter->SetInput(thresholdFilter->GetOutput());
  erodeFilter->SetInput(dilateFilter->GetOutput());
  writer->SetInput(erodeFilter->GetOutput());

  tStart = clock();
  resampleFilter->SetSize(outputSize);
  resampleFilter->SetOutputOrigin(outputOrigin);
  resampleFilter->SetOutputSpacing(outputSpacing);
  resampleFilter->SetOutputDirection(outputDirection);
  resampleFilter->Update();
  tFinish = clock();

  std::cerr << "Resample time: " << tFinish - tStart << std::endl;

  InputImageType::IndexType outputExtractedIndex;
  InputImageType::SizeType outputExtractedSize;
  InputImageType::RegionType outputExtractedRegion;

  outputExtractedSize.Fill(0);
  outputExtractedSize[1] = outputSize[1];
  outputExtractedSize[2] = outputSize[2];
  outputExtractedIndex.Fill(0);
  outputExtractedRegion.SetSize(outputExtractedSize);
  outputExtractedRegion.SetIndex(outputExtractedIndex);

  tStart = clock();
  extractType->SetExtractionRegion(outputExtractedRegion);
  extractType->Update();
  tFinish = clock();

  std::cerr << "Extract time: " << tFinish - tStart << std::endl;

  tStart = clock();
  thresholdFilter->SetOutsideValue(0);
  thresholdFilter->SetInsideValue(1);
  thresholdFilter->SetUpperThreshold(std::numeric_limits<double>::max());
  thresholdFilter->SetLowerThreshold(100);
  thresholdFilter->Update();
  tFinish = clock();

  std::cerr << "Threshold time: " << tFinish - tStart << std::endl;

  tStart = clock();
  dilateFilter->SetNumberOfDilations(1);
  dilateFilter->SetDilateValue(1);
  dilateFilter->Update();
  tFinish = clock();

  std::cerr << "Dilate time: " << tFinish - tStart << std::endl;

  tStart = clock();
  erodeFilter->SetNumberOfErosions(1);
  erodeFilter->Update();
  tFinish = clock();

  std::cerr << "Erode time: " << tFinish - tStart << std::endl;

  tStart = clock();
  writer->SetFileName(argv[2]);
  writer->Update();
  tFinish = clock();

  std::cerr << "Write time: " << tFinish - tStart << std::endl;

  // We are done. Go for coffee.
  return EXIT_SUCCESS;
}
