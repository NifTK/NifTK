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
#include <itkMeanCurvatureImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

/**
 * Basic tests for MeanCurvatureImageFilter
 */
int MeanCurvatureImageFilterTest(int argc, char * argv[])
{
  if (argc != 7)
  {
     std::cerr << "Usage: MeanCurvatureImageFilterTest inputFile outputFile x y z value";
     return EXIT_FAILURE;
  }

  std::string inputFile = argv[1];
  std::string outputFile = argv[2];
  int x = atoi(argv[3]);
  int y = atoi(argv[4]);
  int z = atoi(argv[5]);
  double expectedValue = atof(argv[6]);

  const unsigned int Dimension = 3;
  typedef float PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::MeanCurvatureImageFilter<ImageType, ImageType> FilterType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFile);

  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(reader->GetOutput());

  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(filter->GetOutput());
  writer->SetFileName(outputFile);
  writer->Update();

  ImageType::IndexType voxelIndex;
  voxelIndex[0] = x;
  voxelIndex[1] = y;
  voxelIndex[2] = z;

  std::cerr << "Testing at:" << voxelIndex << std::endl;

  PixelType actualValue = filter->GetOutput()->GetPixel(voxelIndex);
  if (fabs(actualValue - expectedValue) > 0.000001)
  {
    std::cerr << "Expected:" << expectedValue << " for pixel " << x << ", " << y << ", " << z << ", but got:" << actualValue << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
