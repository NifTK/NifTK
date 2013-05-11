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
#include <itkMIDASMaskByRegionImageFilter.h>

/**
 * Basic tests for itkMIDASMaskByRegionFilter.
 * The specification is in the header file "itkMIDASByRegionFilter.h"
 */
int itkMIDASMaskByRegionFilterTest(int argc, char * argv[])
{

  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::MIDASMaskByRegionImageFilter<ImageType, ImageType> FilterType;

  // Create the first image.
  typedef ImageType::IndexType        IndexType;
  typedef ImageType::SizeType         SizeType;
  typedef ImageType::RegionType       RegionType;

  //Create a 8 pixel image for the 3 inputs.
  SizeType imageSize;
  imageSize[0] = 8;
  imageSize[1] = 1;

  IndexType startIndex;
  startIndex[0] = 0;
  startIndex[1] = 0;

  RegionType imageRegion;
  imageRegion.SetIndex(startIndex);
  imageRegion.SetSize(imageSize);

  ImageType::Pointer inputImage = ImageType::New();
  inputImage->SetLargestPossibleRegion(imageRegion);
  inputImage->SetBufferedRegion(imageRegion);
  inputImage->SetRequestedRegion(imageRegion);
  inputImage->Allocate();
  inputImage->FillBuffer(0);

  ImageType::Pointer additionsImage = ImageType::New();
  additionsImage->SetLargestPossibleRegion(imageRegion);
  additionsImage->SetBufferedRegion(imageRegion);
  additionsImage->SetRequestedRegion(imageRegion);
  additionsImage->Allocate();
  additionsImage->FillBuffer(0);

  ImageType::Pointer editsImage = ImageType::New();
  editsImage->SetLargestPossibleRegion(imageRegion);
  editsImage->SetBufferedRegion(imageRegion);
  editsImage->SetRequestedRegion(imageRegion);
  editsImage->Allocate();
  editsImage->FillBuffer(0);

  // Set the input data.
  IndexType pixelIndex;
  pixelIndex[1] = 0;
  pixelIndex[0] = 0; inputImage->SetPixel(pixelIndex, 0); additionsImage->SetPixel(pixelIndex, 0); editsImage->SetPixel(pixelIndex, 0);
  pixelIndex[0] = 1; inputImage->SetPixel(pixelIndex, 0); additionsImage->SetPixel(pixelIndex, 0); editsImage->SetPixel(pixelIndex, 1);
  pixelIndex[0] = 2; inputImage->SetPixel(pixelIndex, 0); additionsImage->SetPixel(pixelIndex, 1); editsImage->SetPixel(pixelIndex, 0);
  pixelIndex[0] = 3; inputImage->SetPixel(pixelIndex, 0); additionsImage->SetPixel(pixelIndex, 1); editsImage->SetPixel(pixelIndex, 1);
  pixelIndex[0] = 4; inputImage->SetPixel(pixelIndex, 1); additionsImage->SetPixel(pixelIndex, 0); editsImage->SetPixel(pixelIndex, 0);
  pixelIndex[0] = 5; inputImage->SetPixel(pixelIndex, 1); additionsImage->SetPixel(pixelIndex, 0); editsImage->SetPixel(pixelIndex, 1);
  pixelIndex[0] = 6; inputImage->SetPixel(pixelIndex, 1); additionsImage->SetPixel(pixelIndex, 1); editsImage->SetPixel(pixelIndex, 0);
  pixelIndex[0] = 7; inputImage->SetPixel(pixelIndex, 1); additionsImage->SetPixel(pixelIndex, 1); editsImage->SetPixel(pixelIndex, 1);

  // First test, with no additionsImage and editsImage
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(inputImage);
  filter->SetNumberOfThreads(1);
  filter->Update();

  ImageType::Pointer outputImage = filter->GetOutput();

  for (unsigned int i = 0; i < 8; i++)
  {
    pixelIndex[0] = i;
    if (i < 4 && outputImage->GetPixel(pixelIndex) != 0)
    {
      std::cerr << "First test, i=" << i << " failed as output was " << outputImage->GetPixel(pixelIndex) << std::endl;
      return EXIT_FAILURE;
    }
    if (i >= 4 && outputImage->GetPixel(pixelIndex) != 1)
    {
      std::cerr << "First test, i=" << i << " failed as output was " << outputImage->GetPixel(pixelIndex) << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Second test, connect the other two images
  filter->SetInput(1, additionsImage);
  filter->SetInput(2, editsImage);
  filter->Update();

  for (unsigned int i = 0; i < 8; i++)
  {
    pixelIndex[0] = i;
    if ((i == 2 || i == 4 || i == 6) && outputImage->GetPixel(pixelIndex) != 1)
    {
      std::cerr << "Second test, i=" << i << " failed as output was " << outputImage->GetPixel(pixelIndex) << std::endl;
      return EXIT_FAILURE;
    }
    if ((i == 0 || i == 1 || i == 3 || i == 5 || i == 7) && outputImage->GetPixel(pixelIndex) != 0)
    {
      std::cerr << "Second test, i=" << i << " failed as output was " << outputImage->GetPixel(pixelIndex) << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
