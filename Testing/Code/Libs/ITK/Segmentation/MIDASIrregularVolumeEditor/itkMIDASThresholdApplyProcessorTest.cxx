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
#include "itkImage.h"
#include "../itkMIDASSegmentationTestUtils.h"
#include "itkMIDASThresholdApplyProcessor.h"

/**
 * Basic tests for MIDASThresholdApplyProcessor
 */
int itkMIDASThresholdApplyProcessorTest(int argc, char * argv[])
{

  typedef itk::Image<unsigned char, 2> ImageType;
  typedef ImageType::RegionType RegionType;
  typedef ImageType::IndexType IndexType;
  typedef ImageType::SizeType SizeType;
  typedef itk::MIDASThresholdApplyProcessor<unsigned char, 2> ProcessorType;
  typedef ProcessorType::Pointer ProcessorPointer;

  IndexType voxelIndex;
  SizeType size;
  RegionType region;

  size.Fill(5);
  voxelIndex.Fill(0);
  region.SetSize(size);
  region.SetIndex(voxelIndex);

  ImageType::Pointer sourceImage = ImageType::New();
  sourceImage->SetRegions(region);
  sourceImage->Allocate();
  sourceImage->FillBuffer(0);

  ImageType::Pointer destinationImage = ImageType::New();
  destinationImage->SetRegions(region);
  destinationImage->Allocate();
  destinationImage->FillBuffer(0);

  // fill test region, like some region growing output, that will be copied from source to destination.
  for (int x = 1; x < 3; x++)
  {
    for (int y = 1; y < 3; y++)
    {
      voxelIndex[0] = x; voxelIndex[1] = y;
      sourceImage->SetPixel(voxelIndex, 1);
    }
  }

  // Check inputs
  int count = CountVoxelsAboveValue<unsigned char, 2>((unsigned char)0, sourceImage.GetPointer());
  if (count != 4)
  {
    std::cerr << "1. Expected 4, but got=" << count << std::endl;
  }
  count = CountVoxelsAboveValue<unsigned char, 2>((unsigned char)0, destinationImage.GetPointer());
  if (count != 0)
  {
    std::cerr << "2. Expected 0, but got=" << count << std::endl;
  }

  // connect filter
  ProcessorPointer processor = ProcessorType::New();
  processor->SetSourceImage(sourceImage);
  processor->SetDestinationImage(destinationImage);
  processor->CalculateRegionOfInterest();
  processor->Redo();

  sourceImage = processor->GetSourceImage();
  destinationImage = processor->GetDestinationImage();

  count = CountVoxelsAboveValue<unsigned char, 2>((unsigned char)0, sourceImage.GetPointer());
  if (count != 0)
  {
    std::cerr << "3. Expected 0, but got=" << count << std::endl;
  }
  count = CountVoxelsAboveValue<unsigned char, 2>((unsigned char)0, destinationImage.GetPointer());
  if (count != 4)
  {
    std::cerr << "4. Expected 4, but got=" << count << std::endl;
  }

  processor->Undo();

  sourceImage = processor->GetSourceImage();
  destinationImage = processor->GetDestinationImage();

  count = CountVoxelsAboveValue<unsigned char, 2>((unsigned char)0, sourceImage.GetPointer());
  if (count != 4)
  {
    std::cerr << "5. Expected 4, but got=" << count << std::endl;
  }
  count = CountVoxelsAboveValue<unsigned char, 2>((unsigned char)0, destinationImage.GetPointer());
  if (count != 0)
  {
    std::cerr << "6. Expected 0, but got=" << count << std::endl;
  }

  return EXIT_SUCCESS;
}
