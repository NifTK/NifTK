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
#include "itkMIDASRetainMarksNoThresholdingProcessor.h"
#include "itkMIDASHelper.h"

/**
 * Basic tests for itkMIDASRetailMarksNoThresholdingProcessor
 */
int itkMIDASRetainMarksNoThresholdingTest(int argc, char * argv[])
{

  typedef itk::MIDASRetainMarksNoThresholdingProcessor<unsigned char, 3> ProcessorType;
  typedef itk::Image<unsigned char, 3> ImageType;
  typedef ImageType::RegionType RegionType;
  typedef ImageType::IndexType IndexType;
  typedef ImageType::SizeType SizeType;

  ImageType::Pointer image = ImageType::New();

  SizeType size;
  size.Fill(3);

  IndexType voxelIndex;
  voxelIndex.Fill(0);

  RegionType region;
  region.SetSize(size);
  region.SetIndex(voxelIndex);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(1);

  RegionType sourceRegion;
  size.Fill(1);
  voxelIndex.Fill(0);
  sourceRegion.SetSize(size);
  sourceRegion.SetIndex(voxelIndex);

  RegionType targetRegion;
  size.Fill(1);
  voxelIndex.Fill(2);
  targetRegion.SetSize(size);
  targetRegion.SetIndex(voxelIndex);

  FillImageRegionWithValue<unsigned char, 3>((unsigned char)2, image, sourceRegion);
  FillImageRegionWithValue<unsigned char, 3>((unsigned char)3, image, targetRegion);
  int counter = CountVoxelsAboveValue<unsigned char, 3>(2, image);
  std::cerr << "Before counter=" << counter << std::endl;

  ProcessorType::Pointer processor = ProcessorType::New();
  processor->DebugOn();
  processor->SetSourceImage(image);
  processor->SetDestinationImage(image);
  processor->SetSlices(itk::ORIENTATION_AXIAL, 0, 2);
  processor->Redo();

  ImageType* outputImage = processor->GetDestinationImage();

  counter = CountVoxelsAboveValue<unsigned char, 3>(2, outputImage);
  if (counter != 0)
  {
    std::cerr << "Expected zero, but got:" << counter << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
