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
#include <itkMIDASImageUpdateClearRegionProcessor.h>

/**
 * Basic tests for itkMIDASImageUpdateClearRegionProcessor
 */
int itkMIDASImageUpdateClearRegionProcessorTest(int argc, char * argv[])
{

  itk::MIDASImageUpdateClearRegionProcessor<unsigned char, 2>::Pointer processor
    = itk::MIDASImageUpdateClearRegionProcessor<unsigned char, 2>::New();

  typedef itk::Image<unsigned char, 2> ImageType;

  ImageType::Pointer image = itk::Image<unsigned char, 2>::New();
  ImageType::IndexType index;
  ImageType::SizeType size;
  ImageType::RegionType region;

  index.Fill(0);
  size.Fill(2);

  region.SetSize(size);
  region.SetIndex(index);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(2);

  ImageType::RegionType regionOfInterest;
  ImageType::SizeType roiSize;
  roiSize.Fill(1);
  index.Fill(1);
  regionOfInterest.SetIndex(index);
  regionOfInterest.SetSize(roiSize);

  processor->SetDestinationImage(image);
  processor->SetDestinationRegionOfInterest(regionOfInterest);
  processor->SetWipeValue(1);
  processor->SetDebug(true);

  std::cerr << "Calling first redo" << std::endl;

  processor->Redo();
  image = processor->GetDestinationImage();

  std::cerr << "Checking first redo" << std::endl;

  index.Fill(1);
  if (image->GetPixel(index) != 1)
  {
    std::cerr << "1. At index=" << index << ", was expecting 1, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (image->GetPixel(index) != 2)
  {
    std::cerr << "2. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Calling first undo" << std::endl;

  processor->Undo();
  image = processor->GetDestinationImage();

  std::cerr << "Checking first undo" << std::endl;

  index.Fill(1);
  if (image->GetPixel(index) != 2)
  {
    std::cerr << "3. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (image->GetPixel(index) != 2)
  {
    std::cerr << "4. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Calling second redo" << std::endl;

  processor->Redo();
  image = processor->GetDestinationImage();

  std::cerr << "Checking second redo" << std::endl;

  index.Fill(1);
  if (image->GetPixel(index) != 1)
  {
    std::cerr << "5. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (image->GetPixel(index) != 2)
  {
    std::cerr << "6. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
