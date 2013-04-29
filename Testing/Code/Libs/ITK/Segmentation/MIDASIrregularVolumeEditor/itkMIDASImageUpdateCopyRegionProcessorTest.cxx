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
#include "itkMIDASImageUpdateCopyRegionProcessor.h"

/**
 * Basic tests for itkMIDASImageUpdateCopyRegionProcessor
 */
int itkMIDASImageUpdateCopyRegionProcessorTest(int argc, char * argv[])
{

  itk::MIDASImageUpdateCopyRegionProcessor<unsigned char, 2>::Pointer processor
    = itk::MIDASImageUpdateCopyRegionProcessor<unsigned char, 2>::New();

  typedef itk::Image<unsigned char, 2> ImageType;


  ImageType::IndexType index;
  ImageType::SizeType size;
  ImageType::RegionType region;

  index.Fill(0);
  size.Fill(2);

  region.SetSize(size);
  region.SetIndex(index);

  ImageType::Pointer destinationImage = itk::Image<unsigned char, 2>::New();
  destinationImage->SetRegions(region);
  destinationImage->Allocate();
  destinationImage->FillBuffer(2);

  ImageType::Pointer sourceImage = itk::Image<unsigned char, 2>::New();
  sourceImage->SetRegions(region);
  sourceImage->Allocate();
  sourceImage->FillBuffer(3);

  ImageType::RegionType regionOfInterest;
  ImageType::SizeType roiSize;
  roiSize.Fill(1);
  index.Fill(1);
  regionOfInterest.SetIndex(index);
  regionOfInterest.SetSize(roiSize);

  processor->SetDestinationImage(destinationImage);
  processor->SetDestinationRegionOfInterest(regionOfInterest);
  processor->SetSourceImage(sourceImage);
  processor->SetSourceRegionOfInterest(regionOfInterest);
  processor->SetDebug(true);

  std::cerr << "Calling first redo" << std::endl;

  processor->Redo();
  destinationImage = processor->GetDestinationImage();

  std::cerr << "Checking first redo" << std::endl;

  index.Fill(1);
  if (destinationImage->GetPixel(index) != 3)
  {
    std::cerr << "1. At index=" << index << ", was expecting 3, but got:" << destinationImage->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (destinationImage->GetPixel(index) != 2)
  {
    std::cerr << "2. At index=" << index << ", was expecting 2, but got:" << destinationImage->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Calling first undo" << std::endl;

  processor->Undo();
  destinationImage = processor->GetDestinationImage();

  std::cerr << "Checking first undo" << std::endl;

  index.Fill(1);
  if (destinationImage->GetPixel(index) != 2)
  {
    std::cerr << "3. At index=" << index << ", was expecting 2, but got:" << destinationImage->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (destinationImage->GetPixel(index) != 2)
  {
    std::cerr << "4. At index=" << index << ", was expecting 2, but got:" << destinationImage->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Calling second redo" << std::endl;

  processor->Redo();
  destinationImage = processor->GetDestinationImage();

  std::cerr << "Checking second redo" << std::endl;

  index.Fill(1);
  if (destinationImage->GetPixel(index) != 3)
  {
    std::cerr << "5. At index=" << index << ", was expecting 3, but got:" << destinationImage->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (destinationImage->GetPixel(index) != 2)
  {
    std::cerr << "6. At index=" << index << ", was expecting 2, but got:" << destinationImage->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
