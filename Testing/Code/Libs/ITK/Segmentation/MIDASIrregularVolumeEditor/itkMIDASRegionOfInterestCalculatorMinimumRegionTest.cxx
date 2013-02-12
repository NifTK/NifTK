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
#include "itkImageRegionIterator.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionOfInterestCalculator.h"

/**
 * Basic tests for itkMIDASRegionOfInterestCalculator
 */
int itkMIDASRegionOfInterestCalculatorMinimumRegionTest(int argc, char * argv[])
{

  typedef itk::Image<unsigned char, 3> ImageType;
  typedef itk::MIDASRegionOfInterestCalculator<unsigned char, 3> CalculatorType;
  typedef ImageType::RegionType RegionType;
  typedef ImageType::SizeType   SizeType;
  typedef ImageType::IndexType  IndexType;

  /**********************************************************
   * Normal, default ITK image, should be RAI
   * i.e.
   * 1 0 0
   * 0 1 0 == RAI
   * 0 0 1
   **********************************************************/
  CalculatorType::Pointer calculator = CalculatorType::New();
  //calculator->DebugOn();

  ImageType::Pointer image = ImageType::New();

  SizeType size;
  size.Fill(10);

  IndexType voxelIndex;
  voxelIndex.Fill(0);

  RegionType region;
  region.SetSize(size);
  region.SetIndex(voxelIndex);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);

  // Fill with some data.
  voxelIndex[0] = 2;
  voxelIndex[1] = 3;
  voxelIndex[2] = 4;

  size[0] = 3;
  size[1] = 4;
  size[2] = 5;

  region.SetSize(size);
  region.SetIndex(voxelIndex);

  itk::ImageRegionIterator<ImageType> iterator(image, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(1);
  }

  region = calculator->GetMinimumRegion(image, 0);
  voxelIndex = region.GetIndex();
  size = region.GetSize();

  if (voxelIndex[0] != 2)
  {
    std::cerr << "Expected 2 but got:" << voxelIndex[0] << std::endl;
    return EXIT_FAILURE;
  }
  if (voxelIndex[1] != 3)
  {
    std::cerr << "Expected 3 but got:" << voxelIndex[1] << std::endl;
    return EXIT_FAILURE;
  }
  if (voxelIndex[2] != 4)
  {
    std::cerr << "Expected 4 but got:" << voxelIndex[2] << std::endl;
    return EXIT_FAILURE;
  }
  if (size[0] != 3)
  {
    std::cerr << "Expected 3 but got:" << size[0] << std::endl;
    return EXIT_FAILURE;
  }
  if (size[1] != 4)
  {
    std::cerr << "Expected 4 but got:" << size[1] << std::endl;
    return EXIT_FAILURE;
  }
  if (size[2] != 5)
  {
    std::cerr << "Expected 5 but got:" << size[2] << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
