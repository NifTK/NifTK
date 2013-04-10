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
#include "itkMIDASHelper.h"
#include "itkMIDASRegionOfInterestCalculator.h"

/**
 * Basic tests for itkMIDASRegionOfInterestCalculator
 */
int itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest(int argc, char * argv[])
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
  calculator->DebugOn();

  ImageType::Pointer image = ImageType::New();

  SizeType size;
  size.Fill(5);

  IndexType voxelIndex;
  voxelIndex.Fill(0);

  RegionType region;
  region.SetSize(size);
  region.SetIndex(voxelIndex);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);

  // Now set a smaller ROI
  size.Fill(3);
  voxelIndex.Fill(1);
  region.SetSize(size);
  region.SetIndex(voxelIndex);

  std::vector<RegionType> regions = calculator->SplitRegionBySlices(region, image, itk::ORIENTATION_AXIAL);

  if (regions.size() != 3)
  {
    std::cerr << "Expected 3 regions, but got:" << regions.size() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
