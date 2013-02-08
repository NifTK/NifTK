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
int itkMIDASRegionOfInterestCalculatorTest(int argc, char * argv[])
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

  std::string orientation = calculator->GetOrientationString(image);
  if (orientation != "RAI")
  {
    std::cerr << "Expected RAI, but got:" << orientation << " from direction=\n" << image->GetDirection() << std::endl;
    return EXIT_FAILURE;
  }

  int axis = calculator->GetAxis(image, itk::ORIENTATION_AXIAL);
  if (axis != 2)
  {
    std::cerr << "Expected 2, but got:" << axis << std::endl;
    return EXIT_FAILURE;
  }

  axis = calculator->GetAxis(image, itk::ORIENTATION_SAGITTAL);
  if (axis != 0)
  {
    std::cerr << "Expected 0, but got:" << axis << std::endl;
    return EXIT_FAILURE;
  }

  axis = calculator->GetAxis(image, itk::ORIENTATION_CORONAL);
  if (axis != 1)
  {
    std::cerr << "Expected 1, but got:" << axis << std::endl;
    return EXIT_FAILURE;
  }

  int direction = 0;
  direction = calculator->GetPlusOrUpDirection(image, itk::ORIENTATION_AXIAL);
  if (direction != -1)
  {
    std::cerr << "Expected -1, but got:" << direction << std::endl;
    return EXIT_FAILURE;
  }

  direction = calculator->GetPlusOrUpDirection(image, itk::ORIENTATION_SAGITTAL);
  if (direction != -1)
  {
    std::cerr << "Expected -1, but got:" << direction << std::endl;
    return EXIT_FAILURE;
  }

  direction = calculator->GetPlusOrUpDirection(image, itk::ORIENTATION_CORONAL);
  if (direction != -1)
  {
    std::cerr << "Expected -1, but got:" << direction << std::endl;
    return EXIT_FAILURE;
  }

  SizeType size;
  size.Fill(256);

  IndexType voxelIndex;
  voxelIndex.Fill(0);

  RegionType region;
  region.SetSize(size);
  region.SetIndex(voxelIndex);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);

  region = calculator->GetPlusOrUpRegion(image, itk::ORIENTATION_AXIAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 256 || size[1] != 256 || size[2] != 10 || voxelIndex[0] != 0 || voxelIndex[1] != 0 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 256, 256, 10, 0, 0, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetMinusOrDownRegion(image, itk::ORIENTATION_AXIAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 256 || size[1] != 256 || size[2] != 245 || voxelIndex[0] != 0 || voxelIndex[1] != 0 || voxelIndex[2] != 11)
  {
    std::cerr << "Expected 256, 256, 245, 0, 0, 11, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetPlusOrUpRegion(image, itk::ORIENTATION_CORONAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 256 || size[1] != 10 || size[2] != 256 || voxelIndex[0] != 0 || voxelIndex[1] != 0 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 256, 10, 256, 0, 0, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetMinusOrDownRegion(image, itk::ORIENTATION_CORONAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 256 || size[1] != 245 || size[2] != 256 || voxelIndex[0] != 0 || voxelIndex[1] != 11 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 256, 245, 256, 0, 11, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetPlusOrUpRegion(image, itk::ORIENTATION_SAGITTAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 10 || size[1] != 256 || size[2] != 256 || voxelIndex[0] != 0 || voxelIndex[1] != 0 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 10, 256, 256, 0, 0, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetMinusOrDownRegion(image, itk::ORIENTATION_SAGITTAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 245 || size[1] != 256 || size[2] != 256 || voxelIndex[0] != 11 || voxelIndex[1] != 0 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 245, 256, 256, 11, 0, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetSliceRegion(image, itk::ORIENTATION_AXIAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 256 || size[1] != 256 || size[2] != 1 || voxelIndex[0] != 0 || voxelIndex[1] != 0 || voxelIndex[2] != 10)
  {
    std::cerr << "Expected 256, 256, 1, 0, 0, 10, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetSliceRegion(image, itk::ORIENTATION_CORONAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 256 || size[1] != 1 || size[2] != 256 || voxelIndex[0] != 0 || voxelIndex[1] != 10 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 256, 1, 256, 0, 10, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  region = calculator->GetSliceRegion(image, itk::ORIENTATION_SAGITTAL, 10);
  size = region.GetSize();
  voxelIndex = region.GetIndex();

  if (size[0] != 1 || size[1] != 256 || size[2] != 256 || voxelIndex[0] != 10 || voxelIndex[1] != 0 || voxelIndex[2] != 0)
  {
    std::cerr << "Expected 1, 256, 256, 10, 0, 0, but got:\n" << region << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
