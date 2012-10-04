/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7522 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkMIDASRegionGrowingImageFilter.h"
#include "../itkMIDASSegmentationTestUtils.h"

/**
 * Basic tests for itkMIDASRegionGrowingImageFilter.
 */
int itkMIDASRegionGrowingImageFilterTest2(int argc, char * argv[])
{

  typedef itk::Image<short, 2>              GreyScaleImageType;
  typedef itk::Image<unsigned char, 2>      SegmentationImageType;
  typedef itk::Point<double, 2>             PointType;
  typedef itk::PointSet<double, 2>          PointSetType;
  typedef SegmentationImageType::RegionType RegionType;
  typedef SegmentationImageType::IndexType  IndexType;
  typedef SegmentationImageType::PointType  PointType;
  typedef SegmentationImageType::SizeType   SizeType;
  typedef GreyScaleImageType::PixelType     GreyPixelType;
  typedef itk::MIDASRegionGrowingImageFilter<GreyScaleImageType, SegmentationImageType, PointSetType> FilterType;

  PointType seedPoint;

  SizeType regionSize;
  regionSize.Fill(7);

  IndexType regionIndex;
  regionIndex.Fill(0);

  RegionType region;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  GreyScaleImageType::Pointer greyImage = GreyScaleImageType::New();
  greyImage->SetRegions(region);
  greyImage->Allocate();
  greyImage->FillBuffer(1);

  SegmentationImageType::Pointer contourImage = SegmentationImageType::New();
  contourImage->SetRegions(region);
  contourImage->Allocate();
  contourImage->FillBuffer(0);

  PointSetType::Pointer points = PointSetType::New();

  FilterType::Pointer filter = FilterType::New();

  // Test 1. No seed. Output image should be blank (zero).
  filter->SetLowerThreshold(0);
  filter->SetUpperThreshold(255);
  filter->SetForegroundValue(255);
  filter->SetBackgroundValue(0);
  filter->SetUseRegionOfInterest(false);
  filter->SetProjectSeedsIntoRegion(false);
  filter->SetSeedPoints(*(points));
  filter->SetContourImage(contourImage);
  filter->SetInput(greyImage);
  filter->Update();

  int numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 0, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 2. Single Seed in middle. Grey scale image is concentric squares of equal intensity. No contours.
  greyImage->FillBuffer(0);
  regionSize.Fill(5);
  regionIndex.Fill(1);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<short, 2>(1, greyImage, region);
  regionSize.Fill(3);
  regionIndex.Fill(2);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<short, 2>(2, greyImage, region);

  regionIndex.Fill(3);
  contourImage->TransformIndexToPhysicalPoint(regionIndex, seedPoint);
  points->GetPoints()->InsertElement(0, seedPoint);

  filter->SetLowerThreshold(2);
  filter->SetUpperThreshold(2);
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 9)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 9, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 3. Single seed in middle. Different Threshold.
  filter->SetLowerThreshold(1);
  filter->SetUpperThreshold(2);
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 25)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 25, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 4. Single seed in middle. Upper threshold is prohibitive so no region growing.
  filter->SetLowerThreshold(1);
  filter->SetUpperThreshold(1);
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 0, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 5. Single seed in middle. Test we go right to edge without crashing.
  filter->SetLowerThreshold(0);
  filter->SetUpperThreshold(2);
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 49)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 49, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 6. Create closed contour in Contour image. Region growing should go up to and including contour.
  contourImage->FillBuffer(0);
  regionSize.Fill(5);
  regionIndex.Fill(1);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<unsigned char, 2>(255, contourImage, region);
  regionSize.Fill(3);
  regionIndex.Fill(2);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<unsigned char, 2>(0, contourImage, region);

  // Check we painted the right number of voxels in the contour image.
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, contourImage);
  if (numberOfVoxels != 16)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 16, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  filter->SetContourImage(contourImage);
  filter->Modified();
  filter->Update();

  // Checking the region growing output stopped at the contour.
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 25)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 25, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 7. Creating a single voxel space in the contour. Then whole image should fill.
  regionIndex[0] = 3;
  regionIndex[1] = 1;
  contourImage->SetPixel(regionIndex, 0);
  filter->SetContourImage(contourImage);
  filter->Modified();
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 49)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 49 after contour broken, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 8. Draw a line across the contour image like when doing editing in MIDAS.
  contourImage->FillBuffer(0);
  regionIndex[0] = 3; regionIndex[1] = 0; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 2; regionIndex[1] = 1; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 3; regionIndex[1] = 1; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 1; regionIndex[1] = 2; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 2; regionIndex[1] = 2; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 1; regionIndex[1] = 3; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 0; regionIndex[1] = 4; contourImage->SetPixel(regionIndex, 255);
  regionIndex[0] = 1; regionIndex[1] = 4; contourImage->SetPixel(regionIndex, 255);
  filter->SetContourImage(contourImage);
  filter->SetLowerThreshold(1);
  filter->SetUpperThreshold(2);
  filter->Modified();
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 24)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 24 after contour painted, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }


  // Test 9. Blank contour image· Define region of interest. Then place seed outside region of interest, and check that
  // the seeds are NOT projected, and hence no region growing occurs. i.e. the seeds are outside region.
  contourImage->FillBuffer(0);

  regionIndex.Fill(0);
  contourImage->TransformIndexToPhysicalPoint(regionIndex, seedPoint);
  points->GetPoints()->InsertElement(0, seedPoint);

  regionSize.Fill(3);
  regionIndex.Fill(3);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  filter->SetRegionOfInterest(region);
  filter->SetUseRegionOfInterest(true);
  filter->SetProjectSeedsIntoRegion(false);
  filter->Modified();
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 0 as seed is outside region, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 10. Now project seeds. Default projection distance is 1 though, so seed should not be projected.
  filter->SetProjectSeedsIntoRegion(true);
  filter->Modified();
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 0 as seed is outside region, and outside projection distance, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }


  // Test 11. Seed distance to 2, so should still be no projection, as we are 1 voxel short.
  filter->SetProjectSeedsIntoRegion(true);
  filter->SetMaximumSeedProjectionDistanceInVoxels(2);
  filter->Modified();
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 0 as seed is outside region, and still outside projection distance, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 12. Seed distance to 3, so point should be projected, and region growing kicks in.
  filter->SetProjectSeedsIntoRegion(true);
  filter->SetMaximumSeedProjectionDistanceInVoxels(3);
  filter->Modified();
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 9)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 9 as seed is projected into region, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 13. Try from the other side of the image. So, repeat test 9-12, using seeds on opposite side of image, to check boundary conditions.
  contourImage->FillBuffer(0);

  regionIndex.Fill(6);
  contourImage->TransformIndexToPhysicalPoint(regionIndex, seedPoint);
  points->GetPoints()->InsertElement(0, seedPoint);

  regionSize.Fill(3);
  regionIndex.Fill(1);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  filter->SetRegionOfInterest(region);
  filter->SetUseRegionOfInterest(true);
  filter->SetProjectSeedsIntoRegion(false);
  filter->Modified();
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: Test 13, expected 0 as seed is outside region, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 14. Now project seeds. Default projection distance is 1 though, so seed should not be projected.
  filter->SetProjectSeedsIntoRegion(true);
  filter->SetMaximumSeedProjectionDistanceInVoxels(1);
  filter->Modified();
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: Test 14, expected 0 as seed is outside region, and outside projection distance, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }


  // Test 15. Seed distance to 2, so should still be no projection, as we are 1 voxel short.
  filter->SetProjectSeedsIntoRegion(true);
  filter->SetMaximumSeedProjectionDistanceInVoxels(2);
  filter->Modified();
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 0)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: Test 15, expected 0 as seed is outside region, and still outside projection distance, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 15. Seed distance to 3, so point should be projected, and region growing kicks in.
  filter->SetProjectSeedsIntoRegion(true);
  filter->SetMaximumSeedProjectionDistanceInVoxels(3);
  filter->Modified();
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 9)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: Test 15, expected 9 as seed is projected into region, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
