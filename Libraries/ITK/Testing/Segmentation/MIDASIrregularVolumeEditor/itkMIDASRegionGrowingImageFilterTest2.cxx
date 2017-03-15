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
#include <itkImageFileWriter.h>
#include <itkPoint.h>
#include <itkPointSet.h>
#include <itkMIDASThresholdingRegionGrowingImageFilter.h>
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
  typedef itk::MIDASThresholdingRegionGrowingImageFilter<GreyScaleImageType, SegmentationImageType, PointSetType> FilterType;

  PointType seedPoint;

  SizeType regionSize;
  regionSize.Fill(9);

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
  filter->SetManualContourImage(contourImage);
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
  regionSize.Fill(7);
  regionIndex.Fill(1);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<short, 2>(1, greyImage, region);
  regionSize.Fill(5);
  regionIndex.Fill(2);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<short, 2>(2, greyImage, region);
  regionSize.Fill(3);
  regionIndex.Fill(3);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<short, 2>(3, greyImage, region);

  regionIndex.Fill(4);
  contourImage->TransformIndexToPhysicalPoint(regionIndex, seedPoint);
  points->GetPoints()->InsertElement(0, seedPoint);

  /*
   * Grey image:
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  8  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  7  |   0   |   1   |   1   |   1   |   1   |   1   |   1   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  6  |   0   |   1   |   2   |   2   |   2   |   2   |   2   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  5  |   0   |   1   |   2   |   3   |   3   |   3   |   2   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  4  |   0   |   1   |   2   |   3   |   3   |   3   |   2   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  3  |   0   |   1   |   2   |   3   |   3   |   3   |   2   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  2  |   0   |   1   |   2   |   2   |   2   |   2   |   2   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  1  |   0   |   1   |   1   |   1   |   1   |   1   |   1   |   1   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  0  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+

         0       1       2       3       4       5       6       7       8

      Seed in the middle, at (4, 4).
  */

  filter->SetLowerThreshold(3);
  filter->SetUpperThreshold(3);
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 9)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 9, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 3. Single seed in middle. Different Threshold.
  filter->SetLowerThreshold(2);
  filter->SetUpperThreshold(3);
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
  filter->SetUpperThreshold(3);
  filter->Update();
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 81)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 81, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 6. Create closed contour in Contour image. Region growing should go up to and including contour.
  contourImage->FillBuffer(0);
  regionSize.Fill(7);
  regionIndex.Fill(1);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<unsigned char, 2>(255, contourImage, region);
  regionSize.Fill(3);
  regionIndex.Fill(3);
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  FillImageRegionWithValue<unsigned char, 2>(0, contourImage, region);
  filter->SetManualContourImageBorderValue(255);

  /*
   * Contour image with contour point set:
   *
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  8  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  7  |   0   |  255  |  255  |  255  |  255  |  255  |  255  |  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------o---o---+---o---+---o---+---o---+---o---o-------+-------+
     |       |       |       |       |       |       |       |       |       |
  6  |   0   |  255  o  255  |  255  |  255  |  255  |  255  o  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  5  |   0   |  255  o  255  |   0   |   0   |   0   |  255  o  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  4  |   0   |  255  o  255  |   0   |   0   |   0   |  255  o  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  3  |   0   |  255  o  255  |   0   |   0   |   0   |  255  o  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  2  |   0   |  255  o  255  |  255  |  255  |  255  |  255  o  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------o---o---+---o---+---o---+---o---+---o---o-------+-------+
     |       |       |       |       |       |       |       |       |       |
  1  |   0   |  255  |  255  |  255  |  255  |  255  |  255  |  255  |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  0  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+

         0       1       2       3       4       5       6       7       8
  */

  // Check we painted the right number of voxels in the contour image.
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, contourImage);
  if (numberOfVoxels != 40)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 40, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }


  std::vector<itk::PolyLineParametricPath<2>::Pointer> contourPaths;
  {
    itk::PolyLineParametricPath<2>::Pointer contourPath = itk::PolyLineParametricPath<2>::New();

    double contourPoints[][2] = {
      {1.5, 1.5}, {2.0, 1.5}, {3.0, 1.5}, {4.0, 1.5}, {5.0, 1.5}, {6.0, 1.5},
      {6.5, 1.5}, {6.5, 2.0}, {6.5, 3.0}, {6.5, 4.0}, {6.5, 5.0}, {6.5, 6.0},
      {6.5, 6.5}, {6.0, 6.5}, {5.0, 6.5}, {4.0, 6.5}, {3.0, 6.5}, {2.0, 6.5},
      {1.5, 6.5}, {1.5, 6.0}, {1.5, 5.0}, {1.5, 4.0}, {1.5, 3.0}, {1.5, 2.0},
      {1.5, 1.5}
    };
    for (int i = 0; i < sizeof(contourPoints) / sizeof(contourPoints[0]); ++i)
    {
      itk::ContinuousIndex<float, 2> index;
      index[0] = contourPoints[i][0];
      index[1] = contourPoints[i][1];

      PointType point;
      contourImage->TransformContinuousIndexToPhysicalPoint(index, point);

      itk::PolyLineParametricPath<2>::ContinuousIndexType idx;
      idx.CastFrom(point);

      contourPath->AddVertex(idx);
    }

    contourPaths.push_back(contourPath);
  }


  filter->SetManualContours(&contourPaths);
  filter->SetManualContourImage(contourImage);
  filter->Modified();
  filter->Update();

  // Checking the region growing output stopped at the contour.
  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 25)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 25, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 7. Creating a gap in the contour. Then whole image should fill.
  regionIndex[0] = 4;
  regionIndex[1] = 1;
  contourImage->SetPixel(regionIndex, 0);
  regionIndex[0] = 4;
  regionIndex[1] = 2;
  contourImage->SetPixel(regionIndex, 0);
  filter->SetManualContourImage(contourImage);
  filter->Modified();
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 81)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 81 after contour broken, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }

  // Test 8. Draw a line across the contour image like when doing editing in MIDAS.

  /*
   * Contour image with contour point set:
   *
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  8  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  7  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  6  |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  5  |  255  |  255  |  255  |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     o---o---+---o---o-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  4  |  255  |  255  o  255  |   0   |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  3  |   0   |  255  o  255  |  255  |   0   |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------o---o---o-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  2  |   0   |  255  |  255  o  255  |  255  |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------o---o---o-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  1  |   0   |   0   |  255  |  255  o  255  |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
     |       |       |       |       |       |       |       |       |       |
  0  |   0   |   0   |   0   |  255  o  255  |   0   |   0   |   0   |   0   |
     |       |       |       |       |       |       |       |       |       |
     +-------+-------+-------+-------o-------+-------+-------+-------+-------+

         0       1       2       3       4       5       6       7       8
  */

  contourImage->FillBuffer(0);
  {
    int lineVoxelIndices[][2] = {
      {0, 5},
      {1, 5},
      {2, 5},
      {0, 4},
      {1, 4},
      {2, 4},
      {1, 3},
      {2, 3},
      {3, 3},
      {1, 2},
      {2, 2},
      {3, 2},
      {4, 2},
      {2, 1},
      {3, 1},
      {4, 1},
      {3, 0},
      {4, 0},
    };
    for (int i = 0; i < sizeof(lineVoxelIndices) / sizeof(lineVoxelIndices[0]); ++i)
    {
      IndexType index;
      index[0] = lineVoxelIndices[i][0];
      index[1] = lineVoxelIndices[i][1];
      contourImage->SetPixel(index, 255);
    }
  }

  filter->SetManualContourImage(contourImage);

  contourPaths.clear();
  {
    itk::PolyLineParametricPath<2>::Pointer contourPath = itk::PolyLineParametricPath<2>::New();

    double contourPoints[][2] = {
      {-0.5, 4.5}, {0.0, 4.5}, {1.0, 4.5},
      {1.5, 4.5}, {1.5, 4.0}, {1.5, 3.0},
      {1.5, 2.5}, {2.0, 2.5},
      {2.5, 2.5}, {2.5, 2.0},
      {2.5, 1.5}, {3.0, 1.5},
      {3.5, 1.5}, {3.5, 1.0}, {3.5, 0.0},
      {3.5, -0.5},
    };
    for (int i = 0; i < sizeof(contourPoints) / sizeof(contourPoints[0]); ++i)
    {
      itk::ContinuousIndex<float, 2> index;
      index[0] = contourPoints[i][0];
      index[1] = contourPoints[i][1];

      PointType point;
      contourImage->TransformContinuousIndexToPhysicalPoint(index, point);

      itk::PolyLineParametricPath<2>::ContinuousIndexType idx;
      idx.CastFrom(point);

      contourPath->AddVertex(idx);
    }

    contourPaths.push_back(contourPath);
  }


  filter->SetManualContours(&contourPaths);

  filter->SetLowerThreshold(1);
  filter->SetUpperThreshold(3);
  filter->Modified();
  filter->Update();

  numberOfVoxels = CountVoxelsAboveValue<unsigned char, 2>(0, filter->GetOutput());
  if (numberOfVoxels != 42)
  {
    std::cerr << "itkMIDASRegionGrowingImageFilterTest2: expected 42 after contour painted, but got " << numberOfVoxels << std::endl;
    return EXIT_FAILURE;
  }


  // Test 9. Blank contour image. Define region of interest. Then place seed outside region of interest, and check that
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
