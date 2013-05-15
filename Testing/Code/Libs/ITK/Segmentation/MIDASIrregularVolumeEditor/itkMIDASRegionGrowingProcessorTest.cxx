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
#include <itkPoint.h>
#include <itkPointSet.h>
#include <itkMIDASRegionGrowingProcessor.h>
#include <itkMIDASRegionOfInterestCalculator.h>
#include "../itkMIDASSegmentationTestUtils.h"

/**
 * Basic tests for itkMIDASRegionGrowingProcessor.
 */
int itkMIDASRegionGrowingProcessorTest(int argc, char * argv[])
{

  typedef itk::Image<unsigned char, 3> SegmentationImageType;
  typedef itk::Image<short, 3>         GreyScaleImageType;
  typedef itk::PointSet<double, 3>     PointSetType;
  typedef itk::Point<double, 3>        PointType;
  typedef SegmentationImageType::RegionType RegionType;
  typedef SegmentationImageType::IndexType  IndexType;
  typedef SegmentationImageType::SizeType   SizeType;
  typedef GreyScaleImageType::PixelType     GreyPixelType;

  typedef itk::MIDASRegionOfInterestCalculator<unsigned char, 3> CalculatorType;

  IndexType imageIndex;
  imageIndex.Fill(0);
  SizeType  imageSize;
  imageSize.Fill(10);
  RegionType imageRegion;
  imageRegion.SetSize(imageSize);
  imageRegion.SetIndex(imageIndex);

  PointType seed;
  seed.Fill(5);

  PointSetType::Pointer seeds = PointSetType::New();
  seeds->GetPoints()->InsertElement(0, seed);

  SegmentationImageType::Pointer destinationImage = SegmentationImageType::New();
  destinationImage->SetRegions(imageRegion);
  destinationImage->Allocate();
  destinationImage->FillBuffer(0);

  GreyScaleImageType::Pointer greyScaleImage = GreyScaleImageType::New();
  greyScaleImage->SetRegions(imageRegion);
  greyScaleImage->Allocate();
  greyScaleImage->FillBuffer(0);

  for (int z = 0; z < 10; z++)
  {
    for (int y = 0; y < 10; y++)
    {
      for (int x = 0; x < 10; x++)
      {
        imageIndex[0] = x;
        imageIndex[1] = y;
        imageIndex[2] = z;
        greyScaleImage->SetPixel(imageIndex, x);
      }
    }
  }

  int sliceNumber = 5;
  GreyPixelType lowerThreshold = 4;
  GreyPixelType upperThreshold = 6;
  RegionType regionOfInterest;
  IndexType regionOfInterestIndex;
  SizeType regionOfInterestSize;

  regionOfInterestIndex[0] = 0; regionOfInterestIndex[1] = 0; regionOfInterestIndex[2] = 5;
  regionOfInterestSize[0] = 10; regionOfInterestSize[1] = 10; regionOfInterestSize[2] = 1;
  regionOfInterest.SetSize(regionOfInterestSize);
  regionOfInterest.SetIndex(regionOfInterestIndex);

  itk::ORIENTATION_ENUM orientation = itk::ORIENTATION_AXIAL;

  std::cerr << "destination image 1=" << destinationImage << std::endl;

  typedef itk::MIDASRegionGrowingProcessor<GreyScaleImageType, SegmentationImageType, PointSetType> ProcessorType;
  ProcessorType::Pointer processor = ProcessorType::New();
  processor->DebugOn();
  processor->SetRegionOfInterest(regionOfInterest);
  processor->SetSliceNumber(sliceNumber);
  processor->SetOrientation(orientation);
  processor->SetLowerThreshold(lowerThreshold);
  processor->SetUpperThreshold(upperThreshold);
  processor->SetGreyScaleImage(greyScaleImage);
  processor->SetDestinationImage(destinationImage);
  processor->SetSeeds(seeds);
  processor->Execute();

  // Note, sliceNumber must match seed number, otherwise seeds are rejected.

  // 1. Basic one slice region test
  // Threshold 4-6 inclusive,  10x10 axial, should colour in 3x10 pixels
  destinationImage = processor->GetDestinationImage();
  int count = CountVoxelsAboveValue<unsigned char, 3>(0, destinationImage);
  if (count != 30)
  {
    std::cerr << "Expected 30, but got count=" << count << std::endl;
  }
  destinationImage->FillBuffer(0);

  std::cerr << "destination image 2=" << destinationImage << std::endl;

  // 2. Multi slice region test,
  // Threshold 4-6 inclusive, 10x10 axial, from slice 2-7, should be 3x10x6 = 180.

  regionOfInterestIndex[0] = 0; regionOfInterestIndex[1] = 0; regionOfInterestIndex[2] = 2;
  regionOfInterestSize[0] = 10; regionOfInterestSize[1] = 10; regionOfInterestSize[2] = 6;
  regionOfInterest.SetSize(regionOfInterestSize);
  regionOfInterest.SetIndex(regionOfInterestIndex);
  processor->SetRegionOfInterest(regionOfInterest);
  processor->SetDestinationImage(destinationImage);
  processor->Execute();
  destinationImage = processor->GetDestinationImage();

  count = CountVoxelsAboveValue<unsigned char, 3>(0, destinationImage);
  if (count != 180)
  {
    std::cerr << "Expected 180, but got count=" << count << std::endl;
    return EXIT_FAILURE;
  }
  destinationImage->FillBuffer(0);

  std::cerr << "destination image 3=" << destinationImage << std::endl;

  // 3. Wrong seeds test
  // Seed on wrong slice to slice number, reject seed, no output.
  sliceNumber = 4;
  processor->SetSliceNumber(sliceNumber);
  processor->SetDestinationImage(destinationImage);
  processor->Execute();
  destinationImage = processor->GetDestinationImage();
  count = CountVoxelsAboveValue<unsigned char, 3>(0, destinationImage);
  if (count != 0)
  {
    std::cerr << "Expected 30, but got count=" << count << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << "destination image 4=" << destinationImage << std::endl;

  return EXIT_SUCCESS;
}
