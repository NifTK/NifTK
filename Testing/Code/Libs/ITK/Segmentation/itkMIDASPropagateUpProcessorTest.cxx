/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7443 $
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
#include "itkPointSet.h"
#include "itkPoint.h"
#include "itkMIDASHelper.h"
#include "itkMIDASPropagateUpProcessor.h"
#include "itkMIDASSegmentationTestUtils.h"

/**
 * Basic tests for itkMIDASPropagateUpProcessor
 */
int itkMIDASPropagateUpProcessorTest(int argc, char * argv[])
{
  typedef itk::Image<unsigned char, 3> SegmentationImageType;
  typedef itk::Image<short, 3>         GreyScaleImageType;
  typedef itk::PointSet<double, 3>     PointSetType;
  typedef itk::Point<double, 3>        PointType;
  typedef SegmentationImageType::RegionType RegionType;
  typedef SegmentationImageType::IndexType  IndexType;
  typedef SegmentationImageType::SizeType   SizeType;
  typedef GreyScaleImageType::PixelType     GreyPixelType;

  IndexType imageIndex;
  imageIndex.Fill(0);
  SizeType  imageSize;
  imageSize.Fill(10);
  RegionType imageRegion;
  imageRegion.SetSize(imageSize);
  imageRegion.SetIndex(imageIndex);

  PointType seed;
  seed.Fill(4);

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

  typedef itk::MIDASPropagateUpProcessor<unsigned char, short, double, 3> ProcessorType;
  ProcessorType::Pointer processor = ProcessorType::New();

  itk::ORIENTATION_ENUM orientation = itk::ORIENTATION_AXIAL;
  int sliceNumber = 4;
  GreyPixelType lowerThreshold = 4;
  GreyPixelType upperThreshold = 6;

  processor->SetGreyScaleImage(greyScaleImage);
  processor->SetDestinationImage(destinationImage);
  processor->SetSeeds(seeds);
  processor->SetLowerThreshold(lowerThreshold);
  processor->SetUpperThreshold(upperThreshold);
  processor->SetOrientationAndSlice(orientation, sliceNumber);
  processor->Redo();

  destinationImage = processor->GetDestinationImage();
  int count = CountVoxelsAboveValue<unsigned char, 3>(0, destinationImage);
  if (count != 120)
  {
    std::cerr << "Expected 120, but got count=" << count << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
