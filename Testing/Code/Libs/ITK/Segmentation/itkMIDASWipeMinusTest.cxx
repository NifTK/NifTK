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
#include "itkMIDASSegmentationTestUtils.h"
#include "itkMIDASHelper.h"
#include "itkMIDASWipeMinusProcessor.h"

/**
 * Basic tests for itkMIDASWipeMinusTest
 */
int itkMIDASWipeMinusTest(int argc, char * argv[])
{

  typedef itk::MIDASWipeMinusProcessor<unsigned char, 3> ProcessorType;
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

  ProcessorType::Pointer processor = ProcessorType::New();
  processor->DebugOn();
  processor->SetDestinationImage(image);
  processor->SetOrientationAndSlice(itk::ORIENTATION_AXIAL, 1);
  processor->Redo();

  unsigned long int count = CountVoxelsAboveValue<unsigned char, 3>(0, processor->GetDestinationImage());
  if (count != 18)
  {
    std::cerr << "Expected 18, but got:" << count << std::endl;
    return EXIT_FAILURE;
  }

  // Check that the right side got edited.
  voxelIndex.Fill(0);
  unsigned char pixelValue = processor->GetDestinationImage()->GetPixel(voxelIndex);
  if (pixelValue != 1)
  {
    std::cerr << "Expected 1 at:" << voxelIndex << ", but got:" << pixelValue << std::endl;
    return EXIT_FAILURE;
  }

  voxelIndex.Fill(2);
  pixelValue = processor->GetDestinationImage()->GetPixel(voxelIndex);
  if (pixelValue != 0)
  {
    std::cerr << "Expected 0 at:" << voxelIndex << ", but got:" << pixelValue << std::endl;
    return EXIT_FAILURE;
  }

  processor->Undo();
  count = CountVoxelsAboveValue<unsigned char, 3>(0, processor->GetDestinationImage());
  if (count != 27)
  {
    std::cerr << "Expected 27, but got:" << count << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}
