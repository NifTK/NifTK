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
