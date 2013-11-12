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
#include <itkVector.h>
#include <itkVectorMagnitudeImageFilter.h>

/**
 * \file VectorMagnitudeImageFilterTest
 * \brief Basic tests for VectorMagnitudeImageFilterTest
 * As of the migration from ITK 3.20 to 4.3, ITK itself
 * now has a VectorMagnitudeImageFilter, so we are keeping
 * this test and removing the filter within NifTK.
 */
int VectorMagnitudeImageFilterTest(int argc, char * argv[])
{
  const unsigned int Dimension = 2;
  typedef float PixelType;
  typedef itk::Vector<PixelType, Dimension> VectorType;
  typedef itk::Image<VectorType, Dimension> VectorImageType;
  typedef itk::Image<PixelType, Dimension>  ScalarImageType;
  typedef ScalarImageType::IndexType         IndexType;
  typedef ScalarImageType::SizeType          SizeType;
  typedef ScalarImageType::RegionType        RegionType;
  typedef ScalarImageType::SpacingType       SpacingType;
  typedef ScalarImageType::PointType         OriginType;
  typedef ScalarImageType::DirectionType     DirectionType;
  
  typedef itk::VectorMagnitudeImageFilter<VectorImageType, ScalarImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  
  // Create a test image
  VectorImageType::Pointer image = VectorImageType::New();

  SizeType size;
  size.Fill(1);
  
  IndexType index;
  index.Fill(0);
  
  RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  
  SpacingType spacing;
  spacing.Fill(1);
  
  OriginType origin;
  origin.Fill(0);
  
  VectorType pixel;
  pixel[0] = 0;
  pixel[1] = 0; 
  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetOrigin(origin);
  image->Allocate();
  image->FillBuffer(pixel);
  
  index[0] = 0;
  index[1] = 0;
  
  pixel[0] = 1.2;
  pixel[1] = 3.4;
  
  image->SetPixel(index, pixel);
  
  filter->SetInput(image);
  filter->Update();
  
  index[0] = 0;
  index[1] = 0;
  if (fabs(filter->GetOutput()->GetPixel(index) - 3.60555) > 0.00001)
    {
      std::cerr << "Expected 3.60555, but got:" << filter->GetOutput()->GetPixel(index) << std::endl;
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
