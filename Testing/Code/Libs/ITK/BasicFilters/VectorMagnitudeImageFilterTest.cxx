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
 * Basic tests for VectorMagnitudeImageFilterTest
 */
int VectorMagnitudeImageFilterTest(int argc, char * argv[])
{
  const unsigned int Dimension = 2;
  typedef float PixelType;
  typedef itk::Vector<PixelType, Dimension> VectorType;
  typedef itk::Image<VectorType, Dimension> ImageType;
  typedef ImageType::IndexType         IndexType;
  typedef ImageType::SizeType          SizeType;
  typedef ImageType::RegionType        RegionType;
  typedef ImageType::SpacingType       SpacingType;
  typedef ImageType::PointType         OriginType;
  typedef ImageType::DirectionType     DirectionType;
  
  typedef itk::VectorMagnitudeImageFilter<PixelType, Dimension> FilterType;
  FilterType::Pointer filter = FilterType::New();
  
  // Create a test image
  ImageType::Pointer image = ImageType::New();

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
  
  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetOrigin(origin);
  image->Allocate();
  image->FillBuffer((float)0.0);
  
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
      std::cerr << "Expected 3.60555, but got:" << image->GetPixel(index) << std::endl;
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
