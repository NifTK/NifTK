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
#include "itkVector.h"
#include "itkSetOutputVectorToCurrentPositionFilter.h"

/**
 * Basic tests for SetOutputVectorToCurrentPositionFilterTest
 */
int SetOutputVectorToCurrentPositionFilterTest(int argc, char * argv[])
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
  
  typedef itk::SetOutputVectorToCurrentPositionFilter<PixelType, Dimension> FilterType;
  FilterType::Pointer filter = FilterType::New();
  
  // Create a test image
  ImageType::Pointer image = ImageType::New();

  SizeType size;
  size.Fill(2);
  
  IndexType index;
  index.Fill(0);
  
  RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  
  SpacingType spacing;
  spacing[0] = 1.1;
  spacing[1] = 2.2;
  
  OriginType origin;
  origin[0] = -0.3;
  origin[1] = 10.4;
  
  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetOrigin(origin);
  image->Allocate();
  image->FillBuffer((float)0.0);
  
  filter->SetInput(image);
  filter->Update();
  
  VectorType pixel;
  
  index[0] = 0;
  index[1] = 0;
  image = filter->GetOutput();
  pixel = image->GetPixel(index);
  if (fabs(pixel[0] - -0.3) > 0.00001)
    {
      std::cerr << "Expected -0.3, but got:" << image->GetPixel(index)[0] << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 1;
  index[1] = 0;
  pixel = image->GetPixel(index);
  if (fabs(pixel[0] - 0.8) > 0.00001)
    {
      std::cerr << "Expected 0.8, but got:" << image->GetPixel(index)[0] << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 1;
  index[1] = 1;
  pixel = image->GetPixel(index);

  if (fabs(pixel[1] - 12.6) > 0.00001)
    {
      std::cerr << "Expected 12.6, but got:" << image->GetPixel(index)[1] << std::endl;
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}
