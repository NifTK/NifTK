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
#include <itkVectorVPlusLambdaUImageFilter.h>

/**
 * Basic tests for VectorVPlusLambdaUImageFilterTest
 */
int VectorVPlusLambdaUImageFilterTest(int argc, char * argv[])
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
  
  typedef itk::VectorVPlusLambdaUImageFilter<PixelType, Dimension> FilterType;
  FilterType::Pointer filter = FilterType::New();
  
  // Create two test images
  ImageType::Pointer vImage = ImageType::New();
  ImageType::Pointer uImage = ImageType::New();
  
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
  pixel.Fill(0);
  
  vImage->SetRegions(region);
  vImage->SetSpacing(spacing);
  vImage->SetOrigin(origin);
  vImage->Allocate();
  vImage->FillBuffer(pixel);

  uImage->SetRegions(region);
  uImage->SetSpacing(spacing);
  uImage->SetOrigin(origin);
  uImage->Allocate();
  uImage->FillBuffer(pixel);

  index[0] = 0;
  index[1] = 0;
  
  pixel[0] = 1.2;
  pixel[1] = 3.4;
  
  vImage->SetPixel(index, pixel);

  pixel[0] = 5.6;
  pixel[1] = 7.8;

  uImage->SetPixel(index, pixel);
  
  filter->SetInput(0, vImage);
  filter->SetInput(1, uImage);
  filter->SetLambda(2.1);
  filter->Update();
  
  
  if (fabs(filter->GetOutput()->GetPixel(index)[0] - 12.96) > 0.00001)
    {
      std::cerr << "Expected 12.96, but got:" << filter->GetOutput()->GetPixel(index) << std::endl;
      return EXIT_FAILURE;
    }

  if (fabs(filter->GetOutput()->GetPixel(index)[1] - 19.78) > 0.00001)
    {
      std::cerr << "Expected 19.78, but got:" << filter->GetOutput()->GetPixel(index) << std::endl;
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
