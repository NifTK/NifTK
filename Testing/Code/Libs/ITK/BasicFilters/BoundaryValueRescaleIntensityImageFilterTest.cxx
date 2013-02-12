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
#include "itkBoundaryValueRescaleIntensityImageFilter.h"

/**
 * Basic tests for BoundaryValueRescaleIntensityImageFilterTest
 */
int BoundaryValueRescaleIntensityImageFilterTest(int argc, char * argv[])
{

  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::BoundaryValueRescaleIntensityImageFilter<ImageType> FilterType;
  
  // Create an image. 
  ImageType::Pointer image = ImageType::New();
  typedef itk::Index<Dimension>  IndexType;
  typedef itk::Size<Dimension> SizeType;
  typedef itk::ImageRegion<Dimension> RegionType;
  PixelType pixel;
  
  SizeType size;
  size[0] = 2;
  size[1] = 2;

  IndexType index;
  index[0] = 0;
  index[1] = 0;

  RegionType region;
  region.SetIndex( index );
  region.SetSize( size );
  
  image->SetLargestPossibleRegion( region );
  image->SetBufferedRegion( region );
  image->SetRequestedRegion( region );
  image->Allocate();
  image->FillBuffer(0);
    
  index[0] = 0;
  index[1] = 0;
  image->SetPixel(index, 1);

  index[0] = 1;
  index[1] = 0;
  image->SetPixel(index, 1000);

  index[0] = 0;
  index[1] = 1;
  image->SetPixel(index, 10);

  index[0] = 1;
  index[1] = 1;
  image->SetPixel(index, 20);
  
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetOutputMinimum(30);
  filter->SetOutputMaximum(50);
  filter->SetOutputBoundaryValue(2);
  filter->SetInputLowerThreshold(5);  // So we should see effect of thresholding
  filter->SetInputUpperThreshold(25); // So we should see effect of thresholding
  filter->Update();
  
  ImageType::Pointer output = filter->GetOutput();
  
  index[0] = 0;
  index[1] = 0;
  pixel = output->GetPixel(index); 
  if (pixel != 2)
    {
      std::cerr << "Expected 2, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 1;
  index[1] = 0;
  pixel = output->GetPixel(index); 
  if (pixel != 2)
    {
      std::cerr << "Expected 2, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 0;
  index[1] = 1;
  pixel = output->GetPixel(index);   
  if (pixel != 30)
    {
      std::cerr << "Expected 30, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 1;
  index[1] = 1;
  pixel = output->GetPixel(index);   
  if (pixel != 50)
    {
      std::cerr << "Expected 50, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  FilterType::Pointer filter2 = FilterType::New();
  filter2->SetInput(image);
  filter2->SetOutputMinimum(30);
  filter2->SetOutputMaximum(50);
  filter2->SetOutputBoundaryValue(2);
  filter2->Update();

  output = filter2->GetOutput();
    
  index[0] = 0;
  index[1] = 0;
  pixel = output->GetPixel(index);   
  if (pixel != 30)
    {
      std::cerr << "Expected 30, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 1;
  index[1] = 0;
  pixel = output->GetPixel(index); 
  if (pixel != 50)
    {
      std::cerr << "Expected 50, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 0;
  index[1] = 1;
  pixel = output->GetPixel(index); 
  if (pixel != 30)
    {
      std::cerr << "Expected 30, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 1;
  index[1] = 1;
  pixel = output->GetPixel(index); 
  if (pixel != 30)
    {
      std::cerr << "Expected 30, got:" << pixel << std::endl;
      return EXIT_FAILURE;
    }

  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
