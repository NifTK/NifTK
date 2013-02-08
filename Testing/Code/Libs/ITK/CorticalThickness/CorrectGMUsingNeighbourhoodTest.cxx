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

#include "itkImage.h"
#include "itkCorrectGMUsingNeighbourhoodFilter.h"

/**
 * Test the CorrectGMUsingneighbourhood filter
 */
int CorrectGMUsingNeighbourhoodTest(int, char* []) 
{
  const unsigned int Dimension = 2;
  typedef float PixelType;
  typedef itk::Image<PixelType, Dimension>  InputImageType;
  typedef InputImageType::IndexType    IndexType;
  typedef InputImageType::SizeType     SizeType;
  typedef InputImageType::SpacingType  SpacingType;
  typedef InputImageType::RegionType   RegionType;
  typedef InputImageType::Pointer      ImageTypePointer;

  // Create an image
  ImageTypePointer segmentedImage = InputImageType::New();
  
  // Define their size, and start index
  SizeType size;
  size[0] = 5;
  size[1] = 5;

  IndexType index;
  index[0] = 0;
  index[1] = 0;

  RegionType region;
  region.SetIndex( index );
  region.SetSize( size );
  
  SpacingType spacing;
  spacing[0] = 1.1;
  spacing[1] = 1.2;
  
  // Initialize Image 
  segmentedImage->SetLargestPossibleRegion( region );
  segmentedImage->SetBufferedRegion( region );
  segmentedImage->SetRequestedRegion( region );
  segmentedImage->SetSpacing(spacing);
  segmentedImage->Allocate();
  segmentedImage->FillBuffer(0);

  // c c c c c
  // c g g c c 
  // c g w g c
  // c g c g c
  // c c c c c 
  
  // So 2 should get reclassified, at position (3,1) and (2,3)
  // c=0, g=1, w=2
  
  index[0] = 1;
  index[1] = 1;
  segmentedImage->SetPixel(index, 1);

  index[0] = 2;
  index[1] = 1;
  segmentedImage->SetPixel(index, 1);

  index[0] = 1;
  index[1] = 2;
  segmentedImage->SetPixel(index, 1);

  index[0] = 2;
  index[1] = 2;
  segmentedImage->SetPixel(index, 2);

  index[0] = 3;
  index[1] = 2;
  segmentedImage->SetPixel(index, 1);

  index[0] = 1;
  index[1] = 3;
  segmentedImage->SetPixel(index, 1);

  index[0] = 3;
  index[1] = 3;
  segmentedImage->SetPixel(index, 1);

  typedef itk::CorrectGMUsingNeighbourhoodFilter<InputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetSegmentedImage(segmentedImage);
  filter->SetLabelThresholds(1, 2, 0);
  filter->Update();
  
  if (filter->GetExtraCerebralMatterLabel() != 0) 
    {
      std::cout << "CSF label should be 0" << std::endl;
      return EXIT_FAILURE;
    }
  
  if (filter->GetGreyMatterLabel() != 1) 
    {
      std::cout << "Grey label should be 1" << std::endl;
      return EXIT_FAILURE;
    }
  
  if (filter->GetWhiteMatterLabel() != 2)
    {
      std::cout << "White label should be 2" << std::endl;
      return EXIT_FAILURE;
    }
  
  index[0] = 3;
  index[1] = 1;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetGreyMatterLabel()) 
    {
      std::cout << "Pixel (3,1) should be grey" << std::endl;
      return EXIT_FAILURE;
    }

  index[0] = 2;
  index[1] = 3;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetGreyMatterLabel()) 
    {
      std::cout << "Pixel (2,3) should be grey" << std::endl;
      return EXIT_FAILURE;
    }

  if (filter->GetNumberReclassified() != 2)
    {
      std::cout << "Number reclassified should be 2" << std::endl;
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
