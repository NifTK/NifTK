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

#include <itkImage.h>
#include <itkCorrectGMUsingPVMapFilter.h>

/**
 * Test the CorrectGMUsingPVMap filter
 */
int CorrectGMUsingPVMapTest(int, char* []) 
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
  ImageTypePointer greyPVMap = InputImageType::New();
  
  // Define their size, and start index
  SizeType size;
  size[0] = 4;
  size[1] = 4;

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
  segmentedImage->FillBuffer(2);
  
  greyPVMap->SetLargestPossibleRegion( region );
  greyPVMap->SetBufferedRegion( region );
  greyPVMap->SetRequestedRegion( region );
  greyPVMap->SetSpacing(spacing);
  greyPVMap->Allocate();
  greyPVMap->FillBuffer(0);
  
  index[0] = 2;
  index[1] = 1;
  segmentedImage->SetPixel(index, 2);
  greyPVMap->SetPixel(index, 1);
  
  index[0] = 1;
  index[1] = 2;
  segmentedImage->SetPixel(index, 1);
  greyPVMap->SetPixel(index, 1);
  
  index[0] = 2;
  index[1] = 2;
  segmentedImage->SetPixel(index, 1);
  greyPVMap->SetPixel(index, 0.9);  // should get reclassified by rule 1 in paper.
  
  index[0] = 0;
  index[1] = 0;
  segmentedImage->SetPixel(index, 0);

  index[0] = 1;
  index[1] = 0;
  segmentedImage->SetPixel(index, 0);

  index[0] = 2;
  index[1] = 0;
  segmentedImage->SetPixel(index, 0);

  index[0] = 3;
  index[1] = 0;
  segmentedImage->SetPixel(index, 0);

  index[0] = 0;
  index[1] = 1;
  segmentedImage->SetPixel(index, 0);

  index[0] = 1;
  index[1] = 1;
  segmentedImage->SetPixel(index, 0);

  index[0] = 0;
  index[1] = 2;
  segmentedImage->SetPixel(index, 0);

  typedef itk::CorrectGMUsingPVMapFilter<InputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetSegmentedImage(segmentedImage);
  filter->SetGMPVMap(greyPVMap);
  filter->SetLabelThresholds(1, 0, 2);
  filter->Update();
  
  if (filter->GetExtraCerebralMatterLabel() != 2) return EXIT_FAILURE;
  if (filter->GetGreyMatterLabel() != 1) return EXIT_FAILURE;
  if (filter->GetWhiteMatterLabel() != 0) return EXIT_FAILURE;
  
  // Boundary wont get changed
  index[0] = 0;
  index[1] = 2;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetWhiteMatterLabel()) return EXIT_FAILURE;

  // Boundary wont get changed
  index[0] = 0;
  index[1] = 3;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetExtraCerebralMatterLabel()) return EXIT_FAILURE;

  // Boundary wont get changed
  index[0] = 3;
  index[1] = 0;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetWhiteMatterLabel()) return EXIT_FAILURE;

  index[0] = 2;
  index[1] = 2;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetExtraCerebralMatterLabel()) return EXIT_FAILURE;

  index[0] = 2;
  index[1] = 1;
  if ( filter->GetOutput()->GetPixel(index) != filter->GetGreyMatterLabel()) return EXIT_FAILURE;

  return EXIT_SUCCESS;

}




