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
#include <itkMIDASMeanIntensityWithinARegionFilter.h>

/**
 * Basic tests for MIDASMeanIntensityWithinARegionFilterTest
 */
int itkMIDASMeanIntensityWithinARegionFilterTest(int argc, char * argv[])
{

  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::MIDASMeanIntensityWithinARegionFilter<ImageType, ImageType, ImageType> MeanIntensityFilterType;

  // Create the first image.
  ImageType::Pointer inputImageMain = ImageType::New();
  typedef ImageType::IndexType        IndexType;
  typedef ImageType::SizeType         SizeType;
  typedef ImageType::RegionType       RegionType;
 
  //Create a 4x4 image
  SizeType imageSize;
  imageSize[0] = 4;
  imageSize[1] = 4;

  IndexType startIndex;
  startIndex[0] = 0;
  startIndex[1] = 0;

  RegionType imageRegion;
  imageRegion.SetIndex(startIndex);
  imageRegion.SetSize(imageSize);
  
  inputImageMain->SetLargestPossibleRegion(imageRegion);
  inputImageMain->SetBufferedRegion(imageRegion);
  inputImageMain->SetRequestedRegion(imageRegion);
  inputImageMain->Allocate();

  int intensityValue = 0;
  //rows
  for(unsigned int i = 0; i < 4; i++)
  {
    //columns
    for(unsigned int j = 0; j < 4; j++)
    {
      intensityValue++;
      IndexType imageIndex;
      imageIndex[0] = i;
      imageIndex[1] = j;
      inputImageMain->SetPixel(imageIndex, intensityValue);
    }
  }

 /********************************************************************/

  // Create the second image, that is, the mask image,
  // of the same size as the main image
  ImageType::Pointer inputImageMask = ImageType::New();

  //Create a 4x4 image
  imageSize[0] = 4;
  imageSize[1] = 4;

  startIndex[0] = 0;
  startIndex[1] = 0;

  imageRegion.SetIndex(startIndex);
  imageRegion.SetSize(imageSize);

  inputImageMask->SetLargestPossibleRegion(imageRegion);
  inputImageMask->SetBufferedRegion(imageRegion);
  inputImageMask->SetRequestedRegion(imageRegion);
  inputImageMask->Allocate();
  inputImageMask->FillBuffer(0);

  int maskValue = 1;
  //rows
  for(unsigned int i = 1; i < 3; i++)
  {
    //columns
    for(unsigned int j = 1; j < 3; j++)
    {
      IndexType imageIndex;
      imageIndex[0] = i;
      imageIndex[1] = j;
      inputImageMask->SetPixel(imageIndex, maskValue);
    }
  }

  MeanIntensityFilterType::Pointer meanIntensityFilter = MeanIntensityFilterType::New();
  meanIntensityFilter->SetInput(0, inputImageMain);
  meanIntensityFilter->SetInput(1, inputImageMask);
  meanIntensityFilter->Update();


  double meanIntensityOfMainImage = meanIntensityFilter->GetMeanIntensityMainImage();

  if(meanIntensityOfMainImage  == 8.5 )
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }

  return EXIT_FAILURE;
}
