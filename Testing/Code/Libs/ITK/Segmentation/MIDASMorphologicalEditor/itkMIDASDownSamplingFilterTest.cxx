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
#include "itkMIDASDownSamplingFilter.h"
#include "itkImageRegionConstIterator.h"

/**
 * Basic tests for MIDASDownSamplingFilterTest
 */
int itkMIDASDownSamplingFilterTest(int argc, char * argv[])
{
  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                   ImageType;
  typedef itk::MIDASDownSamplingFilter<ImageType, ImageType> DownSamplingFilterType;

  // Create the first image.
  ImageType::Pointer inputImage  = ImageType::New();
  typedef ImageType::IndexType     IndexType;
  typedef ImageType::SizeType      SizeType;
  typedef ImageType::RegionType    RegionType;
 
  //Create a 9x9 image
  SizeType imageSize;
  imageSize[0] = 9;
  imageSize[1] = 9;

  IndexType startIndex;
  startIndex[0] = 0;
  startIndex[1] = 0;

  RegionType imageRegion;
  imageRegion.SetIndex(startIndex);
  imageRegion.SetSize(imageSize);
  
  inputImage->SetLargestPossibleRegion(imageRegion);
  inputImage->SetBufferedRegion(imageRegion);
  inputImage->SetRequestedRegion(imageRegion);
  inputImage->Allocate();

  unsigned int value = 0;
  IndexType imageIndex;

  value = 1;
  //columns
  for(unsigned int i = 0; i < 9; i++)
  {
    //rows
    for(unsigned int j = 0; j < 9; j++)
    {
      value = 0;
      imageIndex[0] = i;
      imageIndex[1] = j;

      if( (i == 0) && ( (j == 4) || (j == 5) || (j == 8) ) )
      {
        value = 1;
      }
      if( (i == 1) && ( (j == 1) || (j == 3) ) )
      {
        value = 1;
      }
      if( (i == 2) && ( (j == 6) || (j == 7) || (j == 8) ) )
      {
        value = 1;
      }
      if( (i == 3) && ( (j == 2) || (j == 5) ) )
      {
        value = 1;
      }
      if( (i == 4) && ( (j == 0) || (j == 4) || (j == 6) ) )
      {
        value = 1;
      }
      if( (i == 5) && ( (j == 3) || (j == 7) ) )
      {
        value = 1;
      }
      if( (i == 7) && ( (j == 0) || (j == 1) || (j == 3) || (j == 5) ) )
      {
        value = 1;
      }
      inputImage->SetPixel(imageIndex, value);

    }//end of for loop of rows

  }//end of for loop of columns

  unsigned int downSampleFactor    = 3;
  
  DownSamplingFilterType::Pointer DownSamplingFilter = DownSamplingFilterType::New();
  DownSamplingFilter->SetInput(inputImage);
  DownSamplingFilter->SetDownSamplingFactor(downSampleFactor);
  DownSamplingFilter->Update();

  /**Check if the filter gives the correct output */
  ImageType::Pointer outputImagePtr  = DownSamplingFilter->GetOutput();
  typedef itk::ImageRegionConstIterator<ImageType> OutputImageIterator;
  OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

  bool bFilterStatus   = true;

  std::vector<int> outValueVector;
  for(unsigned int i = 0; i < (3*3); i++)
  {
    outValueVector.push_back(1);
  }
  outValueVector[(3*3) - 1] = 0;


  unsigned int index   = 0;

  outputImageIter.GoToBegin();
  while(!outputImageIter.IsAtEnd())
  {
    std::cerr << "Expected=" << outValueVector[index] << ", actual=" << outputImageIter.Get() << std::endl;

    if(outputImageIter.Get() != outValueVector[index])
    {
      bFilterStatus = false;
      break;
    }
    ++outputImageIter;
    ++index;
  }

  if(bFilterStatus)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;

  return EXIT_FAILURE;

}
