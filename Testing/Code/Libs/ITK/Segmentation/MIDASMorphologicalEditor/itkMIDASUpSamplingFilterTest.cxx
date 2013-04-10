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
#include "itkMIDASUpSamplingFilter.h"
#include "itkImageRegionConstIterator.h"
using namespace std;

/**
 * Basic tests for MIDASUpSamplingFilterTest
 */
int itkMIDASUpSamplingFilterTest(int argc, char * argv[])
{
  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                 ImageType;
  typedef itk::MIDASUpSamplingFilter<ImageType, ImageType> UpSamplingFilterType;

  // Create the input image.
  ImageType::Pointer inputImage  = ImageType::New();
  typedef ImageType::IndexType     IndexType;
  typedef ImageType::SizeType      SizeType;
  typedef ImageType::RegionType    RegionType;

  //Create a 3x3 image
  SizeType inputImageSize;
  inputImageSize[0] = 3;
  inputImageSize[1] = 3;

  IndexType inputImageStartIndex;
  inputImageStartIndex[0] = 0;
  inputImageStartIndex[1] = 0;

  RegionType inputImageRegion;
  inputImageRegion.SetIndex(inputImageStartIndex);
  inputImageRegion.SetSize(inputImageSize);
  
  inputImage->SetLargestPossibleRegion(inputImageRegion);
  inputImage->SetBufferedRegion(inputImageRegion);
  inputImage->SetRequestedRegion(inputImageRegion);
  inputImage->Allocate();

  IndexType inputImageIndex;
  unsigned int value = 0;

  //columns
  for(unsigned int i = 0; i < 3; i++)
  {
    //rows
    for(unsigned int j = 0; j < 3; j++)
    {
      value = 1;
      inputImageIndex[0] = i;
      inputImageIndex[1] = j;

      if((i == 2) && (j == 2))
      {
        value = 0;
      }
      inputImage->SetPixel(inputImageIndex, value);

    }//end of for loop of columns

  }//end of for loop of rows

  inputImageSize[0] = 9;
  inputImageSize[1] = 9;
  inputImageRegion.SetIndex(inputImageStartIndex);
  inputImageRegion.SetSize(inputImageSize);

  ImageType::Pointer inputLargeImage  = ImageType::New();
  inputLargeImage->SetLargestPossibleRegion(inputImageRegion);
  inputLargeImage->SetBufferedRegion(inputImageRegion);
  inputLargeImage->SetRequestedRegion(inputImageRegion);
  inputLargeImage->Allocate();

  unsigned int upSampleFactor      = 3;
  
  UpSamplingFilterType::Pointer UpSamplingFilter = UpSamplingFilterType::New();
  UpSamplingFilter->SetInput(0, inputImage);
  UpSamplingFilter->SetInput(1, inputLargeImage);
  UpSamplingFilter->SetUpSamplingFactor(upSampleFactor);
  UpSamplingFilter->Update();

  /**Check if the filter gives the correct output */
  ImageType::Pointer outputImagePtr  = UpSamplingFilter->GetOutput();
  typedef itk::ImageRegionConstIterator<ImageType> OutputImageIterator;
  OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

  bool bFilterStatus   = true;

  std::vector<int> upSampledValues;
  int val = 1;

  for(unsigned int i = 0; i < 81; i++)
  {
    upSampledValues.push_back(val);
  }
  
  for(unsigned int j = 60; j <= 62; j++)
  {
    upSampledValues[j] = 0;
  }

  for(unsigned int j = 69; j <= 71; j++)
  {
    upSampledValues[j] = 0;
  }

  for(unsigned int j = 78; j <= 80; j++)
  {
    upSampledValues[j] = 0;
  }

  unsigned int index   = 0;

  outputImageIter.GoToBegin();
  while(!outputImageIter.IsAtEnd())
  {
    std::cerr << "index=" << index << ", Expected=" <<  upSampledValues[index] << ", actual=" << outputImageIter.Get() << std::endl;

    if(outputImageIter.Get() != upSampledValues[index])
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
