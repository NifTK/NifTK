/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include "itkImage.h"
#include "itkExcludeImageFilter.h"

/**
 * Basic tests for ExcludeImageFilterTest
 */
int itkExcludeImageFilterTest(int argc, char * argv[])
{

  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::ExcludeImageFilter<ImageType, ImageType, ImageType> ExcludeImageFilterType;

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
      IndexType imageIndex;
      imageIndex[0] = i;
      imageIndex[1] = j;
      if((i == 0) || (i == 3))
      {
    	  intensityValue = 0;
      }
      else
      {
    	  intensityValue = 1;
      }

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

  int maskValue = 0;
  //rows
  for(unsigned int i = 1; i < 3; i++)
  {
    //columns
    for(unsigned int j = 1; j < 3; j++)
    {
      IndexType imageIndex;
      imageIndex[0] = i;
      imageIndex[1] = j;
      if( (i == 1 || i == 2) && (j == 2) )
      {
    	  maskValue = 1;
      }
      else
      {
    	  maskValue = 0;
      }
      inputImageMask->SetPixel(imageIndex, maskValue);
    }
  }

  ExcludeImageFilterType::Pointer excludeImageFilter = ExcludeImageFilterType::New();
  excludeImageFilter->SetInput(0, inputImageMain);
  excludeImageFilter->SetInput(1, inputImageMask);
  excludeImageFilter->Update();


  /**Check if the filter gives the correct output */
  ImageType::Pointer outputImagePtr  = excludeImageFilter->GetOutput();
  typedef itk::ImageRegionConstIterator<ImageType> OutputImageIterator;
  OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

  bool bFilterStatus   = true;

  std::vector<int> outValueVector;
  for(unsigned int i = 0; i < (4*4); i++)
  {
    outValueVector.push_back(0);
  }

  for(unsigned int i = 0; i < (4*4); i++)
  {
    if( (i == 1) || (i == 2) || (i == 5) || (i == 6) || (i == 13) || (i == 14) )
    {
      outValueVector[i] = 1;
    }
  }


  unsigned int index   = 0;

  outputImageIter.GoToBegin();
  while(!outputImageIter.IsAtEnd())
  {
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
