/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

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
#include "itkLargestConnectedComponentFilter.h"
#include "itkImageRegionConstIterator.h"

/**
 * Basic tests for LargestConnectedComponentFilterTest
 */
int itkLargestConnectedComponentFilterTest(int argc, char * argv[])
{
  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::LargestConnectedComponentFilter<ImageType, ImageType> LargestConnectedComponentFilterType;

  // Create the first image.
  ImageType::Pointer inputImage  = ImageType::New();
  typedef ImageType::IndexType     IndexType;
  typedef ImageType::SizeType      SizeType;
  typedef ImageType::RegionType    RegionType;
 
  //Create a 5x5 image
  SizeType imageSize;
  imageSize[0] = 9;
  imageSize[1] = 17;

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
  //filling the rows
  //rows
  for(unsigned int i = 0; i < 9; i++)
  {
    //columns
    for(unsigned int j = 0; j < 17; j++)
    {
      value = 0;
      imageIndex[0] = i;
      imageIndex[1] = j;

      if( (i == 1) && ( (j == 2) || (j == 3) || (j == 6) || (j == 7)
           || (j == 10) || (j == 11) || (j == 14) || (j == 15) ) )
      {
        value = 1;
      }
      else if( (i == 2) && ( ((j >= 1) && (j <= 8)) || ((j >= 11) && (j <= 14)) ) )
      {
        value = 1;
      }
      else if( (i == 3) && ( ((j >= 3) && (j <= 6)) || ((j >= 10) && (j <= 13)) ) )
      {
        value = 1;
      }
      else if( (i == 4) && ( ((j >= 2) && (j <= 5)) || ((j >= 9) && (j <= 11))
               || (j == 14) || (j == 15)) )
      {
        value = 1;
      }
      else if( (i == 5) && ( ((j >= 1) && (j <= 3)) || ((j >= 11) && (j <= 13))
               || (j == 6) || (j == 7)) )
      {
        value = 1;
      }
      else if( (i == 6) && ( (j == 2) || (j == 3) || (j == 9) || (j == 10)
            || (j == 14) || (j == 15) ) )
      {
        value = 1;
      }
      else if( (i == 7) && ( ((j >= 6) && (j <= 9)) || ((j >= 12) && (j <= 15)) ) )
      {
        value = 1;
      }

      inputImage->SetPixel(imageIndex, value);
  
    }//end of for loop of columns

  }//end of for loop of rows

  
  LargestConnectedComponentFilterType::Pointer LargestConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  LargestConnectedComponentFilter->SetInput(0, inputImage);

  LargestConnectedComponentFilter->Update();

  /**Check if the filter gives the correct output */
  ImageType::Pointer outputImagePtr  = LargestConnectedComponentFilter->GetOutput();
  typedef itk::ImageRegionConstIterator<ImageType> OutputImageIterator;
  OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

  bool bFilterStatus   = true;

  std::vector<int> outValueVector;
  for(unsigned int i = 0; i < (17*9); i++)
  {
    outValueVector.push_back(0);
  }

  for(unsigned int i = 0; i < (17*9); i++)
  {
    if( (i == 11) || (i == 14) || (i == 19) || (i == 20) || ((i >= 22) && (i <= 24))
        || ((i >= 19) && (i <= 20)) || ((i >= 22) && (i <= 24)) || ((i >= 28) && (i <= 33)) 
        || ((i >= 38) && (i <= 40)) || ((i >= 47) && (i <= 49)) || ((i >= 55) && (i <= 57))
        || ((i >= 64) && (i <= 65)) || (i == 74) )
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
