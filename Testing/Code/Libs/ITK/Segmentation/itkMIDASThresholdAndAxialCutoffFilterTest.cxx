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
#include "itkMIDASThresholdAndAxialCutoffFilter.h"
#include "itkImageRegionConstIterator.h"

/**
 * Basic tests for MIDASThresholdAndAxialCutoffFilterTest
 */
int itkMIDASThresholdAndAxialCutoffFilterTest(int argc, char * argv[])
{
  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::MIDASThresholdAndAxialCutoffFilter<ImageType> ThresholdAndAxialCutoffFilterType;

  // Create the first image.
  ImageType::Pointer inputImage  = ImageType::New();
  typedef ImageType::IndexType     IndexType;
  typedef ImageType::SizeType      SizeType;
  typedef ImageType::RegionType    RegionType;
 
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
  
  inputImage->SetLargestPossibleRegion(imageRegion);
  inputImage->SetBufferedRegion(imageRegion);
  inputImage->SetRequestedRegion(imageRegion);
  inputImage->Allocate();

  int intensityValue = 0;
  //filling the rows 
  //rows
  for(unsigned int i = 0; i < 4; i++)
  {
    //columns
    for(unsigned int j = 0; j < 4; j++)
    {
      if( (i == 1) || (i == 3) )
        intensityValue--;
      else
        intensityValue++;

      IndexType imageIndex;
      imageIndex[0] = i;
      imageIndex[1] = j;
      inputImage->SetPixel(imageIndex, intensityValue);
    }
  }

/****************************************************************/

  //Create a 2x2 region
  SizeType regionSize;
  regionSize[0] = 2;
  regionSize[1] = 2;

  IndexType startRegionIndex;
  startRegionIndex[0] = 1;
  startRegionIndex[1] = 1;

  RegionType regionToProcess;
  regionToProcess.SetIndex(startRegionIndex);
  regionToProcess.SetSize(regionSize);


  ThresholdAndAxialCutoffFilterType::Pointer thresholdAndAxialCutoffFilter = ThresholdAndAxialCutoffFilterType::New();
  thresholdAndAxialCutoffFilter->SetInput(0, inputImage);
  
  thresholdAndAxialCutoffFilter->SetLowerThreshold(2);
  thresholdAndAxialCutoffFilter->SetUpperThreshold(3);

  thresholdAndAxialCutoffFilter->SetInsideRegionValue(1);
  thresholdAndAxialCutoffFilter->SetOutsideRegionValue(0);

  thresholdAndAxialCutoffFilter->SetRegionToProcess(regionToProcess);
  thresholdAndAxialCutoffFilter->SetUseRegionToProcess(true);

  thresholdAndAxialCutoffFilter->Update();

  /**Check if the filter gives the correct output */
  ImageType::Pointer outputImagePtr  = thresholdAndAxialCutoffFilter->GetOutput();
  typedef itk::ImageRegionConstIterator<ImageType> OutputImageIterator;
  OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

  std::vector<int> outValueVector;
  outValueVector.reserve(16);
  for(unsigned int i = 0; i < 16; i++)
  {
    outValueVector.push_back(0);
  }

  outValueVector[5] =  1;
  outValueVector[6] =  1;
  outValueVector[10] = 1;


  unsigned int index   = 0;
  bool bFilterStatus   = true;

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
