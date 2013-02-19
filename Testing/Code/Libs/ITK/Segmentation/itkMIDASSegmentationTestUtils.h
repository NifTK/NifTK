/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMIDASSEGMENTATIONTESTUTILS_H
#define ITKMIDASSEGMENTATIONTESTUTILS_H

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMIDASSegmentationTestUtils.h"

template <class TPixel, unsigned int VImageDimension>
unsigned long int CountVoxelsAboveValue(TPixel value, itk::Image<TPixel, VImageDimension>* image)
{
  unsigned long int count = 0;
  typedef itk::Image<TPixel, VImageDimension> ImageType;

  itk::ImageRegionConstIterator<ImageType> iterator(image, image->GetLargestPossibleRegion());
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    if (iterator.Get() > value)
    {
      count++;
    }
  }
  return count;
}

template <class TPixel, unsigned int VImageDimension>
void FillImageRegionWithValue(TPixel value, itk::Image<TPixel, VImageDimension>* image, typename itk::Image<TPixel, VImageDimension>::RegionType region)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;

  itk::ImageRegionIteratorWithIndex<ImageType> iterator(image, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(value);
  }
}

#endif // ITKMIDASSEGMENTATIONTESTUTILS_H
