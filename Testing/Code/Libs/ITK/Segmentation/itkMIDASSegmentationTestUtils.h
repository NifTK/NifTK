/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7522 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKMIDASSEGMENTATIONTESTUTILS_H
#define ITKMIDASSEGMENTATIONTESTUTILS_H

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

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

  itk::ImageRegionIterator<ImageType> iterator(image, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(value);
  }
}

#endif // ITKMIDASSEGMENTATIONTESTUTILS_H
