/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-04 15:23:08 +0100 (Wed, 04 May 2011) $
 Revision          : $Revision: 6054 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASHelper_h
#define itkMIDASHelper_h

#include "itkImageRegionIteratorWithIndex.h"

/**
 * File provides useful utility functions that could be used in multiple ITK filters.
 */
namespace itk
{
  /** Enum to define the concept of orientation directions. */
  enum ORIENTATION_ENUM {
    ORIENTATION_AXIAL = 0,
    ORIENTATION_SAGITTAL = 1,
    ORIENTATION_CORONAL = 2,
    ORIENTATION_UNKNOWN = -1
  };

  /**
   * Takes an input mask, and region, and iterates through the whole mask,
   * checking that if a pixel is on (it's not the 'outValue'),
   * it is within the specified region. If not, that pixel is set to
   * the outValue. We assume, and don't check that the region is entirely
   * within the mask.
   */
  template <class TImage>
  ITK_EXPORT void LimitMaskByRegion(TImage* mask,
                         typename TImage::RegionType &region,
                         typename TImage::PixelType outValue
                        )
  {
    int i;
    int dimensions = TImage::ImageDimension;

    unsigned long int regionSize = 1;
    for (i = 0; i < dimensions; i++)
    {
      regionSize *= region.GetSize()[i];
    }

    if (regionSize == 0)
    {
      return;
    }

    typename TImage::PixelType pixel;
    typename TImage::IndexType pixelIndex;
    typename TImage::IndexType minimumRegionIndex = region.GetIndex();
    typename TImage::IndexType maximumRegionIndex = region.GetIndex() + region.GetSize();

    itk::ImageRegionIteratorWithIndex<TImage> iterator(mask, mask->GetLargestPossibleRegion());
    for (iterator.GoToBegin();
        !iterator.IsAtEnd();
        ++iterator)
    {
      pixel = iterator.Get();

      if(pixel != outValue)
      {
        pixelIndex = iterator.GetIndex();

        for (i = 0; i < dimensions; i++)
        {
          if (pixelIndex[i] < minimumRegionIndex[i] || pixelIndex[i] >= maximumRegionIndex[i])
          {
            pixel = outValue;
            iterator.Set(pixel);

          } // end if
        } // end for
      } // end if
    } // end for
  } // end function

} // end namespace

#endif

