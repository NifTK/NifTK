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
#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkConversionUtils.h"
#include "itkSpatialOrientationAdapter.h"
#include "itkSpatialOrientation.h"

/**
 * \file itkMIDASHelper.h
 * \brief Provides useful utility functions that could be used in multiple ITK filters.
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
   * \brief Used to mask an image within a region.
   *
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

  /**
   * \brief Returns the volume (number of voxels * voxel volume), of the
   * number of voxels above zero.
   */
  template<typename TPixel, unsigned int VImageDimension>
  ITK_EXPORT
  void
  GetVolumeFromITKImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    double &imageVolume
    )
  {
    typedef itk::Image<TPixel, VImageDimension> ImageType;
    typedef typename ImageType::SpacingType SpacingType;

    SpacingType imageSpacing = itkImage->GetSpacing();
    double voxelVolume = 1;
    for ( unsigned int i = 0; i < imageSpacing.Size(); i++)
    {
      voxelVolume *= imageSpacing[i];
    }

    unsigned long int numberOfForegroundVoxels = 0;
    itk::ImageRegionConstIterator<ImageType> iter(itkImage, itkImage->GetLargestPossibleRegion());
    for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
    {
      if (iter.Get() > 0)
      {
        numberOfForegroundVoxels++;
      }
    }

    imageVolume = numberOfForegroundVoxels * voxelVolume;
  }

  /**
   * \brief Returns the axis [0=x, 1=y, 2=z, -1=UNKNOWN] corresponding to the specified orientation.
   */
  template<typename TPixel, unsigned int VImageDimension>
  ITK_EXPORT
  void
  GetAxisFromITKImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    itk::ORIENTATION_ENUM orientation,
    int &outputAxis
    )
  {
    outputAxis = -1;

    typename itk::SpatialOrientationAdapter adaptor;
    typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientationFlag;
    orientationFlag = adaptor.FromDirectionCosines(itkImage->GetDirection());
    std::string orientationString = itk::ConvertSpatialOrientationToString(orientationFlag);

    if (orientationString != "UNKNOWN")
    {
      for (int i = 0; i < 3; i++)
      {
        if (orientation == itk::ORIENTATION_AXIAL && (orientationString[i] == 'S' || orientationString[i] == 'I'))
        {
          outputAxis = i;
          break;
        }

        if (orientation == itk::ORIENTATION_CORONAL && (orientationString[i] == 'A' || orientationString[i] == 'P'))
        {
          outputAxis = i;
          break;
        }

        if (orientation == itk::ORIENTATION_SAGITTAL && (orientationString[i] == 'L' || orientationString[i] == 'R'))
        {
          outputAxis = i;
          break;
        }
      }
    }
  }

  /**
   * \brief Returns +1 or -1 (or 0 if unknown) to indicate which way from the centre of the
   * volume is considered "Up", which means anterior in coronal view, superior in axial view
   * and right in sagittal view.
   */
  template<typename TPixel, unsigned int VImageDimension>
  ITK_EXPORT
  void
  GetUpDirectionFromITKImage(
      itk::Image<TPixel, VImageDimension>* itkImage,
      itk::ORIENTATION_ENUM orientation,
      int &upDirection
      )
  {
    upDirection = 0;

    typename itk::SpatialOrientationAdapter adaptor;
    typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientationFlag;
    orientationFlag = adaptor.FromDirectionCosines(itkImage->GetDirection());
    std::string orientationString = itk::ConvertSpatialOrientationToString(orientationFlag);

    int axisOfInterest = -1;
    GetAxisFromITKImage(itkImage, orientation, axisOfInterest);

    // NOTE: ITK convention is that an image that goes from
    // Left to Right in X, Posterior to Anterior in Y and Inferior to Superior in Z
    // is called an LPI, whereas in Nifti speak, that would be RAS.
    if (orientationString != "UNKNOWN" && axisOfInterest != -1)
    {
      char direction = orientationString[axisOfInterest];
      if (direction == 'A' || direction == 'S' || direction == 'R')
      {
        upDirection = -1;
      }
      else if (direction == 'P' || direction == 'I' || direction == 'L')
      {
        upDirection = 1;
      }
    }
  }
} // end namespace

#endif

