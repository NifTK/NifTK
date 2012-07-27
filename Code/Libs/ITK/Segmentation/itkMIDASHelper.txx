/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkMIDASHelper.h"

namespace itk
{

//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
template<unsigned int VImageDimension>
ITK_EXPORT
void
GetOrientationString(
  const itk::Matrix<double, VImageDimension, VImageDimension>& directionMatrix,
  std::string &orientationString
  )
{
  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientationFlag;
  orientationFlag = adaptor.FromDirectionCosines(directionMatrix);
  orientationString = itk::ConvertSpatialOrientationToString(orientationFlag);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
ITK_EXPORT
void
GetOrientationStringFromITKImage(
  const itk::Image<TPixel, VImageDimension>* itkImage,
  std::string &orientationString
  )
{
  itk::Matrix<double, VImageDimension, VImageDimension> direction = itkImage->GetDirection();
  GetOrientationString<VImageDimension>(direction, orientationString);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
ITK_EXPORT
void
GetAxisFromITKImage(
  const itk::Image<TPixel, VImageDimension>* itkImage,
  const itk::ORIENTATION_ENUM orientation,
  int &outputAxis
  )
{
  outputAxis = -1;

  std::string orientationString = "UNKNOWN";
  GetOrientationStringFromITKImage(itkImage, orientationString);

  if (orientationString != "UNKNOWN")
  {
    outputAxis = GetAxisFromOrientationString(orientationString, orientation);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
ITK_EXPORT
void
GetUpDirectionFromITKImage(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    const itk::ORIENTATION_ENUM orientation,
    int &upDirection
    )
{
  upDirection = 0;

  std::string orientationString = "UNKNOWN";
  GetOrientationStringFromITKImage(itkImage, orientationString);
  std::cerr << "Matt, GetUpDirectionFromITKImage geom dir=" << itkImage->GetDirection() << std::endl;
  int axisOfInterest = -1;
  GetAxisFromITKImage(itkImage, orientation, axisOfInterest);

  if (orientationString != "UNKNOWN" && axisOfInterest != -1)
  {
    upDirection = GetUpDirection(orientationString, axisOfInterest);
  }
}

} // end namespace
