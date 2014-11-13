/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
  const itk::Orientation orientation,
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
    const itk::Orientation orientation,
    int &upDirection
    )
{
  upDirection = 0;

  std::string orientationString = "UNKNOWN";
  GetOrientationStringFromITKImage(itkImage, orientationString);

  int axisOfInterest = -1;
  GetAxisFromITKImage(itkImage, orientation, axisOfInterest);

  if (orientationString != "UNKNOWN" && axisOfInterest != -1)
  {
    upDirection = GetUpDirection(orientationString, axisOfInterest);
  }
}

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
ITK_EXPORT
void GetOrientationLabelFromITKImage(const itk::Image<TPixel, VImageDimension>* itkImage, std::string &label)
{
  if (VImageDimension < 3)
    return;
  
  // Get direction cosines from the image
  itk::Image<TPixel, VImageDimension>::DirectionType dirCosines = itkImage->GetDirection();

  // Copy values to a new ITK matrix
  mitk::AffineTransform3D::MatrixType::InternalMatrixType normalisedMatrix;
  for (unsigned int i=0; i < 3; i++)
  {
    for (unsigned int j = 0; j < 3; j++)
    {
      normalisedMatrix[i][j] = dirCosines[i][j];
    }
  }
  // Normalize values
  normalisedMatrix.normalize_columns();

  // Get major axis label
  std::string rowAxis = GetMajorAxisFromPatientRelativeDirectionCosine(normalisedMatrix[0][0], normalisedMatrix[1][0], normalisedMatrix[2][0]);
  std::string colAxis = GetMajorAxisFromPatientRelativeDirectionCosine(normalisedMatrix[0][1], normalisedMatrix[1][1], normalisedMatrix[2][1]);
  
  if (!rowAxis.empty() && !colAxis.empty()) 
  {
    if ((rowAxis == "R" || rowAxis == "L") && (colAxis == "A" || colAxis == "P"))
      label="AXIAL";
    else if ((colAxis == "R" || colAxis == "L") && (rowAxis == "A" || rowAxis == "P"))
      label="AXIAL";
    else if ((rowAxis == "R" || rowAxis == "L") && (colAxis == "H" || colAxis == "F"))
      label="CORONAL";
    else if ((colAxis == "R" || colAxis == "L") && (rowAxis == "H" || rowAxis == "F"))
      label="CORONAL";
    else if ((rowAxis == "A" || rowAxis == "P") && (colAxis == "H" || colAxis == "F"))
      label="SAGITTAL";
    else if ((colAxis == "A" || colAxis == "P") && (rowAxis == "H" || rowAxis == "F"))
      label="SAGITTAL";
  }
  else
  {
    label="OBLIQUE";
  }
}

} // end namespace
