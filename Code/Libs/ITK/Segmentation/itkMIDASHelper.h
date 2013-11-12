/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASHelper_h
#define itkMIDASHelper_h

#include <niftkITKWin32ExportHeader.h>

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkConversionUtils.h>
#include <itkSpatialOrientationAdapter.h>
#include <itkSpatialOrientation.h>
#include <itkMatrix.h>

/**
 * \file itkMIDASHelper.h
 * \brief Provides useful utility functions that could be used in multiple ITK filters.
 */
namespace itk
{
  /** Enum to define the concept of orientation directions. */
  enum Orientation {
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
  NIFTKITK_WINEXPORT ITK_EXPORT void LimitMaskByRegion(TImage* mask,
                         typename TImage::RegionType &region,
                         typename TImage::PixelType outValue
                         );

  /**
   * \brief Returns the volume (number of voxels * voxel volume), of the
   * number of voxels above zero.
   */
  template<typename TPixel, unsigned int VImageDimension>
  NIFTKITK_WINEXPORT ITK_EXPORT
  void
  GetVolumeFromITKImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    double &imageVolume
    );


  /**
   * \brief Gets the orientation string from direction cosines, but only works for 3D.
   */
  template<unsigned int VImageDimension>
  NIFTKITK_WINEXPORT ITK_EXPORT
  void
  GetOrientationString(
    const itk::Matrix<double, VImageDimension, VImageDimension>& directionMatrix,
    std::string &orientationString
    );


  /**
   * \brief Works out the axis of interest from the orientationString (normally derived from direction cosines), and the requested orientation.
   */
  NIFTKITK_WINEXPORT ITK_EXPORT int GetAxisFromOrientationString(const std::string& orientationString, const itk::Orientation& orientation);


  /**
   * \brief Returns either +1, or -1 to indicate in which direction you should change the slice number to go "up".
   * \param orientationString The orientation string such as "RAS", "LPI" etc.
   * \param axisOfInterest Which axis are we looking at within the orientationString.
   * \return -1 or +1 telling you to either increase of decrease the slice number or 0 for "unknown".
   *
   * So, the MIDAS spec is: Shortcut key A=Up, Z=Down which means:
   * <pre>
   * Axial: A=Superior, Z=Inferior
   * Coronal: A=Anterior, Z=Posterior
   * Sagittal: A=Right, Z=Left
   * </pre>
   */
  NIFTKITK_WINEXPORT ITK_EXPORT int GetUpDirection(const std::string& orientationString, const int& axisOfInterest);


  /**
   * \brief Gets the orientation string for a 3D image.
   */
  template<typename TPixel, unsigned int VImageDimension>
  NIFTKITK_WINEXPORT ITK_EXPORT
  void
  GetOrientationStringFromITKImage(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    std::string &orientationString
    );


  /**
   * \brief Returns the axis [0=x, 1=y, 2=z, -1=UNKNOWN] corresponding to the specified orientation for the given image.
   */
  template<typename TPixel, unsigned int VImageDimension>
  NIFTKITK_WINEXPORT ITK_EXPORT
  void
  GetAxisFromITKImage(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    const itk::Orientation orientation,
    int &outputAxis
    );


  /**
   * \brief Returns +1 or -1 (or 0 if unknown) to indicate which way from the centre of the
   * volume is considered "Up", which means anterior in coronal view, superior in axial view and right in sagittal view.
   */
  template<typename TPixel, unsigned int VImageDimension>
  NIFTKITK_WINEXPORT ITK_EXPORT
  void
  GetUpDirectionFromITKImage(
      const itk::Image<TPixel, VImageDimension>* itkImage,
      const itk::Orientation orientation,
      int &upDirection
      );

} // end namespace


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASHelper.txx"
#endif

#endif

