/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASOrientationUtils_h
#define mitkMIDASOrientationUtils_h

#include "niftkCoreExports.h"
#include <mitkImage.h>
#include "mitkMIDASEnums.h"
#include <itkMIDASHelper.h>

/**
 * \file mitkMIDASOrientationUtils.h
 * \brief Some utilities to help with MIDAS conventions on orientation.
 */
namespace mitk
{

/**
 * \brief Converts an MITK orientation enum to an ITK orientation enum, and ideally these types should be merged.
 */
NIFTKCORE_EXPORT itk::Orientation GetItkOrientation(const MIDASOrientation& orientation);


/*
 * \brief Converts an ITK orientation enum to an MITK orientation enum, and ideally these types should be merged.
 */
NIFTKCORE_EXPORT MIDASOrientation GetMitkOrientation(const itk::Orientation& orientation);


/**
 * \brief See GetUpDirection as in effect, we are only using the direction cosines from the geometry.
 */
NIFTKCORE_EXPORT int GetUpDirection(const mitk::Image* image, const MIDASOrientation& orientation);


/**
 * \brief Returns either +1, or -1 to indicate in which direction you should change the slice number to go "up".
 * \param geometry An MITK geometry, not NULL.
 * \param orientation a MIDASOrientation corresponding to Axial, Coronal or Sagittal.
 * \return -1 or +1 telling you to either increase of decrease the slice number or 0 for "unknown".
 *
 * So, the MIDAS spec is: Shortcut key A=Up, Z=Down which means:
 * <pre>
 * Axial: A=Superior, Z=Inferior
 * Coronal: A=Anterior, Z=Posterior
 * Sagittal: A=Right, Z=Left
 * </pre>
 */

NIFTKCORE_EXPORT int GetUpDirection(const mitk::Geometry3D* geometry, const MIDASOrientation& orientation);


/**
 * \brief Returns either -1 (unknown), or [0,1,2] for the x, y, or z axis corresponding to the through plane direction for the specified orientation.
 * \param image An MITK image, not NULL.
 * \param orientation a MIDASOrientation corresponding to Axial, Coronal or Sagittal.
 * \return -1=unknown, or the axis number [0,1,2].
 */
NIFTKCORE_EXPORT int GetThroughPlaneAxis(const mitk::Image* image, const MIDASOrientation& orientation);


/**
 * \brief Returns the Orientation String (RPI, RAS etc).
 * \param image An MITK image, not NULL.
 *
 * NOTE: MIDAS Analyze are flipped in the MITK GUI. This means if you use the default ITK reader
 * which is used for example in the command line app niftkImageInfo, you will get a different answer to
 * this method, as this method will be run from within the MITK GUI, and hence will be using itkDRCAnalyzeImageIO.
 */
NIFTKCORE_EXPORT std::string GetOrientationString(const mitk::Image* image);

/// \brief Converts between voxel coordinate order and world coordinate order.
/// The function writes the axes of the sagittal, coronal and axial dimensions to @a axes,
/// in this order:
///
///     axes[0]: axis of sagittal dimension
///     axes[1]: axis of coronal dimension
///     axes[2]: axis of axial dimension
///
template<typename TPixel, unsigned int VImageDimension>
NIFTKCORE_EXPORT
void GetAxesInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, int axes[3]);

/// \brief Converts between voxel coordinate order and world coordinate order.
/// The function writes the axes of the sagittal, coronal and axial dimensions to @a axes,
/// in this order:
///
///     axes[0]: axis of sagittal dimension
///     axes[1]: axis of coronal dimension
///     axes[2]: axis of axial dimension
///
NIFTKCORE_EXPORT
void GetAxesInWorldCoordinateOrder(const mitk::Image* mitkImage, int axes[3]);

/// \brief Gets the spacing of the image in world coordinate order.
///
///     spacing[0]: spacing along sagittal dimension
///     spacing[1]: spacing along coronal dimension
///     spacing[2]: spacing axial dimension
///
template<typename TPixel, unsigned int VImageDimension>
NIFTKCORE_EXPORT
void GetSpacingInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, mitk::Vector3D& spacing);

/// \brief Gets the spacing of the image in world coordinate order.
///
///     spacing[0]: spacing along sagittal dimension
///     spacing[1]: spacing along coronal dimension
///     spacing[2]: spacing axial dimension
///
NIFTKCORE_EXPORT
void GetSpacingInWorldCoordinateOrder(const mitk::Image* mitkImage, mitk::Vector3D& spacing);

/// \brief Gets the extents (number of voxels) of the image in world coordinate order.
///
///     extentsInVx[0]: extent of sagittal dimension
///     extentsInVx[1]: extent of coronal dimension
///     extentsInVx[2]: extent of axial dimension
///
template<typename TPixel, unsigned int VImageDimension>
NIFTKCORE_EXPORT
void GetExtentsInVxInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, mitk::Vector3D& extentsInVx);

/// \brief Gets the extents (number of voxels) of the image in world coordinate order.
///
///     extentsInVx[0]: extent of sagittal dimension as number of voxels
///     extentsInVx[1]: extent of coronal dimension as number of voxels
///     extentsInVx[2]: extent of axial dimension as number of voxels
///
NIFTKCORE_EXPORT
void GetExtentsInVxInWorldCoordinateOrder(const mitk::Image* mitkImage, mitk::Vector3D& extentsInVx);

/// \brief Gets the extents of the image in millimetres in world coordinate order.
///
///     extentsInMm[0]: extent of sagittal dimension in millimetres
///     extentsInMm[1]: extent of coronal dimension in millimetres
///     extentsInMm[2]: extent of axial dimension in millimetres
///
template<typename TPixel, unsigned int VImageDimension>
NIFTKCORE_EXPORT
void GetExtentsInMmInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, mitk::Vector3D& extentsInMm);

/// \brief Gets the extents of the image in millimetres in world coordinate order.
///
///     extentsInMm[0]: extent of sagittal dimension in millimetres
///     extentsInMm[1]: extent of coronal dimension in millimetres
///     extentsInMm[2]: extent of axial dimension in millimetres
///
NIFTKCORE_EXPORT
void GetExtentsInMmInWorldCoordinateOrder(const mitk::Image* mitkImage, mitk::Vector3D& extentsInMm);

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "mitkMIDASOrientationUtils.txx"
#endif

#endif
