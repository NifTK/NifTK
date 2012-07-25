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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASIMAGEUTILS_H
#define MITKMIDASIMAGEUTILS_H

#include "niftkMitkExtExports.h"
#include "itkConversionUtils.h"
#include "itkSpatialOrientationAdapter.h"
#include "mitkImage.h"
#include "mitkDataNode.h"
#include "mitkPositionEvent.h"
#include "mitkMIDASEnums.h"

/**
 * \file mitkMIDASImageUtils.h
 * \brief Some useful MIDAS related image utilities, such as working out the As Acquired orientation, volumes etc.
 */
namespace mitk
{
  /**
   * \brief ITK method that given an image, returns the MIDASOrientation for the XY plane.
   * \param itkImage an ITK image
   * \param outputOrientation the output MIDASOrientation as either MIDAS_ORIENTATION_AXIAL,
   * MIDAS_ORIENTATION_CORONAL or MIDAS_ORIENTATION_SAGITTAL, or else is unchanged from the input.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKGetAsAcquiredOrientation(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    MIDASOrientation &outputOrientation
  );

  /**
   * \brief Returns the MIDASView corresponding to the XY plane, or else returns the supplied default.
   * \param defaultView A default MIDASView that will be returned if we can't work out the As Acquired view.
   * \param image An image to check.
   * \return MIDASView the As Acquired view, or the defaultView.
   */
  NIFTKMITKEXT_EXPORT MIDASView GetAsAcquiredView(const MIDASView& defaultView, const mitk::Image* image);

  /**
   * \brief Simply returns true if a node contains an image, and false otherwise.
   * \param node An MITK DataNode.
   * \return true if node contains an mitk::Image and false otherwise
   */
  NIFTKMITKEXT_EXPORT bool IsImage(const mitk::DataNode* node);

  /**
   * \brief ITK method that compares if images have the same intensity values,
   * and should only be called from ImagesHaveEqualIntensities. Does not check
   * if the images are actually the same size, so you should check that first.
   *
   * \param itkImage an ITK image
   * \param image2 an MITK image
   * \param output true if images have the same intensities and false otherwise.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKImagesHaveEqualIntensities(
      const itk::Image<TPixel, VImageDimension>* itkImage,
      const mitk::Image* image2,
      bool &output
      );

  /**
   * \brief Utility method that compares if images have the same intensity values.
   * \param image1 an MITK image
   * \param image2 an MITK image
   * \return true if images have the same intensity values, and false otherwise.
   */
  NIFTKMITKEXT_EXPORT bool ImagesHaveEqualIntensities(const mitk::Image* image1, const mitk::Image* image2);

  /**
   * \brief ITK method that compares if images are have the same spatial extent (size, spacing, origin, direction).
   * \param itkImage an ITK image.
   * \param image2 an MITK image.
   * \output true if images have the same spatial extent, and false otherwise.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKImagesHaveSameSpatialExtent(
      const itk::Image<TPixel, VImageDimension>* itkImage,
      const mitk::Image* image2,
      bool &output
      );

  /**
   * \brief Utility method that compares if images have the same spatial extent.
   * \param image1 an MITK image.
   * \param image2 an MITK image.
   * \return true if images have the same spatial extent, and false otherwise.
   */
  NIFTKMITKEXT_EXPORT bool ImagesHaveSameSpatialExtent(const mitk::Image* image1, const mitk::Image* image2);

  /**
   * \brief ITK method that simply iterates through a whole image, filling it with the specified value which is cast to the appropriate pixel type.
   * \param image A non NULL MITK image.
   * \param value A single scalar value that will be cast.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKFillImage(
      itk::Image<TPixel, VImageDimension>* itkImage,
      float &value
      );

  /**
   * \brief Simply iterates through a whole image, filling it with the specified value which is cast to the appropriate pixel type.
   * \param image A non NULL MITK image.
   * \param value A single scalar value that will be cast.
   */
  NIFTKMITKEXT_EXPORT void FillImage(mitk::Image* image, float value);


  /**
   * \brief Simply iterates through a whole image, counting how many intensity values are >= lower and <= upper.
   * \param itkImage An ITK image.
   * \param lower A lower threshold for intensity values
   * \param upper An upper threshold for intensity values
   * \return unsigned long int The number of voxels.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKCountBetweenThreshold(
      const itk::Image<TPixel, VImageDimension>* itkImage,
      const float &lower,
      const float &upper,
      unsigned long int &outputCount
      );

  /**
   * \brief Simply iterates through a whole image, counting how many intensity values are >= lower and <= upper.
   * \param image An MITK Image.
   * \param lower A lower threshold for intensity values
   * \param upper An upper threshold for intensity values
   * \return unsigned long int The number of voxels.
   */
  NIFTKMITKEXT_EXPORT unsigned long int CountBetweenThreshold(const mitk::Image* image, const float& lower, const float& upper);

  /**
   * \brief Returns the number of voxels in an image.
   * \param image An MITK Image.
   * \return unsigned long int The number of voxels.
   */
  NIFTKMITKEXT_EXPORT unsigned long int GetNumberOfVoxels(const mitk::Image* image);

  /**
   * \brief Returns the middle voxel of an image.
   * \param image An MITK image.
   * \return If the image has odd numbers of voxels in an axis, the returned voxel is
   * the middle one, whereas if the image has even number of voxels in an axis, the
   * returned voxel is <code>(int)(number vox - 1)/2.0</code>
   */
  NIFTKMITKEXT_EXPORT mitk::Point3D GetMiddlePointInVoxels(const mitk::Image* image);

  /**
   * \brief Generates a fake position event, (mainly for unit testing), at a given voxel location.
   * \param image An MITK image.
   * \param voxelLocation A voxel location.
   * \return A fake position event, where by "fake" we mean that there is no valid window Id.
   */
  NIFTKMITKEXT_EXPORT mitk::PositionEvent GeneratePositionEvent(const mitk::BaseRenderer* renderer, const mitk::Image* image, const mitk::Point3D& voxelLocation);


  /**
   * \brief ITK method that returns the volume of non-zero voxels in an image.
   * \param itkImage An ITK image.
   * \param imageVolume output volume in millimetres cubed.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKGetVolume(
      itk::Image<TPixel, VImageDimension>* itkImage,
      double &imageVolume
      );

  /**
   * \brief Returns the volume of non-zero voxels in an image.
   * \param image An MITK image.
   * \return double The volume in millimetres cubed.
   */
  NIFTKMITKEXT_EXPORT double GetVolume(const mitk::Image* image);


  /**
   * \brief Calculates the volume of non-zero voxels in image, and creates a property "midas.volume" on the data node.
   * \param image An MITK image, not NULL.
   * \param node An MITK DataNode, not NULL.
   */
  NIFTKMITKEXT_EXPORT void UpdateVolumeProperty(const mitk::Image* image, mitk::DataNode* node);

}

#endif // MITKMIDASIMAGEUTILS_H
