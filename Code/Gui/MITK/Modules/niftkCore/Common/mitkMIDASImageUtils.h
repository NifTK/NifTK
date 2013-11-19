/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASImageUtils_h
#define mitkMIDASImageUtils_h

#include "niftkCoreExports.h"
#include <itkConversionUtils.h>
#include <itkSpatialOrientationAdapter.h>
#include <mitkImage.h>
#include <mitkDataNode.h>
#include <mitkPositionEvent.h>
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
   * \brief Returns the MIDASOrientation corresponding to the XY plane, or else returns the supplied default.
   * \param defaultOrientation A default MIDASOrientation that will be returned if we can't work out the As Acquired orientation.
   * \param image An image to check.
   * \return MIDASOrientation the As Acquired orientation, or the defaultOrientation.
   */
  NIFTKCORE_EXPORT MIDASOrientation GetAsAcquiredOrientation(const MIDASOrientation& defaultOrientation, const mitk::Image* image);


  /**
   * \brief Simply returns true if a node contains an image, and false otherwise.
   * \param node An MITK DataNode.
   * \return true if node contains an mitk::Image and false otherwise
   */
  NIFTKCORE_EXPORT bool IsImage(const mitk::DataNode* node);


  /**
   * \brief \see ImagesHaveEqualIntensities
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
  NIFTKCORE_EXPORT bool ImagesHaveEqualIntensities(const mitk::Image* image1, const mitk::Image* image2);


  /**
   * \brief \see ImagesHaveSameSpatialExtent
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
  NIFTKCORE_EXPORT bool ImagesHaveSameSpatialExtent(const mitk::Image* image1, const mitk::Image* image2);


  /**
   * \brief \see FillImage
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
  NIFTKCORE_EXPORT void FillImage(mitk::Image* image, float value);


  /**
   * \brief \see CountBetweenThreshold
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
  NIFTKCORE_EXPORT unsigned long int CountBetweenThreshold(const mitk::Image* image, const float& lower, const float& upper);


  /**
   * \brief Returns the number of voxels in an image.
   * \param image An MITK Image.
   * \return unsigned long int The number of voxels.
   */
  NIFTKCORE_EXPORT unsigned long int GetNumberOfVoxels(const mitk::Image* image);


  /**
   * \brief Returns the middle voxel of an image.
   * \param image An MITK image.
   * \return If the image has odd numbers of voxels in an axis, the returned voxel is
   * the middle one, whereas if the image has even number of voxels in an axis, the
   * returned voxel is <code>(int)(number vox - 1)/2.0</code>
   */
  NIFTKCORE_EXPORT mitk::Point3D GetMiddlePointInVoxels(const mitk::Image* image);


  /**
   * \brief Generates a fake position event, (mainly for unit testing), at a given voxel location.
   * \param image An MITK image.
   * \param voxelLocation A voxel location.
   * \return A fake position event, where by "fake" we mean that there is no valid window Id.
   */
  NIFTKCORE_EXPORT mitk::PositionEvent GeneratePositionEvent(const mitk::BaseRenderer* renderer, const mitk::Image* image, const mitk::Point3D& voxelLocation);


  /**
   * \brief Returns the volume of non-zero voxels in an image.
   * \param image An MITK image.
   * \return double The volume in millimetres cubed.
   */
  NIFTKCORE_EXPORT double GetVolume(const mitk::Image* image);


  /**
   * \brief Calculates the volume of non-zero voxels in image, and creates a property "midas.volume" on the data node.
   * \param image An MITK image, not NULL.
   * \param node An MITK DataNode, not NULL.
   */
  NIFTKCORE_EXPORT void UpdateVolumeProperty(const mitk::Image* image, mitk::DataNode* node);

  /**
   * \brief ITK function to copy image data, performing C-style casting between data types.
   * \param itkImage1 An ITK image.
   * \param itkImage2 An ITK image.
   */
  template <typename TPixel1, unsigned int VImageDimension1, typename TPixel2, unsigned int VImageDimension2>
  void ITKCopyIntensityData(itk::Image<TPixel1, VImageDimension1>* itkImage1,
                        itk::Image<TPixel2, VImageDimension2>* itkImage2
                       );

  /**
   * \brief Assumes same size image, and same data type, and copies data from the input image to the output image.
   * \param input Input MITK image, not NULL.
   * \param output Output MITK image, not NULL, same size as input.
   */
  NIFTKCORE_EXPORT void CopyIntensityData(const mitk::Image* input, mitk::Image* output);


  /**
   * \brief Writes the image to file.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKDumpImage(
      const itk::Image<TPixel, VImageDimension>* itkImage,
      const std::string& filename
      );

  /**
   * \brief Writes the image to file.
   */
  NIFTKCORE_EXPORT void DumpImage(const mitk::Image *input, const std::string& fileName);

  /**
   * \brief Utility method to calculate the aspect ratio.
   *
   * Retrieves the image scaling, and if X dimension has largest voxels e.g. 2.1mm compared with Y having 1.1mm,
   * will calculate the aspect ratio for X, so that you can rescale the number of pixels in the
   * X direction and leave Y un-altered. And vice-versa for Y.
   *
   * eg. given X = 2.2mm, Y = 1.1mm, will return mitk::Vector3D with vector[0]=2 and vector[1] = 1;
   *
   * i.e. the x axis is twice as big as the y axis.
   */
  NIFTKCORE_EXPORT mitk::Vector3D GetXYAspectRatio(const mitk::Image::Pointer image);

}

#endif
