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
#include "mitkMIDASEnums.h"

/**
 * \file mitkMIDASImageUtils.h
 * \brief Some useful MIDAS related image utilities, such as working out the As Acquired orientation.
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
    itk::Image<TPixel, VImageDimension>* itkImage,
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

}

#endif // MITKMIDASIMAGEUTILS_H
