/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundTransformAndImageMerger_h
#define mitkUltrasoundTransformAndImageMerger_h

#include "niftkOpenCVExports.h"
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \class UltrasoundTransformAndImageMerger
 * \brief Merges a directory of images and tracking data into a .mhd file, according to PLUS meta-data specifications.
 *
 * This takes each 2D slice, and stacks them into a 3D volume. It writes the data out as a .mhd file.
 * .mhd files have the binary image data in a file ending in .raw, and .mhd is an ASCII header.
 * Therefore PLUS have specified additional information to be stored in the header. This additional
 * information pertains to tracking information collected at the time the images were grabbed.
 * The tracking information normally comprises the probe-to-tracker transform, the
 * reference-to-tracker transform and the stylus-to-tracker transform. For our purposes,
 * we just want to get the data into PLUS to use, for example, the fCal calibration or
 * the temporal calibration. So, inputMatrixDirectory refers to a directory full of
 * probe tracking transformations, giving the probe-to-tracker transform.
 * The reference-to-tracker and stylus-to-tracker are set to identity.
 *
 * Furthermore, NifTK will record tracking and image data at whatever framerate
 * the devices support. So, this method takes an image, and will interpolate the
 * timestamps of the tracking data, and hence interpolate the transformation matrices.
 */
class NIFTKOPENCV_EXPORT UltrasoundTransformAndImageMerger : public itk::Object
{

public:

  mitkClassMacro(UltrasoundTransformAndImageMerger, itk::Object);
  itkNewMacro(UltrasoundTransformAndImageMerger);

  /**
   * \brief Does merging.
   */
  void Merge(
      const std::string& inputMatrixDirectory,
      const std::string& inputImageDirectory,
      const std::string& outputImageFileName,
      const std::string& outputDataFileName,
      const std::string& imageOrientation
      );

protected:

  UltrasoundTransformAndImageMerger();
  virtual ~UltrasoundTransformAndImageMerger();

  UltrasoundTransformAndImageMerger(const UltrasoundTransformAndImageMerger&); // Purposefully not implemented.
  UltrasoundTransformAndImageMerger& operator=(const UltrasoundTransformAndImageMerger&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
