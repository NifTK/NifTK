/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCameraCalibrationFromDirectory_h
#define mitkCameraCalibrationFromDirectory_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \class CameraCalibrationFromDirectory
 * \brief Does a camera calibration from a directory containing a set of image files.
 */
class NIFTKOPENCV_EXPORT CameraCalibrationFromDirectory : public itk::Object
{

public:

  mitkClassMacro(CameraCalibrationFromDirectory, itk::Object);
  itkNewMacro(CameraCalibrationFromDirectory);

  /**
   * \brief Calibration function that returns the reprojection error (squared error).
   * \param squareSizeInMillimetres the physical size of the square as printed out on the calibration object.
   * \param pixelScaleFactor the caller can specify a multiplier for the number of pixels in each direction to scale up/down the image.
   */
  double Calibrate(const std::string& fullDirectoryName,
      const int& numberCornersX,
      const int& numberCornersY,
      const float& sizeSquareMillimeters,
      const mitk::Point2D& pixelScaleFactor,
      const std::string& outputFile,
      const bool& writeImages
      );

protected:

  CameraCalibrationFromDirectory();
  virtual ~CameraCalibrationFromDirectory();

  CameraCalibrationFromDirectory(const CameraCalibrationFromDirectory&); // Purposefully not implemented.
  CameraCalibrationFromDirectory& operator=(const CameraCalibrationFromDirectory&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
