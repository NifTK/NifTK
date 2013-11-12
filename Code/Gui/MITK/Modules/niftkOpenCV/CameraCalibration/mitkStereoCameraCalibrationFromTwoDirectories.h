/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkStereoCameraCalibrationFromTwoDirectories_h
#define mitkStereoCameraCalibrationFromTwoDirectories_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \class StereoCameraCalibrationFromTwoDirectories
 * \brief Does a stereo camera calibration from two directories, each containing a set of image files.
 */
class NIFTKOPENCV_EXPORT StereoCameraCalibrationFromTwoDirectories : public itk::Object
{

public:

  mitkClassMacro(StereoCameraCalibrationFromTwoDirectories, itk::Object);
  itkNewMacro(StereoCameraCalibrationFromTwoDirectories);

  /**
   * \brief Calibration function that returns the reprojection error (squared error).
   * \param squareSizeInMillimetres the physical size of the square as printed out on the calibration object.
   * \param pixelScaleFactor the caller can specify a multiplier for the number of pixels in each direction to scale up/down the image.
   */
  double Calibrate(const std::string& leftDirectoryName,
      const std::string& rightDirectoryName,
      const int& numberCornersX,
      const int& numberCornersY,
      const double& sizeSquareMillimeters,
      const mitk::Point2D& pixelScaleFactor,
      const std::string& outputFileName,
      const bool& writeImages
      );

protected:

  StereoCameraCalibrationFromTwoDirectories();
  virtual ~StereoCameraCalibrationFromTwoDirectories();

  StereoCameraCalibrationFromTwoDirectories(const StereoCameraCalibrationFromTwoDirectories&); // Purposefully not implemented.
  StereoCameraCalibrationFromTwoDirectories& operator=(const StereoCameraCalibrationFromTwoDirectories&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
