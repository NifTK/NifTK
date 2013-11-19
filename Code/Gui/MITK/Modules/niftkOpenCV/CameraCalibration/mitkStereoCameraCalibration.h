/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkStereoCameraCalibration_h
#define mitkStereoCameraCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \class StereoCameraCalibration
 * \brief Does a stereo camera calibration from one/two directories, each containing a set of image files.
 */
class NIFTKOPENCV_EXPORT StereoCameraCalibration : public itk::Object
{

public:

  mitkClassMacro(StereoCameraCalibration, itk::Object);
  itkNewMacro(StereoCameraCalibration);

  /**
   * \brief Calibration function that returns the reprojection error (squared error).
   * \param numberOfFrames if != 0, will pick either left or right directory, scan for image pairs (sequential files),
   * try to extract chessboards on all frames, and build a list of suitable pairs, and then randomly select a suitable number of frames.
   * \param squareSizeInMillimetres the physical size of the square as printed out on the calibration object.
   * \param pixelScaleFactor the caller can specify a multiplier for the number of pixels in each direction to scale up/down the image.
   */
  double Calibrate(const std::string& leftDirectoryName,
      const std::string& rightDirectoryName,
      const int& numberOfFrames,
      const int& numberCornersX,
      const int& numberCornersY,
      const double& sizeSquareMillimeters,
      const mitk::Point2D& pixelScaleFactor,
      const std::string& outputFileName,
      const bool& writeImages
      );

protected:

  StereoCameraCalibration();
  virtual ~StereoCameraCalibration();

  StereoCameraCalibration(const StereoCameraCalibration&); // Purposefully not implemented.
  StereoCameraCalibration& operator=(const StereoCameraCalibration&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
