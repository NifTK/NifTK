/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkStereoHandeyeFromTwoDirectories_h
#define __mitkStereoHandeyeFromTwoDirectories_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class StereoHandeyeFromTwoDirectories_h
 * \brief Does an intrinsic and handeye calibration for both channels of a stereo scope
 * using StereoCameraCalibrationFromTwoDirectories and Handeye
 */
class NIFTKOPENCV_EXPORT StereoHandeyeFromTwoDirectories : public itk::Object
{

public:

  mitkClassMacro(StereoHandeyeFromTwoDirectories, itk::Object);
  itkNewMacro(StereoHandeyeFromTwoDirectories);

  /**
   * \brief Calibration function that returns the reprojection error (squared error).
   */
  double Calibrate(const std::string& leftDirectoryName,
      const std::string& rightDirectoryName,
      const int& numberCornersX,
      const int& numberCornersY,
      const float& sizeSquareMillimeters,
      const std::string& outputFileName,
      const bool& writeImages
      );

protected:

  StereoHandeyeFromTwoDirectories();
  virtual ~StereoHandeyeFromTwoDirectories();

  StereoHandeyeFromTwoDirectories(const StereoHandeyeFromTwoDirectories&); // Purposefully not implemented.
  StereoHandeyeFromTwoDirectories& operator=(const StereoHandeyeFromTwoDirectories&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
