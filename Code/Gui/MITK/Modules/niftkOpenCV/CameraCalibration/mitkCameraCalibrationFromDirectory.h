/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKCAMERACALIBRATIONFROMDIRECTORY_H
#define MITKCAMERACALIBRATIONFROMDIRECTORY_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

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

  bool Calibrate(const std::string& fullDirectoryName,
      const int& numberCornersX,
      const int& numberCornersY,
      const float& sizeSquareMillimeters,
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
