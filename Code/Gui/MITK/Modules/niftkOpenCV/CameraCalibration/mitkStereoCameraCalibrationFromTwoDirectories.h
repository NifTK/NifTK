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

#ifndef MITKSTEREOCAMERACALIBRATIONFROMTWODIRECTORIES_H
#define MITKSTEREOCAMERACALIBRATIONFROMTWODIRECTORIES_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

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

  bool Calibrate(const std::string& leftDirectoryName,
      const std::string& rightDirectoryName,
      const int& numberCornersX,
      const int& numberCornersY,
      const float& sizeSquareMillimeters,
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
