/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundPinCalibration_h
#define mitkUltrasoundPinCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <cv.h>

namespace mitk {

/**
 * \class UltrasoundPinCalibration
 * \brief Does an ultrasound probe calibration from an ordered list of tracker matrices, and pin locations (x,y pixels).
 */
class NIFTKOPENCV_EXPORT UltrasoundPinCalibration : public itk::Object
{

public:

  mitkClassMacro(UltrasoundPinCalibration, itk::Object);
  itkNewMacro(UltrasoundPinCalibration);

  bool CalibrateUsingInvariantPointAndFilesInTwoDirectories(
      const std::string& matrixDirectory,
      const std::string& pointDirectory,
      const mitk::Point3D& invariantPoint,
      const mitk::Point2D& originInImagePlaneInPixels,
      const mitk::Point2D& millimetresPerPixel,
      const std::vector<double>& initialGuessOfTransformation,
      const bool& optimiseScaling,
      const bool& optimiseInvariantPoint,
      double &residualError,
      const std::string& outputFileName
      );

  bool Calibrate(
      const std::vector< cv::Mat >& matrices,
      const std::vector< cv::Point2d >& points,
      const cv::Point3d& invariantPoint,
      const cv::Point2d& originInImagePlaneInPixels,
      const cv::Point2d& millimetresPerPixel,
      const std::vector<double>& initialGuessOfTransformation,
      const bool& optimiseScaling,
      const bool& optimiseInvariantPoint,
      double& residualError,
      cv::Matx44d& outputMatrix
      );

protected:

  UltrasoundPinCalibration();
  virtual ~UltrasoundPinCalibration();

  UltrasoundPinCalibration(const UltrasoundPinCalibration&); // Purposefully not implemented.
  UltrasoundPinCalibration& operator=(const UltrasoundPinCalibration&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
