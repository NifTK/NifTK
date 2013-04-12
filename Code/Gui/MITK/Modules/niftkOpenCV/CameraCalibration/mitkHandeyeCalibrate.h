/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkHandeyeCalibrate_h
#define __mitkHandeyeCalibarte_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>

namespace mitk {

/**
 * \class HandeyeCalibrate_h
 * \brief Computes the handeye calibration using extrinsic calibration and tracking
 * data, using TSAI's least squares method
 */
class NIFTKOPENCV_EXPORT HandeyeCalibrate : public itk::Object
{

public:

  mitkClassMacro(HandeyeCalibrate, itk::Object);
  itkNewMacro(HandeyeCalibrate);

  /**
   * \brief Calibration function that returns the reprojection error (squared error).
   */
  double Calibrate(const std::vector<cv::Mat> MarkerToWorld,
      const std::vector <cv::Mat> GridToCamera,
      cv::Mat CameraToMarker
      );

  /**
   * \brief Read a set of matrices from a directory and 
   * put them in a vector of 4x4 cvMats
   */
  std::vector<cv::Mat> LoadMatricesFromDirectory (const std::string& fullDirectoryName);

  /**
   * \brief Load a set of matrices from a file describing the 
   * extrinsic parameters of a standard camera calibration
   */
  std::vector<cv::Mat> LoadMatricesFromExtrinsicFile (const std::string& fullFileName);

protected:

  HandeyeCalibrate();
  virtual ~HandeyeCalibrate();

  HandeyeCalibrate(const HandeyeCalibrate&); // Purposefully not implemented.
  HandeyeCalibrate& operator=(const HandeyeCalibrate&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
