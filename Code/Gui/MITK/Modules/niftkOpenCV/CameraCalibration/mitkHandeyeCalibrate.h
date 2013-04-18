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
  cv::Mat Calibrate(const std::vector<cv::Mat> MarkerToWorld,
      const std::vector <cv::Mat> GridToCamera, 
      std::vector<double>* Residuals = NULL);

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

  /**
   * \brief Flips the matrices in the vector from left handed coordinate 
   * system to right handed and vice versa
   */
  std::vector<cv::Mat> FlipMatrices (const std::vector<cv::Mat> Matrices);

 /**
   * \brief Sorts the matrices based on the translations , and returns the order
   */
  std::vector<int> SortMatricesByDistance (const std::vector<cv::Mat> Matrices);
 
  /**
   * \brief Sorts the matrices based on the rotations, and returns the order
   */
  std::vector<int> SortMatricesByAngle (const std::vector<cv::Mat> Matrices);

protected:

  HandeyeCalibrate();
  virtual ~HandeyeCalibrate();

  HandeyeCalibrate(const HandeyeCalibrate&); // Purposefully not implemented.
  HandeyeCalibrate& operator=(const HandeyeCalibrate&); // Purposefully not implemented.

private:
  /**
   * \brief Returns the angular distance between two rotation matrices
   */
  double AngleBetweenMatrices(cv::Mat Mat1 , cv::Mat Mat2);
  /**
   * \brief Converts a 3x3 rotation matrix to a quaternion
   */
  cv::Mat DirectionCosineToQuaternion(cv::Mat dc_Matrix);
  /**
   * \brief Returns -1.0 if value < 0 or 1.0 if value >= 0
   */
  double ModifiedSignum(double value);
  /**
   * \brief Returns 0.0 of value < 0 or sqrt(value) if value >= 0
   */
  double SafeSQRT(double value);

}; // end class

} // end namespace

#endif
