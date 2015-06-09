/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkHandeyeCalibrateUsingRegistration_h
#define niftkHandeyeCalibrateUsingRegistration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>
#include <cv.h>

namespace niftk {

/**
* \class HandeyeCalibrateUsingRegistration
* \brief Computes hand-eye using registration.
*/
class NIFTKOPENCV_EXPORT HandeyeCalibrateUsingRegistration : public itk::Object
{

public:

  mitkClassMacro(HandeyeCalibrateUsingRegistration, itk::Object);
  itkNewMacro(HandeyeCalibrateUsingRegistration);
  
  /**
  * \brief Calibration function that computes a hand-eye matrix using registration.
  * \param modelInputFile a filename of an mitk::PointSet containing chessboard positions.
  * \param modelTrackingDirectory if specified a directory containing tracking matrices of tracked chessboard,
  *        where all points in modelInputFile will be multiplied by these matrices
  *        to give coordinates in tracker (world) space.
  * \param cameraPointsDirectory reconstructed chessboard points from the camera view.
  * \param distanceThreshold if the distance from the camera is >= this threshold then that view is ignored.
  * \param fiducialRegistrationThreshold if the FRE is >= this threshold, then that view is ignored.
  * \param outputMatrixFile the output matrix relating tracker (hand) to camera (eye).
  *
  * The number of matrices in modelTrackingDirectory must be zero (or the modelTrackingDirectory.length() == 0),
  * or match the number of matrices in cameraPointsDirectory. It is up to whatever generated
  * the matrices and points to make sure that they are all synchronised in time.
  */
  void Calibrate (
    const std::string& modelInputFile,
    const std::string& modelTrackingDirectory,
    const std::string& cameraPointsDirectory,
    const std::string& handTrackingDirectory,
    const double& distanceThreshold,
    const double& fiducialRegistrationThreshold,
    const std::string& outputMatrixFile
  );

  /**
  * \brief The actual calibration method.
  * \param modelPointSet representing calibration object (chessboard)
  * \param trackingMatrices vector of 4x4 matrices, which should be empty, or the same length as pointsInCameraSpace.
  * \param pointsInCameraSpace vector of camera views of the chessboard, where points are in camera space.
  * \param distanceThreshold camera views are rejected if the distance from the camera is >= distanceThreshold.
  * \param fiducialRegistrationThreshold camera views are rejected if the FRE is >= fiducialRegistrationThreshold.
  */
  void Calibrate (
    const mitk::PointSet& modelPointSet,
    const std::vector<cv::Mat>& modelTrackingMatrices,
    const std::vector<cv::Mat>& handTrackingMatrices,
    const std::vector<mitk::PointSet::Pointer>& pointsInCameraSpace,
    const double& distanceThreshold,
    const double& fiducialRegistrationThreshold,
    vtkMatrix4x4& outputMatrix
  );

protected:

  HandeyeCalibrateUsingRegistration();
  virtual ~HandeyeCalibrateUsingRegistration();

  HandeyeCalibrateUsingRegistration(const HandeyeCalibrateUsingRegistration&);
  HandeyeCalibrateUsingRegistration& operator=(const HandeyeCalibrateUsingRegistration&);

private:

}; // end class

} // end namespace

#endif
