/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPivotCalibration_h
#define mitkPivotCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <cv.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class PivotCalibration
 * \brief Does a pivot calibration from a vector of 4x4 matrices,
 * such as might be used to calibrate an image guided surgery pointing device.
 *
 * Inspired by: Feuerstein et. al. Intraoperative Laparoscope Augmentation
 * for Port Placement and Resection Planning. IEEE TMI Vol 27, No 3. March 2008.
 */
class NIFTKOPENCV_EXPORT PivotCalibration : public itk::Object
{

public:

  mitkClassMacro(PivotCalibration, itk::Object);
  itkNewMacro(PivotCalibration);

  itkSetMacro(SingularValueThreshold, double);
  itkGetMacro(SingularValueThreshold, double);

  /**
   * \brief Method that provides directory scanning before calling the other calibrate method.
   *
   * More specifically, it is assumed that a directory is provided containing JUST
   * the 4x4 matrices, where each matrix is a separate plain text file, containing 4 rows
   * of 4 floating point numbers, each number separated by spaces.
   *
   * \param[In] matrixDirectory directory containing tracking matrices
   * \param[Out] residualError the root mean square distance of each re-constructed point from the invariant point.
   * \param[Out] outputMatrix the output transformation.
   * \param[In] percentage if < 100 and > 0 will take a randomly chosen number of matrices and run calibration,
   * and measure the mean and standard deviation of the RMS reconstrution error.
   * \param[In] reruns the number of re-runs to do when you are randomly selecting data.
   */
  void CalibrateUsingFilesInDirectories(
      const std::string& matrixDirectory,
      double &residualError,
      vtkMatrix4x4& outputMatrix,
      const int& percentage = 100,
      const int& reruns = 100
      );

  /**
   * \brief Performs pivot (invariant point) based calibration.
   * \param[In] matrices a vector of 4x4 matrices representing rigid body tracking transformation.
   * \param[Out] outputMatrix the calibration matrix
   * \param[Out] residualError the root mean square distance of each re-constructed point from the invariant point.
   * \param[In] percentage if < 100 and > 0 will take a randomly chosen number of matrices and run calibration,
   * and measure the mean and standard deviation of the RMS reconstrution error.
   * \param[In] reruns the number of re-runs to do when you are randomly selecting data.
   */
  void Calibrate(
      const std::vector< cv::Mat >& matrices,
      cv::Matx44d& outputMatrix,
      double& residualError,
      const int& percentage = 100,
      const int& reruns = 100
      );

protected:

  PivotCalibration();
  virtual ~PivotCalibration();

  PivotCalibration(const PivotCalibration&); // Purposefully not implemented.
  PivotCalibration& operator=(const PivotCalibration&); // Purposefully not implemented.

private:

  void DoCalibration(
    const std::vector< cv::Mat >& matrices,
    cv::Matx44d& outputMatrix,
    double& residualError
    );

  double m_SingularValueThreshold;

}; // end class

} // end namespace

#endif
