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
#include <vtkMatrix4x4.h>

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

  /**
   * \brief Method that provides directory scanning before calling the other calibrate method.
   *
   * More specifically, it will look in 2 directories, where 1 directory contains JUST the tracking
   * matrices, with each matrix in a plain text file of 4 rows of 4 columns and the second directory
   * contains just the points, each point in a separate plain text file. The filenames are used
   * for sorting so that there must be the same number of files in each directory, and the sort
   * order must correspond.
   *
   * \param[In] matrixDirectory directory containing tracking matrices
   * \param[In] pointDirectory directory containing 2D pixel location of a pin-head.
   * \param[In] optimiseScaling if true the scaling will be optimised along with the 6DOF calibration matrix.
   * \param[In] optimiseInvariantPoint if true the position of the invariant point will be optimised.
   * \param[Out] rigidBodyTransformation rx, ry, rz, tx, ty, tz where rotations in radians and translations in millimetres.
   * \param[Out] invariantPoint an initial guess at the invariant point, or equivalently the tracker to pin-head transformation. i.e. the pin-head in tracker coordinates.
   * \param[Out] millimetresPerPixel scale factors for the ultrasound image in both x and y direction.
   * \param[Out] residualError the root mean square distance of each constructed point from the theoretical pin position (0, 0, 0).
   * \param[Out] outputMatrix the output transformation.
   */
  bool CalibrateUsingInvariantPointAndFilesInTwoDirectories(
      const std::string& matrixDirectory,
      const std::string& pointDirectory,
      const bool& optimiseScaling,
      const bool& optimiseInvariantPoint,
      std::vector<double>& rigidBodyTransformation,
      mitk::Point3D& invariantPoint,
      mitk::Point2D& millimetresPerPixel,
      double &residualError,
      vtkMatrix4x4& outputMatrix
      );

  /**
   * \brief Performs pin-head (invariant-point) calibration.
   * \param[In] matrices a vector of 4x4 matrices representing rigid body tracking transformation.
   * \param[In] points a vector of 2D pixel locations in the same order as the tracking transformations.
   * \param[In] optimiseScaling if true the scaling will be optimised along with the 6DOF calibration matrix.
   * \param[In] optimiseInvariantPoint if true the position of the invariant point will be optimised.
   * \param[Out] rigidBodyTransformation rx, ry, rz, tx, ty, tz where rotations in radians and translations in millimetres.
   * \param[Out] invariantPoint an initial guess at the invariant point, or equivalently the tracker to pin-head transformation. i.e. the pin-head in tracker coordinates.
   * \param[Out] millimetresPerPixel scale factors for the ultrasound image in both x and y direction.
   * \param[Out] outputMatrix the calibration matrix
   * \param[Out] residualError the root mean square distance of each constructed point from the theoretical pin position (0, 0, 0).
   */
  bool Calibrate(
      const std::vector< cv::Mat >& matrices,
      const std::vector< cv::Point2d >& points,
      const bool& optimiseScaling,
      const bool& optimiseInvariantPoint,
      std::vector<double>& rigidBodyTransformation,
      cv::Point3d& invariantPoint,
      cv::Point2d& millimetresPerPixel,
      cv::Matx44d& outputMatrix,
      double& residualError
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
