/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkOpenCVMaths_h
#define mitkOpenCVMaths_h

#include "niftkOpenCVExports.h"
#include <cv.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

/**
 * \file mitkOpenCVMaths.h
 * \brief Various simple mathematically based functions using OpenCV data types.
 * We try to minimise the amount of code exported from this module.
 */
namespace mitk {

/**
 * \brief Calculates the centroid of a vector of points.
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point3d GetCentroid(const std::vector<cv::Point3d>& points);


/**
 * \brief Subtracts a point (e.g. the centroid) from a list of points.
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<cv::Point3d> SubtractPointFromPoints(const std::vector<cv::Point3d> listOfPoints, const cv::Point3d& point);


/**
 * \brief Converts mitk::PointSet to vector of cv::Point3d, but you lose the point ID contained within the mitk::PointSet.
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<cv::Point3d> PointSetToVector(const mitk::PointSet::Pointer& pointSet);


/**
 * \brief Returns true if fabs(value) is less than a small tolerance, which defaults to 0.000001.
 */
extern "C++" NIFTKOPENCV_EXPORT bool IsCloseToZero(const double& value, const double& tolerance = 0.000001);


/**
 * \brief Haven't found a direct method to do this yet.
 */
extern "C++" NIFTKOPENCV_EXPORT void MakeIdentity(cv::Matx44d& outputMatrix);


/**
 * \brief Calculates 1/N Sum (q_i * qPrime_i^t) where q_i and qPrime_i are column vectors, so the product is a 3x3 matrix.
 * \see Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987, DOI=10.1109/TPAMI.1987.4767965, matrix H.
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d CalculateCrossCovarianceH(const std::vector<cv::Point3d>& q, const std::vector<cv::Point3d>& qPrime);


/**
 * \brief Helper method to do the main point based registration, and handle error conditions.
 */
extern "C++" NIFTKOPENCV_EXPORT bool DoSVDPointBasedRegistration(const std::vector<cv::Point3d>& fixedPoints,
  const std::vector<cv::Point3d>& movingPoints,
  cv::Matx33d& H,
  cv::Point3d &p,
  cv::Point3d& pPrime,
  cv::Matx44d& outputMatrix,
  double &fiducialRegistrationError
  );


/**
 * \brief Calculates Fiducial Registration Error by multiplying the movingPoints by the matrix, and comparing with fixedPoints.
 */
extern "C++" NIFTKOPENCV_EXPORT double CalculateFiducialRegistrationError(const std::vector<cv::Point3d>& fixedPoints,
  const std::vector<cv::Point3d>& movingPoints,
  const cv::Matx44d& matrix
  );


/**
 * \brief Converts format of input to call the other CalculateFiducialRegistrationError method.
 */
extern "C++" NIFTKOPENCV_EXPORT double CalculateFiducialRegistrationError(const mitk::PointSet::Pointer& fixedPointSet,
  const mitk::PointSet::Pointer& movingPointSet,
  vtkMatrix4x4& vtkMatrix
  );


/**
 * \brief Simply copies the translation vector and rotation matrix into the 4x4 matrix.
 */
extern "C++" NIFTKOPENCV_EXPORT void ConstructAffineMatrix(const cv::Matx31d& translation, const cv::Matx33d& rotation, cv::Matx44d& matrix);


/**
 * \brief Copies matrix to vtkMatrix.
 */
extern "C++" NIFTKOPENCV_EXPORT void CopyToVTK4x4Matrix(const cv::Matx44d& matrix, vtkMatrix4x4& vtkMatrix);


/**
 * \brief Copies matrix to openCVMatrix.
 */
extern "C++" NIFTKOPENCV_EXPORT void CopyToOpenCVMatrix(const vtkMatrix4x4& matrix, cv::Matx44d& openCVMatrix);


/**
 * \brief Generates a rotation about X-axis, given a Euler angle in radians.
 * \param rx angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRxMatrix(const double& rx);


/**
 * \brief Generates a rotation about Y-axis, given a Euler angle in radians.
 * \param ry angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRyMatrix(const double& ry);


/**
 * \brief Generates a rotation about Z-axis, given a Euler angle in radians.
 * \param rz angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRzMatrix(const double& rz);


/**
 * \brief Generates a rotation matrix, given Euler angles in radians.
 * \param rx angle in radians
 * \param ry angle in radians
 * \param rz angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRotationMatrix(const double& rx, const double& ry, const double& rz);


/**
 * \brief Converts Euler angles in radians to the Rodrigues rotation vector (axis-angle convention) mentioned in OpenCV.
 * \param rx Euler angle rotation about x-axis in radians
 * \param ry Euler angle rotation about y-axis in radians
 * \param rz Euler angle rotation about z-axis in radians
 * \return A new [1x3] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx13d ConvertEulerToRodrigues(
  const double& rx,
  const double& ry,
  const double& rz
  );


/**
 * \brief From rotations in radians and translations in millimetres, constructs a 4x4 transformation matrix, using OpenCV conventions.
 * \param rx Euler rotation about x-axis in radians
 * \param ry Euler rotation about y-axis in radians
 * \param rz Euler rotation about z-axis in radians
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructRigidTransformationMatrix(
  const double& rx,
  const double& ry,
  const double& rz,
  const double& tx,
  const double& ty,
  const double& tz
  );


/**
 * \brief Constructs a scaling matrix from sx, sy, sz where the scale factors simply appear on the diagonal.
 * \param sx scale factor in x direction
 * \param sy scale factor in y direction
 * \param sz scale factor in z direction
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructScalingTransformation(const double& sx, const double& sy, const double& sz = 1);


/**
 * \brief Constructs an affine transformation, without skew using the specified parameters, where rotations are in degrees.
 * \param rx Euler rotation about x-axis in radians
 * \param ry Euler rotation about y-axis in radians
 * \param rz Euler rotation about z-axis in radians
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \param sx scale factor in x direction
 * \param sy scale factor in y direction
 * \param sz scale factor in z direction
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructSimilarityTransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz,
    const double& sx,
    const double& sy,
    const double& sz
    );
} // end namespace

#endif



