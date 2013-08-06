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
cv::Point3d GetCentroid(const std::vector<cv::Point3d>& points);

/**
 * \brief Subtracts a point (e.g. the centroid) from a list of points.
 */
std::vector<cv::Point3d> SubtractPointFromPoints(const std::vector<cv::Point3d> listOfPoints, const cv::Point3d& point);

/**
 * \brief Converts mitk::PointSet to vector of cv::Point3d, but you lose the point ID.
 */
std::vector<cv::Point3d> PointSetToVector(const mitk::PointSet::Pointer& pointSet);

/**
 * \brief Returns true if fabs(value) is less than a small tolerance (see code).
 */
bool IsCloseToZero(const double& value);

/**
 * \brief Haven't found a direct method to do this yet.
 */
void MakeIdentity(cv::Matx44d& outputMatrix);

/**
 * \brief Calculates 1/N Sum (q_i * qPrime_i^t) where q_i and qPrime_i are column vectors, so the product is a 3x3 matrix.
 * \see Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987, DOI=10.1109/TPAMI.1987.4767965, matrix H.
 */
cv::Matx33d CalculateCrossCovarianceH(const std::vector<cv::Point3d>& q, const std::vector<cv::Point3d>& qPrime);

/**
 * \brief Helper method to do the main point based registration, and handle error conditions.
 */
bool DoSVDPointBasedRegistration(const std::vector<cv::Point3d>& fixedPoints,
                                 const std::vector<cv::Point3d>& movingPoints,
                                 cv::Matx33d& H,
                                 cv::Point3d &p,
                                 cv::Point3d& pPrime,
                                 cv::Matx44d& outputMatrix,
                                 double &fiducialRegistrationError);

/**
 * \brief Calculates Fiducial Registration Error by multiplying the movingPoints by the matrix, and comparing with fixedPoints.
 */
double CalculateFiducialRegistrationError(const std::vector<cv::Point3d>& fixedPoints,
                                          const std::vector<cv::Point3d>& movingPoints,
                                          const cv::Matx44d& matrix
                                          );

/**
 * \brief Converts format of input to call the other CalculateFiducialRegistrationError method.
 */
NIFTKOPENCV_EXPORT double CalculateFiducialRegistrationError(const mitk::PointSet::Pointer& fixedPointSet,
                                                             const mitk::PointSet::Pointer& movingPointSet,
                                                             vtkMatrix4x4& vtkMatrix);

/**
 * \brief Simply copies the translation vector and rotation matrix into the 4x4 matrix.
 */
void Setup4x4Matrix(const cv::Matx31d& translation, const cv::Matx33d& rotation, cv::Matx44d& matrix);

/**
 * \brief Copies matrix to vtkMatrix.
 */
void CopyToVTK4x4Matrix(const cv::Matx44d& matrix, vtkMatrix4x4& vtkMatrix);

/**
 * \brief Copies matrix to openCVMatrix.
 */
void CopyToOpenCVMatrix(const vtkMatrix4x4& matrix, cv::Matx44d& openCVMatrix);


/**
 * \brief Generates a rotation about X-axis, given a Euler angle in radians.
 * \param rx angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRxMatrix(const double& rx, CvMat &matrix3x3);


/**
 * \brief Generates a rotation about Y-axis, given a Euler angle in radians.
 * \param ry angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRyMatrix(const double& ry, CvMat &matrix3x3);


/**
 * \brief Generates a rotation about Z-axis, given a Euler angle in radians.
 * \param rz angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRzMatrix(const double& rz, CvMat &matrix3x3);


/**
 * \brief Generates a rotation matrix, given Euler angles in radians.
 * \param rx angle in radians
 * \param ry angle in radians
 * \param rz angle in radians
 * \param matrix3x3 pre-allocated [3x3] matrix
 */
void GenerateEulerRotationMatrix(const double& rx, const double& ry, const double& rz, CvMat &matrix3x3);

/**
 * \brief Converts Euler angles in radians to the Rodrigues rotation vector (axis-angle convention) mentioned in OpenCV.
 * \param rx Euler angle rotation about x-axis in radians
 * \param ry Euler angle rotation about y-axis in radians
 * \param rz Euler angle rotation about z-axis in radians
 * \return A new [1x3] matrix that the caller must then de-allocate.
 */
CvMat* ConvertEulerToRodrigues(
    const double& rx,
    const double& ry,
    const double& rz
    );


/**
 * \brief Converts from the Rodrigues rotation vector mentioned in OpenCV to 3 Euler angles in radians.
 * \param rotationVector a [1x3] Rodrigues rotation vector
 * \return A new [1x3] matrix that the caller must then de-allocate containing rx, ry, rz in radians, as a result of calling OpenCV's RQDecomp3x3.
 */
CvMat* ConvertRodriguesToEuler(
    const CvMat& rotationVector
    );


/**
 * \brief From rotations in radians and translations in millimetres, constructs a 4x4 transformation matrix, using OpenCV conventions.
 * \param rx Euler rotation about x-axis in radians
 * \param ry Euler rotation about y-axis in radians
 * \param rz Euler rotation about z-axis in radians
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \return a new [4x4] matrix that the caller must then de-allocate.
 */
CvMat* Construct4x4TransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    );


/**
 * \brief From rotations in degrees (+/- 180), converts to radians, then passes on to Construct4x4TransformationMatrix.
 * \return a new [4x4] matrix that the caller must then de-allocate.
 */
CvMat* Construct4x4TransformationMatrixFromDegrees(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    );

/**
 * \brief Transforms a [Nx3] matrix of points (or normals) by a [4x4] matrix.
 * \param input3D [Nx3] matrix of points/normals
 * \param matrix4x4 [4x4] transformation matrix
 * \param isNormals if true, will assume that the input3D are normals, and if false, just 3D points.
 * \param output3DPoints [Nx3] matrix of points as output.
 */
void TransformBy4x4Matrix(
    const CvMat &input3D,
    const CvMat &matrix4x4,
    const bool &isNormals,
    CvMat& output3DPoints
    );

} // end namespace

#endif



