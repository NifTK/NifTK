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
 * \brief multiplies a set of points by a 4x4 transformation matrix
 */
std::vector <cv::Point3f> operator*(cv::Mat M, const std::vector<cv::Point3f>& p);
/**
 * \brief multiplies a  point by a 4x4 transformation matrix
 */
cv::Point3f operator*(cv::Mat M, const cv::Point3f& p);

/**
 * \ brief Finds the intersection point of two 2D lines defined as cv::Vec41
 */
cv::Point2f FindIntersect(cv::Vec4i , cv::Vec4i ,bool RejectIfNotOnALine = false, bool RejectIfNotPerpendicular = false);

/**
 * \ brief Finds all the intersection points of a vector of  2D lines defined as cv::Vec41
 */
std::vector <cv::Point2f> FindIntersects (std::vector <cv::Vec4i>, 
    bool RejectIfNotOnALine = false , bool RejectIfNotPerpendicular = false);
/**
 * \brief Calculates the centroid of a vector of points.
 */
cv::Point2f GetCentroid(const std::vector<cv::Point2f>& points, bool RefineForOutliers = false);
/**
 * \brief Calculates the centroid of a vector of points.
 */
cv::Point3f GetCentroid(const std::vector<cv::Point3f>& points, bool RefineForOutliers = false, cv::Point3f* StandardDeviation = NULL);




} // end namespace

#endif



