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

#include <cv.h>
#include <mitkPointSet.h>

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

} // end namespace

#endif



