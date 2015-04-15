/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointRegMaths_h
#define niftkPointRegMaths_h

#include "niftkPointRegExports.h"
#include <cv.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

/**
 * \file niftkPointRegMaths.h
 * \brief Math functions to support point based registration.
 */
namespace niftk {

/**
 * \brief Calculates 1/N Sum (q_i * qPrime_i^t) where q_i and qPrime_i are column vectors, so the product is a 3x3 matrix.
 * \see Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987, DOI=10.1109/TPAMI.1987.4767965, where this calculates matrix H.
 */
extern "C++" NIFTKPOINTREG_EXPORT
cv::Matx33d CalculateCrossCovarianceH(
  const std::vector<cv::Point3d>& q,
  const std::vector<cv::Point3d>& qPrime);


/**
 * \brief Does the main SVD bit of the point based registration, and handles the degenerate conditions mentioned in Aruns paper.
 */
extern "C++" NIFTKPOINTREG_EXPORT
double DoSVDPointBasedRegistration(
  const std::vector<cv::Point3d>& fixedPoints,
  const std::vector<cv::Point3d>& movingPoints,
  cv::Matx33d& H,
  cv::Point3d &p,
  cv::Point3d& pPrime,
  cv::Matx44d& outputMatrix);


/**
 * \brief Calculates Fiducial Registration Error by multiplying the movingPoints by the matrix, and comparing with fixedPoints.
 */
extern "C++" NIFTKPOINTREG_EXPORT
double CalculateFiducialRegistrationError(
  const std::vector<cv::Point3d>& fixedPoints,
  const std::vector<cv::Point3d>& movingPoints,
  const cv::Matx44d& matrix
  );


/**
 * \brief Converts format of input to call the other CalculateFiducialRegistrationError method.
 */
extern "C++" NIFTKPOINTREG_EXPORT
double CalculateFiducialRegistrationError(
  const mitk::PointSet::Pointer& fixedPointSet,
  const mitk::PointSet::Pointer& movingPointSet,
  vtkMatrix4x4& vtkMatrix
  );

} // end namespace

#endif



