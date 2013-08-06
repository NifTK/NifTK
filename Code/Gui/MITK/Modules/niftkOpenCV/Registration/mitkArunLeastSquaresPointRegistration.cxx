/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkArunLeastSquaresPointRegistration.h"
#include <mitkOpenCVMaths.h>

namespace mitk {

//-----------------------------------------------------------------------------
ArunLeastSquaresPointRegistration::ArunLeastSquaresPointRegistration()
{
}


//-----------------------------------------------------------------------------
ArunLeastSquaresPointRegistration::~ArunLeastSquaresPointRegistration()
{
}


//-----------------------------------------------------------------------------
bool ArunLeastSquaresPointRegistration::Update(const std::vector<cv::Point3d>& fixedPoints,
                                               const std::vector<cv::Point3d>& movingPoints,
                                               cv::Matx44d& outputMatrix,
                                               double &fiducialRegistrationError
                                                 )
{
  assert(fixedPoints.size() == movingPoints.size());

  bool success = false;

  // Arun Equation 4.
  cv::Point3d pPrime = GetCentroid(fixedPoints);

  // Arun Equation 6.
  cv::Point3d p = GetCentroid(movingPoints);

  // Arun Equation 7.
  std::vector<cv::Point3d> q = SubtractPointFromPoints(movingPoints, p);

  // Arun Equation 8.
  std::vector<cv::Point3d> qPrime = SubtractPointFromPoints(fixedPoints, pPrime);

  // Arun Equation 11.
  cv::Matx33d H = CalculateCrossCovarianceH(q, qPrime);

  // Delegate to helper method for the rest of the implementation.
  success = DoSVDPointBasedRegistration(fixedPoints, movingPoints, H, p, pPrime, outputMatrix, fiducialRegistrationError);

  return success;
}


//-----------------------------------------------------------------------------
} // end namespace





