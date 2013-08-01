/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLiuLeastSquaresWithNormalsRegistration.h"
#include <mitkOpenCVMaths.h>

namespace mitk {

//-----------------------------------------------------------------------------
LiuLeastSquaresWithNormalsRegistration::LiuLeastSquaresWithNormalsRegistration()
{
}


//-----------------------------------------------------------------------------
LiuLeastSquaresWithNormalsRegistration::~LiuLeastSquaresWithNormalsRegistration()
{
}


//-----------------------------------------------------------------------------
bool LiuLeastSquaresWithNormalsRegistration::Update(const std::vector<cv::Point3d>& fixedPoints,
                                                    const std::vector<cv::Point3d>& fixedNormals,
                                                    const std::vector<cv::Point3d>& movingPoints,
                                                    const std::vector<cv::Point3d>& movingNormals,
                                                    cv::Matx44d& outputMatrix,
                                                    double &fiducialRegistrationError)
{
  assert(fixedPoints.size() == fixedNormals.size());
  assert(fixedPoints.size() == movingPoints.size());
  assert(fixedPoints.size() == movingNormals.size());

  // See: http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=758228

  bool success = false;

  // Liu Equation 12.
  cv::Point3d pBar = GetCentroid(movingPoints);
  cv::Point3d qBar = GetCentroid(fixedPoints);

  // Liu Equation 14.
  std::vector<cv::Point3d> pTilde = SubtractPointFromPoints(movingPoints, pBar);
  std::vector<cv::Point3d> qTilde = SubtractPointFromPoints(fixedPoints, qBar);

  // Liu Equation 13.
  cv::Matx33d H1 = CalculateCrossCovarianceH(pTilde, qTilde);
  cv::Matx33d H2 = CalculateCrossCovarianceH(movingNormals, fixedNormals);
  cv::Matx33d H = H1 + H2;

  // Delegate to helper method for the rest of the implementation.
  success = DoSVDPointBasedRegistration(fixedPoints, movingPoints, H, qBar, qBar, outputMatrix, fiducialRegistrationError);

  return success;
}


//-----------------------------------------------------------------------------
} // end namespace





