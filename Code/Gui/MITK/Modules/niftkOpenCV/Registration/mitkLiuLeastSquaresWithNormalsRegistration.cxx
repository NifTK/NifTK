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

  // Equation 12.
  cv::Point3d pBar = GetCentroid(movingPoints);
  cv::Point3d qBar = GetCentroid(fixedPoints);

  // Equation 14.
  std::vector<cv::Point3d> pTilde = SubtractPointFromPoints(movingPoints, pBar);
  std::vector<cv::Point3d> qTilde = SubtractPointFromPoints(fixedPoints, qBar);

  // Equation 13.
  cv::Matx33d H1 = CalculateCrossCovarianceH(pTilde, qTilde);
  cv::Matx33d H2 = CalculateCrossCovarianceH(movingNormals, fixedNormals);
  cv::Matx33d H = H1 + H2;

  // Do SVD.
  cv::SVD svd(H);
  cv::Mat VU = svd.vt.t() * svd.u;
  double detVU = cv::determinant(VU);
  cv::Matx33d diag = cv::Matx33d::zeros();
  diag(0,0) = 1;
  diag(1,1) = 1;
  diag(2,2) = detVU;
  cv::Mat diagonal(diag);
  cv::Mat R = svd.u.t() * diagonal * svd.vt.t();
  double detR = cv::determinant(R);

  if (detR > 0)
  {
    // Equation 11.
    // Compare this with the Arun method???

    cv::Matx31d T, tmpQBar, tmpPBar;
    tmpQBar(0,0) = qBar.x;
    tmpQBar(1,0) = qBar.y;
    tmpQBar(2,0) = qBar.z;
    tmpPBar(0,0) = pBar.x;
    tmpPBar(1,0) = pBar.y;
    tmpPBar(2,0) = pBar.z;
    T = tmpQBar - tmpPBar;

    Setup4x4Matrix(T, R, outputMatrix);
    fiducialRegistrationError = CalculateFiducialRegistrationError(fixedPoints, movingPoints, outputMatrix);

    success = true;
  }
  else
  {
    MakeIdentity(outputMatrix);
  }

  return success;
}


//-----------------------------------------------------------------------------
} // end namespace





