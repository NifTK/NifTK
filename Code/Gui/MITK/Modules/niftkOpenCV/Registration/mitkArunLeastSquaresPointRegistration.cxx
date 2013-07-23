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
  // See:
  // http://eecs.vanderbilt.edu/people/mikefitzpatrick/papers/2009_Medim_Fitzpatrick_TRE_FRE_uncorrelated_as_published.pdf
  // Then:
  // http://tango.andrew.cmu.edu/~gustavor/42431-intro-bioimaging/readings/ch8.pdf

  assert(fixedPoints.size() == movingPoints.size());

  bool success = false;

  // Equation 4.
  cv::Point3d pPrime = GetCentroid(fixedPoints);

  // Equation 6.
  cv::Point3d p = GetCentroid(movingPoints);

  // Equation 7.
  std::vector<cv::Point3d> q = SubtractPointFromPoints(movingPoints, p);

  // Equation 8.
  std::vector<cv::Point3d> qPrime = SubtractPointFromPoints(fixedPoints, pPrime);

  // Arun Equation 11.
  cv::Matx33d H = CalculateCrossCovarianceH(q, qPrime);

  // Arun Equation 12.
  cv::SVD svd(H);

  // Arun Equation 13.
  cv::Mat X = svd.vt.t() * svd.u.t();

  // Replace with Fitzpatrick, chapter 8, page 470.
  cv::Mat VU = svd.vt.t() * svd.u;
  double detVU = cv::determinant(VU);
  cv::Matx33d diag = cv::Matx33d::zeros();
  diag(0,0) = 1;
  diag(1,1) = 1;
  diag(2,2) = detVU;
  cv::Mat diagonal(diag);
  X = (svd.vt.t() * (diagonal * svd.u.t()));

  // Arun Step 5.

  double detX = cv::determinant(X);
  bool haveTriedToFixDeterminantIssue = false;

  if ( detX < 0
       && (   IsCloseToZero(svd.w.at<double>(0,0))
           || IsCloseToZero(svd.w.at<double>(1,1))
           || IsCloseToZero(svd.w.at<double>(2,2))
          )
     )
  {
    // Implement 2a in section VI in paper.

    cv::Mat VPrime = svd.vt.t();
    VPrime.at<double>(0,2) = -1.0 * VPrime.at<double>(0,2);
    VPrime.at<double>(1,2) = -1.0 * VPrime.at<double>(1,2);
    VPrime.at<double>(2,2) = -1.0 * VPrime.at<double>(2,2);

    X = VPrime * svd.u.t();
    haveTriedToFixDeterminantIssue = true;
  }

  if (detX > 0 || haveTriedToFixDeterminantIssue)
  {
    // Equation 10.
    cv::Matx31d T, tmpP, tmpPPrime;
    cv::Matx33d R(X);
    tmpP(0,0) = p.x;
    tmpP(1,0) = p.y;
    tmpP(2,0) = p.z;
    tmpPPrime(0,0) = pPrime.x;
    tmpPPrime(1,0) = pPrime.y;
    tmpPPrime(2,0) = pPrime.z;
    T = tmpPPrime - R*tmpP;

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





