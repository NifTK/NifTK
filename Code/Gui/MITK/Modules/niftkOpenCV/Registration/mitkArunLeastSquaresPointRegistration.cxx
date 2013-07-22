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
cv::Matx33d ArunLeastSquaresPointRegistration::CalculateH(
    const std::vector<cv::Point3d>& q,
    const std::vector<cv::Point3d>& qPrime)
{
  cv::Matx33d result = cv::Matx33d::zeros();

  for (unsigned int i = 0; i < q.size(); ++i)
  {
    cv::Matx33d tmp(
          q[i].x*qPrime[i].x, q[i].x*qPrime[i].y, q[i].x*qPrime[i].z,
          q[i].y*qPrime[i].x, q[i].y*qPrime[i].y, q[i].y*qPrime[i].z,
          q[i].z*qPrime[i].x, q[i].z*qPrime[i].y, q[i].z*qPrime[i].z
        );

    result += tmp;
  }

  return result;
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
  unsigned int numberOfPoints = fixedPoints.size();

  // Equation 4.
  cv::Point3d pPrime = GetCentroid(movingPoints);

  // Equation 6.
  cv::Point3d p = GetCentroid(fixedPoints);

  // Equation 7.
  std::vector<cv::Point3d> q = SubtractPointFromPoints(fixedPoints, p);

  // Equation 8.
  std::vector<cv::Point3d> qPrime = SubtractPointFromPoints(movingPoints, pPrime);

  // Arun Equation 11.
  cv::Matx33d H = this->CalculateH(q, qPrime);

  // Arun Equation 12.
  cv::SVD::SVD svd(H);

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

    for (unsigned int i = 0; i < 3; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        outputMatrix(i,j) = R(i,j);
      }
      outputMatrix(i, 3) = T(i, 0);
    }
    outputMatrix(3,0) = 0;
    outputMatrix(3,1) = 0;
    outputMatrix(3,2) = 0;
    outputMatrix(3,3) = 1;

    fiducialRegistrationError = 0;
    for (unsigned int i = 0; i < numberOfPoints; ++i)
    {
      cv::Matx41d f, m, mPrime;
      f(0,0) = fixedPoints[i].x;
      f(1,0) = fixedPoints[i].y;
      f(2,0) = fixedPoints[i].z;
      f(3,0) = 1;
      m(0,0) = movingPoints[i].x;
      m(1,0) = movingPoints[i].y;
      m(2,0) = movingPoints[i].z;
      m(3,0) = 1;
      mPrime = outputMatrix * m;
      double squaredError =   (f(0,0) - mPrime(0,0)) * (f(0,0) - mPrime(0,0))
                            + (f(1,0) - mPrime(1,0)) * (f(1,0) - mPrime(1,0))
                            + (f(2,0) - mPrime(2,0)) * (f(2,0) - mPrime(2,0))
                            ;
      fiducialRegistrationError += squaredError;
    }
    if (numberOfPoints > 0)
    {
      fiducialRegistrationError /= (double)numberOfPoints;
    }
    fiducialRegistrationError = sqrt(fiducialRegistrationError);
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





