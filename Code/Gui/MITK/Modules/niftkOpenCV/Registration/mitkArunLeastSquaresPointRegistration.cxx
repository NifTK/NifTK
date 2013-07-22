/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkArunLeastSquaresPointRegistration.h"

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
cv::Point3d ArunLeastSquaresPointRegistration::GetCentroid(const std::vector<cv::Point3d>& points)
{
  cv::Point3d centroid;
  centroid.x = 0;
  centroid.y = 0;
  centroid.z = 0;

  unsigned int numberOfPoints = points.size();

  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
    centroid.x += points[i].x;
    centroid.y += points[i].y;
    centroid.z += points[i].z;
  }

  centroid.x /= (double) numberOfPoints;
  centroid.y /= (double) numberOfPoints;
  centroid.z /= (double) numberOfPoints;

  return centroid;
}


//-----------------------------------------------------------------------------
std::vector<cv::Point3d> ArunLeastSquaresPointRegistration::Subtract(const std::vector<cv::Point3d> listOfPoints, const cv::Point3d& centroid)
{
  std::vector<cv::Point3d> result;

  for (unsigned int i = 0; i < listOfPoints.size(); ++i)
  {
    cv::Point3d c;

    c.x = listOfPoints[i].x - centroid.x;
    c.y = listOfPoints[i].y - centroid.y;
    c.z = listOfPoints[i].z - centroid.z;

    result.push_back(c);
  }

  return result;
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
double ArunLeastSquaresPointRegistration::Update(const std::vector<cv::Point3d>& fixedPoints,
                                                 const std::vector<cv::Point3d>& movingPoints,
                                                 cv::Matx44d& outputMatrix)
{
  unsigned int numberOfPoints = fixedPoints.size();

  // Equation 4.
  cv::Point3d pPrime = this->GetCentroid(movingPoints);

  // Equation 6.
  cv::Point3d p = this->GetCentroid(fixedPoints);

  // Equation 7.
  std::vector<cv::Point3d> q = this->Subtract(fixedPoints, p);

  // Equation 8.
  std::vector<cv::Point3d> qPrime = this->Subtract(movingPoints, pPrime);

  // Equation 11.
  cv::Matx33d H = this->CalculateH(q, qPrime);

  // Equation 12.
  cv::SVD::SVD svd(H);

  // Equation 13.
  cv::Mat X = svd.vt.t() * svd.u.t();

  // Step 5.
  double determinant =   X.at<double>(0,0)*(X.at<double>(1,1)*X.at<double>(2,2) - X.at<double>(2,1)*X.at<double>(1,2))
                       - X.at<double>(0,1)*(X.at<double>(1,0)*X.at<double>(2,2) - X.at<double>(2,0)*X.at<double>(1,2))
                       + X.at<double>(0,2)*(X.at<double>(1,0)*X.at<double>(2,1) - X.at<double>(2,0)*X.at<double>(1,1))
                       ;

  if (determinant > 0)
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

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++i)
      {
        outputMatrix(i,j) = R(i,j);
      }
      outputMatrix(i, 3) = T(i, 0);
    }
    outputMatrix(3,0) = 0;
    outputMatrix(3,1) = 0;
    outputMatrix(3,2) = 0;
    outputMatrix(3,3) = 1;
  }
  else
  {
    outputMatrix = cv::Matx44d::zeros();
    outputMatrix(0,0) = 1;
    outputMatrix(1,1) = 1;
    outputMatrix(2,2) = 1;
    outputMatrix(3,3) = 1;
  }

  // Calculate FRE.
  double FRE = 0;
  for (int i = 0; i < numberOfPoints; i++)
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
    double squaredError = (f(0,0) - mPrime(0,0)) * (f(0,0) - mPrime(0,0))
                          + (f(1,0) - mPrime(1,0)) * (f(1,0) - mPrime(1,0))
                          + (f(2,0) - mPrime(2,0)) * (f(2,0) - mPrime(2,0))
                          ;
    FRE += squaredError;
  }
  if (numberOfPoints > 0)
  {
    FRE /= (double)numberOfPoints;
  }
  FRE = sqrt(FRE);
  return FRE;
}


//-----------------------------------------------------------------------------
} // end namespace





