/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVMaths.h"

namespace mitk {

//-----------------------------------------------------------------------------
cv::Point3d GetCentroid(const std::vector<cv::Point3d>& points)
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
std::vector<cv::Point3d> SubtractPointFromPoints(const std::vector<cv::Point3d> listOfPoints, const cv::Point3d& centroid)
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
std::vector<cv::Point3d> PointSetToVector(const mitk::PointSet::Pointer& pointSet)
{
  std::vector<cv::Point3d> result;

  mitk::PointSet::DataType* itkPointSet = pointSet->GetPointSet(0);
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointType point;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    point = pIt->Value();
    cv::Point3d cvPoint;

    cvPoint.x = point[0];
    cvPoint.y = point[1];
    cvPoint.z = point[2];
    result.push_back(cvPoint);
  }

  return result;
}


//-----------------------------------------------------------------------------
bool IsCloseToZero(const double& value)
{
  if (fabs(value) < 0.000001)
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------------
void MakeIdentity(cv::Matx44d& outputMatrix)
{
  // ToDo: Surely this is already implemented in OpenCV?
  outputMatrix = cv::Matx44d::zeros();
  outputMatrix(0,0) = 1;
  outputMatrix(1,1) = 1;
  outputMatrix(2,2) = 1;
  outputMatrix(3,3) = 1;
}



//-----------------------------------------------------------------------------
cv::Matx33d CalculateCrossCovarianceH(
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
bool DoSVDPointBasedRegistration(const std::vector<cv::Point3d>& fixedPoints,
                                 const std::vector<cv::Point3d>& movingPoints,
                                 cv::Matx33d& H,
                                 cv::Point3d &p,
                                 cv::Point3d& pPrime,
                                 cv::Matx44d& outputMatrix,
                                 double &fiducialRegistrationError)
{
  // Based on Arun's method:
  // Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987,
  // 10.1109/TPAMI.1987.4767965
  //
  // Also See:
  // http://eecs.vanderbilt.edu/people/mikefitzpatrick/papers/2009_Medim_Fitzpatrick_TRE_FRE_uncorrelated_as_published.pdf
  // Then:
  // http://tango.andrew.cmu.edu/~gustavor/42431-intro-bioimaging/readings/ch8.pdf

  bool success = false;

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
    // Implement 2a in section VI in Arun paper.

    cv::Mat VPrime = svd.vt.t();
    VPrime.at<double>(0,2) = -1.0 * VPrime.at<double>(0,2);
    VPrime.at<double>(1,2) = -1.0 * VPrime.at<double>(1,2);
    VPrime.at<double>(2,2) = -1.0 * VPrime.at<double>(2,2);

    X = VPrime * svd.u.t();
    haveTriedToFixDeterminantIssue = true;
  }

  if (detX > 0 || haveTriedToFixDeterminantIssue)
  {
    // Arun Equation 10.
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
double CalculateFiducialRegistrationError(const std::vector<cv::Point3d>& fixedPoints,
                                          const std::vector<cv::Point3d>& movingPoints,
                                          const cv::Matx44d& matrix
                                          )
{
  assert(fixedPoints.size() == movingPoints.size());

  unsigned int numberOfPoints = fixedPoints.size();
  double fiducialRegistrationError = 0;

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
    mPrime = matrix * m;
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
  return fiducialRegistrationError;
}


//-----------------------------------------------------------------------------
void Setup4x4Matrix(const cv::Matx31d& translation, const cv::Matx33d& rotation, cv::Matx44d& matrix)
{
  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      matrix(i,j) = rotation(i,j);
    }
    matrix(i, 3) = translation(i, 0);
  }
  matrix(3,0) = 0;
  matrix(3,1) = 0;
  matrix(3,2) = 0;
  matrix(3,3) = 1;
}


//-----------------------------------------------------------------------------
void CopyToVTK4x4Matrix(const cv::Matx44d& matrix, vtkMatrix4x4& vtkMatrix)
{
  for (unsigned int i = 0; i < 4; ++i)
  {
    for (unsigned int j = 0; j < 4; ++j)
    {
      vtkMatrix.SetElement(i, j, matrix(i,j));
    }
  }
}


//-----------------------------------------------------------------------------
} // end namespace


