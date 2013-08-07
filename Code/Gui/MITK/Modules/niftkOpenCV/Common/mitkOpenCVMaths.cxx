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
bool IsCloseToZero(const double& value, const double& tolerance)
{
  if (fabs(value) < tolerance)
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
double CalculateFiducialRegistrationError(const mitk::PointSet::Pointer& fixedPointSet,
                                          const mitk::PointSet::Pointer& movingPointSet,
                                          vtkMatrix4x4& vtkMatrix)
{
  std::vector<cv::Point3d> fixedPoints = PointSetToVector(fixedPointSet);
  std::vector<cv::Point3d> movingPoints = PointSetToVector(movingPointSet);
  cv::Matx44d matrix;
  CopyToOpenCVMatrix(vtkMatrix, matrix);

  double fiducialRegistrationError = CalculateFiducialRegistrationError(fixedPoints, movingPoints, matrix);
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
void CopyToOpenCVMatrix(const vtkMatrix4x4& matrix, cv::Matx44d& openCVMatrix)
{
  for (unsigned int i = 0; i < 4; ++i)
  {
    for (unsigned int j = 0; j < 4; ++j)
    {
      openCVMatrix(i, j) = matrix.GetElement(i, j);
    }
  }
}


//-----------------------------------------------------------------------------
void GenerateEulerRxMatrix(const double& rx, cv::Matx33d& matrix3x3)
{
  double cosRx = cos(rx);
  double sinRx = sin(rx);
  matrix3x3.eye();
  matrix3x3(1, 1) = cosRx;
  matrix3x3(1, 2) = sinRx;
  matrix3x3(2, 1) = -sinRx;
  matrix3x3(2, 2) = cosRx;
}


//-----------------------------------------------------------------------------
void GenerateEulerRyMatrix(const double& ry, cv::Matx33d& matrix3x3)
{
  double cosRy = cos(ry);
  double sinRy = sin(ry);
  matrix3x3.eye();
  matrix3x3(0, 0) = cosRy;
  matrix3x3(0, 2) = -sinRy;
  matrix3x3(2, 0) = sinRy;
  matrix3x3(2, 2) = cosRy;
}


//-----------------------------------------------------------------------------
void GenerateEulerRzMatrix(const double& rz, cv::Matx33d& matrix3x3)
{
  double cosRz = cos(rz);
  double sinRz = sin(rz);
  matrix3x3.eye();
  matrix3x3(0, 0) = cosRz;
  matrix3x3(0, 1) = sinRz;
  matrix3x3(1, 0) = -sinRz;
  matrix3x3(1, 1) = cosRz;
}


//-----------------------------------------------------------------------------
void GenerateEulerRotationMatrix(const double& rx, const double& ry, const double& rz, cv::Matx33d& matrix3x3)
{
  cv::Matx33d rotationAboutX;
  cv::Matx33d rotationAboutY;
  cv::Matx33d rotationAboutZ;

  GenerateEulerRxMatrix(rx, rotationAboutX);
  GenerateEulerRyMatrix(ry, rotationAboutY);
  GenerateEulerRzMatrix(rz, rotationAboutZ);

  matrix3x3 = (rotationAboutZ * (rotationAboutY * rotationAboutX));
}


//-----------------------------------------------------------------------------
cv::Matx13d ConvertEulerToRodrigues(
    const double& rx,
    const double& ry,
    const double& rz
    )
{

  cv::Matx33d rotationMatrix;
  cv::Matx13d rotationVector;

  GenerateEulerRotationMatrix(rx, ry, rz, rotationMatrix);
  cv::Rodrigues(rotationMatrix, rotationVector);

  return rotationVector;
}


//-----------------------------------------------------------------------------
cv::Matx44d Construct4x4TransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    )
{
  cv::Matx44d transformation;
  cv::Matx33d rotationMatrix;

  GenerateEulerRotationMatrix(rx, ry, rz, rotationMatrix);

  transformation.eye();

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      transformation(i, j) = rotationMatrix(i, j);
    }
  }
  transformation(0, 3) = tx;
  transformation(1, 3) = ty;
  transformation(2, 3) = tz;

  return transformation;
}


//-----------------------------------------------------------------------------
cv::Matx44d Construct4x4TransformationMatrixFromDegrees(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    )
{
  const double pi = 3.14159265358979323846;

  double radians[3];
  radians[0] = rx * pi / 180;
  radians[1] = ry * pi / 180;
  radians[2] = rz * pi / 180;

  return Construct4x4TransformationMatrix(radians[0], radians[1], radians[2], tx, ty, tz);
}

//-----------------------------------------------------------------------------
} // end namespace


