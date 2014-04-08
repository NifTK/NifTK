/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVMaths.h"
#include <boost/math/special_functions/fpclassify.hpp>
#include <numeric>
#include <algorithm>
#include <functional>

namespace mitk {

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

    ConstructAffineMatrix(T, R, outputMatrix);
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
void ConstructAffineMatrix(const cv::Matx31d& translation, const cv::Matx33d& rotation, cv::Matx44d& matrix)
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
std::vector <std::pair <cv::Point3d, cv::Scalar> > operator*(cv::Mat M, const std::vector< std::pair < cv::Point3d, cv::Scalar > > & p)
{
  cv::Mat src ( 4, p.size(), CV_64F );
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    src.at<double>(0,i) = p[i].first.x;
    src.at<double>(1,i) = p[i].first.y;
    src.at<double>(2,i) = p[i].first.z;
    src.at<double>(3,i) = 1.0;
  }
  cv::Mat dst = M*src;
  std::vector < std::pair <cv::Point3d, cv::Scalar > > returnPoints;
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    cv::Point3d point;
    point.x = dst.at<double>(0,i);
    point.y = dst.at<double>(1,i);
    point.z = dst.at<double>(2,i);
    returnPoints.push_back(std::pair<cv::Point3d, cv::Scalar> (point, p[i].second));
  }
  return returnPoints;
}

//-----------------------------------------------------------------------------
std::pair <cv::Point3d, cv::Scalar>  operator*(cv::Mat M, const  std::pair < cv::Point3d, cv::Scalar >  & p)
{
  cv::Mat src ( 4, 1 , CV_64F );
  src.at<double>(0,0) = p.first.x;
  src.at<double>(1,0) = p.first.y;
  src.at<double>(2,0) = p.first.z;
  src.at<double>(3,0) = 1.0;

  cv::Mat dst = M*src;
  std::pair <cv::Point3d, cv::Scalar >  returnPoint;
   
  cv::Point3d point;
  point.x = dst.at<double>(0,0);
  point.y = dst.at<double>(1,0);
  point.z = dst.at<double>(2,0);
  returnPoint = std::pair<cv::Point3d, cv::Scalar> (point, p.second);
  
  return returnPoint;
}




//-----------------------------------------------------------------------------
std::vector <cv::Point3d> operator*(cv::Mat M, const std::vector<cv::Point3d>& p)
{
  cv::Mat src ( 4, p.size(), CV_64F );
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    src.at<double>(0,i) = p[i].x;
    src.at<double>(1,i) = p[i].y;
    src.at<double>(2,i) = p[i].z;
    src.at<double>(3,i) = 1.0;
  }
  cv::Mat dst = M*src;
  std::vector <cv::Point3d> returnPoints;
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    cv::Point3d point;
    point.x = dst.at<double>(0,i);
    point.y = dst.at<double>(1,i);
    point.z = dst.at<double>(2,i);
    returnPoints.push_back(point);
  }
  return returnPoints;
}


//-----------------------------------------------------------------------------
cv::Point3d operator*(cv::Mat M, const cv::Point3d& p)
{
  cv::Mat src ( 4, 1, CV_64F );
  src.at<double>(0,0) = p.x;
  src.at<double>(1,0) = p.y;
  src.at<double>(2,0) = p.z;
  src.at<double>(3,0) = 1.0;
    
  cv::Mat dst = M*src;
  cv::Point3d returnPoint;
  
  returnPoint.x = dst.at<double>(0,0);
  returnPoint.y = dst.at<double>(1,0);
  returnPoint.z = dst.at<double>(2,0);

  return returnPoint;
}


//-----------------------------------------------------------------------------
cv::Point2d FindIntersect (cv::Vec4i line1, cv::Vec4i line2, bool RejectIfNotOnALine,
    bool RejectIfNotPerpendicular)
{
  double a1;
  double a2;
  double b1;
  double b2;
  cv::Point2d returnPoint;
  returnPoint.x = -100.0;
  returnPoint.y = -100.0;

  if ( line1[2] == line1[0]  || line2[2] == line2[0]  ) 
  {
    MITK_ERROR << "Intersect for vertical lines not implemented";
    return returnPoint;
  }
  else
  {
    a1 =( static_cast<double>(line1[3]) - static_cast<double>(line1[1]) ) / 
      ( static_cast<double>(line1[2]) - static_cast<double>(line1[0]) );
    a2 =( static_cast<double>(line2[3]) - static_cast<double>(line2[1]) ) / 
      ( static_cast<double>(line2[2]) - static_cast<double>(line2[0]) );
    b1 = static_cast<double>(line1[1]) - a1 * static_cast<double>(line1[0]);
    b2 = static_cast<double>(line2[1]) - a2 * static_cast<double>(line2[0]);
  }
  returnPoint.x = ( b2 - b1 )/(a1 - a2 );
  returnPoint.y = a1 * returnPoint.x + b1;

  bool ok = true;
  if ( RejectIfNotOnALine )
  {
    if ( ((returnPoint.x >= line1[2]) && (returnPoint.x <= line1[0])) || 
         ((returnPoint.x >= line1[0]) && (returnPoint.x <= line1[2])) ||
         ((returnPoint.x >= line2[2]) && (returnPoint.x <= line2[0])) ||
         ((returnPoint.x >= line2[0]) && (returnPoint.x <= line2[2])) )
    {
      ok = true;
    }
    else
    {
      ok = false;
    }
  }
  if ( RejectIfNotPerpendicular ) 
  {
    //if there perpendicular a1 * a2 should be approximately 1
    double Angle = fabs(a1 * a2);
    if ( ! ( (Angle < 3.0) && (Angle > 0.1) ) )
    {
      ok = false;
    }
  }
  if ( ok == false ) 
  {
    return ( cv::Point2d (-100.0, -100.0) );
  }
  else 
  {
    return returnPoint;
  }

}


//-----------------------------------------------------------------------------
std::vector <cv::Point2d> FindIntersects (std::vector <cv::Vec4i> lines  , bool RejectIfNotOnALine, bool RejectIfNotPerpendicular) 
{
  std::vector<cv::Point2d> returnPoints; 
  for ( unsigned int i = 0 ; i < lines.size() ; i ++ ) 
  {
    for ( unsigned int j = i + 1 ; j < lines.size() ; j ++ ) 
    {
      cv::Point2d point =  FindIntersect (lines[i], lines[j], RejectIfNotOnALine, RejectIfNotPerpendicular);
      if ( ! ( point.x == -100.0 && point.y == -100.0 ) )
      {
        returnPoints.push_back ( FindIntersect (lines[i], lines[j], RejectIfNotOnALine, RejectIfNotPerpendicular)) ;
      }
    }
  }
  return returnPoints;
}


//-----------------------------------------------------------------------------
cv::Point2d GetCentroid(const std::vector<cv::Point2d>& points, bool RefineForOutliers, 
    cv::Point2d * StandardDeviation)
{
  cv::Point2d centroid;
  centroid.x = 0.0;
  centroid.y = 0.0;

  unsigned int  numberOfPoints = points.size();

  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
    centroid.x += points[i].x;
    centroid.y += points[i].y;
  }

  centroid.x /= (double) numberOfPoints;
  centroid.y /= (double) numberOfPoints;
  if ( ( ! RefineForOutliers ) && ( StandardDeviation == NULL ) )
  {
    return centroid;
  }
  
  cv::Point2d standardDeviation;
  standardDeviation.x = 0.0;
  standardDeviation.y = 0.0;

  for (unsigned int i = 0; i < numberOfPoints ; ++i )
  {
    standardDeviation.x += ( points[i].x - centroid.x ) * (points[i].x - centroid.x);
    standardDeviation.y += ( points[i].y - centroid.y ) * (points[i].y - centroid.y);
  }
  standardDeviation.x = sqrt ( standardDeviation.x/ (double) numberOfPoints ) ;
  standardDeviation.y = sqrt ( standardDeviation.y/ (double) numberOfPoints ) ;

  if ( ! RefineForOutliers ) 
  {
    *StandardDeviation = standardDeviation;
    return centroid;
  }

  cv::Point2d highLimit (centroid.x + 2 * standardDeviation.x , centroid.y + 2 * standardDeviation.y);
  cv::Point2d lowLimit (centroid.x - 2 * standardDeviation.x , centroid.y - 2 * standardDeviation.y);

  centroid.x = 0.0;
  centroid.y = 0.0;
  unsigned int goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
    if ( ( points[i].x <= highLimit.x ) && ( points[i].x >= lowLimit.x ) &&
         ( points[i].y <= highLimit.y ) && ( points[i].y >= lowLimit.y ) ) 
    {
      centroid.x += points[i].x;
      centroid.y += points[i].y;
      goodPoints++;
    }
  }

  centroid.x /= (double) goodPoints;
  centroid.y /= (double) goodPoints;

  if ( StandardDeviation == NULL ) 
  {
    return centroid;
  }
  standardDeviation.x = 0.0;
  standardDeviation.y = 0.0;
  goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints ; ++i )
  {
    if ( ( points[i].x <= highLimit.x ) && ( points[i].x >= lowLimit.x ) &&
         ( points[i].y <= highLimit.y ) && ( points[i].y >= lowLimit.y ) ) 
    {
      standardDeviation.x += ( points[i].x - centroid.x ) * (points[i].x - centroid.x);
      standardDeviation.y += ( points[i].y - centroid.y ) * (points[i].y - centroid.y);
      goodPoints++;
    }
  }
  standardDeviation.x = sqrt ( standardDeviation.x/ (double) goodPoints ) ;
  standardDeviation.y = sqrt ( standardDeviation.y/ (double) goodPoints ) ;
  
  *StandardDeviation = standardDeviation;
  return centroid;
}


//-----------------------------------------------------------------------------
cv::Point3d GetCentroid(const std::vector<cv::Point3d>& points, bool RefineForOutliers , cv::Point3d* StandardDeviation)
{
  cv::Point3d centroid;
  centroid.x = 0.0;
  centroid.y = 0.0;
  centroid.z = 0.0;

  unsigned int  numberOfPoints = points.size();

  unsigned int goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
  
    if ( ! ( boost::math::isnan(points[i].x) || boost::math::isnan(points[i].y) || boost::math::isnan(points[i].z) ) )
    { 
      centroid.x += points[i].x;
      centroid.y += points[i].y;
      centroid.z += points[i].z;
      goodPoints++;
    }
  }

  centroid.x /= (double) goodPoints;
  centroid.y /= (double) goodPoints;
  centroid.z /= (double) goodPoints;

  if ( ! RefineForOutliers  && StandardDeviation == NULL)
  {
    return centroid;
  }
  
  cv::Point3d standardDeviation;
  standardDeviation.x = 0.0;
  standardDeviation.y = 0.0;
  standardDeviation.z = 0.0;

  goodPoints = 0;
  for (unsigned int i = 0; i < numberOfPoints ; ++i )
  {
    if ( ! ( boost::math::isnan(points[i].x) || boost::math::isnan(points[i].y) || boost::math::isnan(points[i].z) ) )
    {
      standardDeviation.x += ( points[i].x - centroid.x ) * (points[i].x - centroid.x);
      standardDeviation.y += ( points[i].y - centroid.y ) * (points[i].y - centroid.y);
      standardDeviation.z += ( points[i].z - centroid.z ) * (points[i].z - centroid.z);
      goodPoints++;
    }
  }
  standardDeviation.x = sqrt ( standardDeviation.x/ (double) goodPoints ) ;
  standardDeviation.y = sqrt ( standardDeviation.y/ (double) goodPoints ) ;
  standardDeviation.z = sqrt ( standardDeviation.z/ (double) goodPoints ) ;
  
  if ( ! RefineForOutliers )
  {
    *StandardDeviation = standardDeviation;
    return centroid;
  }
  cv::Point3d highLimit (centroid.x + 2 * standardDeviation.x , 
      centroid.y + 2 * standardDeviation.y, centroid.z + standardDeviation.z);
  cv::Point3d lowLimit (centroid.x - 2 * standardDeviation.x , 
      centroid.y - 2 * standardDeviation.y, centroid.z - standardDeviation.z);

  centroid.x = 0.0;
  centroid.y = 0.0;
  centroid.z = 0.0;
  goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
    if ( ( ! ( boost::math::isnan(points[i].x) || boost::math::isnan(points[i].y) || boost::math::isnan(points[i].z) ) ) &&
         ( points[i].x <= highLimit.x ) && ( points[i].x >= lowLimit.x ) &&
         ( points[i].y <= highLimit.y ) && ( points[i].y >= lowLimit.y ) &&
         ( points[i].z <= highLimit.z ) && ( points[i].z >= lowLimit.z )) 
    {
      centroid.x += points[i].x;
      centroid.y += points[i].y;
      centroid.z += points[i].z;
      goodPoints++;
    }
  }

  centroid.x /= (double) goodPoints;
  centroid.y /= (double) goodPoints;
  centroid.z /= (double) goodPoints;

  if ( StandardDeviation == NULL ) 
  {
    return centroid;
  }
  goodPoints = 0 ;
  standardDeviation.x = 0.0;
  standardDeviation.y = 0.0;
  standardDeviation.z = 0.0;

  for (unsigned int i = 0; i < numberOfPoints ; ++i )
  {
    if ( ( ! ( boost::math::isnan(points[i].x) || boost::math::isnan(points[i].y) || boost::math::isnan(points[i].z) ) ) &&
         ( points[i].x <= highLimit.x ) && ( points[i].x >= lowLimit.x ) &&
         ( points[i].y <= highLimit.y ) && ( points[i].y >= lowLimit.y ) &&
         ( points[i].z <= highLimit.z ) && ( points[i].z >= lowLimit.z )) 
    { 
      standardDeviation.x += ( points[i].x - centroid.x ) * (points[i].x - centroid.x);
      standardDeviation.y += ( points[i].y - centroid.y ) * (points[i].y - centroid.y);
      standardDeviation.z += ( points[i].z - centroid.z ) * (points[i].z - centroid.z);
      goodPoints++;
    }
  }
  standardDeviation.x = sqrt ( standardDeviation.x/ (double) goodPoints ) ;
  standardDeviation.y = sqrt ( standardDeviation.y/ (double) goodPoints ) ;
  standardDeviation.z = sqrt ( standardDeviation.z/ (double) goodPoints ) ;
  *StandardDeviation = standardDeviation;
  return centroid;
}


//-----------------------------------------------------------------------------
cv::Matx33d ConstructEulerRxMatrix(const double& rx)
{
  cv::Matx33d result;

  double cosRx = cos(rx);
  double sinRx = sin(rx);

  result = result.eye();
  result(1, 1) = cosRx;
  result(1, 2) = sinRx;
  result(2, 1) = -sinRx;
  result(2, 2) = cosRx;
  result(0, 0) = 1;

  return result;
}


//-----------------------------------------------------------------------------
cv::Matx33d ConstructEulerRyMatrix(const double& ry)
{
  cv::Matx33d result;

  double cosRy = cos(ry);
  double sinRy = sin(ry);

  result = result.eye();
  result(0, 0) = cosRy;
  result(0, 2) = -sinRy;
  result(2, 0) = sinRy;
  result(2, 2) = cosRy;
  result(1, 1) = 1;

  return result;
}


//-----------------------------------------------------------------------------
cv::Matx33d ConstructEulerRzMatrix(const double& rz)
{
  cv::Matx33d result;

  double cosRz = cos(rz);
  double sinRz = sin(rz);

  result = result.eye();
  result(0, 0) = cosRz;
  result(0, 1) = sinRz;
  result(1, 0) = -sinRz;
  result(1, 1) = cosRz;
  result(2, 2) = 1;
  
  return result;
}


//-----------------------------------------------------------------------------
cv::Matx33d ConstructEulerRotationMatrix(const double& rx, const double& ry, const double& rz)
{
  cv::Matx33d result;

  cv::Matx33d rotationAboutX = ConstructEulerRxMatrix(rx);
  cv::Matx33d rotationAboutY = ConstructEulerRyMatrix(ry);
  cv::Matx33d rotationAboutZ = ConstructEulerRzMatrix(rz);

  result = (rotationAboutZ * (rotationAboutY * rotationAboutX));
  return result;
}


//-----------------------------------------------------------------------------
cv::Matx13d ConvertEulerToRodrigues(
    const double& rx,
    const double& ry,
    const double& rz
    )
{
  cv::Matx13d rotationVector;

  cv::Matx33d rotationMatrix = ConstructEulerRotationMatrix(rx, ry, rz);
  cv::Rodrigues(rotationMatrix, rotationVector);

  return rotationVector;
}


//-----------------------------------------------------------------------------
cv::Matx44d ConstructRigidTransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz
    )
{
  cv::Matx44d transformation;
  mitk::MakeIdentity(transformation);

  cv::Matx33d rotationMatrix = ConstructEulerRotationMatrix(rx, ry, rz);

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
cv::Matx44d ConstructRodriguesTransformationMatrix(
    const double& r1,
    const double& r2,
    const double& r3,
    const double& tx,
    const double& ty,
    const double& tz
    )
{
  cv::Matx44d transformation;
  mitk::MakeIdentity(transformation);

  cv::Matx13d rotationVector;
  rotationVector(0,0) = r1;
  rotationVector(0,1) = r2;
  rotationVector(0,2) = r3;

  cv::Matx33d rotationMatrix;
  cv::Rodrigues(rotationVector, rotationMatrix);

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
cv::Matx44d ConstructScalingTransformation(const double& sx, const double& sy, const double& sz)
{
  cv::Matx44d scaling;
  mitk::MakeIdentity(scaling);

  scaling(0,0) = sx;
  scaling(1,1) = sy;
  scaling(2,2) = sz;

  return scaling;
}


//-----------------------------------------------------------------------------
cv::Matx44d ConstructSimilarityTransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz,
    const double& sx,
    const double& sy,
    const double& sz
    )
{
  cv::Matx44d scaling;
  cv::Matx44d rigid;
  cv::Matx44d result;

  rigid = ConstructRigidTransformationMatrix(rx, ry, rz, tx, ty, tz);
  scaling = ConstructScalingTransformation(sx, sy, sz);

  result = scaling * rigid;
  return result;
}


//-----------------------------------------------------------------------------
cv::Point3d FindMinimumValues ( std::vector < cv::Point3d > inputValues, cv::Point3i * indexes )
{
  cv::Point3d minimumValues;

  if ( inputValues.size() > 0 ) 
  {
    minimumValues.x = inputValues[0].x;
    minimumValues.y = inputValues[0].y;
    minimumValues.z = inputValues[0].z;

    if ( indexes != NULL )
    {
      indexes->x = 0;
      indexes->y = 0;
      indexes->z = 0;
    }
  }
  for ( unsigned int i = 0 ; i < inputValues.size() ; i ++ )
  {
    std::cerr << i << std::endl;
    if ( inputValues[i].x < minimumValues.x )
    {
      minimumValues.x = inputValues[i].x;
      if ( indexes != NULL )
      {
        indexes->x = i;
      }
    }
    if ( inputValues[i].y < minimumValues.y )
    {
      minimumValues.y = inputValues[i].y;
      if ( indexes != NULL )
      {
        indexes->y = i;
      }
    }
    if ( inputValues[i].z < minimumValues.z )
    {
      minimumValues.z = inputValues[i].z;
      if ( indexes != NULL )
      {
        indexes->z = i;
      }
    }

  }
  return minimumValues;
}  
//-----------------------------------------------------------------------------
std::pair <double, double >  FindMinimumValues ( std::vector < std::pair < double, double > > inputValues, std::pair < unsigned int , unsigned int >  * indexes )
{
  std::pair < double , double > minimumValues;

  if ( inputValues.size() > 0 ) 
  {
    minimumValues.first = inputValues[0].first;
    minimumValues.second = inputValues[0].second;

    if ( indexes != NULL )
    {
      indexes->first = 0;
      indexes->second = 0;
    }
  }
  for ( unsigned int i = 0 ; i < inputValues.size() ; i ++ )
  {
    if ( inputValues[i].first < minimumValues.first )
    {
      minimumValues.first = inputValues[i].first;
      if ( indexes != NULL )
      {
        indexes->first = i;
      }
    }
    if ( inputValues[i].second < minimumValues.second )
    {
      minimumValues.second = inputValues[i].second;
      if ( indexes != NULL )
      {
        indexes->second = i;
      }
    }
  }
  return minimumValues;
}

//-----------------------------------------------------------------------------
std::pair < double, double >  RMSError (std::vector < std::vector < std::pair <cv::Point2d, cv::Point2d> > >  measured , std::vector < std::vector <std::pair<cv::Point2d, cv::Point2d> > > actual , 
    int indexToUse , double outlierSD)
{
  assert ( measured.size() == actual.size() * 2 );

  std::pair < double, double>  RMSError;
  
  RMSError.first = 0.0 ;
  RMSError.second = 0.0 ;
 
  std::pair < cv::Point2d, cv::Point2d > errorStandardDeviations;
  std::pair < cv::Point2d, cv::Point2d > errorMeans;
  errorMeans = mitk::MeanError (measured, actual, &errorStandardDeviations, indexToUse);
  std::pair < cv::Point2d , cv::Point2d > lowLimit;
  std::pair < cv::Point2d , cv::Point2d > highLimit;
  lowLimit.first.x = errorMeans.first.x - outlierSD * errorStandardDeviations.first.x; 
  lowLimit.first.y = errorMeans.first.y - outlierSD * errorStandardDeviations.first.y; 
  lowLimit.second.x = errorMeans.second.x - outlierSD * errorStandardDeviations.second.x; 
  lowLimit.second.y = errorMeans.second.y - outlierSD * errorStandardDeviations.second.y; 
  highLimit.first.x = errorMeans.first.x + outlierSD * errorStandardDeviations.first.x; 
  highLimit.first.y = errorMeans.first.y + outlierSD * errorStandardDeviations.first.y; 
  highLimit.second.x = errorMeans.second.x + outlierSD * errorStandardDeviations.second.x; 
  highLimit.second.y = errorMeans.second.y + outlierSD * errorStandardDeviations.second.y; 

  std::pair < int , int > count;
  count.first = 0;
  count.second = 0;
  int lowIndex = 0;
  int highIndex = measured[0].size();
  if ( indexToUse != -1 )
  {
    lowIndex = indexToUse; 
    highIndex = indexToUse;
  }
  for ( int index = lowIndex; index < highIndex ; index ++ ) 
  {
    for ( unsigned int i = 0 ; i < actual.size() ; i ++ ) 
    {
      if ( ! ( boost::math::isnan(measured[i*2][index].first.x) || boost::math::isnan(measured[i*2][index].first.y) ||
          boost::math::isnan(actual[i][index].first.x) || boost::math::isnan(actual[i][index].first.y) ) )
      {
        double xerror = actual[i][index].first.x - measured[i*2][index].first.x;
        double yerror = actual[i][index].first.y - measured[i*2][index].first.y;
        if ( ( xerror > lowLimit.first.x ) && ( xerror < highLimit.first.x ) &&
             ( yerror > lowLimit.first.y ) && ( yerror < highLimit.first.y ) )
        {
          RMSError.first += ( xerror * xerror ) + ( yerror * yerror );
          count.first ++;
        }
      }
      if ( ! ( boost::math::isnan(measured[i*2][index].second.x) || boost::math::isnan(measured[i*2][index].second.y) ||
          boost::math::isnan(actual[i][index].second.x) || boost::math::isnan(actual[i][index].second.y) ) )
      {
        double xerror = actual[i][index].second.x - measured[i*2][index].second.x;
        double yerror = actual[i][index].second.y - measured[i*2][index].second.y;
        if ( ( xerror > lowLimit.second.x ) && ( xerror < highLimit.second.x ) &&
             ( yerror > lowLimit.second.y ) && ( yerror < highLimit.second.y ) )
        {
          RMSError.second += ( xerror * xerror ) + ( yerror * yerror );
          count.second ++;
        }
      }
    }
  }
  if ( count.first > 0 ) 
  {
    RMSError.first = sqrt ( RMSError.first / count.first );
  }
  if ( count.second > 0 ) 
  {
    RMSError.second = sqrt ( RMSError.second / count.second );
  }
  return RMSError;
}
//-----------------------------------------------------------------------------
std::pair < double, double >  RMSError (std::vector < std::pair < long long , std::vector < std::pair <cv::Point2d, cv::Point2d> > > >  measured , std::vector < std::vector <std::pair<cv::Point2d, cv::Point2d> > > actual , 
    int indexToUse , double outlierSD, long long allowableTimingError )
{
  std::vector < std::vector < std::pair < cv::Point2d, cv::Point2d > > > culledMeasured;
  for ( unsigned int i = 0 ; i < measured.size() ; i ++ ) 
  {
    if ( measured[i].first < abs (allowableTimingError) )
    {
      culledMeasured.push_back( measured[i].second );
    }
    else 
    {
      MITK_WARN << "Dropping point pair " << i << " due to high timing error " << measured[i].first << " > " << allowableTimingError;
    }
  }
  return mitk::RMSError ( culledMeasured, actual, indexToUse, outlierSD );
}

//-----------------------------------------------------------------------------
std::pair < cv::Point2d, cv::Point2d >  MeanError (
    std::vector < std::vector < std::pair <cv::Point2d, cv::Point2d> > >  measured , 
    std::vector < std::vector <std::pair<cv::Point2d, cv::Point2d> > > actual , 
    std::pair < cv::Point2d, cv::Point2d >* StandardDeviations, int indexToUse)
{
  assert ( measured.size() == actual.size() * 2 );

  std::pair < cv::Point2d, cv::Point2d>  meanError;
  
  meanError.first.x = 0.0 ;
  meanError.first.y = 0.0 ;
  meanError.second.x = 0.0 ;
  meanError.second.y = 0.0 ;
  
  std::pair < int , int > count;
  count.first = 0;
  count.second = 0;
  int lowIndex = 0;
  int highIndex = measured[0].size();
  if ( indexToUse != -1 )
  {
    lowIndex = indexToUse; 
    highIndex = indexToUse;
  }
  for ( int index = lowIndex; index < highIndex ; index ++ ) 
  {
    for ( unsigned int i = 0 ; i < actual.size() ; i ++ ) 
    {
      if ( ! ( boost::math::isnan(measured[i*2][index].first.x) || boost::math::isnan(measured[i*2][index].first.y) ||
          boost::math::isnan(actual[i][index].first.x) || boost::math::isnan(actual[i][index].first.y) ) )
      {
        meanError.first.x +=  actual[i][index].first.x - measured[i*2][index].first.x ;
        meanError.first.y +=  actual[i][index].first.y - measured[i*2][index].first.y ;
        count.first ++;
      }
      if ( ! ( boost::math::isnan(measured[i*2][index].second.x) || boost::math::isnan(measured[i*2][index].second.y) ||
          boost::math::isnan(actual[i][index].second.x) || boost::math::isnan(actual[i][index].second.y) ) )
      {
        meanError.second.x +=  actual[i][index].second.x - measured[i*2][index].second.x ;
        meanError.second.y +=  actual[i][index].second.y - measured[i*2][index].second.y ;
        count.second ++;
      }
    }
  }
  if ( count.first > 0 ) 
  {
    meanError.first.x =  meanError.first.x / count.first ;
    meanError.first.y =  meanError.first.y / count.first ;
  }
  if ( count.second > 0 ) 
  {
    meanError.second.x =  meanError.second.x / count.second ;
    meanError.second.y =  meanError.second.y / count.second ;
  }
  if ( StandardDeviations == NULL ) 
  {
    return meanError;
  }
  else
  {
    StandardDeviations->first.x = 0.0;
    StandardDeviations->first.y = 0.0;
    StandardDeviations->second.x = 0.0;
    StandardDeviations->second.y = 0.0;
    for ( int index = lowIndex; index < highIndex ; index ++ ) 
    {
      for ( unsigned int i = 0 ; i < actual.size() ; i ++ ) 
      {
        if ( ! ( boost::math::isnan(measured[i*2][index].first.x) || boost::math::isnan(measured[i*2][index].first.y) ||
            boost::math::isnan(actual[i][index].first.x) || boost::math::isnan(actual[i][index].first.y) ) )
        {
          double xerror = actual[i][index].first.x - measured[i*2][index].first.x - meanError.first.x;
          double yerror = actual[i][index].first.y - measured[i*2][index].first.y - meanError.first.y;
          StandardDeviations->first.x += xerror * xerror;
          StandardDeviations->first.y += yerror * yerror;
          count.first ++;
        }
        if ( ! ( boost::math::isnan(measured[i*2][index].second.x) || boost::math::isnan(measured[i*2][index].second.y) ||
            boost::math::isnan(actual[i][index].second.x) || boost::math::isnan(actual[i][index].second.y) ) )
        {
          double xerror = actual[i][index].second.x - measured[i*2][index].second.x - meanError.second.x;
          double yerror = actual[i][index].second.y - measured[i*2][index].second.y - meanError.second.y;
          StandardDeviations->second.x += xerror * xerror;
          StandardDeviations->second.y += yerror * yerror;
          count.second ++;
        }
      }
    }
    if ( count.first > 0 ) 
    {
      StandardDeviations->first.x =  sqrt(StandardDeviations->first.x / count.first);
      StandardDeviations->first.y =  sqrt(StandardDeviations->first.y / count.first) ;
    }
    if ( count.second > 0 ) 
    {
      StandardDeviations->second.x = sqrt( StandardDeviations->second.x / count.second) ;
      StandardDeviations->second.y = sqrt( StandardDeviations->second.y / count.second) ;
    }

  }
  return meanError;

}

//-----------------------------------------------------------------------------
cv::Mat PerturbTransform (const cv::Mat transformIn , 
    const double tx, const double ty, const double tz,
    const double rx, const double ry, const double rz)
{

  cv::Mat rotationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat rotationMatrix = cv::Mat (3,3,CV_64FC1);
  cv::Mat perturbationMatrix = cv::Mat (4,4,CV_64FC1);
  rotationVector.at<double>(0,0) = rx * CV_PI/180;
  rotationVector.at<double>(1,0) = ry * CV_PI/180;
  rotationVector.at<double>(2,0) = rz * CV_PI/180;
  
  cv::Rodrigues ( rotationVector,rotationMatrix );
  for ( int row = 0 ; row < 3 ; row ++ )
  {
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      perturbationMatrix.at<double>(row,col) = rotationMatrix.at<double>(row,col);
    }
  }
  perturbationMatrix.at<double>(0,3) = tx;
  perturbationMatrix.at<double>(1,3) = ty;
  perturbationMatrix.at<double>(2,3) = tz;
  perturbationMatrix.at<double>(3,0) = 0.0;
  perturbationMatrix.at<double>(3,1) = 0.0;
  perturbationMatrix.at<double>(3,2) = 0.0;
  perturbationMatrix.at<double>(3,3) = 1.0;

  return transformIn * perturbationMatrix;
}


//-----------------------------------------------------------------------------
double RMS(const std::vector<double>& input)
{
  double mean = Mean(input);
  return sqrt(mean);
}


//-----------------------------------------------------------------------------
double Mean(const std::vector<double>& input)
{
  if (input.size() == 0)
  {
    return 0;
  }
  double sum = std::accumulate(input.begin(), input.end(), 0.0);
  double mean = sum / input.size();
  return mean;  
}


//-----------------------------------------------------------------------------
double StdDev(const std::vector<double>& input)
{
  if (input.size() == 0)
  {
    return 0;
  }
  
  double mean = mitk::Mean(input);
  
  std::vector<double> diff(input.size());
  std::transform(input.begin(), input.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
  double squared = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(squared / ((double)(input.size()) - 1.0));
  return stdev;
}

//-----------------------------------------------------------------------------
cv::Point2d FindNearestPoint ( const cv::Point2d& point, 
    const std::vector < cv::Point2d>& matchingPoints, double * minRatio , unsigned int * Index ) 
{
  std::vector <cv::Point2d>  sortedMatches;
  for ( unsigned int i = 0 ; i < matchingPoints.size() ; i ++ )
  {
    sortedMatches.push_back ( point - matchingPoints[i] );
  }
  
  if ( Index != NULL ) 
  {
    *Index = std::min_element(sortedMatches.begin(), sortedMatches.end(), DistanceCompare) -
      sortedMatches.begin();
  }

  std::sort ( sortedMatches.begin(), sortedMatches.end () , DistanceCompare );

  if ( minRatio != NULL )
  {
    if ( sortedMatches.size() > 1 )
    {
      *minRatio =  
        sqrt(sortedMatches[1].x * sortedMatches[1].x + sortedMatches[1].y * sortedMatches[1].y ) /
        sqrt(sortedMatches[0].x * sortedMatches[0].x + sortedMatches[0].y * sortedMatches[0].y ); 
    }
    else 
    {
      *minRatio = 0.0;
    }
  }
  if (boost::math::isinf (sortedMatches[0].x))
  {
    *minRatio =  0.0;
  }
  return  point - sortedMatches [0];
}

//-----------------------------------------------------------------------------
bool DistanceCompare ( const cv::Point2d& p1, const cv::Point2d& p2 )
{
  double d1 = sqrt( p1.x * p1.x + p1.y * p1.y );
  double d2 = sqrt( p2.x * p2.x + p2.y * p2.y );
  return d1 < d2;
}

//-----------------------------------------------------------------------------
bool CompareGSPointPair ( const std::pair < unsigned int , cv::Point2d >& p1, 
    const std::pair < unsigned int , cv::Point2d> & p2 )
{
  return p1.first < p2.first;
}

//-----------------------------------------------------------------------------
cv::Mat HandeyeRotation ( const std::vector<cv::Mat>& Tracker1, 
    const std::vector<cv::Mat>& Tracker2, double& Residual)
{
 
  if ( Tracker1.size() != Tracker2.size() ) 
  {
    MITK_ERROR << "Called HandeyeRotation with unequal matrix vectors";
    Residual = -1.0;
    return cv::Mat();
  }
  int numberOfViews = Tracker1.size();

  cv::Mat A = cvCreateMat ( 3 * (numberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (numberOfViews - 1), 1, CV_64FC1 );
  
  for ( int i = 0; i < numberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = Tracker1[i+1].inv() * Tracker1[i];
    mat2 = Tracker2[i+1] * Tracker2[i].inv();

    cv::Mat rotationMat1 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationMat2 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationVector1 = cvCreateMat(3,1,CV_64FC1);
    cv::Mat rotationVector2 = cvCreateMat(3,1,CV_64FC1);
    for ( int row = 0; row < 3; row ++ )
    {
      for ( int col = 0; col < 3; col ++ )
      {
        rotationMat1.at<double>(row,col) = mat1.at<double>(row,col);
        rotationMat2.at<double>(row,col) = mat2.at<double>(row,col);
      }
    }
    cv::Rodrigues (rotationMat1, rotationVector1 );
    cv::Rodrigues (rotationMat2, rotationVector2 );

    double norm1 = cv::norm(rotationVector1);
    double norm2 = cv::norm(rotationVector2);

    rotationVector1 *= 2*sin(norm1/2) / norm1;
    rotationVector2 *= 2*sin(norm2/2) / norm2;

    cv::Mat sum = rotationVector1 + rotationVector2;
    cv::Mat diff = rotationVector2 - rotationVector1;

    A.at<double>(i*3+0,0)=0.0;
    A.at<double>(i*3+0,1)=-(sum.at<double>(2,0));
    A.at<double>(i*3+0,2)=sum.at<double>(1,0);
    A.at<double>(i*3+1,0)=sum.at<double>(2,0);
    A.at<double>(i*3+1,1)=0.0;
    A.at<double>(i*3+1,2)=-(sum.at<double>(0,0));
    A.at<double>(i*3+2,0)=-(sum.at<double>(1,0));
    A.at<double>(i*3+2,1)=sum.at<double>(0,0);
    A.at<double>(i*3+2,2)=0.0;
 
    b.at<double>(i*3+0,0)=diff.at<double>(0,0);
    b.at<double>(i*3+1,0)=diff.at<double>(1,0);
    b.at<double>(i*3+2,0)=diff.at<double>(2,0);
  
  }
  
  cv::Mat PseudoInverse = cvCreateMat(3,3,CV_64FC1);
  cv::invert(A,PseudoInverse,CV_SVD);
 
  cv::Mat pcgPrime = PseudoInverse * b;

  cv::Mat Error = A * pcgPrime-b;
 
  cv::Mat ErrorTransMult = cvCreateMat(Error.cols, Error.cols, CV_64FC1);
 
  cv::mulTransposed (Error, ErrorTransMult, true);
     
  Residual = sqrt(ErrorTransMult.at<double>(0,0)/(numberOfViews-1));
 
  cv::Mat pcg = 2 * pcgPrime / ( sqrt(1 + cv::norm(pcgPrime) * cv::norm(pcgPrime)) );
  cv::Mat id3 = cvCreateMat(3,3,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      if ( row == col )
      {
        id3.at<double>(row,col) = 1.0;
      }
      else
      {
        id3.at<double>(row,col) = 0.0;
      }
    }
  }
      
  cv::Mat pcg_crossproduct = cvCreateMat(3,3,CV_64FC1);
  pcg_crossproduct.at<double>(0,0)=0.0;
  pcg_crossproduct.at<double>(0,1)=-(pcg.at<double>(2,0));
  pcg_crossproduct.at<double>(0,2)=(pcg.at<double>(1,0));
  pcg_crossproduct.at<double>(1,0)=(pcg.at<double>(2,0));
  pcg_crossproduct.at<double>(1,1)=0.0;
  pcg_crossproduct.at<double>(1,2)=-(pcg.at<double>(0,0));
  pcg_crossproduct.at<double>(2,0)=-(pcg.at<double>(1,0));
  pcg_crossproduct.at<double>(2,1)=(pcg.at<double>(0,0));
  pcg_crossproduct.at<double>(2,2)=0.0;
 
  cv::Mat pcg_mulTransposed = cvCreateMat(pcg.rows, pcg.rows, CV_64FC1);
  cv::mulTransposed (pcg, pcg_mulTransposed, false);
  cv::Mat rcg = ( 1 - cv::norm(pcg) * norm(pcg) /2 ) * id3
    + 0.5 * ( pcg_mulTransposed + sqrt(4 - norm(pcg) * norm(pcg))*pcg_crossproduct);
  return rcg;
}
//-----------------------------------------------------------------------------
cv::Mat HandeyeTranslation ( const std::vector<cv::Mat>& Tracker1, 
     const std::vector<cv::Mat>& Tracker2, double& Residual, const cv::Mat& rcg)
{
  if ( Tracker1.size() != Tracker2.size() ) 
  {
    MITK_ERROR << "Called HandeyeTranslation with unequal matrix vectors";
    Residual = -1.0;
    return cv::Mat();
  }
  int numberOfViews = Tracker1.size();

  cv::Mat A = cvCreateMat ( 3 * (numberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (numberOfViews - 1), 1, CV_64FC1 );

  for ( int i = 0; i < numberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = Tracker1[i+1].inv() * Tracker1[i];
    mat2 = Tracker2[i+1] * Tracker2[i].inv();

    A.at<double>(i*3+0,0)=mat1.at<double>(0,0) - 1.0;
    A.at<double>(i*3+0,1)=mat1.at<double>(0,1) - 0.0;
    A.at<double>(i*3+0,2)=mat1.at<double>(0,2) - 0.0;
    A.at<double>(i*3+1,0)=mat1.at<double>(1,0) - 0.0;
    A.at<double>(i*3+1,1)=mat1.at<double>(1,1) - 1.0;
    A.at<double>(i*3+1,2)=mat1.at<double>(1,2) - 0.0;
    A.at<double>(i*3+2,0)=mat1.at<double>(2,0) - 0.0;
    A.at<double>(i*3+2,1)=mat1.at<double>(2,1) - 0.0;
    A.at<double>(i*3+2,2)=mat1.at<double>(2,2) - 1.0;

    cv::Mat m1_t = cvCreateMat(3,1,CV_64FC1);
    cv::Mat m2_t = cvCreateMat(3,1,CV_64FC1);
    for ( int j = 0; j < 3; j ++ )
    {
      m1_t.at<double>(j,0) = mat1.at<double>(j,3);
      m2_t.at<double>(j,0) = mat2.at<double>(j,3);
    }
    cv::Mat b_t = rcg * m2_t - m1_t;

    b.at<double>(i*3+0,0)=b_t.at<double>(0,0);
    b.at<double>(i*3+1,0)=b_t.at<double>(1,0);
    b.at<double>(i*3+2,0)=b_t.at<double>(2,0);

  }
  cv::Mat PseudoInverse = cvCreateMat(3,3,CV_64FC1);
  cv::invert(A,PseudoInverse,CV_SVD);
  cv::Mat tcg = PseudoInverse * b;
  
  cv::Mat Error = A * tcg -b;
  cv::Mat ErrorTransMult = cvCreateMat(Error.cols, Error.cols, CV_64FC1);
  cv::mulTransposed (Error, ErrorTransMult, true);
  Residual = sqrt(ErrorTransMult.at<double>(0,0)/(numberOfViews-1));
  return tcg;
}
//-----------------------------------------------------------------------------
cv::Mat HandeyeRotationAndTranslation ( const std::vector<cv::Mat>& Tracker1, 
     const std::vector<cv::Mat>& Tracker2, std::vector<double>& Residuals, 
     cv::Mat * World2ToWorld1)
{
  Residuals.clear();
  //init residuals with negative number to stop unit test passing
  //  //if Load result and calibration both produce zero.
  Residuals.push_back(-100.0);
  Residuals.push_back(-100.0);

  double RotationalResidual;
  cv::Mat rcg = mitk::HandeyeRotation ( Tracker1, Tracker2, RotationalResidual);
  double TranslationalResidual;
  cv::Mat tcg = mitk::HandeyeTranslation (Tracker1, Tracker2, TranslationalResidual, rcg);

  Residuals[0] = RotationalResidual;
  Residuals[1] = TranslationalResidual;

  cv::Mat handeye = cvCreateMat(4,4,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      handeye.at<double>(row,col) = rcg.at<double>(row,col);
    }
  }
  for ( int row = 0; row < 3; row ++ )
  {
    handeye.at<double>(row,3) = tcg.at<double>(row,0);
  }
  for ( int col = 0; col < 3; col ++ )
  {
    handeye.at<double>(3,col) = 0.0;
  }
  handeye.at<double>(3,3)=1.0;

  if ( World2ToWorld1 != NULL )
  {
    std::vector<cv::Mat> world2ToWorld1s;
    world2ToWorld1s.clear();
    for ( int i = 0; i < Tracker1.size() ; i ++ )
    {
      cv::Mat world2ToWorld1 = cvCreateMat(4,4,CV_64FC1);
      cv::Mat tracker2ToWorld1 = cvCreateMat(4,4,CV_64FC1);

      tracker2ToWorld1 =  Tracker1[i]*(handeye);
      world2ToWorld1 = tracker2ToWorld1 *(Tracker2[i]);
      world2ToWorld1s.push_back(world2ToWorld1);
    }
    *World2ToWorld1 = mitk::AverageMatrices (world2ToWorld1s);
  }
  else 
  {
    MITK_INFO << "Grid to world NULL ";
  }
  return handeye;
} 

//-----------------------------------------------------------------------------------------
cv::Mat AverageMatrices ( std::vector <cv::Mat> Matrices )
{
  cv::Mat temp = cvCreateMat(3,3,CV_64FC1);
  cv::Mat temp_T = cvCreateMat (3,1,CV_64FC1);
  for ( int row = 0 ; row < 3 ; row++ )
  {
    for ( int col = 0 ; col < 3 ; col++ ) 
    {
      temp.at<double>(row,col) = 0.0;
    }
    temp_T.at<double>(row,0) = 0.0;
  }
  for ( unsigned int i = 0 ; i < Matrices.size() ; i ++ ) 
  {
    for ( int row = 0 ; row < 3 ; row++ )
    {
      for ( int col = 0 ; col < 3 ; col++ ) 
      {
        double whatItWas = temp.at<double>(row,col);
        double whatToAdd = Matrices[i].at<double>(row,col);
        temp.at<double>(row,col) = whatItWas +  whatToAdd;
      }
      temp_T.at<double>(row,0) += Matrices[i].at<double>(row,3);
    }
    
    //we write temp out, not because it's interesting but because it 
    //seems to fix a bug in the averaging code, trac 2895
    MITK_INFO << "temp " << temp;
  }
  
  temp_T = temp_T / static_cast<double>(Matrices.size());
  temp = temp / static_cast<double>(Matrices.size());


  cv::Mat rtr = temp.t() * temp;

  cv::Mat eigenvectors = cvCreateMat(3,3,CV_64FC1);
  cv::Mat eigenvalues = cvCreateMat(3,1,CV_64FC1);
  cv::eigen(rtr , eigenvalues, eigenvectors);
  cv::Mat rootedEigenValues = cvCreateMat(3,3,CV_64FC1);
  //write out the vectors and values, because it might be interesting, trac 2972
  MITK_INFO << "eigenvalues " << eigenvalues;
  MITK_INFO << "eigenvectors " << eigenvectors;
  for ( int row = 0 ; row < 3 ; row ++ ) 
  {
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      if ( row == col )
      {
        rootedEigenValues.at<double>(row,col) = sqrt(1.0/eigenvalues.at<double>(row,0));
      }
      else
      {
        rootedEigenValues.at<double>(row,col) = 0.0;
      }
    }
  }
  //write out the rooted eigenValues trac 2972
  MITK_INFO << " rooted eigenvalues " << rootedEigenValues;

  cv::Mat returnMat = cvCreateMat (4,4,CV_64FC1);
  cv::Mat temp2 = cvCreateMat(3,3,CV_64FC1);
  temp2 = temp * ( eigenvectors * rootedEigenValues * eigenvectors.t() );
  for ( int row = 0 ; row < 3 ; row ++ ) 
  {
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      returnMat.at<double>(row,col) = temp2.at<double>(row,col);
    }
    returnMat.at<double>(row,3) = temp_T.at<double>(row,0);
  }
  returnMat.at<double>(3,0) = 0.0;
  returnMat.at<double>(3,1) = 0.0;
  returnMat.at<double>(3,2) = 0.0;
  returnMat.at<double>(3,3)  = 1.0;
  return returnMat;
    
} 

//-----------------------------------------------------------------------------
std::vector<cv::Mat> FlipMatrices (const std::vector<cv::Mat> Matrices)
{
  std::vector<cv::Mat>  OutMatrices;
  for ( unsigned int i = 0; i < Matrices.size(); i ++ )
  {
    if ( Matrices[i].type() == CV_64FC1 )
    {
      cv::Mat FlipMat = cvCreateMat(4,4,CV_64FC1);
      FlipMat.at<double>(0,0) = Matrices[i].at<double>(0,0);
      FlipMat.at<double>(0,1) = Matrices[i].at<double>(0,1);
      FlipMat.at<double>(0,2) = Matrices[i].at<double>(0,2) * -1;
      FlipMat.at<double>(0,3) = Matrices[i].at<double>(0,3);

      FlipMat.at<double>(1,0) = Matrices[i].at<double>(1,0);
      FlipMat.at<double>(1,1) = Matrices[i].at<double>(1,1);
      FlipMat.at<double>(1,2) = Matrices[i].at<double>(1,2) * -1;
      FlipMat.at<double>(1,3) = Matrices[i].at<double>(1,3);

      FlipMat.at<double>(2,0) = Matrices[i].at<double>(2,0) * -1;
      FlipMat.at<double>(2,1) = Matrices[i].at<double>(2,1) * -1;
      FlipMat.at<double>(2,2) = Matrices[i].at<double>(2,2);
      FlipMat.at<double>(2,3) = Matrices[i].at<double>(2,3) * -1;

      FlipMat.at<double>(3,0) = Matrices[i].at<double>(3,0);
      FlipMat.at<double>(3,1) = Matrices[i].at<double>(3,1);
      FlipMat.at<double>(3,2) = Matrices[i].at<double>(3,2);
      FlipMat.at<double>(3,3) = Matrices[i].at<double>(3,3);

      OutMatrices.push_back(FlipMat);
    }
    else if ( Matrices[i].type() == CV_32FC1 )
    {
      cv::Mat FlipMat = cvCreateMat(4,4,CV_32FC1);
      FlipMat.at<float>(0,0) = Matrices[i].at<float>(0,0);
      FlipMat.at<float>(0,1) = Matrices[i].at<float>(0,1);
      FlipMat.at<float>(0,2) = Matrices[i].at<float>(0,2) * -1;
      FlipMat.at<float>(0,3) = Matrices[i].at<float>(0,3);

      FlipMat.at<float>(1,0) = Matrices[i].at<float>(1,0);
      FlipMat.at<float>(1,1) = Matrices[i].at<float>(1,1);
      FlipMat.at<float>(1,2) = Matrices[i].at<float>(1,2) * -1;
      FlipMat.at<float>(1,3) = Matrices[i].at<float>(1,3);

      FlipMat.at<float>(2,0) = Matrices[i].at<float>(2,0) * -1;
      FlipMat.at<float>(2,1) = Matrices[i].at<float>(2,1) * -1;
      FlipMat.at<float>(2,2) = Matrices[i].at<float>(2,2);
      FlipMat.at<float>(2,3) = Matrices[i].at<float>(2,3) * -1;

      FlipMat.at<float>(3,0) = Matrices[i].at<float>(3,0);
      FlipMat.at<float>(3,1) = Matrices[i].at<float>(3,1);
      FlipMat.at<float>(3,2) = Matrices[i].at<float>(3,2);
      FlipMat.at<float>(3,3) = Matrices[i].at<float>(3,3);

      OutMatrices.push_back(FlipMat);
    }
  }
  return OutMatrices;
}

//-----------------------------------------------------------------------------
std::vector<int> SortMatricesByDistance(const std::vector<cv::Mat>  Matrices)
{
  int NumberOfViews = Matrices.size();

  std::vector<int> used;
  std::vector<int> index;
  for ( int i = 0; i < NumberOfViews; i++ )
  {
    used.push_back(i);
    index.push_back(0);
  }

  int counter = 0;
  int startIndex = 0;
  double distance = 1e-10;

  while ( fabs(distance) > 0 )
  {
    cv::Mat t1 = cvCreateMat(3,1,CV_64FC1);
    cv::Mat t2 = cvCreateMat(3,1,CV_64FC1);
   
    for ( int row = 0; row < 3; row ++ )
    {
      t1.at<double>(row,0) = Matrices[startIndex].at<double>(row,3);
    }
    used [startIndex] = 0;
    index [counter] = startIndex;
    counter++;
    distance = 0.0;
    int CurrentIndex=0;
    for ( int i = 0; i < NumberOfViews; i ++ )
    {
      if ( ( startIndex != i ) && ( used[i] != 0 ))
      {
        for ( int row = 0; row < 3; row ++ )
        {
          t2.at<double>(row,0) = Matrices[i].at<double>(row,3);
        }
        double d = cv::norm(t1-t2);
        if ( d > distance )
        {
          distance = d;
          CurrentIndex=i;
        }
      }
    }
    if ( counter < NumberOfViews )
    {
      index[counter] = CurrentIndex;
    }
    startIndex = CurrentIndex;

   
  }
  return index;
}

//-----------------------------------------------------------------------------
std::vector<int> SortMatricesByAngle(const std::vector<cv::Mat>  Matrices)
{
  int NumberOfViews = Matrices.size();

  std::vector<int> used;
  std::vector<int> index;
  for ( int i = 0; i < NumberOfViews; i++ )
  {
    used.push_back(i);
    index.push_back(0);
  }

  int counter = 0;
  int startIndex = 0;
  double distance = 1e-10;

  while ( fabs(distance) > 0.0 )
  {
    cv::Mat t1 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat t2 = cvCreateMat(3,3,CV_64FC1);
   
    for ( int row = 0; row < 3; row ++ )
    {
      for ( int col = 0; col < 3; col ++ )
      {
        t1.at<double>(row,col) = Matrices[startIndex].at<double>(row,col);
      }
    }
    used [startIndex] = 0;
    index [counter] = startIndex;
    counter++;
    distance = 0.0;
    int CurrentIndex=0;
    for ( int i = 0; i < NumberOfViews; i ++ )
    {
      if ( ( startIndex != i ) && ( used[i] != 0 ))
      {
        for ( int row = 0; row < 3; row ++ )
        {
          for ( int col = 0; col < 3; col ++ )
          {
            t2.at<double>(row,col) = Matrices[i].at<double>(row,col);
          }
        }
        double d = AngleBetweenMatrices(t1,t2);
        if ( d > distance )
        {
          distance = d;
          CurrentIndex=i;
        }
      }
    }
    if ( counter < NumberOfViews )
    {
      index[counter] = CurrentIndex;
    }
    startIndex = CurrentIndex;
  }
  return index;
}

//-----------------------------------------------------------------------------
double AngleBetweenMatrices(cv::Mat Mat1 , cv::Mat Mat2)
{
  //turn them into quaternions first
  cv::Mat q1 = DirectionCosineToQuaternion(Mat1);
  cv::Mat q2 = DirectionCosineToQuaternion(Mat2);
 
  return 2 * acos (q1.at<double>(3,0) * q2.at<double>(3,0)
      + q1.at<double>(0,0) * q2.at<double>(0,0)
      + q1.at<double>(1,0) * q2.at<double>(1,0)
      + q1.at<double>(2,0) * q2.at<double>(2,0));

}
//-----------------------------------------------------------------------------
cv::Mat DirectionCosineToQuaternion(cv::Mat dc_Matrix)
{
  cv::Mat q = cvCreateMat(4,1,CV_64FC1);
  q.at<double>(0,0) = 0.5 * SafeSQRT ( 1 + dc_Matrix.at<double>(0,0) -
  dc_Matrix.at<double>(1,1) - dc_Matrix.at<double>(2,2) ) *
  ModifiedSignum ( dc_Matrix.at<double>(1,2) - dc_Matrix.at<double>(2,1));

  q.at<double>(1,0) = 0.5 * SafeSQRT ( 1 - dc_Matrix.at<double>(0,0) +
  dc_Matrix.at<double>(1,1) - dc_Matrix.at<double>(2,2) ) *
  ModifiedSignum ( dc_Matrix.at<double>(2,0) - dc_Matrix.at<double>(0,2));

  q.at<double>(2,0) = 0.5 * SafeSQRT ( 1 - dc_Matrix.at<double>(0,0) -
  dc_Matrix.at<double>(1,1) + dc_Matrix.at<double>(2,2) ) *
  ModifiedSignum ( dc_Matrix.at<double>(0,1) - dc_Matrix.at<double>(1,0));

  q.at<double>(3,0) = 0.5 * SafeSQRT ( 1 + dc_Matrix.at<double>(0,0) +
  dc_Matrix.at<double>(1,1) + dc_Matrix.at<double>(2,2) );

  return q;
}

//-----------------------------------------------------------------------------
double ModifiedSignum(double value)
{
  if ( value < 0.0 )
  {
    return -1.0;
  }
  return 1.0;
}

//-----------------------------------------------------------------------------
double SafeSQRT(double value)
{
  if ( value < 0 )
  {
    return 0.0;
  }
  return sqrt(value);
}


} // end namespace
