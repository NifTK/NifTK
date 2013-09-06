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
std::vector <cv::Point3f> operator*(cv::Mat M, const std::vector<cv::Point3f>& p)
{
  cv::Mat src ( 4, p.size(), CV_32F );
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    src.at<float>(0,i) = p[i].x;
    src.at<float>(1,i) = p[i].y;
    src.at<float>(2,i) = p[i].z;
    src.at<float>(3,i) = 1.0;
  }
  cv::Mat dst = M*src;
  std::vector <cv::Point3f> returnPoints;
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    cv::Point3f point;
    point.x = dst.at<float>(0,i);
    point.y = dst.at<float>(1,i);
    point.z = dst.at<float>(2,i);
    returnPoints.push_back(point);
  }
  return returnPoints;
}
//-----------------------------------------------------------------------------
cv::Point3f operator*(cv::Mat M, const cv::Point3f& p)
{
  cv::Mat src ( 4, 1, CV_32F );
  src.at<float>(0,0) = p.x;
  src.at<float>(1,0) = p.y;
  src.at<float>(2,0) = p.z;
  src.at<float>(3,0) = 1.0;
    
  cv::Mat dst = M*src;
  cv::Point3f returnPoint;
  
  returnPoint.x = dst.at<float>(0,0);
  returnPoint.y = dst.at<float>(1,0);
  returnPoint.z = dst.at<float>(2,0);

  return returnPoint;
}
//-----------------------------------------------------------------------------
cv::Point2f FindIntersect (cv::Vec4i line1, cv::Vec4i line2, bool RejectIfNotOnALine,
    bool RejectIfNotPerpendicular)
{
  double a1;
  double a2;
  double b1;
  double b2;
  cv::Point2f returnPoint;
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
    return ( cv::Point2f (-100.0, -100.0) );
  }
  else 
  {
    return returnPoint;
  }

}

//-----------------------------------------------------------------------------
std::vector <cv::Point2f> FindIntersects (std::vector <cv::Vec4i> lines  , bool RejectIfNotOnALine, bool RejectIfNotPerpendicular) 
{
  std::vector<cv::Point2f> returnPoints; 
  for ( unsigned int i = 0 ; i < lines.size() ; i ++ ) 
  {
    for ( unsigned int j = i + 1 ; j < lines.size() ; j ++ ) 
    {
      cv::Point2f point =  FindIntersect (lines[i], lines[j], RejectIfNotOnALine, RejectIfNotPerpendicular);
      if ( ! ( point.x == -100.0 && point.y == -100.0 ) )
      {
        returnPoints.push_back ( FindIntersect (lines[i], lines[j], RejectIfNotOnALine, RejectIfNotPerpendicular)) ;
      }
    }
  }
  return returnPoints;
}
//-----------------------------------------------------------------------------
cv::Point2f GetCentroid(const std::vector<cv::Point2f>& points, bool RefineForOutliers)
{
  cv::Point2f centroid;
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
  if ( ! RefineForOutliers )
  {
    return centroid;
  }
  
  cv::Point2f standardDeviation;
  standardDeviation.x = 0.0;
  standardDeviation.y = 0.0;

  for (unsigned int i = 0; i < numberOfPoints ; ++i )
  {
    standardDeviation.x += ( points[i].x - centroid.x ) * (points[i].x - centroid.x);
    standardDeviation.y += ( points[i].y - centroid.y ) * (points[i].y - centroid.y);
  }
  standardDeviation.x = sqrt ( standardDeviation.x/ (double) numberOfPoints ) ;
  standardDeviation.y = sqrt ( standardDeviation.y/ (double) numberOfPoints ) ;

  cv::Point2f highLimit (centroid.x + 2 * standardDeviation.x , centroid.y + 2 * standardDeviation.y);
  cv::Point2f lowLimit (centroid.x - 2 * standardDeviation.x , centroid.y - 2 * standardDeviation.y);

  centroid.x = 0.0;
  centroid.y = 0.0;
  unsigned int goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
    if ( ( points[i].x < highLimit.x ) && ( points[i].x > lowLimit.x ) &&
         ( points[i].y < highLimit.y ) && ( points[i].y > lowLimit.y ) ) 
    {
      centroid.x += points[i].x;
      centroid.y += points[i].y;
      goodPoints++;
    }
  }

  centroid.x /= (double) goodPoints;
  centroid.y /= (double) goodPoints;

  return centroid;
}
//-----------------------------------------------------------------------------
cv::Point3f GetCentroid(const std::vector<cv::Point3f>& points, bool RefineForOutliers , cv::Point3f* StandardDeviation)
{
  cv::Point3f centroid;
  centroid.x = 0.0;
  centroid.y = 0.0;
  centroid.z = 0.0;

  unsigned int  numberOfPoints = points.size();

  unsigned int goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
  
    if ( ! ( isnan(points[i].x) || isnan(points[i].y) || isnan(points[i].z) ) )
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
  
  cv::Point3f standardDeviation;
  standardDeviation.x = 0.0;
  standardDeviation.y = 0.0;
  standardDeviation.z = 0.0;

  goodPoints = 0;
  for (unsigned int i = 0; i < numberOfPoints ; ++i )
  {
    if ( ! ( isnan(points[i].x) || isnan(points[i].y) || isnan(points[i].z) ) )
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
  cv::Point3f highLimit (centroid.x + 2 * standardDeviation.x , 
      centroid.y + 2 * standardDeviation.y, centroid.z + standardDeviation.z);
  cv::Point3f lowLimit (centroid.x - 2 * standardDeviation.x , 
      centroid.y - 2 * standardDeviation.y, centroid.z - standardDeviation.z);

  centroid.x = 0.0;
  centroid.y = 0.0;
  centroid.z = 0.0;
  goodPoints = 0 ;
  for (unsigned int i = 0; i < numberOfPoints; ++i)
  {
    if ( ( ! ( isnan(points[i].x) || isnan(points[i].y) || isnan(points[i].z) ) ) &&
         ( points[i].x < highLimit.x ) && ( points[i].x > lowLimit.x ) &&
         ( points[i].y < highLimit.y ) && ( points[i].y > lowLimit.y ) &&
         ( points[i].z < highLimit.z ) && ( points[i].z > lowLimit.z )) 
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
    if ( ( ! ( isnan(points[i].x) || isnan(points[i].y) || isnan(points[i].z) ) ) &&
         ( points[i].x < highLimit.x ) && ( points[i].x > lowLimit.x ) &&
         ( points[i].y < highLimit.y ) && ( points[i].y > lowLimit.y ) &&
         ( points[i].z < highLimit.z ) && ( points[i].z > lowLimit.z )) 
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

  result.eye();
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

  result.eye();
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

  result.eye();
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
} // end namespace


