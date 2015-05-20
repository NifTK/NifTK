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
#include <mitkMathsUtils.h>
#include <mitkExceptionMacro.h>
#include <niftkVTKFunctions.h>

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
void CopyToVTK4x4Matrix(const cv::Mat& input, vtkMatrix4x4& output)
{
  if (input.rows != 4)
  {
    mitkThrow() << "Input matrix does not have 4 rows." << std::endl;
  }
  if (input.cols != 4)
  {
    mitkThrow() << "Input matrix does not have 4 columns." << std::endl;
  }

  for (unsigned int i = 0; i < 4; ++i)
  {
    for (unsigned int j = 0; j < 4; ++j)
    {
      output.SetElement(i, j, input.at<double>(i,j));
    }
  }
}


//-----------------------------------------------------------------------------
void CopyToOpenCVMatrix(const vtkMatrix4x4& input, cv::Mat& output)
{
  if (output.rows != 4)
  {
    mitkThrow() << "Output matrix does not have 4 rows." << std::endl;
  }
  if (output.cols != 4)
  {
    mitkThrow() << "Output matrix does not have 4 columns." << std::endl;
  }

  for (unsigned int i = 0; i < 4; ++i)
  {
    for (unsigned int j = 0; j < 4; ++j)
    {
      output.at<double>(i,j) = input.GetElement(i,j);
    }
  }
}


//-----------------------------------------------------------------------------
void CopyToOpenCVMatrix(const cv::Matx44d& input, cv::Mat& output)
{
  if (output.rows != 4)
  {
    mitkThrow() << "Output matrix does not have 4 rows." << std::endl;
  }
  if (output.cols != 4)
  {
    mitkThrow() << "Output matrix does not have 4 columns." << std::endl;
  }

  for (unsigned int i = 0; i < 4; ++i)
  {
    for (unsigned int j = 0; j < 4; ++j)
    {
      output.at<double>(i,j) = input(i,j);
    }
  }
}


//-----------------------------------------------------------------------------
std::vector < mitk::WorldPoint > operator*(const cv::Mat& M, const std::vector< mitk::WorldPoint > & p)
{
  cv::Mat src ( 4, p.size(), CV_64F );
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    src.at<double>(0,i) = p[i].m_Point.x;
    src.at<double>(1,i) = p[i].m_Point.y;
    src.at<double>(2,i) = p[i].m_Point.z;
    src.at<double>(3,i) = 1.0;
  }
  cv::Mat dst = M*src;
  std::vector < mitk::WorldPoint > returnPoints;
  for ( unsigned int i = 0 ; i < p.size() ; i ++ ) 
  {
    cv::Point3d point;
    point.x = dst.at<double>(0,i) / dst.at<double>(3,i);
    point.y = dst.at<double>(1,i) / dst.at<double>(3,i);
    point.z = dst.at<double>(2,i) / dst.at<double>(3,i);
    returnPoints.push_back(mitk::WorldPoint (point, p[i].m_Scalar));
  }
  return returnPoints;
}


//-----------------------------------------------------------------------------
std::vector<mitk::WorldPoint> operator*(const cv::Matx44d& M, const std::vector<mitk::WorldPoint>& p)
{
  return operator*(cv::Mat(4, 4, CV_64F, (void*) &M.val[0]), p);
}


//-----------------------------------------------------------------------------
mitk::WorldPoint  operator*(const cv::Mat& M, const  mitk::WorldPoint & p)
{
  cv::Mat src ( 4, 1 , CV_64F );
  src.at<double>(0,0) = p.m_Point.x;
  src.at<double>(1,0) = p.m_Point.y;
  src.at<double>(2,0) = p.m_Point.z;
  src.at<double>(3,0) = 1.0;

  cv::Mat dst = M*src;
  mitk::WorldPoint  returnPoint;
   
  cv::Point3d point;
  point.x = dst.at<double>(0,0) / dst.at<double>(3, 0);
  point.y = dst.at<double>(1,0) / dst.at<double>(3, 0);
  point.z = dst.at<double>(2,0) / dst.at<double>(3, 0);
  returnPoint = mitk::WorldPoint (point, p.m_Scalar);
  
  return returnPoint;
}


//-----------------------------------------------------------------------------
mitk::WorldPoint  operator*(const cv::Matx44d& M, const mitk::WorldPoint& p)
{
  return operator*(cv::Mat(4, 4, CV_64F, (void*) &M.val[0]), p);
}


//-----------------------------------------------------------------------------
std::vector <cv::Point3d> operator*(const cv::Mat& M, const std::vector<cv::Point3d>& p)
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
    point.x = dst.at<double>(0,i) / dst.at<double>(3, i);
    point.y = dst.at<double>(1,i) / dst.at<double>(3, i);
    point.z = dst.at<double>(2,i) / dst.at<double>(3, i);
    returnPoints.push_back(point);
  }
  return returnPoints;
}


//-----------------------------------------------------------------------------
std::vector <cv::Point3d> operator*(const cv::Matx44d& M, const std::vector<cv::Point3d>& p)
{
  return operator*(cv::Mat(4, 4, CV_64F, (void*) &M.val[0]), p);
}


//-----------------------------------------------------------------------------
cv::Point3d operator*(const cv::Mat& M, const cv::Point3d& p)
{
  cv::Mat src ( 4, 1, CV_64F );
  src.at<double>(0,0) = p.x;
  src.at<double>(1,0) = p.y;
  src.at<double>(2,0) = p.z;
  src.at<double>(3,0) = 1.0;
    
  cv::Mat dst = M*src;
  cv::Point3d returnPoint;
  
  returnPoint.x = dst.at<double>(0,0) / dst.at<double>(3, 0);
  returnPoint.y = dst.at<double>(1,0) / dst.at<double>(3, 0);
  returnPoint.z = dst.at<double>(2,0) / dst.at<double>(3, 0);

  return returnPoint;
}


//-----------------------------------------------------------------------------
cv::Point3d operator*(const cv::Matx44d& M, const cv::Point3d& p)
{
  return operator*(cv::Mat(4, 4, CV_64F, (void*) &M.val[0]), p);
}

//-----------------------------------------------------------------------------
std::pair < cv::Point3d , cv::Point3d > TransformPointPair(const cv::Matx44d& M, const std::pair < cv::Point3d, cv::Point3d >& p)
{
  return std::pair < cv::Point3d, cv::Point3d > ( M * p.first, M*p.second );
}


//-----------------------------------------------------------------------------
bool NearlyEqual(const cv::Point2d& p1, const cv::Point2d& p2, const double& tolerance )
{
  if ( fabs(( ( p1.x - p2.x ) + ( p2.y - p2.y ) )) < tolerance )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//-----------------------------------------------------------------------------
bool NearlyEqual(const cv::Point3d& p1, const cv::Point3d& p2, const double& tolerance )
{
  if ( fabs(( ( p1.x - p2.x ) + ( p2.y - p2.y ) + ( p1.z - p2.z ) )) < tolerance )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//-----------------------------------------------------------------------------
bool ImageHeadersEqual ( const cv::Mat& m1 , const cv::Mat& m2 )
{
  bool equal = true;
  if ( ! ( m1.type() == m2.type() ) )
  {
    equal = false;
  }

  if ( !( ( m1.rows == m2.rows ) && (m1.cols == m2.cols )  ) )
  {
    equal = false;
  }
  return equal;
}

//-----------------------------------------------------------------------------
bool ImageDataEqual ( const cv::Mat& m1 , const cv::Mat& m2 , const double& tolerance) 
{
  bool equal = ImageHeadersEqual (m1, m2 );
  
  if ( ! equal )
  {
    MITK_WARN << "Attempted to compare data in matrices of different types or sizes";
    return equal;
  }
  
  double error = 0 ;
  for ( unsigned int i = 0 ; i < m1.rows ; i ++ ) 
  {
    for ( unsigned int j = 0 ; j < m1.cols ; j ++ )
    {
      for ( unsigned int channel = 0 ; channel < m1.channels() ; channel++ )
      {
        switch ( m1.depth() )
        {
          case  CV_8U:
          {
            error += static_cast<double>(m1.ptr<unsigned char> (i,j)[channel]) - static_cast<double>(m2.ptr<unsigned char> (i,j)[channel]) ;
            break;
          }
          case CV_8S:
          {
            error += static_cast<double>(m1.ptr<char> (i,j)[channel]) - static_cast<double>(m2.ptr<char> (i,j)[channel]) ;
            break;
          }
          case CV_16U:
          {
            error += static_cast<double>(m1.ptr<unsigned int> (i,j)[channel]) - static_cast<double>(m2.ptr<unsigned int> (i,j)[channel]) ;
            break;
          }
          case CV_16S:
          {
            error += static_cast<double>(m1.ptr<int> (i,j)[channel]) - static_cast<double>(m2.ptr<int> (i,j)[channel]) ;
            break;
          }
          case CV_32S:
          { 
            error += static_cast<double>(m1.ptr<long int> (i,j)[channel]) - static_cast<double>(m2.ptr<long int> (i,j)[channel]) ;
            break;
          }
          case CV_32F:
          {
            error += static_cast<double>(m1.ptr<float> (i,j)[channel]) - static_cast<double>(m2.ptr<float> (i,j)[channel]) ;
            break;
          }
          case CV_64F:
          {
            error += static_cast<double>(m1.ptr<double> (i,j)[channel]) - static_cast<double>(m2.ptr<double> (i,j)[channel]) ;
            break;
          }
          default:
          {
            MITK_WARN << "Called compare data in matrices of unknown depth " << m1.depth();
            equal = false;
            return equal;
          }
        }
      }
    }
  }
  if ( error > tolerance )
  {
    equal = false;
  }
  return equal;
}


//-----------------------------------------------------------------------------
cv::Point2d operator/(const cv::Point2d& p1, const int& n)
{
  return cv::Point2d ( p1.x / static_cast<double>(n) , p1.y / static_cast<double>(n) );
}

//-----------------------------------------------------------------------------
cv::Point2d operator*(const cv::Point2d& p1, const cv::Point2d& p2)
{
  return cv::Point2d ( p1.x * p2.x , p1.y * p2.y );
}

//-----------------------------------------------------------------------------
cv::Point2d FindIntersect (const cv::Vec4i& line1, const cv::Vec4i& line2 )
{
  double a1;
  double a2;
  double b1;
  double b2;
  cv::Point2d returnPoint;
  returnPoint.x = std::numeric_limits<double>::quiet_NaN();
  returnPoint.y = std::numeric_limits<double>::quiet_NaN();
  
  if ( ! ( fabs ( mitk::AngleBetweenLines(line1,line2) ) > 1e-6 ) )
  {
    return returnPoint;
  }
  if ( ( line1[2] == line1[0] )  || ( line2[2] == line2[0] )  ) 
  {
    if ( line1[2] == line1[0] )
    {
      //line1 is vertical so substitute x = line1[0] into equation for line2
      a2 =( static_cast<double>(line2[3]) - static_cast<double>(line2[1]) ) /
              ( static_cast<double>(line2[2]) - static_cast<double>(line2[0]) );
      b2 = static_cast<double>(line2[1]) - a2 * static_cast<double>(line2[0]);
      returnPoint.x = line1[0];
      returnPoint.y = a2 * returnPoint.x + b2;
    }
    else
    {
      //line2 is vertical so substitute x = line2[0] into equation for line1
      a1 =( static_cast<double>(line1[3]) - static_cast<double>(line1[1]) ) /
              ( static_cast<double>(line1[2]) - static_cast<double>(line1[0]) );
      b1 = static_cast<double>(line1[1]) - a1 * static_cast<double>(line1[0]);
      returnPoint.x = line2[0];
      returnPoint.y = a1 * returnPoint.x + b1;
    }
  }
  else
  {
    a1 =( static_cast<double>(line1[3]) - static_cast<double>(line1[1]) ) / 
      ( static_cast<double>(line1[2]) - static_cast<double>(line1[0]) );
    a2 =( static_cast<double>(line2[3]) - static_cast<double>(line2[1]) ) / 
      ( static_cast<double>(line2[2]) - static_cast<double>(line2[0]) );
    b1 = static_cast<double>(line1[1]) - a1 * static_cast<double>(line1[0]);
    b2 = static_cast<double>(line2[1]) - a2 * static_cast<double>(line2[0]);
    returnPoint.x = ( b2 - b1 )/(a1 - a2 );
    returnPoint.y = a1 * returnPoint.x + b1;
  }

  return returnPoint;
}

//-----------------------------------------------------------------------------
bool PointInInterval ( const cv::Point2d& point , const cv::Vec4i& interval ) 
{
  if ( (((point.x >= static_cast<double>(interval[2])) && (point.x <= static_cast<double>(interval[0]))) || 
    ((point.x >= static_cast<double>(interval[0])) && (point.x <= static_cast<double>(interval[2]))))  &&
    (((point.y >= static_cast<double>(interval[3])) && (point.y <= static_cast<double>(interval[1]))) ||
    ((point.y >= static_cast<double>(interval[1])) && (point.y <= static_cast<double>(interval[3])))) )
  {
    return true;
  }
  else
  {
    return false;
  }
}
//-----------------------------------------------------------------------------
bool CheckIfLinesArePerpendicular ( cv::Vec4i line1, cv::Vec4i line2 , double tolerance )
{
  if ( fabs ( mitk::AngleBetweenLines ( line1, line2 ) - (CV_PI/2.0) ) <= (tolerance * CV_PI/180.0) )
  {
    return true;
  }
  else
  {
    return false;
  }
}
 
//-----------------------------------------------------------------------------
double AngleBetweenLines ( cv::Vec4i line1, cv::Vec4i line2 )
{
  double u1 = static_cast<double>(line1[2]) - static_cast<double>(line1[0]);
  double u2 = static_cast<double>(line1[3]) - static_cast<double>(line1[1]);
  double v1 = static_cast<double>(line2[2]) - static_cast<double>(line2[0]);
  double v2 = static_cast<double>(line2[3]) - static_cast<double>(line2[1]);
 
  double cosAngle = fabs ( u1 * v1 + u2 * v2 ) /
    ( (sqrt( u1*u1 + u2*u2)) * (sqrt( v1*v1 + v2*v2 )) );
  return acos (cosAngle);
}


//-----------------------------------------------------------------------------
std::vector <cv::Point2d> FindIntersects (const std::vector <cv::Vec4i>& lines  , const bool& rejectIfPointNotOnBothLines,
    const bool& rejectIfNotPerpendicular, const double& angleTolerance) 
{
  std::vector<cv::Point2d> returnPoints;
  if ( lines.size () < 2 ) 
  {
    MITK_WARN << "Called FindIntersects with only " << lines.size() << " lines";
    return returnPoints;
  }
  for ( unsigned int i = 0 ; i < lines.size() - 1 ; i ++ ) 
  {
    for ( unsigned int j = i + 1 ; j < lines.size() ; j ++ ) 
    {
      if ( (!rejectIfNotPerpendicular) || CheckIfLinesArePerpendicular( lines[i], lines[j] , angleTolerance) )
      {
        cv::Point2d point =  FindIntersect (lines[i], lines[j]);
        if (  (! rejectIfPointNotOnBothLines) ||  
          ( (mitk::PointInInterval ( point, lines[i] ) ) && ( PointInInterval ( point , lines[j] ) ) ) )
        {
          if ( ! ( boost::math::isnan(point.x) || boost::math::isnan(point.y) ) )
          {
            returnPoints.push_back ( point ) ;
          }
        }
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
std::pair < double, double >  RMSError (
    std::vector < mitk::ProjectedPointPairsWithTimingError >  measured , 
    std::vector < mitk::ProjectedPointPairsWithTimingError > actual , 
    int indexToUse , cv::Point2d outlierSD, long long allowableTimingError,
    bool duplicateLines )
{
  assert ( measured.size() == actual.size() );

  std::pair < double, double>  RMSError;
  
  RMSError.first = 0.0 ;
  RMSError.second = 0.0 ;
 
  mitk::ProjectedPointPair errorStandardDeviations;
  mitk::ProjectedPointPair  errorMeans;
  errorMeans = mitk::MeanError (measured, actual, &errorStandardDeviations,
      indexToUse, allowableTimingError);
  mitk::ProjectedPointPair lowLimit;
  mitk::ProjectedPointPair highLimit;
  lowLimit.m_Left = errorMeans.m_Left - (outlierSD * errorStandardDeviations.m_Left); 
  lowLimit.m_Right = errorMeans.m_Right - (outlierSD * errorStandardDeviations.m_Right); 
  highLimit.m_Left = errorMeans.m_Left + (outlierSD * errorStandardDeviations.m_Left); 
  highLimit.m_Right = errorMeans.m_Right + (outlierSD * errorStandardDeviations.m_Right); 

  std::pair < int , int > count;
  count.first = 0;
  count.second = 0;
  int lowIndex = 0;
  int highIndex = measured[0].m_Points.size();
  if ( indexToUse != -1 )
  {
    lowIndex = indexToUse; 
    highIndex = indexToUse;
  }
  for ( int index = lowIndex; index < highIndex ; index ++ ) 
  {
    unsigned int increment=1;
    if ( duplicateLines )
    {
      increment = 2;
    }
    for ( unsigned int frame = 0 ; frame < actual.size() ; frame += increment ) 
    {
      if ( measured[frame].m_TimingError < abs (allowableTimingError) )
      {
        if ( ! ( measured[frame].m_Points[index].LeftNaNOrInf() ) || actual[frame].m_Points[index].LeftNaNOrInf() ) 
        {
          cv::Point2d error = 
            actual[frame].m_Points[index].m_Left - measured[frame].m_Points[index].m_Left;
          
          if ( ( error.x > lowLimit.m_Left.x ) && ( error.x < highLimit.m_Left.x ) &&
             ( error.y > lowLimit.m_Left.y ) && ( error.y < highLimit.m_Left.y ) )
          {
            RMSError.first += ( error.x * error.x ) + ( error.y * error.y );
            count.first ++;
          }
        }
      
        if ( ! ( measured[frame].m_Points[index].RightNaNOrInf() ) || actual[frame].m_Points[index].RightNaNOrInf() ) 
        {
          cv::Point2d error = 
            actual[frame].m_Points[index].m_Right - measured[frame].m_Points[index].m_Right;
          
          if ( ( error.x > lowLimit.m_Right.x ) && ( error.x < highLimit.m_Right.x ) &&
             ( error.y > lowLimit.m_Right.y ) && ( error.y < highLimit.m_Right.y ) )
          {
            RMSError.second += ( error.x * error.x ) + ( error.y * error.y );
            count.second ++;
          }
        }
      }
      else
      {
        if ( index == lowIndex )
        {
          MITK_WARN << "mitk::RMSError Dropping point pair " << frame << "," << (frame)+1  << " due to high timing error " << measured[frame].m_TimingError << " > " << allowableTimingError;
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
mitk::ProjectedPointPair MeanError (
    std::vector < mitk::ProjectedPointPairsWithTimingError > measured , 
    std::vector < mitk::ProjectedPointPairsWithTimingError > actual , 
    mitk::ProjectedPointPair * StandardDeviations, int indexToUse,
    long long allowableTimingError, bool duplicateLines)
{
  assert ( measured.size() == actual.size() );

  mitk::ProjectedPointPair meanError;
  
  meanError.m_Left.x = 0.0;
  meanError.m_Left.y = 0.0;
  meanError.m_Right.x = 0.0;
  meanError.m_Right.y = 0.0;
  
  std::pair < int , int > count;
  count.first = 0;
  count.second = 0;
  int lowIndex = 0;
  int highIndex = measured[0].m_Points.size();
  if ( indexToUse != -1 )
  {
    lowIndex = indexToUse; 
    highIndex = indexToUse;
  }
  
  for ( int index = lowIndex; index < highIndex ; index ++ ) 
  {
    unsigned int increment=1;
    if ( duplicateLines )
    {
      increment = 2;
    }
    for ( unsigned int frame = 0 ; frame < actual.size() ; frame += increment ) 
    {
      if ( measured[frame].m_TimingError < abs (allowableTimingError) )
      {
        if ( ! ( measured[frame].m_Points[index].LeftNaNOrInf()  || actual[frame].m_Points[index].LeftNaNOrInf() ) ) 
        {
          meanError.m_Left += 
            actual[frame].m_Points[index].m_Left - measured[frame].m_Points[index].m_Left ;
          count.first ++;
        }
        if ( ! ( measured[frame].m_Points[index].RightNaNOrInf() || actual[frame].m_Points[index].RightNaNOrInf() ) )
        {
          meanError.m_Right += 
            actual[frame].m_Points[index].m_Right - measured[frame].m_Points[index].m_Right ;
          count.second ++;
        }
      }
      else
      {
        if ( index == lowIndex )
        {
          MITK_WARN << "mitk::MeanError Dropping point pair " << frame << "," << (frame)+1  << " due to high timing error " << measured[frame].m_TimingError << " > " << allowableTimingError;
        }
      }
    }
  }
  if ( count.first > 0 ) 
  {
    meanError.m_Left =  meanError.m_Left / count.first ;
  }
  if ( count.second > 0 ) 
  {
    meanError.m_Right =  meanError.m_Right / count.second ;
  }
  if ( StandardDeviations == NULL ) 
  {
    return meanError;
  }
  else
  {
    StandardDeviations->m_Left.x = 0.0;
    StandardDeviations->m_Left.y = 0.0;
    StandardDeviations->m_Right.x = 0.0;
    StandardDeviations->m_Right.y = 0.0;
    for ( int index = lowIndex; index < highIndex ; index ++ ) 
    {
      for ( unsigned int frame = 0 ; frame < actual.size() ; frame ++ ) 
      {
        if ( measured[frame].m_TimingError < abs (allowableTimingError) )
        {
          if ( ! ( measured[frame].m_Points[index].LeftNaNOrInf() || actual[frame].m_Points[index].LeftNaNOrInf() ) ) 
          {
            cv::Point2d error = 
              actual[frame].m_Points[index].m_Left - measured[frame].m_Points[index].m_Left - meanError.m_Left;
            StandardDeviations->m_Left += error * error;
            count.first ++;
          }
          if ( ! ( measured[frame].m_Points[index].RightNaNOrInf() || actual[frame].m_Points[index].RightNaNOrInf() ) )
          {
            cv::Point2d error = 
              actual[frame].m_Points[index].m_Right - measured[frame].m_Points[index].m_Right - meanError.m_Right;
            StandardDeviations->m_Right += error * error;
            count.second ++;
          }
        }
      }
    }
    if ( count.first > 0 ) 
    {
      StandardDeviations->m_Left.x =  sqrt(StandardDeviations->m_Left.x / count.first);
      StandardDeviations->m_Left.y =  sqrt(StandardDeviations->m_Left.y / count.first) ;
    }
    if ( count.second > 0 ) 
    {
      StandardDeviations->m_Right.x = sqrt( StandardDeviations->m_Right.x / count.second) ;
      StandardDeviations->m_Right.y = sqrt( StandardDeviations->m_Right.y / count.second) ;
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
mitk::PickedObject FindNearestPoint ( const mitk::PickedObject& point, const std::vector <mitk::PickedObject>& matchingPoints , 
    double* minRatio )
{
  mitk::PickedObject nearestPoint;
  mitk::PickedObject localPoint = point;
  double nearestDistance = std::numeric_limits<double>::infinity();
  double nextNearestDistance = std::numeric_limits<double>::infinity();

  for ( std::vector<mitk::PickedObject>::const_iterator it = matchingPoints.begin() ; it < matchingPoints.end() ; it++ )
  {
    double distance = localPoint.DistanceTo(*it);
    if ( distance < nearestDistance ) 
    {
      nextNearestDistance = nearestDistance;
      nearestDistance = distance;
      nearestPoint = *it;
    }
  }
  if ( minRatio != NULL )
  {
    *minRatio = nextNearestDistance / nearestDistance;
  }
  return nearestPoint;
}

//-----------------------------------------------------------------------------
bool DistanceCompare ( const cv::Point2d& p1, const cv::Point2d& p2 )
{
  double d1 = sqrt( p1.x * p1.x + p1.y * p1.y );
  double d2 = sqrt( p2.x * p2.x + p2.y * p2.y );
  return d1 < d2;
}

//-----------------------------------------------------------------------------
cv::Mat Tracker2ToTracker1Rotation ( const std::vector<cv::Mat>& Tracker1ToWorld1, 
    const std::vector<cv::Mat>& World2ToTracker2, double& Residual)
{
 
  if ( Tracker1ToWorld1.size() != World2ToTracker2.size() ) 
  {
    MITK_ERROR << "Called HandeyeRotation with unequal matrix vectors";
    Residual = -1.0;
    return cv::Mat();
  }
  int numberOfViews = Tracker1ToWorld1.size();

  cv::Mat A = cvCreateMat ( 3 * (numberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (numberOfViews - 1), 1, CV_64FC1 );
  
  for ( int i = 0; i < numberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = Tracker1ToWorld1[i+1].inv() * Tracker1ToWorld1[i];
    mat2 = World2ToTracker2[i+1] * World2ToTracker2[i].inv();

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
cv::Mat Tracker2ToTracker1Translation ( const std::vector<cv::Mat>& Tracker1ToWorld1, 
     const std::vector<cv::Mat>& World2ToTracker2, double& Residual, const cv::Mat& rcg)
{
  if ( Tracker1ToWorld1.size() != World2ToTracker2.size() ) 
  {
    MITK_ERROR << "Called HandeyeTranslation with unequal matrix vectors";
    Residual = -1.0;
    return cv::Mat();
  }
  int numberOfViews = Tracker1ToWorld1.size();

  cv::Mat A = cvCreateMat ( 3 * (numberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (numberOfViews - 1), 1, CV_64FC1 );

  for ( int i = 0; i < numberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = Tracker1ToWorld1[i+1].inv() * Tracker1ToWorld1[i];
    mat2 = World2ToTracker2[i+1] * World2ToTracker2[i].inv();

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
cv::Mat Tracker2ToTracker1RotationAndTranslation ( const std::vector<cv::Mat>& Tracker1ToWorld1, 
     const std::vector<cv::Mat>& World2ToTracker2, std::vector<double>& Residuals, 
     cv::Mat * World2ToWorld1)
{
  Residuals.clear();
  //init residuals with negative number to stop unit test passing
  //  //if Load result and calibration both produce zero.
  Residuals.push_back(-100.0);
  Residuals.push_back(-100.0);

  double RotationalResidual;
  cv::Mat rcg = mitk::Tracker2ToTracker1Rotation ( Tracker1ToWorld1, World2ToTracker2, RotationalResidual);
  double TranslationalResidual;
  cv::Mat tcg = mitk::Tracker2ToTracker1Translation (Tracker1ToWorld1, World2ToTracker2, TranslationalResidual, rcg);

  Residuals[0] = RotationalResidual;
  Residuals[1] = TranslationalResidual;

  cv::Mat tracker2ToTracker1 = cvCreateMat(4,4,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      tracker2ToTracker1.at<double>(row,col) = rcg.at<double>(row,col);
    }
  }
  for ( int row = 0; row < 3; row ++ )
  {
    tracker2ToTracker1.at<double>(row,3) = tcg.at<double>(row,0);
  }
  for ( int col = 0; col < 3; col ++ )
  {
    tracker2ToTracker1.at<double>(3,col) = 0.0;
  }
  tracker2ToTracker1.at<double>(3,3)=1.0;

  if ( World2ToWorld1 != NULL )
  {
    std::vector<cv::Mat> world2ToWorld1s;
    world2ToWorld1s.clear();
    for ( int i = 0; i < Tracker1ToWorld1.size() ; i ++ )
    {
      cv::Mat world2ToWorld1 = cvCreateMat(4,4,CV_64FC1);
      cv::Mat tracker2ToWorld1 = cvCreateMat(4,4,CV_64FC1);

      tracker2ToWorld1 =  Tracker1ToWorld1[i]*(tracker2ToTracker1);
      world2ToWorld1 = tracker2ToWorld1 *(World2ToTracker2[i]);
      world2ToWorld1s.push_back(world2ToWorld1);
    }
    *World2ToWorld1 = mitk::AverageMatrices (world2ToWorld1s);
    //lets do a check To get Tracker2 into Tracker1 
    //Tracker1InWorld1 = (Tracker2InWorld2 * world2ToWorld1) * tracker2toTracker1
    for ( int i = 0 ; i < Tracker1ToWorld1.size() ; i++ )
    {
      if ( i == 0 ) 
      {
        MITK_INFO << "Tracker 1: " << i ;
        MITK_INFO << Tracker1ToWorld1[i];
        MITK_INFO << "Tracker 2 to World 1 " << i ;
        MITK_INFO << (*World2ToWorld1) * World2ToTracker2[i].inv(); 
        MITK_INFO << "Tracker 1 to world 1 "  << i ;
        MITK_INFO <<  ((*World2ToWorld1) * World2ToTracker2[i].inv()) * tracker2ToTracker1.inv();
      }
      MITK_INFO << "Difference " << i ; 
      MITK_INFO << (((*World2ToWorld1) * World2ToTracker2[i].inv()) * tracker2ToTracker1.inv())- Tracker1ToWorld1[i];
    }
  }
  else 
  {
    MITK_INFO << "Grid to world NULL ";
  }
  return tracker2ToTracker1;
} 

//-----------------------------------------------------------------------------------------
cv::Mat AverageMatrices (const std::vector <cv::Mat>& Matrices )
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
  cv::Mat t1 = cvCreateMat(3,1,CV_64FC1);
  cv::Mat t2 = cvCreateMat(3,1,CV_64FC1);
  double d;

  while ( fabs(distance) > 0 )
  {
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
          t1.at<double>(row,0) = Matrices[startIndex].at<double>(row,3);
          t2.at<double>(row,0) = Matrices[i].at<double>(row,3);
        }
        d = cv::norm(t1-t2);

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
  t1.release();
  t2.release();
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

  cv::Mat t1 = cvCreateMat(3,3,CV_64FC1);
  cv::Mat t2 = cvCreateMat(3,3,CV_64FC1);
  cv::Mat t1q = cvCreateMat(4,1,CV_64FC1);
  cv::Mat t2q = cvCreateMat(4,1,CV_64FC1);
  double d;
  while ( fabs(distance) > 0.0 )
  {
   
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

        t1q = DirectionCosineToQuaternion(t1);
        t2q = DirectionCosineToQuaternion(t2);
        d = 2 * acos (t1q.at<double>(3,0) * t2q.at<double>(3,0)
          + t1q.at<double>(0,0) * t2q.at<double>(0,0)
          + t1q.at<double>(1,0) * t2q.at<double>(1,0)
          + t1q.at<double>(2,0) * t2q.at<double>(2,0));
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
  t1.release();
  t2.release();
  t1q.release();
  t2q.release();
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
double DistanceBetweenMatrices(cv::Mat Mat1 , cv::Mat Mat2)
{
  cv::Mat t1 = cvCreateMat(3,1,CV_64FC1);
  cv::Mat t2 = cvCreateMat(3,1,CV_64FC1);
   
  for ( int row = 0; row < 3; row ++ )
  {
    t1.at<double>(row,0) = Mat1.at<double>(row,3);
    t2.at<double>(row,0) = Mat2.at<double>(row,3);
  }
  double returnVal = cv::norm(t1-t2);
  //This function still leaks memory, I'm not the following statements are 
  //working
  t1.release();
  t2.release();
  return returnVal;
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
void InvertRigid4x4Matrix(const CvMat& input, CvMat& output)
{
  if (input.rows != 4)
  {
    mitkThrow() << "Input matrix must have 4 rows." << std::endl;
  }
  if (input.cols != 4)
  {
    mitkThrow() << "Input matrix must have 4 columns." << std::endl;
  }
  if (output.rows != 4)
  {
    mitkThrow() << "Output matrix must have 4 rows." << std::endl;
  }
  if (output.cols != 4)
  {
    mitkThrow() << "Output matrix must have 4 columns." << std::endl;
  }

  CvMat *inputRotationMatrix = cvCreateMat(3,3,CV_64FC1);
  CvMat *inputRotationMatrixTransposed = cvCreateMat(3,3,CV_64FC1);
  CvMat *inputTranslationVector = cvCreateMat(3,1,CV_64FC1);
  CvMat *inputTranslationVectorInverted = cvCreateMat(3,1,CV_64FC1);

  // Copy from 4x4 to separate rotation matrix and translation vector.
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      CV_MAT_ELEM(*inputRotationMatrix, double, r, c) = CV_MAT_ELEM(input, double, r, c);
    }
    CV_MAT_ELEM(*inputTranslationVector, double, r, 0) = CV_MAT_ELEM(input, double, r, 3);
  }

  cvTranspose(inputRotationMatrix, inputRotationMatrixTransposed);
  cvGEMM(inputRotationMatrixTransposed, inputTranslationVector, -1, NULL, 0, inputTranslationVectorInverted);

  // Copy inverted matrix to output.
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      CV_MAT_ELEM(output, double, r, c) = CV_MAT_ELEM(*inputRotationMatrixTransposed, double, r, c);
    }
    CV_MAT_ELEM(output, double, r, 3) = CV_MAT_ELEM(*inputTranslationVectorInverted, double, r, 0);
  }

  CV_MAT_ELEM(output, double, 3, 0) = 0;
  CV_MAT_ELEM(output, double, 3, 1) = 0;
  CV_MAT_ELEM(output, double, 3, 2) = 0;
  CV_MAT_ELEM(output, double, 3, 3) = 1;

  cvReleaseMat(&inputRotationMatrix);
  cvReleaseMat(&inputRotationMatrixTransposed);
  cvReleaseMat(&inputTranslationVector);
  cvReleaseMat(&inputTranslationVectorInverted);
}


//-----------------------------------------------------------------------------
void InvertRigid4x4Matrix(const cv::Mat& input, cv::Mat& output)
{
  const CvMat inputCv = input;
  CvMat outputCv = output;
  InvertRigid4x4Matrix(inputCv, outputCv);
}


//-----------------------------------------------------------------------------
void InvertRigid4x4Matrix(const cv::Matx44d& input, cv::Matx44d& output)
{
  cv::Mat tmpInput = cvCreateMat(4,4,CV_64FC1);
  cv::Mat tmpOutput = cvCreateMat(4,4,CV_64FC1);
  for (unsigned int r = 0; r < 4; r++)
  {
    for (unsigned int c = 0; c < 4; c++)
    {
      tmpInput.at<double>(r,c) = input(r,c);
    }
  }
  InvertRigid4x4Matrix(tmpInput, tmpOutput);
  output = tmpOutput;
}


//-----------------------------------------------------------------------------
void InterpolateTransformationMatrix(const cv::Mat& before, const cv::Mat& after, const double& proportion, cv::Mat& output)
{
  vtkSmartPointer<vtkMatrix4x4> b = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> a = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> interp = vtkSmartPointer<vtkMatrix4x4>::New();

  mitk::CopyToVTK4x4Matrix(before, *b);
  mitk::CopyToVTK4x4Matrix(after, *a);

  niftk::InterpolateTransformationMatrix(*b, *a, proportion, *interp);

  mitk::CopyToOpenCVMatrix(*interp, output);
}


//-----------------------------------------------------------------------------
void InterpolateTransformationMatrix(const cv::Matx44d& before, const cv::Matx44d& after, const double& proportion, cv::Matx44d& output)
{
  vtkSmartPointer<vtkMatrix4x4> b = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> a = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> interp = vtkSmartPointer<vtkMatrix4x4>::New();

  mitk::CopyToVTK4x4Matrix(before, *b);
  mitk::CopyToVTK4x4Matrix(after, *a);

  niftk::InterpolateTransformationMatrix(*b, *a, proportion, *interp);

  mitk::CopyToOpenCVMatrix(*interp, output);
}

//-----------------------------------------------------------------------------
std::string MatrixType ( const cv::Mat& matrix)
{
  std::string returnString;
  
  switch ( matrix.type() ) 
  {
    case ( CV_8SC1 ): 
      returnString = "CV_8SC1";
      break;
    case ( CV_8SC2 ): 
      returnString = "CV_8SC2";
      break;
    case ( CV_8SC3 ): 
      returnString = "CV_8SC3";
      break;
    case ( CV_8SC4 ): 
      returnString = "CV_8SC4";
      break;

    case ( CV_8UC1 ): 
      returnString = "CV_8UC1";
      break;
    case ( CV_8UC2 ): 
      returnString = "CV_8UC2";
      break;
    case ( CV_8UC3 ): 
      returnString = "CV_8UC3";
      break;
    case ( CV_8UC4 ): 
      returnString = "CV_8UC4";
      break;

    case ( CV_16SC1 ): 
      returnString = "CV_16SC1";
      break;
    case ( CV_16SC2 ): 
      returnString = "CV_16SC2";
      break;
    case ( CV_16SC3 ): 
      returnString = "CV_16SC3";
      break;
    case ( CV_16SC4 ): 
      returnString = "CV_16SC4";
      break;

    case ( CV_16UC1 ): 
      returnString = "CV_16UC1";
      break;
    case ( CV_16UC2 ): 
      returnString = "CV_16UC2";
      break;
    case ( CV_16UC3 ): 
      returnString = "CV_16UC3";
      break;
    case ( CV_16UC4 ): 
      returnString = "CV_16UC4";
      break;

    case ( CV_32FC1 ): 
      returnString = "CV_32FC1";
      break;
    case ( CV_32FC2 ): 
      returnString = "CV_32FC2";
      break;
    case ( CV_32FC3 ): 
      returnString = "CV_32FC3";
      break;
    case ( CV_32FC4 ): 
      returnString = "CV_32FC4";
      break;

    case ( CV_64FC1 ): 
      returnString = "CV_64FC1";
      break;
    case ( CV_64FC2 ): 
      returnString = "CV_64FC2";
      break;
    case ( CV_64FC3 ): 
      returnString = "CV_64FC3";
      break;
    case ( CV_64FC4 ): 
      returnString = "CV_64FC4";
      break;
    default:
      returnString = "Don't know";
  }
  return returnString;

}

//-----------------------------------------------------------------------------
bool IsNaN ( const cv::Point2d& point)
{
  if ( ( boost::math::isnan ( point.x ))  || (boost::math::isnan (point.y)) )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//-----------------------------------------------------------------------------
bool IsNotNaNorInf ( const cv::Point2d& point)
{
  bool ok = true;
  if ( ( boost::math::isnan ( point.x ))  || (boost::math::isnan (point.y)) )
  {
    ok = false;
  }
  if ( ( boost::math::isinf ( point.x ))  || (boost::math::isinf (point.y)) )
  {
    ok = false;
  }
  return ok;
}

//-----------------------------------------------------------------------------
bool IsNotNaNorInf ( const cv::Point3d& point)
{
  bool ok = true;
  if ( ( boost::math::isnan ( point.x ))  || (boost::math::isnan (point.y)) || (boost::math::isnan(point.z)) )
  {
    ok = false;
  }
  if ( ( boost::math::isinf ( point.x ))  || (boost::math::isinf (point.y)) || (boost::math::isinf(point.z)) )
  {
    ok = false;
  }
  return ok;
}

//-----------------------------------------------------------------------------
double DistanceToLine ( const std::pair<cv::Point3d, cv::Point3d>& line, const cv::Point3d& x0 )
{
  //courtesy Wolfram Mathworld
  cv::Point3d x1;
  cv::Point3d x2; 

  x1 = line.first;
  x2 = line.second;

  cv::Point3d d1 = x1-x0;
  cv::Point3d d2 = x2-x1;

  return mitk::Norm ( mitk::CrossProduct ( d2,d1 )) / (mitk::Norm(d2));
}

//-----------------------------------------------------------------------------
double DistanceBetweenTwoPoints ( const cv::Point3d& p1 , const cv::Point3d& p2 )
{
  return mitk::Norm ( p1 - p2 );
}

//-----------------------------------------------------------------------------
double DistanceBetweenTwoSplines ( const std::vector <cv::Point3d>& s1 , const std::vector <cv::Point3d>& s2, 
    unsigned int splineOrder )
{
  if ( ( s1.size() < 1) || (s2.size() < 2) )
  {
    MITK_WARN << "Called mitk::DistanceBetweenTwoSplines with insufficient points, returning inf.: " << s1.size() << ", " << s2.size();
    return std::numeric_limits<double>::infinity();
  }
  if ( splineOrder == 1 )
  {
    double sumOfSquares = 0;
    for ( std::vector<cv::Point3d>::const_iterator it_1 = s1.begin() ; it_1 < s1.end() ; it_1 ++ )
    {
      double shortestDistance = std::numeric_limits<double>::infinity();
      for ( std::vector<cv::Point3d>::const_iterator it_2 = s2.begin() + 1 ; it_2 < s2.end() ; it_2 ++ )
      {
        double distance = mitk::DistanceToLineSegment ( std::pair < cv::Point3d, cv::Point3d >(*(it_2) , *(it_2-1)), *it_1 );
        if ( distance < shortestDistance )
        {
          shortestDistance = distance;
        }
      }
      sumOfSquares += shortestDistance;
    }
    return sqrt( sumOfSquares / s1.size() );
  }
  else
  {
    MITK_WARN << "Called mitk::DistanceBetweenTwoSplines with invalid splineOrder, returning inf.: " << splineOrder;
    return std::numeric_limits<double>::infinity();
  }
}

//-----------------------------------------------------------------------------
double DistanceToLineSegment ( const std::pair<cv::Point3d, cv::Point3d>& line, const cv::Point3d& x0 )
{
  //courtesy Wolfram Mathworld
  cv::Point3d x1;
  cv::Point3d x2; 

  x1 = line.first;
  x2 = line.second;

  cv::Point3d d1 = x2-x0;
  cv::Point3d d2 = x2-x1;
  
  double lambda = mitk::DotProduct ( d2, d1 ) /  mitk::DotProduct ( d2,d2 );

  if ( lambda < 0 ) //were beyond x2
  {
    return mitk::Norm ( x2 - x0 );
  }
  if ( lambda > 1 ) //we're beyond x1
  {
    return mitk::Norm ( x1 - x0 );
  }
  //else we're on the line segment
  
  return mitk::Norm ( mitk::CrossProduct ( d2,d1 )) / (mitk::Norm(d2));

}


//-----------------------------------------------------------------------------
double DistanceBetweenLines ( const cv::Point3d& P0, const cv::Point3d& u, const cv::Point3d& Q0, const cv::Point3d& v , 
    cv::Point3d& midpoint)
{
  // Method 1. Solve for shortest line joining two rays, then get midpoint.
  // Taken from: http://geomalgorithms.com/a07-_distance.html
  double sc, tc, a, b, c, d, e;
  double distance;

  cv::Point3d Psc;
  cv::Point3d Qtc;
  cv::Point3d W0;

  // Difference of two origins

  W0.x = P0.x - Q0.x;
  W0.y = P0.y - Q0.y;
  W0.z = P0.z - Q0.z;

  a = u.x*u.x + u.y*u.y + u.z*u.z;
  b = u.x*v.x + u.y*v.y + u.z*v.z;
  c = v.x*v.x + v.y*v.y + v.z*v.z;
  d = u.x*W0.x + u.y*W0.y + u.z*W0.z;
  e = v.x*W0.x + v.y*W0.y + v.z*W0.z;
  sc = (b*e - c*d) / (a*c - b*b);
  tc = (a*e - b*d) / (a*c - b*b);

  if ( boost::math::isnan(sc) || boost::math::isnan(tc) || boost::math::isinf(sc) || boost::math::isinf(tc) )
  {
    //lines are parallel
    distance = sqrt(W0.x*W0.x + W0.y*W0.y + W0.z * W0.z);
    midpoint.x = std::numeric_limits<double>::quiet_NaN();
    midpoint.y = std::numeric_limits<double>::quiet_NaN();
    midpoint.z = std::numeric_limits<double>::quiet_NaN();
    return distance;
  }
  Psc.x = P0.x + sc*u.x;
  Psc.y = P0.y + sc*u.y;
  Psc.z = P0.z + sc*u.z;
  Qtc.x = Q0.x + tc*v.x;
  Qtc.y = Q0.y + tc*v.y;
  Qtc.z = Q0.z + tc*v.z;

  distance = sqrt((Psc.x - Qtc.x)*(Psc.x - Qtc.x)
                        +(Psc.y - Qtc.y)*(Psc.y - Qtc.y)
                        +(Psc.z - Qtc.z)*(Psc.z - Qtc.z));

  midpoint.x = (Psc.x + Qtc.x)/2.0;
  midpoint.y = (Psc.y + Qtc.y)/2.0;
  midpoint.z = (Psc.z + Qtc.z)/2.0;
               
  return distance;
}

//-----------------------------------------------------------------------------
std::pair < cv::Point3d , cv::Point3d > TwoPointsToPLambda ( const std::pair < cv::Point3d , cv::Point3d >& twoPointLine ) 
{
  cv::Point3d delta = twoPointLine.first - twoPointLine.second;;
  double length = sqrt ( ( delta.x * delta.x ) + ( delta.y * delta.y ) + (delta.z * delta.z) );
  
  cv::Point3d u = cv::Point3d (delta.x / length, delta.y/length, delta.z/length) ;

  return ( std::pair < cv::Point3d , cv::Point3d > ( twoPointLine.first, u ) );
}

//-----------------------------------------------------------------------------
cv::Point3d CrossProduct (const cv::Point3d& p1 , const cv::Point3d& p2)
{
  cv::Point3d cp;
  cp.x = p1.y * p2.z - p1.z * p2.y;
  cp.y = p1.z * p2.x - p1.x * p2.z;
  cp.z = p1.x * p2.y - p1.y * p2.x;
  return cp;
}

//-----------------------------------------------------------------------------
double DotProduct (const cv::Point3d& p1 , const cv::Point3d& p2)
{
  return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

//-----------------------------------------------------------------------------
double Norm (const cv::Point3d& p1)
{
  return sqrt ( p1.x * p1.x + p1.y * p1.y + p1.z*p1.z);
}


//-----------------------------------------------------------------------------
class out_of_bounds
{
  const double m_XLow;
  const double m_XHigh;
  const double m_YLow;
  const double m_YHigh;
  const double m_ZLow;
  const double m_ZHigh;

public:
  out_of_bounds ( const double& xLow, const double& xHigh, const double& yLow, const double& yHigh, const double& zLow, const double zHigh)
  : m_XLow (xLow)
  , m_XHigh (xHigh)
  , m_YLow (yLow)
  , m_YHigh (yHigh)
  , m_ZLow (zLow)
  , m_ZHigh (zHigh)
  {}

  bool operator () ( const cv::Point3d& point ) const
  {
    return ( ( point.x < m_XLow ) || ( point.x > m_XHigh ) 
        || ( point.y < m_YLow ) || ( point.y > m_YHigh ) 
        || ( point.z < m_ZLow ) || ( point.z > m_ZHigh ) );
  }
};

//-----------------------------------------------------------------------------
unsigned int RemoveOutliers ( std::vector <cv::Point3d>& points, 
    const double& xLow, const double& xHigh, 
    const double& yLow, const double& yHigh, 
    const double& zLow, const double& zHigh)
{
  unsigned int originalSize = points.size();
  points.erase ( std::remove_if ( points.begin(), points.end(), out_of_bounds (xLow, xHigh, yLow, yHigh, zLow, zHigh )), points.end() );
  return originalSize - points.size();
}

} // end namespace
