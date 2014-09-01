/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkOpenCVMaths_h
#define mitkOpenCVMaths_h

#include "niftkOpenCVExports.h"
#include "mitkOpenCVPointTypes.h"
#include <cv.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

/**
 * \file mitkOpenCVMaths.h
 * \brief Various simple mathematically based functions using OpenCV data types.
 */
namespace mitk {

/**
 * \brief Subtracts a point (e.g. the centroid) from a list of points.
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<cv::Point3d> SubtractPointFromPoints(const std::vector<cv::Point3d> listOfPoints, const cv::Point3d& point);


/**
 * \brief Converts mitk::PointSet to vector of cv::Point3d, but you lose the point ID contained within the mitk::PointSet.
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<cv::Point3d> PointSetToVector(const mitk::PointSet::Pointer& pointSet);


/**
 * \brief Haven't found a direct method to do this yet.
 */
extern "C++" NIFTKOPENCV_EXPORT void MakeIdentity(cv::Matx44d& outputMatrix);


/**
 * \brief Calculates 1/N Sum (q_i * qPrime_i^t) where q_i and qPrime_i are column vectors, so the product is a 3x3 matrix.
 * \see Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987, DOI=10.1109/TPAMI.1987.4767965, where this calculates matrix H.
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d CalculateCrossCovarianceH(const std::vector<cv::Point3d>& q, const std::vector<cv::Point3d>& qPrime);


/**
 * \brief Helper method to do the main SVD bit of the point based registration, and handle the degenerate conditions mentioned in Aruns paper.
 */
extern "C++" NIFTKOPENCV_EXPORT bool DoSVDPointBasedRegistration(const std::vector<cv::Point3d>& fixedPoints,
  const std::vector<cv::Point3d>& movingPoints,
  cv::Matx33d& H,
  cv::Point3d &p,
  cv::Point3d& pPrime,
  cv::Matx44d& outputMatrix,
  double &fiducialRegistrationError
  );


/**
 * \brief Calculates Fiducial Registration Error by multiplying the movingPoints by the matrix, and comparing with fixedPoints.
 */
extern "C++" NIFTKOPENCV_EXPORT double CalculateFiducialRegistrationError(const std::vector<cv::Point3d>& fixedPoints,
  const std::vector<cv::Point3d>& movingPoints,
  const cv::Matx44d& matrix
  );


/**
 * \brief Converts format of input to call the other CalculateFiducialRegistrationError method.
 */
extern "C++" NIFTKOPENCV_EXPORT double CalculateFiducialRegistrationError(const mitk::PointSet::Pointer& fixedPointSet,
  const mitk::PointSet::Pointer& movingPointSet,
  vtkMatrix4x4& vtkMatrix
  );


/**
 * \brief Simply copies the translation vector and rotation matrix into the 4x4 matrix.
 */
extern "C++" NIFTKOPENCV_EXPORT void ConstructAffineMatrix(const cv::Matx31d& translation, const cv::Matx33d& rotation, cv::Matx44d& matrix);


/**
 * \brief Copies matrix to vtkMatrix.
 */
extern "C++" NIFTKOPENCV_EXPORT void CopyToVTK4x4Matrix(const cv::Matx44d& matrix, vtkMatrix4x4& vtkMatrix);


/**
 * \brief Copies matrix to openCVMatrix.
 */
extern "C++" NIFTKOPENCV_EXPORT void CopyToOpenCVMatrix(const vtkMatrix4x4& matrix, cv::Matx44d& openCVMatrix);


/**
 * \brief Copies to VTK matrix, throwing exceptions if input is not 4x4.
 */
extern "C++" NIFTKOPENCV_EXPORT void CopyToVTK4x4Matrix(const cv::Mat& input, vtkMatrix4x4& output);


/**
 * \brief Copies to OpenCV matrix, throwing exceptions if output is not 4x4.
 */
extern "C++" NIFTKOPENCV_EXPORT void CopyToOpenCVMatrix(const vtkMatrix4x4& input, cv::Mat& output);


/**
 * \brief Generates a rotation about X-axis, given a Euler angle in radians.
 * \param rx angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRxMatrix(const double& rx);


/**
 * \brief Generates a rotation about Y-axis, given a Euler angle in radians.
 * \param ry angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRyMatrix(const double& ry);


/**
 * \brief Generates a rotation about Z-axis, given a Euler angle in radians.
 * \param rz angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRzMatrix(const double& rz);


/**
 * \brief Generates a rotation matrix, given Euler angles in radians.
 * \param rx angle in radians
 * \param ry angle in radians
 * \param rz angle in radians
 * \return a new [3x3] rotation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx33d ConstructEulerRotationMatrix(const double& rx, const double& ry, const double& rz);


/**
 * \brief Converts Euler angles in radians to the Rodrigues rotation vector (axis-angle convention) mentioned in OpenCV.
 * \param rx Euler angle rotation about x-axis in radians
 * \param ry Euler angle rotation about y-axis in radians
 * \param rz Euler angle rotation about z-axis in radians
 * \return A new [1x3] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx13d ConvertEulerToRodrigues(
  const double& rx,
  const double& ry,
  const double& rz
  );


/**
 * \brief multiplies a set of points by a 4x4 transformation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector <cv::Point3d> operator*(cv::Mat M, const std::vector<cv::Point3d>& p);


/**
 * \brief multiplies a set of points and corresponding scalar values by a 4x4 transformation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector < mitk::WorldPoint > operator*(cv::Mat M, 
    const std::vector< mitk::WorldPoint >& p);


/**
 * \brief multiplies a point and corresponding scalar value by a 4x4 transformation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT mitk::WorldPoint  operator*(cv::Mat M, 
    const mitk::WorldPoint & p);


/**
 * \brief multiplies a  point by a 4x4 transformation matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point3d operator*(cv::Mat M, const cv::Point3d& p);


/**
 * \brief Tests equality of 2 2d points. The openCV == operator struggles on floating points, 
 * this uses a tolerance of 1e-
 */
extern "C++" NIFTKOPENCV_EXPORT bool NearlyEqual(const cv::Point2d& p1, const cv::Point2d& p2);


/**
 * \brief Divides a 2d point by an integer (x=x1/n, y=y1/2)
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point2d operator/(const cv::Point2d& p, const int& n);


/**
 * \brief Multiplies the components of a 2d point by an integer (x=x1*x2, y=y1*y2)
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point2d operator*(const cv::Point2d& p1, const cv::Point2d& p2);


/**
 * \ brief Finds the intersection point of two 2D lines defined as cv::Vec41
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point2d FindIntersect(cv::Vec4i , cv::Vec4i ,bool RejectIfNotOnALine = false, bool RejectIfNotPerpendicular = false);


/**
 * \ brief Finds all the intersection points of a vector of  2D lines defined as cv::Vec41
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector <cv::Point2d> FindIntersects (std::vector <cv::Vec4i>, 
    bool RejectIfNotOnALine = false , bool RejectIfNotPerpendicular = false);


/**
 * \brief Calculates the centroid of a vector of points.
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point2d GetCentroid(const std::vector<cv::Point2d>& points, bool RefineForOutliers = false, cv::Point2d* StandardDeviation = NULL);


/**
 * \brief Calculates the centroid of a vector of points.
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point3d GetCentroid(const std::vector<cv::Point3d>& points, bool RefineForOutliers = false, cv::Point3d* StandardDeviation = NULL);


/**
 * \brief From rotations in radians and translations in millimetres, constructs a 4x4 transformation matrix.
 * \param rx Euler rotation about x-axis in radians
 * \param ry Euler rotation about y-axis in radians
 * \param rz Euler rotation about z-axis in radians
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructRigidTransformationMatrix(
  const double& rx,
  const double& ry,
  const double& rz,
  const double& tx,
  const double& ty,
  const double& tz
  );


/**
 * \brief From Rodrigues rotation parameters and translations in millimetres, constructs a 4x4 transformation matrix.
 * \param r1 Rodrigues rotation
 * \param r2 Rodrigues rotation
 * \param r3 Rodrigues rotation
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructRodriguesTransformationMatrix(
  const double& r1,
  const double& r2,
  const double& r3,
  const double& tx,
  const double& ty,
  const double& tz
  );


/**
 * \brief Constructs a scaling matrix from sx, sy, sz where the scale factors simply appear on the diagonal.
 * \param sx scale factor in x direction
 * \param sy scale factor in y direction
 * \param sz scale factor in z direction
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructScalingTransformation(const double& sx, const double& sy, const double& sz = 1);


/**
 * \brief Constructs an affine transformation, without skew using the specified parameters, where rotations are in degrees.
 * \param rx Euler rotation about x-axis in radians
 * \param ry Euler rotation about y-axis in radians
 * \param rz Euler rotation about z-axis in radians
 * \param tx translation in millimetres along x-axis
 * \param ty translation in millimetres along y-axis
 * \param tz translation in millimetres along z-axis
 * \param sx scale factor in x direction
 * \param sy scale factor in y direction
 * \param sz scale factor in z direction
 * \return a new [4x4] matrix
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Matx44d ConstructSimilarityTransformationMatrix(
    const double& rx,
    const double& ry,
    const double& rz,
    const double& tx,
    const double& ty,
    const double& tz,
    const double& sx,
    const double& sy,
    const double& sz
    );


/**
 * \brief Takes a point vector and finds the minimum value in each dimension. Returns the 
 * minimum values. Optionally returns the indexes of the minium values.
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point3d FindMinimumValues ( std::vector < cv::Point3d > inputValues, cv::Point3i * indexes = NULL ); 


/**
 * \brief Returns the mean pixel errors for the right and left sets of projected points
 * \param the measured projected points
 * \param the actual projected points
 * \param optional pointer to return standard deviations
 * \param optionally constrain calculation for only one projected point pair in each vector,
 * if -1 all projected point pairs are used
 * \param discard point pairs with timing errors in excess of allowableTimingError
 * \param if duplicateLines true, only every second entry in measured and actual is used, 
 * this is useful when running from stereo video and tracking data.
 */
extern "C++" NIFTKOPENCV_EXPORT mitk::ProjectedPointPair MeanError ( 
    std::vector < mitk::ProjectedPointPairsWithTimingError > measured , 
    std::vector < mitk::ProjectedPointPairsWithTimingError > actual, 
    mitk::ProjectedPointPair * StandardDeviations = NULL , int index = -1,
    long long allowableTimingError = 30e6, bool duplicateLines = true );


/** 
 * \brief Returns the RMS error between two projected point vectors
 * \param the measured projected points
 * \param the actual projected points
 * \param optionally constrain calculation for only one projected point pair in each vector,
 * if -1 all projected point pairs are used
 * \param discard point pairs where the error is above the mean error +/- n standard deviations.
 * \param discard point pairs with timing errors in excess of allowableTimingError
 * \param if duplicateLines true, only every second entry in measured and actual is used, 
 * this is useful when running from stereo video and tracking data.
 */
extern "C++" NIFTKOPENCV_EXPORT std::pair <double,double> RMSError
  (std::vector < mitk::ProjectedPointPairsWithTimingError > measured , 
    std::vector < mitk::ProjectedPointPairsWithTimingError > actual, int index = -1 ,
    cv::Point2d outlierSD = cv::Point2d (2.0,2.0) , long long allowableTimingError = 30e6,
    bool duplicateLines = true);


/**
 * \brief perturbs a 4x4 matrix with a 6 dof rigid transform. The transform is
 * defined by three translations and 3 rotations, in Degrees
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Mat PerturbTransform (
    const cv::Mat transformIn,
    const double tx, const double ty, const double tz, 
    const double rx, const double ry, const double rz );


/** 
 * \brief Searches through vector of 2D points to find the one closest (by distance)
 * to the passed point, and returns the index of that point
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Point2d FindNearestPoint ( const cv::Point2d& point,
    const std::vector < cv::Point2d >& matchingPonints , 
    double* minRatio = NULL , unsigned int * index = NULL );


/**
 * \brief Compare two cv point based on their distance from 0,0
 */
extern "C++" NIFTKOPENCV_EXPORT bool DistanceCompare ( const cv::Point2d& p1, 
    const cv::Point2d& p2 );


/**
 * \brief works out the rigid rotation correspondence between two sets of corresponding 
 * rigid body transforms
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Mat Tracker2ToTracker1Rotation ( 
    const std::vector<cv::Mat>& Tracker1ToWorld1, const std::vector<cv::Mat>& World2ToTracker2,
    double& Residual);


/**
 * \brief works out the rigid translation correspondence between two sets of corresponding 
 * rigid body transforms
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Mat Tracker2ToTracker1Translation ( 
    const std::vector<cv::Mat>& Tracker1ToWorld1, const std::vector<cv::Mat>& World2ToTracker2,
    double& Residual, const cv::Mat & rcg);


/**
 * \brief works out the rigid rotation and translation correspondence between two sets of corresponding 
 * rigid body transforms
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Mat Tracker2ToTracker1RotationAndTranslation ( 
    const std::vector<cv::Mat>& Tracker1ToWorld1, const std::vector<cv::Mat>& World2ToTracker2,
    std::vector<double>& Residuals, cv::Mat* World2ToWorld1 = NULL );


/**
 * \brief Flips the matrices in the vector from left handed coordinate
 * system to right handed and vice versa
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<cv::Mat> FlipMatrices (const std::vector<cv::Mat> matrices);


/**
 * \brief find the average of a vector of 4x4 matrices
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Mat AverageMatrices(const std::vector<cv::Mat>& matrices);


 /**
  * \brief Sorts the matrices based on the translations , and returns the order
  */
extern "C++" NIFTKOPENCV_EXPORT std::vector<int> SortMatricesByDistance (const std::vector<cv::Mat> matrices);


/**
 * \brief Sorts the matrices based on the rotations, and returns the order
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<int> SortMatricesByAngle (const std::vector<cv::Mat> matrices);


/**
 * \brief Returns the angular distance between two rotation matrices
 */
extern "C++" NIFTKOPENCV_EXPORT double AngleBetweenMatrices(cv::Mat Mat1 , cv::Mat Mat2);


/**
 * \brief Returns the distance between two 4x4 matrices
 */
extern "C++" NIFTKOPENCV_EXPORT double DistanceBetweenMatrices(cv::Mat Mat1 , cv::Mat Mat2);


/**
 * \brief Converts a 3x3 rotation matrix to a quaternion
 */
extern "C++" NIFTKOPENCV_EXPORT cv::Mat DirectionCosineToQuaternion(cv::Mat dc_Matrix);


/**
 * \brief Specific method that inverts a matrix without SVD or decomposition,
 * because the input is known to be orthonormal.
 */
extern "C++" NIFTKOPENCV_EXPORT void InvertRigid4x4Matrix(const CvMat& input, CvMat& output);


/**
 * \brief Overloaded invert method that calls the C-looking one.
 */
extern "C++" NIFTKOPENCV_EXPORT void InvertRigid4x4Matrix(const cv::Matx44d& input, cv::Matx44d& output);


/**
 * \brief Overloaded invert method that calls the C-looking one.
 */
extern "C++" NIFTKOPENCV_EXPORT void InvertRigid4x4Matrix(const cv::Mat& input, cv::Mat& output);


/**
 * \brief Interpolates between two matrices.
 * \param proportion is defined as between [0 and 1], where 0 gives exactly the before matrix,
 * 1 gives exactly the after matrix, and the proportion is a linear proportion between them over which to interpolate.
 */
extern "C++" NIFTKOPENCV_EXPORT void InterpolateTransformationMatrix(const cv::Mat& before, const cv::Mat& after, const double& proportion, cv::Mat& output);


/**
 * \see InterpolateTransformationMatrix(const cv::Mat& before, const cv::Mat& after, const double& proportion, cv::Mat& output)
 */
extern "C++" NIFTKOPENCV_EXPORT void InterpolateTransformationMatrix(const cv::Matx44d& before, const cv::Matx44d& after, const double& proportion, cv::Matx44d& output);

} // end namespace

#endif



