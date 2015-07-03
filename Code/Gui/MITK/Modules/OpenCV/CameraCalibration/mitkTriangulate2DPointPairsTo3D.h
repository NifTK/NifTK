/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTriangulate2DPointPairsTo3D_h
#define mitkTriangulate2DPointPairsTo3D_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>

namespace mitk {

/**
 * \class Triangulate2DPointPairsTo3D
 * \brief Takes an input file containing 4 numbers on each line corresponding
 * to the x and y image coordinates for the left and then right image of a stereo
 * video pair, and all the calibration data to enable a reconstruction of 3D points.
 *
 * Currently the image that you are dealing with and hence the 2D pixel coordinates are assumed to be distortion corrected.
 */
class NIFTKOPENCV_EXPORT Triangulate2DPointPairsTo3D : public itk::Object
{

public:

  mitkClassMacroItkParent(Triangulate2DPointPairsTo3D, itk::Object);
  itkNewMacro(Triangulate2DPointPairsTo3D);

  itkSetMacro (Input2DPointPairsFileName, std::string);
  itkSetMacro (IntrinsicLeftFileName, std::string);
  itkSetMacro (IntrinsicRightFileName, std::string);
  itkSetMacro (RightToLeftExtrinsics, std::string);
  itkSetMacro (OutputFileName, std::string);
  itkSetMacro (LeftMaskFileName, std::string);
  itkSetMacro (RightMaskFileName, std::string);
  itkSetMacro (OutputMaskImagePrefix, std::string);
  itkSetMacro (UndistortBeforeTriangulation, bool);
  itkSetMacro (TrackingMatrixFileName, std::string);
  itkSetMacro (HandeyeMatrixFileName, std::string); 
  itkSetMacro (MinimumDistanceFromLens, double);
  itkSetMacro (MaximumDistanceFromLens, double);

  bool Triangulate();

protected:

  Triangulate2DPointPairsTo3D();
  virtual ~Triangulate2DPointPairsTo3D();

  Triangulate2DPointPairsTo3D(const Triangulate2DPointPairsTo3D&); // Purposefully not implemented.
  Triangulate2DPointPairsTo3D& operator=(const Triangulate2DPointPairsTo3D&); // Purposefully not implemented.

private:
  
  std::string m_LeftMaskFileName; // the left mask image
  std::string m_RightMaskFileName; // the right mask image
  std::string m_LeftLensToWorldFileName; // the transform to put the triangulated points in world coordinates
  
  std::string m_Input2DPointPairsFileName; //the input file name
  std::string m_IntrinsicLeftFileName; // the left camera intrinsic parameters
  std::string m_IntrinsicRightFileName; // the right camera intrinsic parameters
  std::string m_RightToLeftExtrinsics; // the right to left camera transformation
  std::string m_OutputFileName; // the output file name
  std::string m_OutputMaskImagePrefix; // optional prefix to write out masking images
  std::string m_TrackingMatrixFileName; // the optional tracking matrix name
  std::string m_HandeyeMatrixFileName; // the optional handeye (leftLensToTracker) matrix file name

  bool m_UndistortBeforeTriangulation; //optionally undistort point pairs before triangulation
  std::vector< std::pair<cv::Point2d, cv::Point2d> > m_PointPairs;
  std::vector< cv::Point3d > m_PointsIn3D; 

  unsigned int m_BlankValue; // the value used by the mask for blanking
  void ApplyMasks (); 
  void CullOnDistance (); 
  void WritePointsAsImage (const std::string& prefix,  const cv::Mat& templateMat );

  double m_MinimumDistanceFromLens; //reconstructed points closer to the lens than this will be removed
  double m_MaximumDistanceFromLens; //reconstructed points further from the lens than this will be removed

  
}; // end class

} // end namespace

#endif
