/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTagTrackingFacade.h"
#include "mitkCameraCalibrationFacade.h"
#include <aruco/aruco.h>

namespace mitk
{

//-----------------------------------------------------------------------------
std::map<int, cv::Point2f> DetectMarkers(
    cv::Mat& inImage,
    const float& minSize,
    const float& maxSize,
    const bool& drawOutlines,
    const bool& drawCentre
    )
{
  cv::Size size(inImage.rows, inImage.cols);
  aruco::CameraParameters cameraParams;

  std::vector<aruco::Marker> markers;
  aruco::MarkerDetector detector;

  detector.setMinMaxSize(minSize, maxSize);
  detector.detect(inImage, markers, cameraParams);

  if (drawOutlines)
  {
    for (unsigned int i=0; i < markers.size(); i++)
    {
      markers[i].draw(inImage, cv::Scalar(255,0,0,255), 1, true);
    }
  }
  if (drawCentre)
  {
    for (unsigned int i=0; i < markers.size(); i++)
    {
      cv::circle(inImage, markers[i].getCenter(), 3, cv::Scalar(0,0,255,255), 3);
    }
  }

  std::map<int, cv::Point2f> result;
  for (unsigned int i=0; i < markers.size(); i++)
  {
    result.insert(std::pair<int, cv::Point2f>(markers[i].id, markers[i].getCenter()));
  }
  return result;
}


//-----------------------------------------------------------------------------
std::map<int, cv::Point3f> DetectMarkerPairs(
    cv::Mat& inImageLeft,
    cv::Mat& inImageRight,
    const cv::Mat& intrinsicParamsLeft,
    const cv::Mat& intrinsicParamsRight,
    const cv::Mat& rightToLeftRotationVector,
    const cv::Mat& rightToLeftTranslationVector,
    const float& minSize,
    const float& maxSize,
    const bool& drawOutlines,
    const bool& drawCentre
    )
{
  std::map<int, cv::Point2f> leftPoints = DetectMarkers(inImageLeft, minSize, maxSize, drawOutlines, drawCentre);
  std::map<int, cv::Point2f> rightPoints = DetectMarkers(inImageRight, minSize, maxSize, drawOutlines, drawCentre);

  std::map<int, cv::Point3f> result;

  std::map<int, cv::Point2f>::iterator leftIter;
  std::map<int, cv::Point2f>::iterator rightIter;
  for (leftIter = leftPoints.begin(); leftIter != leftPoints.end(); leftIter++)
  {
    int leftId = (*leftIter).first;
    cv::Point2f leftPoint = (*leftIter).second;

    rightIter = rightPoints.find(leftId);
    if (rightIter != rightPoints.end())
    {
      cv::Point2f rightPoint = (*rightIter).second;

      std::pair<cv::Point2f, cv::Point2f> pair(leftPoint, rightPoint);
      std::vector<std::pair<cv::Point2f, cv::Point2f> > pairs;
      pairs.push_back(pair);

      std::vector<cv::Point3f> pointIn3D = mitk::TriangulatePointPairs(
          pairs,
          intrinsicParamsLeft,
          intrinsicParamsRight,
          rightToLeftRotationVector,
          rightToLeftTranslationVector
          );

      result.insert(std::pair<int, cv::Point3f>(leftId, pointIn3D[0]));
    }
  }
  return result;
}


} // end namespace
