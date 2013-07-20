/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTagTrackingFacade.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkPointUtils.h>
#include <aruco/aruco.h>

namespace mitk
{

//-----------------------------------------------------------------------------
std::map<int, cv::Point2f> DetectMarkers(
    cv::Mat& inImage,
    const float& minSize,
    const float& maxSize,
    const double& blockSize,
    const double& offset,
    const bool& drawOutlines,
    const bool& drawCentre
    )
{
  aruco::CameraParameters cameraParams;

  std::vector<aruco::Marker> markers;
  aruco::MarkerDetector detector;

  detector.setMinMaxSize(minSize, maxSize);
  detector.setThresholdMethod(aruco::MarkerDetector::ADPT_THRES);
  detector.setThresholdParams(blockSize, offset);
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
    const double& blockSize,
    const double& offset,
    const bool& drawOutlines,
    const bool& drawCentre
    )
{
  std::map<int, cv::Point2f> leftPoints = DetectMarkers(inImageLeft, minSize, maxSize, blockSize, offset, drawOutlines, drawCentre);
  std::map<int, cv::Point2f> rightPoints = DetectMarkers(inImageRight, minSize, maxSize, blockSize, offset, drawOutlines, drawCentre);

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

      std::vector<cv::Point3f> pointsIn3D = mitk::TriangulatePointPairs(
          pairs,
          intrinsicParamsLeft,
          intrinsicParamsRight,
          rightToLeftRotationVector,
          rightToLeftTranslationVector
          );

      if (pointsIn3D.size() > 0) // should only ever be 1, as we are doing 1 at a time.
      {
        cv::Point3f pointIn3D = pointsIn3D[0];
        result.insert(std::pair<int, cv::Point3f>(leftId, pointIn3D));

        std::cout << "Marker id=" << leftId << ", Left=(" << leftPoint.x << ", " << leftPoint.y << "), Right=(" << rightPoint.x << ", " << rightPoint.y << "), 3D=(" << pointIn3D.x << ", " << pointIn3D.y << ", " << pointIn3D.z << ")" << std::endl;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
std::map<int, mitk::Point6D> DetectMarkerPairsAndNormals(
   cv::Mat& inImageLeft,
   cv::Mat& inImageRight,
   const cv::Mat& intrinsicParamsLeft,
   const cv::Mat& intrinsicParamsRight,
   const cv::Mat& rightToLeftRotationVector,
   const cv::Mat& rightToLeftTranslationVector,
   const float& minSize,
   const float& maxSize,
   const double& blockSize,
   const double& offset
   )
{
  // Output map, containing point ID, and a 6-tuple of point and normal.
  std::map<int, mitk::Point6D> output;

  // First detect all markers
  aruco::CameraParameters cameraParams;

  std::vector<aruco::Marker> leftMarkers;
  aruco::MarkerDetector leftDetector;

  leftDetector.setMinMaxSize(minSize, maxSize);
  leftDetector.setThresholdMethod(aruco::MarkerDetector::ADPT_THRES);
  leftDetector.setThresholdParams(blockSize, offset);
  leftDetector.detect(inImageLeft, leftMarkers, cameraParams);

  std::vector<aruco::Marker> rightMarkers;
  aruco::MarkerDetector rightDetector;

  rightDetector.setMinMaxSize(minSize, maxSize);
  rightDetector.setThresholdMethod(aruco::MarkerDetector::ADPT_THRES);
  rightDetector.setThresholdParams(blockSize, offset);
  rightDetector.detect(inImageRight, rightMarkers, cameraParams);

  // Now we find corresponding markers
  for (unsigned int i = 0; i < leftMarkers.size(); ++i)
  {
    int pointID = leftMarkers[i].id;

    for (unsigned int j = 0; j < rightMarkers.size(); ++j)
    {
      // Check if we have valid points detected in both left and right views.
      if (rightMarkers[j].id == pointID && leftMarkers[i].isValid() && rightMarkers[j].isValid())
      {
        // Extract corresponding points. Assuming that each marker has ordered points???
        std::vector<std::pair<cv::Point2f, cv::Point2f> > pairs;
        for (int k = 0; k < 4; k++)
        {
          std::pair<cv::Point2f, cv::Point2f> pair(leftMarkers[i][k], rightMarkers[j][k]);
          pairs.push_back(pair);
        }
        std::pair<cv::Point2f, cv::Point2f> centrePoint(leftMarkers[i].getCenter(), rightMarkers[j].getCenter());
        pairs.push_back(centrePoint);

        std::vector<cv::Point3f> pointsIn3D = mitk::TriangulatePointPairs(
            pairs,
            intrinsicParamsLeft,
            intrinsicParamsRight,
            rightToLeftRotationVector,
            rightToLeftTranslationVector
            );

        // Now we have 5 x 3D points. So, we need the surface normal and centre.
        if (pointsIn3D.size() > 0)
        {
          mitk::Point3D a, b, c, normal;
        a[0] = pointsIn3D[0].x;
        a[1] = pointsIn3D[0].y;
        a[2] = pointsIn3D[0].z;
        b[0] = pointsIn3D[1].x;
        b[1] = pointsIn3D[1].y;
        b[2] = pointsIn3D[1].z;
        c[0] = pointsIn3D[2].x;
        c[1] = pointsIn3D[2].y;
        c[2] = pointsIn3D[2].z;
        mitk::ComputeNormalFromPoints(a, b, c, normal);

        mitk::Point6D outputPoint;
        outputPoint[0] = pointsIn3D[4].x;
        outputPoint[1] = pointsIn3D[4].y;
        outputPoint[2] = pointsIn3D[4].z;
        outputPoint[3] = normal[0];
        outputPoint[4] = normal[1];
        outputPoint[5] = normal[2];

        // Store the result in the map.
        output.insert(std::pair<int, mitk::Point6D>(pointID, outputPoint));
        }


      } // end if valid point
    } // end for each right marker
  } // end for each left marker

  return output;
}

//-----------------------------------------------------------------------------
} // end namespace
