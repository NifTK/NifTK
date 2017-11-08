/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkHandEyeEvaluationInterface.h"
#include <niftkNiftyCalTypes.h>
#include <niftkIOUtilities.h>
#include <niftkMatrixUtilities.h>
#include <niftkPoseFromPoints.h>
#include <niftkFileHelper.h>
#include <mitkTrackingAndTimeStampsContainer.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkExceptionMacro.h>
#include <cv.h>
#include <vector>

namespace niftk
{

//-----------------------------------------------------------------------------
int EvaluateHandeyeFromPoints(const std::string& trackingDir,
                              const std::string& pointsDir,
                              const std::string& modelFile,
                              const std::string& intrinsicsFile,
                              const std::string& handeyeFile,
                              const std::string& registrationFile,
                              double &rmsError
                             )
{
  rmsError = 0;

  mitk::TrackingAndTimeStampsContainer trackingContainer;
  trackingContainer.LoadFromDirectory(trackingDir, false);
  if (trackingContainer.GetSize() == 0)
  {
    mitkThrow() << "Failed to load tracking data from " << trackingDir << std::endl;
  }

  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(pointsDir);
  std::sort(pointFiles.begin(), pointFiles.end(), niftk::NumericStringCompare);

  std::list<PointSet> imagePoints;
  std::vector<unsigned long long> imagePointsTimeStamps;

  for (std::vector<std::string>::size_type i = 0; i < pointFiles.size(); i++)
  {
    niftk::PointSet p = niftk::LoadPointSet(pointFiles[i]);
    if (p.size() == 0)
    {
      mitkThrow() << "Empty file, or failed to read points from:" << pointFiles[i] << std::endl;
    }
    unsigned long long t = mitk::ExtractTimeStampOrThrow(pointFiles[i]);

    imagePoints.push_back(p);
    imagePointsTimeStamps.push_back(t);
  }

  niftk::Model3D model = niftk::LoadModel3D(modelFile);

  cv::Mat cameraIntrinsic = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat cameraDistortion = cv::Mat::zeros(1, 5, CV_64FC1);
  mitk::LoadCameraIntrinsicsFromPlainText(intrinsicsFile, &cameraIntrinsic, &cameraDistortion);

  cv::Matx44d handeyeMatrix = cv::Matx44d::eye();
  mitk::ReadTrackerMatrix(handeyeFile, handeyeMatrix);

  cv::Matx44d modelToWorldViaRegistration = cv::Matx44d::eye();
  mitk::ReadTrackerMatrix(registrationFile, modelToWorldViaRegistration);

  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  niftk::PoseFromPoints(model, imagePoints, cameraIntrinsic, cameraDistortion, rvecs, tvecs);

  int bestLag = 0;
  double rms = 0;
  bool isInBounds = false;
  double bestRMSSoFar = std::numeric_limits<double>::max();
  double squaredError;
  long long timingError;

  for (int lag = 0; lag <= 100; lag++)
  {
    niftk::Point3D p3;
    unsigned int pointSetCounter = 0;
    unsigned long long pointCounter = 0;

    unsigned long long time = 0;

    std::list<PointSet>::const_iterator pointSetIter;
    for (pointSetIter = imagePoints.begin(); pointSetIter != imagePoints.end(); ++pointSetIter)
    {
      cv::Matx44d modelToCamera = niftk::RotationAndTranslationToMatrix(rvecs[pointSetCounter], tvecs[pointSetCounter]);
      time = imagePointsTimeStamps[pointSetCounter] - lag;

      cv::Matx44d trackingMatrix = trackingContainer.GetNearestMatrix(time, timingError, isInBounds);
      cv::Matx44d modelToWorldViaHandEye = trackingMatrix * handeyeMatrix * modelToCamera;

      cv::Matx41d modelPoint;
      modelPoint(3, 0) = 1;

      cv::Matx41d worldPointViaRegistration;
      cv::Matx41d worldPointViaHandEye;

      niftk::PointSet::const_iterator pointIter;
      for (pointIter = pointSetIter->begin(); pointIter != pointSetIter->end(); ++pointIter)
      {
        p3.id = pointIter->first;
        p3.point = model[p3.id].point;
        modelPoint(0, 0) = p3.point.x;
        modelPoint(1, 0) = p3.point.y;
        modelPoint(2, 0) = p3.point.z;

        worldPointViaRegistration = modelToWorldViaRegistration * modelPoint;
        worldPointViaHandEye = modelToWorldViaHandEye * modelPoint;

        squaredError += (worldPointViaHandEye(0, 0) - worldPointViaRegistration(0, 0))
                         *
                        (worldPointViaHandEye(0, 0) - worldPointViaRegistration(0, 0))
                        +
                        (worldPointViaHandEye(1, 0) - worldPointViaRegistration(1, 0))
                         *
                        (worldPointViaHandEye(1, 0) - worldPointViaRegistration(1, 0))
                        +
                        (worldPointViaHandEye(2, 0) - worldPointViaRegistration(2, 0))
                        *
                        (worldPointViaHandEye(2, 0) - worldPointViaRegistration(2, 0));
        pointCounter++;
      }
    }
    if (pointCounter > 0)
    {
      rms = squaredError / static_cast<double>(pointCounter);
    }

    std::cout << "EvaluateHandeyeFromPoints: lag=" << lag << ", rms=" << rms << std::endl;
    if (rms < bestRMSSoFar)
    {
      bestRMSSoFar = rms;
      bestLag = lag;
      rmsError = rms;
    }
  }

  return -bestLag;
}

} // end namespace
