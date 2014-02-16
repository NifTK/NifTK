/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPinCalibrationEvaluation.h"
#include <mitkExceptionMacro.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkUltrasoundCalibration.h>
#include <algorithm>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibrationEvaluation::~UltrasoundPinCalibrationEvaluation()
{
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationEvaluation::UltrasoundPinCalibrationEvaluation()
{
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationEvaluation::Evaluate(
    const std::string& matrixDirectory,
    const std::string& pointDirectory,
    const mitk::Point3D& invariantPoint,
    const mitk::Point2D& millimetresPerPixel,
    const std::string& calibrationMatrixFileName,
    const std::string& cameraToWorldMatrixFileName
    )
{
  cv::Mat calibMatrix = cvCreateMat(4,4,CV_64FC1);
  if (calibrationMatrixFileName.size() != 0)
  {
    if (!mitk::ReadTrackerMatrix(calibrationMatrixFileName, calibMatrix))
    {
      mitkThrow() << "Failed to read calibration matrix " << calibrationMatrixFileName << std::endl;
    }
  }

  cv::Mat cameraToWorldMatrix = cvCreateMat(4,4,CV_64FC1);
  if (cameraToWorldMatrixFileName.size() != 0)
  {
    if (!mitk::ReadTrackerMatrix(cameraToWorldMatrixFileName, cameraToWorldMatrix))
    {
      mitkThrow() << "Failed to read camera-to-world matrix " << cameraToWorldMatrixFileName << std::endl;
    }
  }

  std::vector< cv::Mat > matrices;
  std::vector< std::pair<int, cv::Point2d> > points;

  mitk::UltrasoundCalibration::LoadDataFromDirectories(matrixDirectory, pointDirectory, false, matrices, points);

  double squaredDistance = 0;
  std::vector<double> distancesFromInvariantPoint;
  std::vector<double> squaredDistancesFromInvariantPoint;
  cv::Matx41d transformedPoint;

  cv::Matx44d scaling = mitk::ConstructScalingTransformation(
                          millimetresPerPixel[0],
                          millimetresPerPixel[1],
                          1
                          );

  cv::Matx44d calibMat(calibMatrix);
  cv::Matx44d camToWorldMat(cameraToWorldMatrix);

  for (unsigned int i = 0; i < points.size(); i++)
  {
    cv::Matx44d trackerMatrix = matrices[i];

    cv::Matx41d ultrasoundPoint;
    ultrasoundPoint(0,0) = points[i].second.x;
    ultrasoundPoint(1,0) = points[i].second.y;
    ultrasoundPoint(2,0) = 0;
    ultrasoundPoint(3,0) = 1;

    transformedPoint = (camToWorldMat * (trackerMatrix * (calibMat * (scaling * ultrasoundPoint))));

    squaredDistance =  (transformedPoint(0,0) - invariantPoint[0])*(transformedPoint(0,0) - invariantPoint[0])
                     + (transformedPoint(1,0) - invariantPoint[1])*(transformedPoint(1,0) - invariantPoint[1])
                     + (transformedPoint(2,0) - invariantPoint[2])*(transformedPoint(2,0) - invariantPoint[2])
                     ;

    distancesFromInvariantPoint.push_back(sqrt(squaredDistance));
    squaredDistancesFromInvariantPoint.push_back(squaredDistance);
  }

  std::cout << "Distance error:" << std::endl;
  std::cout << "  Mean   = " << mitk::Mean(distancesFromInvariantPoint) << std::endl;
  std::cout << "  StdDev = " << mitk::StdDev(distancesFromInvariantPoint) << std::endl;
  std::cout << "  Min    = " << *std::min_element(distancesFromInvariantPoint.begin(), distancesFromInvariantPoint.end()) << std::endl;
  std::cout << "  Max    = " << *std::max_element(distancesFromInvariantPoint.begin(), distancesFromInvariantPoint.end()) << std::endl;
  std::cout << "  RMS    = " << mitk::RMS(squaredDistancesFromInvariantPoint) << std::endl;
}


//-----------------------------------------------------------------------------
} // end namespace
