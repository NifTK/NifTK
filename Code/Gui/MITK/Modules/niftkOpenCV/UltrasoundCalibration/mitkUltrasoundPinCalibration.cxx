/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPinCalibration.h"
#include <mitkFileIOUtils.h>
#include <FileHelper.h>
#include <itkMultipleValuedCostFunction.h>
#include <mitkCameraCalibrationFacade.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibration::UltrasoundPinCalibration()
{
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibration::~UltrasoundPinCalibration()
{
}


//-----------------------------------------------------------------------------
bool UltrasoundPinCalibration::CalibrateUsingInvariantPointAndFilesInTwoDirectories(
    const std::string& matrixDirectory,
    const std::string& pointDirectory,
    const bool& optimiseScaling,
    const bool& optimiseInvariantPoint,
    std::vector<double>& rigidBodyTransformation,
    mitk::Point3D& invariantPoint,
    mitk::Point2D& millimetresPerPixel,
    double &residualError,
    vtkMatrix4x4& outputMatrix
    )
{
  std::vector<std::string> matrixFiles = niftk::GetFilesInDirectory(matrixDirectory);
  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(pointDirectory);

  if (matrixFiles.size() != pointFiles.size())
  {
    MITK_ERROR << "ERROR: The matrix directory:" << std::endl << "  " << matrixDirectory << std::endl << "and the point directory:" << std::endl << "  " << pointDirectory << "contain a different number of files!" << std::endl;
    return false;
  }

  std::vector<cv::Mat> matrices = LoadOpenCVMatricesFromDirectory (matrixDirectory);

  std::vector<cv::Point2d> points;
  for (unsigned int i = 0; i < pointFiles.size(); i++)
  {
    mitk::Point2D point;
    if (mitk::Load2DPointFromFile(pointFiles[i], point))
    {
      cv::Point2d cvPoint;
      cvPoint.x = point[0];
      cvPoint.y = point[1];
      points.push_back(cvPoint);
    }
  }

  cv::Matx44d transformationMatrix;
  cv::Point3d invPoint(invariantPoint[0], invariantPoint[1], invariantPoint[2]);
  cv::Point2d mmPerPix(millimetresPerPixel[0], millimetresPerPixel[1]);

  bool calibratedSuccessfully = this->Calibrate(
      matrices,
      points,
      optimiseScaling,
      optimiseInvariantPoint,
      rigidBodyTransformation,
      invPoint,
      mmPerPix,
      transformationMatrix,
      residualError
      );

  if (!calibratedSuccessfully)
  {
    MITK_ERROR << "CalibrateUsingTrackerPointAndFilesInTwoDirectories: Failed to calibrate successfully" << std::endl;
    return false;
  }

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      outputMatrix.SetElement(i, j, transformationMatrix(i, j));
    }
  }
  invariantPoint[0] = invPoint.x;
  invariantPoint[1] = invPoint.y;
  invariantPoint[2] = invPoint.z;
  millimetresPerPixel[0] = mmPerPix.x;
  millimetresPerPixel[1] = mmPerPix.y;

  return true;
}


//-----------------------------------------------------------------------------
bool UltrasoundPinCalibration::Calibrate(
    const std::vector< cv::Mat >& matrices,
    const std::vector< cv::Point2d >& points,
    const bool& optimiseScaling,
    const bool& optimiseInvariantPoint,
    std::vector<double>& rigidBodyTransformation,
    cv::Point3d& invariantPoint,
    cv::Point2d& millimetresPerPixel,
    cv::Matx44d& outputMatrix,
    double& residualError
    )
{
  bool isSuccessful = false;

  std::cout << "UltrasoundPinCalibration:Result = " << std::endl;
  for (int i = 0; i < 4; i++)
  {
    std::cout << outputMatrix(i, 0) << " " << outputMatrix(i, 1) << " " << outputMatrix(i, 2) << " " << outputMatrix(i, 3) << std::endl;
  }
  std::cout << "UltrasoundPinCalibration:Scaling = " << millimetresPerPixel.x << ", " << millimetresPerPixel.y << std::endl;
  std::cout << "UltrasoundPinCalibration:Residual error = " << residualError << std::endl;
  std::cout << "UltrasoundPinCalibration:Success = " << isSuccessful << std::endl;

  return isSuccessful;
}

//-----------------------------------------------------------------------------
} // end namespace
