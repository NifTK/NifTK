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
    const mitk::Point3D& invariantPoint,
    const mitk::Point2D& originInImagePlaneInPixels,
    const mitk::Point2D& millimetresPerPixel,
    const std::vector<double>& initialGuessOfTransformation,
    const bool& optimiseScaling,
    const bool& optimiseInvariantPoint,
    double &residualError,
    const std::string& outputFileName
    )
{
  std::vector<std::string> matrixFiles = niftk::GetFilesInDirectory(matrixDirectory);
  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(pointDirectory);

  if (matrixFiles.size() != pointFiles.size())
  {
    MITK_ERROR << "ERROR: The matrix directory:" << std::endl << "  " << matrixDirectory << std::endl << "and the point directory:" << std::endl << "  " << pointDirectory << "contain different number of files!" << std::endl;
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

  cv::Matx44d outputMatrix;
  cv::Point3d invPoint(invariantPoint[0], invariantPoint[1], invariantPoint[2]);
  cv::Point2d originInPixels(originInImagePlaneInPixels[0], originInImagePlaneInPixels[1]);
  cv::Point2d mmPerPix(millimetresPerPixel[0], millimetresPerPixel[1]);

  bool calibratedSuccessfully = this->Calibrate(
      matrices,
      points,
      invPoint,
      originInPixels,
      mmPerPix,
      initialGuessOfTransformation,
      optimiseScaling,
      optimiseInvariantPoint,
      residualError,
      outputMatrix
      );

  if (!calibratedSuccessfully)
  {
    MITK_ERROR << "CalibrateUsingTrackerPointAndFilesInTwoDirectories: Failed to calibrate successfully" << std::endl;
    return false;
  }

  if (outputFileName.size() > 0)
  {
    vtkSmartPointer<vtkMatrix4x4> vtkMatrix = vtkMatrix4x4::New();
    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        vtkMatrix->SetElement(i, j, outputMatrix(i, j));
      }
    }
    bool savedFile = mitk::SaveVtkMatrix4x4ToFile(outputFileName, *vtkMatrix);
    if (!savedFile)
    {
      MITK_ERROR << "CalibrateUsingTrackerPointAndFilesInTwoDirectories: Failed to save file " << outputFileName << std::endl;
    }
  }

  std::cout << "UltrasoundPinCalibration:Result = " << std::endl;
  for (int i = 0; i < 4; i++)
  {
    std::cout << outputMatrix(i, 0) << " " << outputMatrix(i, 1) << " " << outputMatrix(i, 2) << " " << outputMatrix(i, 3) << std::endl;
  }

  return true;
}


//-----------------------------------------------------------------------------
bool UltrasoundPinCalibration::Calibrate(
    const std::vector< cv::Mat>& matrices,
    const std::vector< cv::Point2d >& points,
    const cv::Point3d& invariantPoint,
    const cv::Point2d& originInImagePlaneInPixels,
    const cv::Point2d& millimetresPerPixel,
    const std::vector<double>& initialGuessOfTransformation,
    const bool& optimiseScaling,
    const bool& optimiseInvariantPoint,
    double& residualError,
    cv::Matx44d& outputMatrix
    )
{
  bool isSuccessful = false;

  return isSuccessful;
}

//-----------------------------------------------------------------------------
} // end namespace
