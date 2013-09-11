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
#include <niftkFileHelper.h>
#include <itkUltrasoundPinCalibrationCostFunction.h>
#include <itkLevenbergMarquardtOptimizer.h>
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

  std::vector<cv::Mat> matrices = LoadMatricesFromDirectory (matrixDirectory);

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

  if (matrices.size() != matrixFiles.size())
  {
    MITK_ERROR << "ERROR: Failed to load all the matrices in directory:" << matrixDirectory << std::endl;
    return false;
  }

  if (points.size() != pointFiles.size())
  {
    MITK_ERROR << "ERROR: Failed to load all the points in directory:" << pointDirectory << std::endl;
    return false;
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

  itk::UltrasoundPinCalibrationCostFunction::ParametersType parameters;

  if (!optimiseScaling && !optimiseInvariantPoint)
  {
    parameters.SetSize(6);
  }
  else if (!optimiseScaling && optimiseInvariantPoint)
  {
    parameters.SetSize(9);
    parameters[6] = invariantPoint.x;
    parameters[7] = invariantPoint.y;
    parameters[8] = invariantPoint.z;
  }
  else if (optimiseScaling && !optimiseInvariantPoint)
  {
    parameters.SetSize(8);
    parameters[6] = millimetresPerPixel.x;
    parameters[7] = millimetresPerPixel.y;
  }
  else if (optimiseScaling && optimiseInvariantPoint)
  {
    parameters.SetSize(11);
    parameters[6] = millimetresPerPixel.x;
    parameters[7] = millimetresPerPixel.y;
    parameters[8] = invariantPoint.x;
    parameters[9] = invariantPoint.y;
    parameters[10] = invariantPoint.z;
  }
  parameters[0] = rigidBodyTransformation[0];
  parameters[1] = rigidBodyTransformation[1];
  parameters[2] = rigidBodyTransformation[2];
  parameters[3] = rigidBodyTransformation[3];
  parameters[4] = rigidBodyTransformation[4];
  parameters[5] = rigidBodyTransformation[5];

  std::cout << "UltrasoundPinCalibration:Start parameters = " << parameters << std::endl;

  itk::UltrasoundPinCalibrationCostFunction::Pointer costFunction = itk::UltrasoundPinCalibrationCostFunction::New();
  costFunction->SetMatrices(matrices);
  costFunction->SetPoints(points);
  costFunction->SetNumberOfParameters(parameters.GetSize());
  costFunction->SetInvariantPoint(invariantPoint);
  costFunction->SetMillimetresPerPixel(millimetresPerPixel);

  itk::LevenbergMarquardtOptimizer::Pointer optimizer = itk::LevenbergMarquardtOptimizer::New();
  optimizer->SetCostFunction(costFunction);
  optimizer->SetInitialPosition(parameters);
  optimizer->StartOptimization();
  parameters = optimizer->GetCurrentPosition();

  itk::UltrasoundPinCalibrationCostFunction::MeasureType values = costFunction->GetValue(parameters);
  residualError = costFunction->GetResidual(values);
  outputMatrix = costFunction->GetCalibrationTransformation(parameters);
  isSuccessful = true;

  std::cout << "UltrasoundPinCalibration:Final parameters = " << parameters << std::endl;
  std::cout << "UltrasoundPinCalibration:Residual error   = " << residualError << std::endl;
  std::cout << "UltrasoundPinCalibration:Result:" << std::endl;
  for (int i = 0; i < 4; i++)
  {
    std::cout << outputMatrix(i, 0) << " " << outputMatrix(i, 1) << " " << outputMatrix(i, 2) << " " << outputMatrix(i, 3) << std::endl;
  }
  return isSuccessful;
}

//-----------------------------------------------------------------------------
} // end namespace
