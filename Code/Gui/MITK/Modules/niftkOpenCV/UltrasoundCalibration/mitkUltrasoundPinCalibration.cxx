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
bool UltrasoundPinCalibration::CalibrateUsingTrackerPointAndFilesInTwoDirectories(
    const std::string& matrixDirectory,
    const std::string& pointDirectory,
    const std::string& outputFileName,
    const mitk::Point3D& invariantPoint,
    const mitk::Point2D& originInImagePlaneInPixels,
    double &residualError
    )
{
  std::vector<std::string> matrixFiles = niftk::GetFilesInDirectory(matrixDirectory);
  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(pointDirectory);

  if (matrixFiles.size() != pointFiles.size())
  {
    MITK_ERROR << "ERROR: The matrix directory:" << std::endl << "  " << matrixDirectory << std::endl << "and the point directory:" << std::endl << "  " << pointDirectory << "contain different number of files!" << std::endl;
    return false;
  }

  std::vector< vtkSmartPointer<vtkMatrix4x4> > matrices;
  for (unsigned int i = 0; i < matrixFiles.size(); i++)
  {
    vtkSmartPointer<vtkMatrix4x4> matrix = LoadVtkMatrix4x4FromFile(matrixFiles[i]);
    matrices.push_back(matrix);
  }

  std::vector<mitk::Point3D> points;
  for (unsigned int i = 0; i < pointFiles.size(); i++)
  {
    mitk::Point3D point;
    bool isSuccessful = mitk::Load3DPointFromFile(pointFiles[i], point);
    if (isSuccessful)
    {
      points.push_back(point);
    }
  }

  vtkSmartPointer<vtkMatrix4x4> outputMatrix = vtkMatrix4x4::New();
  outputMatrix->Identity();

  bool calibratedSuccessfully = this->CalibrateUsingTrackerPoint(
      matrices,
      points,
      invariantPoint,
      originInImagePlaneInPixels,
      residualError,
      *outputMatrix
      );

  if (!calibratedSuccessfully)
  {
    MITK_ERROR << "CalibrateUsingTrackerPointAndFilesInTwoDirectories: Failed to calibrate successfully" << std::endl;
    return false;
  }

  bool savedFileSuccessfully = SaveVtkMatrix4x4ToFile(outputFileName, *outputMatrix);

  if (!savedFileSuccessfully)
  {
    MITK_ERROR << "CalibrateUsingTrackerPointAndFilesInTwoDirectories: Failed to save matrix to file:" << outputFileName << std::endl;
    return false;
  }

  return true;
}


//-----------------------------------------------------------------------------
bool UltrasoundPinCalibration::CalibrateUsingTrackerPoint(
    const std::vector< vtkSmartPointer<vtkMatrix4x4> >& matrices,
    const std::vector<mitk::Point3D>& points,
    const mitk::Point3D& invariantPoint,
    const mitk::Point2D& originInImagePlaneInPixels,
    double &residualError,
    vtkMatrix4x4 &outputMatrix
    )
{
  vtkSmartPointer<vtkMatrix4x4> trackerToPinMatrix = vtkMatrix4x4::New();
  trackerToPinMatrix->Identity();

  bool isSuccessful = this->Calibrate(matrices, points, *trackerToPinMatrix, invariantPoint, originInImagePlaneInPixels, residualError, outputMatrix);
  return isSuccessful;
}


//-----------------------------------------------------------------------------
bool UltrasoundPinCalibration::Calibrate(
    const std::vector< vtkSmartPointer<vtkMatrix4x4> >& matrices,
    const std::vector<mitk::Point3D>& points,
    const vtkMatrix4x4& worldToPhantomMatrix,
    const mitk::Point3D& invariantPoint,
    const mitk::Point2D& originInImagePlaneInPixels,
    double &residualError,
    vtkMatrix4x4 &outputMatrix
    )
{
  bool isSuccessful = false;

  return isSuccessful;
}

//-----------------------------------------------------------------------------
} // end namespace
