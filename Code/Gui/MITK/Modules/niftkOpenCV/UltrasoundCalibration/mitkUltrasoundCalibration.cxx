/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundCalibration.h"
#include <mitkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <niftkVTKFunctions.h>
#include <mitkCameraCalibrationFacade.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkOpenCVMaths.h>
#include <mitkExceptionMacro.h>
#include <iostream>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundCalibration::UltrasoundCalibration()
: m_OptimiseScaling(false)
, m_RetrievePointIdentifier(false)
{
  m_MillimetresPerPixel[0] = 1;
  m_MillimetresPerPixel[1] = 1;
  m_InitialGuess.resize(6);
}


//-----------------------------------------------------------------------------
UltrasoundCalibration::~UltrasoundCalibration()
{
}


//-----------------------------------------------------------------------------
void UltrasoundCalibration::InitialiseMillimetresPerPixel(const std::vector<float>& commandLineArgs)
{
  if (commandLineArgs.size() == 2)
  {
    m_MillimetresPerPixel[0] = commandLineArgs[0];
    m_MillimetresPerPixel[1] = commandLineArgs[1];
  }
  else
  {
    m_MillimetresPerPixel[0] = 1;
    m_MillimetresPerPixel[1] = 1;
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibration::InitialiseInitialGuess(const std::string& fileName)
{
  vtkSmartPointer<vtkMatrix4x4> initialMatrix = vtkMatrix4x4::New();
  initialMatrix->Identity();

  if(fileName.size() != 0)
  {
    initialMatrix = niftk::LoadMatrix4x4FromFile(fileName, false);
  }

  this->SetInitialGuess(*initialMatrix);
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibration::SetInitialGuess(const vtkMatrix4x4& matrix)
{
  cv::Matx33d rotationMatrix;
  cv::Matx31d rotationVector;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      rotationMatrix(i,j) = matrix.GetElement(i,j);
    }
  }
  cv::Rodrigues(rotationMatrix, rotationVector);

  m_InitialGuess.clear();

  m_InitialGuess.push_back(rotationVector(0,0));
  m_InitialGuess.push_back(rotationVector(1,0));
  m_InitialGuess.push_back(rotationVector(2,0));
  m_InitialGuess.push_back(matrix.GetElement(0,3));
  m_InitialGuess.push_back(matrix.GetElement(1,3));
  m_InitialGuess.push_back(matrix.GetElement(2,3));

  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibration::LoadDataFromDirectories(const std::string& matrixDirectory,
                                                    const std::string& pointDirectory,
                                                    const bool& retrievePointIdentifier,
                                                    std::vector< cv::Mat >& matrices,
                                                    std::vector< std::pair<int, cv::Point2d> >& points
                                                    )
{
  std::vector<std::string> matrixFiles = niftk::GetFilesInDirectory(matrixDirectory);
  std::sort(matrixFiles.begin(), matrixFiles.end());

  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(pointDirectory);
  std::sort(pointFiles.begin(), pointFiles.end());

  if (matrixFiles.size() != pointFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "The matrix directory:" << std::endl << "  " << matrixDirectory << std::endl << "and the point directory:" << std::endl << "  " << pointDirectory << "contain a different number of files!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  matrices = LoadMatricesFromDirectory (matrixDirectory);

  for (unsigned int i = 0; i < pointFiles.size(); i++)
  {
    int pointIdentifier;
    cv::Point2d point;
    std::string fileName = pointFiles[i];

    if(fileName.size() > 0)
    {
      ifstream myfile(fileName.c_str());
      if (myfile.is_open())
      {
        if (retrievePointIdentifier)
        {
          myfile >> pointIdentifier;
        }
        myfile >> point.x;
        myfile >> point.y;

        if (myfile.bad() || myfile.eof() || myfile.fail())
        {
          std::ostringstream errorMessage;
          errorMessage << "Could not load point file:" << fileName << std::endl;
          mitkThrow() << errorMessage.str();
        }
        myfile.close();
      }
    }
    points.push_back(std::pair<int, cv::Point2d>(pointIdentifier, point));
  }

  if (matrices.size() != matrixFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Failed to load all the matrices in directory:" << matrixDirectory << std::endl;
    mitkThrow() << errorMessage.str();
  }

  if (points.size() != pointFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Failed to load all the points in directory:" << pointDirectory << std::endl;
    mitkThrow() << errorMessage.str();
  }
}


//-----------------------------------------------------------------------------
double UltrasoundCalibration::CalibrateFromDirectories(
  const std::string& matrixDirectory,
  const std::string& pointDirectory,
  vtkMatrix4x4& outputMatrix
  )
{

  std::vector< cv::Mat > matrices;
  std::vector< std::pair<int, cv::Point2d> > points;

  UltrasoundCalibration::LoadDataFromDirectories(matrixDirectory, pointDirectory, this->m_RetrievePointIdentifier, matrices, points);

  cv::Matx44d transformationMatrix;
  mitk::MakeIdentity(transformationMatrix);

  double residualError = this->Calibrate(matrices, points, transformationMatrix);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      outputMatrix.SetElement(i, j, transformationMatrix(i, j));
    }
  }

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
