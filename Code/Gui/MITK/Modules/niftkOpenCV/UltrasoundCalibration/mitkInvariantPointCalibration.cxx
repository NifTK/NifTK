/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkInvariantPointCalibration.h"
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
InvariantPointCalibration::InvariantPointCalibration()
{
  m_InitialGuess.resize(6);
  m_InvariantPoint[0] = 0;
  m_InvariantPoint[1] = 0;
  m_InvariantPoint[2] = 0;
}


//-----------------------------------------------------------------------------
InvariantPointCalibration::~InvariantPointCalibration()
{
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::InitialiseInitialGuess(const std::string& fileName)
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
void InvariantPointCalibration::SetInitialGuess(const vtkMatrix4x4& matrix)
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
} // end namespace
