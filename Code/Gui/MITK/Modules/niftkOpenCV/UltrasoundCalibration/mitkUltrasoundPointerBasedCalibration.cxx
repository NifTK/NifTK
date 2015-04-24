/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPointerBasedCalibration.h"
#include <mitkExceptionMacro.h>

namespace mitk
{

//-----------------------------------------------------------------------------
UltrasoundPointerBasedCalibration::UltrasoundPointerBasedCalibration()
{
  m_ScalingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_ScalingMatrix->Identity();
  m_RigidBodyMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_RigidBodyMatrix->Identity();
  m_UltrasoundImagePoints = mitk::PointSet::New();
  m_SensorPoints = mitk::PointSet::New();
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasoundPointerBasedCalibration::~UltrasoundPointerBasedCalibration()
{
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerBasedCalibration::GetCalibrationMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> rigid = vtkSmartPointer<vtkMatrix4x4>::New();
  rigid->DeepCopy(m_RigidBodyMatrix);

  vtkSmartPointer<vtkMatrix4x4> scaling = vtkSmartPointer<vtkMatrix4x4>::New();
  scaling->DeepCopy(m_ScalingMatrix);

  vtkSmartPointer<vtkMatrix4x4> calibration = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkMatrix4x4::Multiply4x4(rigid, scaling, calibration);

  return calibration;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerBasedCalibration::GetRigidBodyMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> clone = vtkSmartPointer<vtkMatrix4x4>::New();
  clone->DeepCopy(m_RigidBodyMatrix);
  return clone;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerBasedCalibration::GetScalingMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> clone = vtkSmartPointer<vtkMatrix4x4>::New();
  clone->DeepCopy(m_ScalingMatrix);
  return clone;
}


//-----------------------------------------------------------------------------
void UltrasoundPointerBasedCalibration::SetSensorPoints(mitk::PointSet::Pointer points)
{
  m_SensorPoints = points;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPointerBasedCalibration::SetImagePoints(mitk::PointSet::Pointer points)
{
  m_SensorPoints = points;
  this->Modified();
}


//-----------------------------------------------------------------------------
double UltrasoundPointerBasedCalibration::DoPointerBasedCalibration()
{
  double fiducialRegistrationError = std::numeric_limits<double>::max();

  this->Modified();
  return fiducialRegistrationError;
}

//-----------------------------------------------------------------------------
} // end namespace

