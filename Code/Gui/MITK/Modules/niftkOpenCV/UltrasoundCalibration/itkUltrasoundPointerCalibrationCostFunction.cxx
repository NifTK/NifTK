/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundPointerCalibrationCostFunction.h"
#include <mitkOpenCVMaths.h>

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundPointerCalibrationCostFunction::UltrasoundPointerCalibrationCostFunction()
{
  mitk::MakeIdentity(m_PointerTrackerToProbeTrackerTransform);
  mitk::MakeIdentity(m_ProbeToProbeTrackerTransform);
  m_PointerOffset[0] = 0;
  m_PointerOffset[1] = 0;
  m_PointerOffset[2] = 0;
}


//-----------------------------------------------------------------------------
UltrasoundPointerCalibrationCostFunction::~UltrasoundPointerCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::SetPointerOffset(const mitk::Point3D& pointerOffset)
{
  m_PointerOffset = pointerOffset;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::Point3D UltrasoundPointerCalibrationCostFunction:: GetPointerOffset() const
{
  return m_PointerOffset;
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::SetPointerTrackerToProbeTrackerTransform(const vtkMatrix4x4& matrix)
{
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
    {
      m_PointerTrackerToProbeTrackerTransform(r,c) = matrix.GetElement(r,c);
    }
  }
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::SetProbeToProbeTrackerTransform(const vtkMatrix4x4& matrix)
{
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
    {
      m_ProbeToProbeTrackerTransform(r,c) = matrix.GetElement(r,c);
    }
  }
}


//-----------------------------------------------------------------------------
UltrasoundPointerCalibrationCostFunction::MeasureType UltrasoundPointerCalibrationCostFunction::GetValue(
  const ParametersType & parameters
  ) const
{
  this->ValidateSizeOfParametersArray(parameters);

  cv::Matx44d rigidTransformation = GetCalibrationTransformation(parameters);

  cv::Matx44d scalingTransformation;
  mitk::MakeIdentity(scalingTransformation);

  if (parameters.size() == 8)
  {
    scalingTransformation(0, 0) = parameters[6];
    scalingTransformation(1, 1) = parameters[7];
  }
  else
  {
    // i.e. its not being optimised.
    scalingTransformation(0, 0) = this->m_MillimetresPerPixel[0];
    scalingTransformation(1, 1) = this->m_MillimetresPerPixel[1];
  }

  MeasureType value;
  value.SetSize(this->m_NumberOfValues);

  cv::Matx41d point;
  cv::Matx44d trackingTransformation;
  cv::Matx44d tipToProbeTrackerTransformation;
  cv::Matx44d ultrasoundToProbeTrackerTransformation;
  cv::Matx41d tipPositionInProbeTrackerCoordinates;
  cv::Matx41d ultrasoundPositionInProbeTrackerCoordinates;

  for (unsigned int i = 0; i < this->m_Matrices.size(); i++)
  {
    // First calculate position of pointer tip point,
    // in the coordinate system of the tracker that
    // tracks the ultrasound probe.
    point(0,0) = m_PointerOffset[0];
    point(1,0) = m_PointerOffset[1];
    point(2,0) = m_PointerOffset[2];
    point(3,0) = 1;
    trackingTransformation = this->m_Matrices[i];
    tipToProbeTrackerTransformation = m_PointerTrackerToProbeTrackerTransform * trackingTransformation;
    tipPositionInProbeTrackerCoordinates = tipToProbeTrackerTransformation * point;

    // Then calculate the ultrasound point.
    point(0,0) = this->m_Points[i].second.x;
    point(1,0) = this->m_Points[i].second.y;
    point(2,0) = 0;
    point(3,0) = 1;
    ultrasoundToProbeTrackerTransformation = m_ProbeToProbeTrackerTransform * (rigidTransformation * scalingTransformation);
    ultrasoundPositionInProbeTrackerCoordinates = ultrasoundToProbeTrackerTransformation * point;

    // Then subtract the two.
    value[i*3 + 0] = ultrasoundPositionInProbeTrackerCoordinates(0, 0) - tipPositionInProbeTrackerCoordinates(0,0);
    value[i*3 + 1] = ultrasoundPositionInProbeTrackerCoordinates(1, 0) - tipPositionInProbeTrackerCoordinates(1,0);
    value[i*3 + 2] = ultrasoundPositionInProbeTrackerCoordinates(2, 0) - tipPositionInProbeTrackerCoordinates(2,0);
  }

  double residual = this->GetResidual(value);
  std::cout << "UltrasoundPointerCalibrationCostFunction::GetValue(" << parameters << ") = " << residual << std::endl;

  return value;
}


//-----------------------------------------------------------------------------
} // end namespace
