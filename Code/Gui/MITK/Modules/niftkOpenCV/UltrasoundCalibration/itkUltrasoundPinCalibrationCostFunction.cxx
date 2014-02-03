/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundPinCalibrationCostFunction.h"
#include <sstream>
#include <mitkOpenCVMaths.h>
#include <mitkExceptionMacro.h>

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::~UltrasoundPinCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::UltrasoundPinCalibrationCostFunction() 
{
  this->SetNumberOfInvariantPoints(1);
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetNumberOfInvariantPoints(const unsigned int &numberOfInvariantPoints)
{
  if (numberOfInvariantPoints < 1)
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::SetNumberOfInvariantPoints numberOfInvariantPoints=" << numberOfInvariantPoints << ", which should be >= 1." << std::endl;
    mitkThrow() << oss.str();
  }

  m_InvariantPoints.resize(numberOfInvariantPoints);
  for (unsigned int i = 0; i < m_InvariantPoints.size(); i++)
  {
    m_InvariantPoints[i][0] = 0;
    m_InvariantPoints[i][1] = 0;
    m_InvariantPoints[i][2] = 0;
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPinCalibrationCostFunction::GetNumberOfInvariantPoints() const
{
  return m_InvariantPoints.size();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetInvariantPoint(const unsigned int &pointNumber, const mitk::Point3D &invariantPoint)
{
  if (pointNumber >= m_InvariantPoints.size())
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::SetInvariantPoint pointNumber=" << pointNumber << ", which is out of range [0.." << m_InvariantPoints.size()-1 << "]." << std::endl;
    mitkThrow() << oss.str();
  }

  m_InvariantPoints[pointNumber] = invariantPoint;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::Point3D UltrasoundPinCalibrationCostFunction::GetInvariantPoint(const unsigned int &pointNumber) const
{
  if (pointNumber >= m_InvariantPoints.size())
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::GetInvariantPoint pointNumber=" << pointNumber << ", which is out of range [0.." << m_InvariantPoints.size()-1 << "]." << std::endl;
    mitkThrow() << oss.str();
  }

  return m_InvariantPoints[pointNumber];
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::MeasureType UltrasoundPinCalibrationCostFunction::GetValue(
  const ParametersType & parameters
  ) const
{

  this->ValidateSizeOfParametersArray(parameters);

  cv::Matx44d rigidTransformation = GetCalibrationTransformation(parameters);

  cv::Matx44d scalingTransformation;
  mitk::MakeIdentity(scalingTransformation);

  int invariantPointOffset = 6;
  if ((parameters.size() - 6)%3 == 2) // 6 for calibration, 3 for each invariant point, 2 remainder must be for scaling.
  {
    scalingTransformation(0, 0) = parameters[6];
    scalingTransformation(1, 1) = parameters[7];
    invariantPointOffset = 8;
  }
  else
  {
    // i.e. its not being optimised.
    scalingTransformation(0, 0) = this->m_MillimetresPerPixel[0]; // in base class
    scalingTransformation(1, 1) = this->m_MillimetresPerPixel[1]; // in base class
  }

  // Check if we have the right number of invariant points.
  int parametersForInvariantPoints = parameters.size() - invariantPointOffset;
  if (parametersForInvariantPoints < 0)
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::GetValue parametersForInvariantPoints=" << parametersForInvariantPoints << ", which implies the size of the parameters array is wrong." << std::endl;
    mitkThrow() << oss.str();
  }

  if (   parametersForInvariantPoints != 0
      && parametersForInvariantPoints%3 != 0
     )
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::GetValue parametersForInvariantPoints=" << parametersForInvariantPoints << ", which is not a multiple of 3." << std::endl;
    mitkThrow() << oss.str();
  }

  MeasureType value;
  value.SetSize(m_NumberOfValues);
  mitk::Point3D invariantPoint;
  unsigned int invariantPointIndex = 0;

  for (unsigned int i = 0; i < this->m_Matrices.size(); i++)
  {
    cv::Matx44d trackerTransformation(this->m_Matrices[i]);
    cv::Matx44d combinedTransformation = trackerTransformation * (rigidTransformation * scalingTransformation);
    cv::Matx41d point, transformedPoint;

    point(0,0) = m_Points[i].second.x;
    point(1,0) = m_Points[i].second.y;
    point(2,0) = 0;
    point(3,0) = 1;

    transformedPoint = combinedTransformation * point;

    // Sort out invariant point
    invariantPointIndex = m_Points[i].first;
    if (invariantPointIndex >= m_NumberOfInvariantPoints)
    {
      std::ostringstream oss;
      oss << "UltrasoundPinCalibrationCostFunction::GetValue invariantPointIndex=" << invariantPointIndex << ", which is out of range [0.." << m_NumberOfInvariantPoints-1 << "]." << std::endl;
      mitkThrow() << oss.str();
    }
    if (parametersForInvariantPoints != 0)
    {
      invariantPoint[0] = parameters[invariantPointOffset + invariantPointIndex*3 + 0];
      invariantPoint[1] = parameters[invariantPointOffset + invariantPointIndex*3 + 1];
      invariantPoint[2] = parameters[invariantPointOffset + invariantPointIndex*3 + 2];
    }
    else
    {
      // i.e. its not being optimised.
      // There may still be multiple points, all of which are not optimised.
      invariantPoint = this->m_InvariantPoints[m_Points[i].first];
    }
    value[i*3 + 0] = transformedPoint(0, 0) - invariantPoint[0];
    value[i*3 + 1] = transformedPoint(1, 0) - invariantPoint[1];
    value[i*3 + 2] = transformedPoint(2, 0) - invariantPoint[2];
  }

  double residual = this->GetResidual(value);
  std::cout << "UltrasoundPinCalibrationCostFunction::GetValue(" << parameters << ") = " << residual << std::endl;

  return value;
}


//-----------------------------------------------------------------------------
} // end namespace
