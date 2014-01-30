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
UltrasoundPinCalibrationCostFunction::UltrasoundPinCalibrationCostFunction() 
{
  m_InvariantPoint[0] = 0;
  m_InvariantPoint[1] = 0;
  m_InvariantPoint[2] = 0;
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::~UltrasoundPinCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetInvariantPoint(const mitk::Point3D &invariantPoint)
{
  m_InvariantPoint = invariantPoint;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::Point3D UltrasoundPinCalibrationCostFunction::GetInvariantPoint() const
{
  return m_InvariantPoint;
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

  cv::Matx44d invariantPointTranslation;
  mitk::MakeIdentity(invariantPointTranslation);

  if (parameters.size() == 8 || parameters.size() == 11)
  {
    scalingTransformation(0, 0) = parameters[6];
    scalingTransformation(1, 1) = parameters[7];
  }
  else
  {
    // i.e. its not being optimised.
    scalingTransformation(0, 0) = this->m_MillimetresPerPixel[0]; // in base class
    scalingTransformation(1, 1) = this->m_MillimetresPerPixel[1]; // in base class
  }

  if (parameters.size() == 9 || parameters.size() == 11)
  {
    if (parameters.size() == 9)
    {
      invariantPointTranslation(0, 3) = -parameters[6];
      invariantPointTranslation(1, 3) = -parameters[7];
      invariantPointTranslation(2, 3) = -parameters[8];
    }
    else
    {
      invariantPointTranslation(0, 3) = -parameters[8];
      invariantPointTranslation(1, 3) = -parameters[9];
      invariantPointTranslation(2, 3) = -parameters[10];
    }

    m_InvariantPoint[0] = invariantPointTranslation(0, 3);
    m_InvariantPoint[1] = invariantPointTranslation(1, 3);
    m_InvariantPoint[2] = invariantPointTranslation(2, 3);
  }
  else
  {
    // i.e. its not being optimised.
    invariantPointTranslation(0, 3) = -m_InvariantPoint[0];
    invariantPointTranslation(1, 3) = -m_InvariantPoint[1];
    invariantPointTranslation(2, 3) = -m_InvariantPoint[2];
  }

  MeasureType value;
  value.SetSize(m_NumberOfValues);

  for (unsigned int i = 0; i < this->m_Matrices.size(); i++)
  {
    cv::Matx44d trackerTransformation(this->m_Matrices[i]);
    cv::Matx44d combinedTransformation = invariantPointTranslation * (trackerTransformation * (rigidTransformation * scalingTransformation));
    cv::Matx41d point, transformedPoint;

    point(0,0) = m_Points[i].x;
    point(1,0) = m_Points[i].y;
    point(2,0) = 0;
    point(3,0) = 1;

    transformedPoint = combinedTransformation * point;

    value[i*3 + 0] = transformedPoint(0, 0);
    value[i*3 + 1] = transformedPoint(1, 0);
    value[i*3 + 2] = transformedPoint(2, 0);
  }

  double residual = this->GetResidual(value);
  std::cout << "UltrasoundPinCalibrationCostFunction::GetValue(" << parameters << ") = " << residual << std::endl;

  return value;
}


//-----------------------------------------------------------------------------
} // end namespace
