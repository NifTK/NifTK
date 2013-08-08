/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundPinCalibrationCostFunction.h"

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::UltrasoundPinCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::~UltrasoundPinCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPinCalibrationCostFunction::GetNumberOfParameters(void) const
{
  return m_NumberOfParameters;
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPinCalibrationCostFunction::GetNumberOfValues(void) const
{
  return m_NumberOfValues;
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetMatrices(const std::vector< cv::Mat >& matrices)
{
  m_Matrices = matrices;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetPoints(const std::vector< cv::Point2d > points)
{
  m_Points = points;
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::MeasureType UltrasoundPinCalibrationCostFunction::GetValue(
  const ParametersType & parameters
  ) const
{
  MeasureType value;
  return value;
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::GetDerivative(
  const ParametersType & parameters,
  DerivativeType  & derivative
  ) const
{

}



} // end namespace
