/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundCalibrationCostFunction.h"
#include <mitkOpenCVMaths.h>
#include <sstream>
#include <mitkExceptionMacro.h>

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundCalibrationCostFunction::UltrasoundCalibrationCostFunction()
: m_OptimiseScaling(false)
, m_NumberOfValues(1)
, m_NumberOfParameters(1)
{
  m_InitialGuess.resize(6);
  m_MillimetresPerPixel[0] = 1;
  m_MillimetresPerPixel[1] = 1;
}


//-----------------------------------------------------------------------------
UltrasoundCalibrationCostFunction::~UltrasoundCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundCalibrationCostFunction::GetNumberOfValues(void) const
{
  return m_NumberOfValues;
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundCalibrationCostFunction::GetNumberOfParameters(void) const
{
  return m_NumberOfParameters;
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::SetNumberOfParameters(const int& numberOfParameters)
{
  m_NumberOfParameters = numberOfParameters;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::SetInitialGuess(const std::vector<double>& initialGuess)
{
  m_InitialGuess = initialGuess;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::SetMatrices(const std::vector< cv::Mat >& matrices)
{
  m_Matrices = matrices;
  m_NumberOfValues = matrices.size() * 3;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::SetPoints(const std::vector<std::pair<int, cv::Point2d> > &points)
{
  m_Points = points;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::SetMillimetresPerPixel(const mitk::Point2D& mmPerPix)
{
  m_MillimetresPerPixel = mmPerPix;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::SetScales(const ParametersType& scales)
{
  m_Scales = scales;
  this->Modified();
}


//-----------------------------------------------------------------------------
cv::Matx44d UltrasoundCalibrationCostFunction::GetCalibrationTransformation(const ParametersType & parameters) const
{
  cv::Matx44d rigidTransformation = mitk::ConstructRodriguesTransformationMatrix(
    parameters[0],
    parameters[1],
    parameters[2],
    parameters[3],
    parameters[4],
    parameters[5]
  );

  return rigidTransformation;
}


//-----------------------------------------------------------------------------
double UltrasoundCalibrationCostFunction::GetResidual(const MeasureType& values) const
{
  double residual = 0;
  unsigned int numberOfValues = values.size();

  for (unsigned int i = 0; i < numberOfValues; i++)
  {
    residual += (values[i]*values[i]);
  }

  if (numberOfValues > 0)
  {
    residual /= static_cast<double>(numberOfValues);
  }
  residual = sqrt(residual);

  return residual;
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::ValidateSizeOfParametersArray(const ParametersType & parameters) const
{
  if (parameters.GetSize() != m_NumberOfParameters)
  {
    std::ostringstream oss;
    oss << "UltrasoundCalibrationCostFunction::GetValue given " << parameters.GetSize() << ", but was expecting " << this->m_NumberOfParameters << " parameters." << std::endl;
    mitkThrow() << oss.str();
  }

  if (m_Matrices.size() == 0)
  {
    mitkThrow() << "UltrasoundCalibrationCostFunction::GetValue(): No matrices available." << std::endl;
  }

  if (m_Points.size() == 0)
  {
    mitkThrow() << "UltrasoundCalibrationCostFunction::GetValue(): No points available." << std::endl;
  }

  if (m_Matrices.size() != m_Points.size())
  {
    std::ostringstream oss;
    oss << "UltrasoundCalibrationCostFunction::GetValue(): The number of matrices (" << this->m_Matrices.size() << ") differs from the number of points (" << m_Points.size() << ")." << std::endl;
    mitkThrow() << oss.str();
  }
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::ValidateSizeOfScalesArray(const ParametersType & parameters) const
{
  if (parameters.GetSize() != m_Scales.GetSize())
  {
    std::ostringstream oss;
    oss << "UltrasoundCalibrationCostFunction::ValidateSizeOfScalesArray given " << parameters.GetSize() << " parameters, but the scale factors array has " << m_Scales.GetSize() << " parameters." << std::endl;
    mitkThrow() << oss.str();
  }
}


//-----------------------------------------------------------------------------
void UltrasoundCalibrationCostFunction::GetDerivative(
  const ParametersType & parameters,
  DerivativeType  & derivative
  ) const
{
  this->ValidateSizeOfScalesArray(parameters);

  MeasureType forwardValue;
  MeasureType backwardValue;

  ParametersType offsetParameters;
  derivative.SetSize(m_NumberOfParameters, m_NumberOfValues);

  for (unsigned int i = 0; i < m_NumberOfParameters; i++)
  {
    offsetParameters = parameters;
    offsetParameters[i] += m_Scales[i];
    forwardValue = this->GetValue(offsetParameters);

    offsetParameters = parameters;
    offsetParameters[i] -= m_Scales[i];
    backwardValue = this->GetValue(offsetParameters);

    for (unsigned int j = 0; j < m_NumberOfValues; j++)
    {
      derivative[i][j] = (forwardValue[j] - backwardValue[j])/2.0;
    }
  }

  // Normalise
  double norm = 0;
  for (unsigned int j = 0; j < m_NumberOfValues; j++)
  {
    norm = 0;
    for (unsigned int i = 0; i < m_NumberOfParameters; i++)
    {
      norm += (derivative[i][j]*derivative[i][j]);
    }
    norm = sqrt(norm);

    for (unsigned int i = 0; i < m_NumberOfParameters; i++)
    {
      derivative[i][j] = derivative[i][j]*m_Scales[i]/norm;
    }
  }
}

//-----------------------------------------------------------------------------
} // end namespace
