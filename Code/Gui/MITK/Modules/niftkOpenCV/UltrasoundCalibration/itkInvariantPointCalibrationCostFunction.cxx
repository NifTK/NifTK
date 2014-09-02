/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkInvariantPointCalibrationCostFunction.h"
#include <mitkOpenCVMaths.h>
#include <sstream>
#include <mitkExceptionMacro.h>

namespace itk {

//-----------------------------------------------------------------------------
InvariantPointCalibrationCostFunction::InvariantPointCalibrationCostFunction()
: m_NumberOfValues(1)
, m_NumberOfParameters(1)
, m_PointData(NULL)
, m_TrackingData(NULL)
{
}


//-----------------------------------------------------------------------------
InvariantPointCalibrationCostFunction::~InvariantPointCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
unsigned int InvariantPointCalibrationCostFunction::GetNumberOfValues(void) const
{
  return m_NumberOfValues * 3;
}


//-----------------------------------------------------------------------------
unsigned int InvariantPointCalibrationCostFunction::GetNumberOfParameters(void) const
{
  return m_NumberOfParameters;
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::GetDerivative(
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
cv::Matx44d InvariantPointCalibrationCostFunction::GetRigidTransformation(const ParametersType & parameters) const
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
cv::Matx44d InvariantPointCalibrationCostFunction::GetTranslationTransformation(const ParametersType & parameters) const
{
  cv::Matx44d translateToInvariantPoint;
  mitk::MakeIdentity(translateToInvariantPoint);

  if (parameters.GetSize() >= 9 && this->GetOptimiseInvariantPoint())
  {
    translateToInvariantPoint(0,3) = parameters[6];
    translateToInvariantPoint(1,3) = parameters[7];
    translateToInvariantPoint(2,3) = parameters[8];
  }
  else
  {
    translateToInvariantPoint(0,3) = m_InvariantPoint[0];
    translateToInvariantPoint(1,3) = m_InvariantPoint[1];
    translateToInvariantPoint(2,3) = m_InvariantPoint[2];
  }

  return translateToInvariantPoint;
}


//-----------------------------------------------------------------------------
InvariantPointCalibrationCostFunction::TimeStampType InvariantPointCalibrationCostFunction::GetLag(const ParametersType & parameters) const
{
  TimeStampType lag = 0;
  if (this->GetNumberOfParameters() == 10)
  {
    lag = parameters[9];
  }
  else if (this->GetNumberOfParameters() == 11)
  {
    lag = parameters[11];
  }
  return lag;
}


//-----------------------------------------------------------------------------
double InvariantPointCalibrationCostFunction::GetResidual(const MeasureType& values) const
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
void InvariantPointCalibrationCostFunction::SetScales(const ParametersType& scales)
{
  m_Scales = scales;
  this->Modified();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::SetNumberOfParameters(const int& numberOfParameters)
{
  m_NumberOfParameters = numberOfParameters;
  this->Modified();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::SetTrackingData(mitk::TrackingAndTimeStampsContainer* trackingData)
{
  m_TrackingData = trackingData;
  this->Modified();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::SetPointData(std::vector< std::pair<unsigned long long, cv::Point3d> >* pointData)
{
  m_PointData = pointData;
  this->Modified();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::ValidateSizeOfParametersArray(const ParametersType & parameters) const
{
  if (parameters.GetSize() != m_NumberOfParameters)
  {
    std::ostringstream oss;
    oss << "InvariantPointCalibrationCostFunction::ValidateSizeOfParametersArray given " << parameters.GetSize() << ", but was expecting " << this->m_NumberOfParameters << " parameters." << std::endl;
    mitkThrow() << oss.str();
  }

  if (m_PointData == NULL)
  {
    mitkThrow() << "InvariantPointCalibrationCostFunction::ValidateSizeOfParametersArray(): No points available." << std::endl;
  }

  if (m_TrackingData == NULL)
  {
    mitkThrow() << "InvariantPointCalibrationCostFunction::ValidateSizeOfParametersArray(): No tracking data available." << std::endl;
  }
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::ValidateSizeOfScalesArray(const ParametersType & parameters) const
{
  if (parameters.GetSize() != m_Scales.GetSize())
  {
    std::ostringstream oss;
    oss << "InvariantPointCalibrationCostFunction::ValidateSizeOfScalesArray given " << parameters.GetSize() << " parameters, but the scale factors array has " << m_Scales.GetSize() << " parameters." << std::endl;
    mitkThrow() << oss.str();
  }
}



//-----------------------------------------------------------------------------
InvariantPointCalibrationCostFunction::MeasureType InvariantPointCalibrationCostFunction::GetValue(const ParametersType & parameters) const
{
  assert(m_PointData);
  assert(m_TrackingData);

  this->ValidateSizeOfParametersArray(parameters);

  MeasureType value;
  value.SetSize(m_NumberOfValues);

  cv::Matx44d similarityTransformation = this->GetCalibrationTransformation(parameters);
  cv::Matx44d translationTransformation = this->GetTranslationTransformation(parameters);
  TimeStampType lag = this->GetLag(parameters);
  TimeStampType timeStamp = 0;

  for (unsigned int i = 0; i < this->m_PointData->size(); i++)
  {
    timeStamp = (*this->m_PointData)[i].first;
    timeStamp -= lag;

    cv::Matx44d trackingTransformation = m_TrackingData->InterpolateMatrix(timeStamp);
    cv::Matx44d combinedTransformation = translationTransformation * (trackingTransformation * (similarityTransformation));
    cv::Matx41d point, transformedPoint;

    point(0,0) = (*this->m_PointData)[i].second.x;
    point(1,0) = (*this->m_PointData)[i].second.y;
    point(2,0) = (*this->m_PointData)[i].second.z;
    point(3,0) = 1;

    transformedPoint = combinedTransformation * point;

    value[i*3 + 0] = transformedPoint(0, 0) - m_InvariantPoint[0];
    value[i*3 + 1] = transformedPoint(1, 0) - m_InvariantPoint[1];
    value[i*3 + 2] = transformedPoint(2, 0) - m_InvariantPoint[2];
  }

  double residual = this->GetResidual(value);
  std::cout << "UltrasoundPinCalibrationCostFunction::GetValue(" << parameters << ") = " << residual << std::endl;

  return value;
}


//-----------------------------------------------------------------------------
} // end namespace
