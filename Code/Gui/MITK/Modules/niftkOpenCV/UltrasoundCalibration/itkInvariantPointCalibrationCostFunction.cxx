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
: m_OptimiseInvariantPoint(true)
, m_TimingLag(0)
, m_OptimiseTimingLag(false)
, m_OptimiseRigidTransformation(true)
, m_NumberOfValues(1)
, m_NumberOfParameters(1)
, m_PointData(NULL)
, m_TrackingData(NULL)
, m_Verbose(false)
{
  m_InvariantPoint[0] = 0;
  m_InvariantPoint[1] = 0;
  m_InvariantPoint[2] = 0;

  m_RigidTransformation.clear();
  for (unsigned int i = 0; i < 6; i++)
  {
    m_RigidTransformation.push_back(0);
  }
}


//-----------------------------------------------------------------------------
InvariantPointCalibrationCostFunction::~InvariantPointCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
unsigned int InvariantPointCalibrationCostFunction::GetNumberOfValues(void) const
{
  return m_NumberOfValues;
}


//-----------------------------------------------------------------------------
unsigned int InvariantPointCalibrationCostFunction::GetNumberOfParameters(void) const
{
  return m_NumberOfParameters;
}



//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::SetRigidTransformation(const cv::Matx44d& rigidBodyTrans)
{
  cv::Matx33d rotationMatrix;
  cv::Matx31d rotationVector;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      rotationMatrix(i,j) = rigidBodyTrans(i,j);
    }
  }
  cv::Rodrigues(rotationMatrix, rotationVector);

  m_RigidTransformation.clear();
  m_RigidTransformation.push_back(rotationVector(0,0));
  m_RigidTransformation.push_back(rotationVector(1,0));
  m_RigidTransformation.push_back(rotationVector(2,0));
  m_RigidTransformation.push_back(rigidBodyTrans(0,3));
  m_RigidTransformation.push_back(rigidBodyTrans(1,3));
  m_RigidTransformation.push_back(rigidBodyTrans(2,3));

  this->Modified();
}


//-----------------------------------------------------------------------------
cv::Matx44d InvariantPointCalibrationCostFunction::GetRigidTransformation() const
{
  assert(m_RigidTransformation.size() == 6);

  cv::Matx44d result;
  mitk::MakeIdentity(result);

  cv::Matx33d rotationMatrix;
  cv::Matx31d rotationVector;

  rotationVector(0, 0) = m_RigidTransformation[0];
  rotationVector(1, 0) = m_RigidTransformation[1];
  rotationVector(2, 0) = m_RigidTransformation[2];
  cv::Rodrigues(rotationVector, rotationMatrix);

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      result(i,j) = rotationMatrix(i,j);
    }
    result(i,3) = m_RigidTransformation[i+3];
  }
  return result;
}


//-----------------------------------------------------------------------------
std::vector<double> InvariantPointCalibrationCostFunction::GetRigidTransformationParameters() const
{
  return m_RigidTransformation;
}


//-----------------------------------------------------------------------------
void InvariantPointCalibrationCostFunction::SetRigidTransformationParameters(const std::vector<double>& params)
{
  m_RigidTransformation = params;
  this->Modified();
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
    if (norm != 0)
    {
      norm = sqrt(norm);

      for (unsigned int i = 0; i < m_NumberOfParameters; i++)
      {
        derivative[i][j] = derivative[i][j]*m_Scales[i]/norm;
      }
    }
  }
}


//-----------------------------------------------------------------------------
cv::Matx44d InvariantPointCalibrationCostFunction::GetRigidTransformation(const ParametersType & parameters) const
{
  cv::Matx44d rigidTransformation;
  mitk::MakeIdentity(rigidTransformation);

  if (parameters.GetSize() >= 6 && this->GetOptimiseRigidTransformation())
  {
    rigidTransformation  = mitk::ConstructRodriguesTransformationMatrix(
      parameters[0],
      parameters[1],
      parameters[2],
      parameters[3],
      parameters[4],
      parameters[5]
      );
  }
  else
  {
    rigidTransformation = mitk::ConstructRodriguesTransformationMatrix(
      m_RigidTransformation[0],
      m_RigidTransformation[1],
      m_RigidTransformation[2],
      m_RigidTransformation[3],
      m_RigidTransformation[4],
      m_RigidTransformation[5]
      );
  }

  return rigidTransformation;
}


//-----------------------------------------------------------------------------
cv::Matx44d InvariantPointCalibrationCostFunction::GetTranslationTransformation(const ParametersType & parameters) const
{
  cv::Matx44d translateToInvariantPoint;
  mitk::MakeIdentity(translateToInvariantPoint);

  if (parameters.GetSize() >= 9 && this->GetOptimiseInvariantPoint())
  {
    translateToInvariantPoint(0,3) = -parameters[6];
    translateToInvariantPoint(1,3) = -parameters[7];
    translateToInvariantPoint(2,3) = -parameters[8];
  }
  else
  {
    translateToInvariantPoint(0,3) = -m_InvariantPoint[0];
    translateToInvariantPoint(1,3) = -m_InvariantPoint[1];
    translateToInvariantPoint(2,3) = -m_InvariantPoint[2];
  }

  return translateToInvariantPoint;
}


//-----------------------------------------------------------------------------
double InvariantPointCalibrationCostFunction::GetLag(const ParametersType & parameters) const
{
  double lag = 0;
  if (this->GetOptimiseTimingLag())
  {
    if (this->GetNumberOfParameters() == 1)
    {
      lag = parameters[0];
    }
    else if (this->GetNumberOfParameters() == 10)
    {
      lag = parameters[9];
    }
    else if (this->GetNumberOfParameters() == 12)
    {
      lag = parameters[11];
    }
    else
    {
      mitkThrow() << "Cannot optimise the lag, with " << this->GetNumberOfParameters() << " parameters." << std::endl;
    }
  }
  else
  {
    lag = m_TimingLag;
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
  m_NumberOfValues = pointData->size() * 3;
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

  double lag = this->GetLag(parameters); // seconds
  long long lagInNanoSeconds = static_cast<long long>(lag*1000000000);

  TimeStampType timeStamp = 0;
  TimeStampType timingError = 0;
  bool inBounds;

  for (unsigned int i = 0; i < this->m_PointData->size(); i++)
  {
    timeStamp = (*this->m_PointData)[i].first;
    timeStamp -= lagInNanoSeconds;
    
    cv::Matx44d trackingTransformation = m_TrackingData->InterpolateMatrix(timeStamp, timingError, inBounds );
    cv::Matx44d combinedTransformation = translationTransformation * (trackingTransformation * (similarityTransformation));

    cv::Matx41d point;
    cv::Matx41d residual;
    cv::Matx41d pointInWorld;

    point(0,0) = (*this->m_PointData)[i].second.x;
    point(1,0) = (*this->m_PointData)[i].second.y;
    point(2,0) = (*this->m_PointData)[i].second.z;
    point(3,0) = 1;

    pointInWorld = (trackingTransformation * similarityTransformation) * point;
    residual = translationTransformation * pointInWorld;
  
    value[i*3 + 0] = residual(0, 0);
    value[i*3 + 1] = residual(1, 0);
    value[i*3 + 2] = residual(2, 0);
  }

  if (m_Verbose)
  {
    double residual = this->GetResidual(value);
    std::cout << "InvariantPointCalibrationCostFunction::GetValue(";
    for (int j = 0; j < parameters.GetSize(); j++)
    {
      std::cout << parameters[j];
      if (j != (parameters.GetSize() -1))
      {
        std::cout << ", ";
      }
    }
    std::cout << ") = " << residual << std::endl;
  }

  return value;
}


//-----------------------------------------------------------------------------
} // end namespace
